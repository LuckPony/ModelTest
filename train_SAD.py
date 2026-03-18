import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# 【核心修改 1】：导入最新的数据集和交替式 AS-2.5D 网络
from Dataset.dataset_SAD import DenoiseDataset, ValDataset 
from model.dncnn_SAD import AS_Alternating_2p5D

import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import nibabel as nib
from datetime import datetime
import numpy as np

def plot_loss(loss_list, noise_level, save_loss_dir):
    plt.plot(range(1, len(loss_list)+1), loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss of AS-2.5D Net on {noise_level}% noise')
    plt.savefig(f'{save_loss_dir}/{noise_level}p_noise_loss_{len(loss_list)}epochs.png')
    plt.close()

# 【核心修改 2】：增加 shell_name 参数，防止不同壳层的图片互相覆盖
def plot_val(clean_slice, noisy_slice, denoised_slice, mask_slice, noise_level, epochs, shell_name, save_val_dir):
    psnr_noisy = masked_psnr(clean_slice, noisy_slice, mask_slice, data_range=2.0)
    psnr_denoised = masked_psnr(clean_slice, denoised_slice, mask_slice, data_range=2.0)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    imgs = [noisy_slice, denoised_slice, clean_slice]
    titles = [
        f"Noisy\nPSNR={psnr_noisy:.2f} dB",
        f"Denoised\nPSNR={psnr_denoised:.2f} dB",
        "Ground Truth"
    ]

    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

    plt.suptitle(f"Shell: {shell_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{save_val_dir}/{noise_level}p_noise_{epochs}epochs_{shell_name}.png')
    plt.close()

def masked_psnr(clean, denoised, mask, data_range=2.0):
    mse = ((clean - denoised) ** 2 * mask).sum() / (mask.sum() + 1e-8)
    return 10 * np.log10(data_range**2 / mse)


def train(train_noisy_files, train_clean_files, train_mask_files, train_bval_files, num_epochs, save_model_path, noise_level, val_noisy_files, val_clean_files, val_mask_files, val_bval_files, resume_checkpoint=None):
    folder_name = Path(save_model_path).name 
    
    save_val_dir = f'result/val/{folder_name}/'
    Path(save_val_dir).mkdir(parents=True, exist_ok=True)
    save_loss_dir = f'result/loss/{folder_name}/'
    Path(save_loss_dir).mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n======================================")
    print(f"🚀 AS-2.5D 模型训练开始！当前使用设备: {device}")
    print(f"======================================\n")
    
    train_dataset = DenoiseDataset(train_noisy_files, train_clean_files, train_mask_files, train_bval_files, patch_size=64)
    # 注意：Windows下如果报错，将 num_workers 保持为 0，保留 pin_memory 加速
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, pin_memory=True)
    
    val_dataset = ValDataset(val_noisy_files, val_clean_files, val_mask_files, val_bval_files)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # ==========================================
    # 初始化 AS-2.5D 模型
    # ==========================================
    model = AS_Alternating_2p5D(num_blocks=1, hidden_spatial=16).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    loss_log = []
    start_epoch = 0

    if resume_checkpoint is not None and os.path.isfile(resume_checkpoint):
        print(f"=> 发现检查点: '{resume_checkpoint}'，正在恢复训练...")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            loss_log = checkpoint.get('loss_log', [])
            print(f"=> 成功加载完整检查点！从 Epoch {start_epoch + 1} 继续。")
    else:
        print("=> 将从头开始训练角域-空域交替网络。")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0
        loop = tqdm(train_loader, leave=True, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for noisy, clean, mask in loop:
            noisy, clean, mask = noisy.to(device), clean.to(device), mask.to(device)
            optimizer.zero_grad()
            denoised = model(noisy)
            
            diff = (denoised - clean)**2
            
            # ==========================================
            # 【核心修改 3：壳层加权均衡 Loss】
            # 强制网络平等对待三个壳层的误差
            # ==========================================
            diff_b0 = diff[:, :18, :, :]
            diff_b1k = diff[:, 18:108, :, :]
            diff_b2k = diff[:, 108:, :, :]
            
            # mask 的形状是 (B, 1, H, W)，利用广播机制直接相乘
            mse_b0 = (diff_b0 * mask).sum() / (mask.sum() * 18 + 1e-8)
            mse_b1k = (diff_b1k * mask).sum() / (mask.sum() * 90 + 1e-8)
            mse_b2k = (diff_b2k * mask).sum() / (mask.sum() * 90 + 1e-8)
            
            # 三者等权重相加
            loss = mse_b0 + mse_b1k + mse_b2k
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            avg_loss = total_loss / (loop.n + 1)
            
            loop.set_postfix(epoch=epoch+1, loss=loss.item(), avg_loss=avg_loss, lr=optimizer.param_groups[0]['lr'])
            
        loss_log.append(avg_loss)
        scheduler.step()

        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss_log': loss_log
        }
        torch.save(checkpoint_state, f'{save_model_path}/checkpoint_latest.pth')

        # 每 10 个 epoch 出一次验证图 (如果想看频繁点可以改成 5)
        if (epoch+1) % 10 == 0:
            plot_loss(loss_log, noise_level, save_loss_dir)
            torch.save(checkpoint_state, f'{save_model_path}/checkpoint_epoch_{epoch+1}.pth')
            
            model.eval()
            with torch.no_grad():
                for noisy, clean, mask in val_loader:
                    noisy, clean, mask = noisy.to(device), clean.to(device), mask.to(device)
                    
                    denoised_out = model(noisy).cpu().numpy().squeeze(0) # Shape: (198, H, W)
                    clean_out = clean.cpu().numpy().squeeze(0)
                    mask_vis = mask.cpu().numpy().squeeze(0)[0, :, :] 
                    
                    # 取出输入的中心层用于对比画图: noisy 是 [B, 3, 198, H, W]
                    noisy_center = noisy[:, 1, :, :, :].cpu().numpy().squeeze(0) 

                    # ==========================================
                    # 【核心修改 4：同时输出三个壳层的代表方向图像！】
                    # ==========================================
                    vis_shells = [("b0", 0), ("b1000", 18 + 45), ("b2000", 108 + 45)]
                    
                    for shell_name, idx in vis_shells:
                        n_v = noisy_center[idx, :, :]
                        c_v = clean_out[idx, :, :]
                        d_v = denoised_out[idx, :, :] * mask_vis + (-1) * (1 - mask_vis)
                        
                        plot_val(c_v, n_v, d_v, mask_vis, noise_level, epoch+1, shell_name, save_val_dir)

    torch.save(model.state_dict(), f'{save_model_path}/model_{noise_level}%noise_final.pth')

def main():
    epochs = 100
    noise_level = 2
    
    is_resume = False 
    resume_folder = "" 
    
    if is_resume:
        folder_name = resume_folder
        resume_ckpt = f'result/model/{folder_name}/checkpoint_latest.pth'
    else:
        # 文件夹命名更新为 AS2p5D
        folder_name = datetime.now().strftime("%Y%m%d%H%M")+f'_AS2p5D_{noise_level}p'
        resume_ckpt = None
        
    save_model_path = f'result/model/{folder_name}/'
    Path(save_model_path).mkdir(parents=True, exist_ok=True)
    
    train_noisy_path = f'data/{noise_level}_percent_noise/' 
    train_clean_path = 'data/gt/'
    train_mask_path = 'data/mask/'
    train_bval_path = 'data/bvals/'
    
    train_noisy_files, train_clean_files, train_mask_files, train_bval_files = [], [], [], []
    
    for file in os.listdir(train_clean_path):
        train_clean_files.append(train_clean_path+file)
        train_noisy_files.append(train_noisy_path+Path(Path(file).stem).stem+f'_{noise_level}%noise.nii.gz')
        train_mask_files.append(train_mask_path+Path(Path(file).stem).stem+f'_mask.nii.gz')
        train_bval_files.append(train_bval_path+Path(Path(file).stem).stem+'_bval') 
        
    val_noisy_files = [f"data/val/sub-125525__dwi_filtered_{noise_level}%noise.nii.gz"]
    val_clean_files = ["data/val/sub-125525__dwi_filtered.nii.gz"]
    val_mask_files = ["data/val/sub-125525__dwi_filtered_mask.nii.gz"]
    val_bval_files = ["data/val/sub-125525__dwi_filtered_bval"]

    train(train_noisy_files, train_clean_files, train_mask_files, train_bval_files, epochs, save_model_path, noise_level, val_noisy_files, val_clean_files, val_mask_files, val_bval_files, resume_checkpoint=resume_ckpt)

if __name__ == '__main__':
    main()