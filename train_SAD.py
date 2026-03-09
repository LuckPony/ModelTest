import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# 【修改为导入解耦架构的网络和Dataset】
from Dataset.dataset_SAD import DenoiseDataset, ValDataset 
from model.dncnn_SAD import DnCNN_Decoupled

import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.metrics import peak_signal_noise_ratio as psnr
from datetime import datetime
import numpy as np

def plot_loss(loss_list, noise_level, save_loss_dir):
    plt.plot(range(1, len(loss_list)+1), loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss of Decoupled Net on {noise_level}% noise')
    plt.savefig(f'{save_loss_dir}/{noise_level}p_noise_loss_{len(loss_list)}epochs.png')
    plt.close()

def plot_val(clean_slice, noisy_slice, denoised_slice, mask_slice, noise_level, epochs, save_val_dir):
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

    plt.tight_layout()
    plt.savefig(f'{save_val_dir}/{noise_level}p_noise_{epochs}epochs.png')
    plt.close()

def masked_psnr(clean, denoised, mask, data_range=2.0):
    # 这里处理单张 2D 切片的 PSNR 计算
    mse = ((clean - denoised) ** 2 * mask).sum() / (mask.sum() + 1e-8)
    return 10 * np.log10(data_range**2 / mse)

def train(train_noisy_files, train_clean_files, train_mask_files, num_epochs, save_model_path, noise_level, val_noisy_files, val_clean_files, val_mask_files, resume_checkpoint=None):
    folder_name = Path(save_model_path).name # 修复时间戳不同步的 bug
    
    save_val_dir = f'result/val/{folder_name}/'
    Path(save_val_dir).mkdir(parents=True, exist_ok=True)
    save_loss_dir = f'result/loss/{folder_name}/'
    Path(save_loss_dir).mkdir(parents=True, exist_ok=True)
    
    train_dataset = DenoiseDataset(train_noisy_files, train_clean_files, train_mask_files, patch_size=64)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataset = ValDataset(val_noisy_files, val_clean_files, val_mask_files)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # ==========================================
    # 初始化解耦模型 (固定 288 个方向)
    # ==========================================
    model = DnCNN_Decoupled(num_directions=288, num_layers=12, features=64).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    loss_log = []
    start_epoch = 0

    # 断点续训逻辑
    if resume_checkpoint is not None and os.path.isfile(resume_checkpoint):
        print(f"=> 发现检查点: '{resume_checkpoint}'，正在恢复训练...")
        checkpoint = torch.load(resume_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            loss_log = checkpoint.get('loss_log', [])
            print(f"=> 成功加载完整检查点！从 Epoch {start_epoch + 1} 继续。")
    else:
        print("=> 将从头开始训练解耦网络。")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0
        loop = tqdm(train_loader, leave=True, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for noisy, clean, mask in loop:
            noisy, clean, mask = noisy.cuda(), clean.cuda(), mask.cuda()
            optimizer.zero_grad()
            denoised = model(noisy)
            
            diff = (denoised - clean)**2
            
            # 【核心修改】：因为误差是在 288 个通道上累加的，分母必须乘 288
            loss = (diff * mask).sum() / (mask.sum() * 288 + 1e-8)
            
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

        if (epoch+1) % 10 == 0:
            plot_loss(loss_log, noise_level, save_loss_dir)
            torch.save(checkpoint_state, f'{save_model_path}/checkpoint_epoch_{epoch+1}.pth')
            
            model.eval()
            with torch.no_grad():
                for noisy, clean, mask in val_loader:
                    noisy, clean, mask = noisy.cuda(), clean.cuda(), mask.cuda()
                    denoised = model(noisy).cpu().numpy().squeeze(0) # Shape: (288, H, W)
                    clean = clean.cpu().numpy().squeeze(0)
                    mask = mask.cpu().numpy().squeeze(0)

                    # 【核心修改】：画图时固定抽取第 144 个梯度方向进行可视化对比
                    vis_idx = 144
                    noisy_np = noisy.cpu().numpy().squeeze(0)
                    
                    noisy_vis = noisy_np[vis_idx, :, :]
                    clean_vis = clean[vis_idx, :, :]
                    denoised_vis = denoised[vis_idx, :, :]
                    
                    mask_vis = mask[0, :, :] # mask shape is (1, H, W)
                    
                    denoised_vis = denoised_vis * mask_vis + (-1) * (1 - mask_vis)
                    plot_val(clean_vis, noisy_vis, denoised_vis, mask_vis, noise_level, epoch+1, save_val_dir)

    torch.save(model.state_dict(), f'{save_model_path}/model_{noise_level}%noise_final.pth')

def main():
    epochs = 100
    noise_level = 6
    
    # 填入独立的时间戳或指定续训的文件夹
    is_resume = False 
    resume_folder = "20260309_decoupled" 
    
    if is_resume:
        folder_name = resume_folder
        resume_ckpt = f'result/model/{folder_name}/checkpoint_latest.pth'
    else:
        folder_name = datetime.now().strftime("%Y%m%d%H%M")+f'_decoupled_{noise_level}p'
        resume_ckpt = None
        
    save_model_path = f'result/model/{folder_name}/'
    Path(save_model_path).mkdir(parents=True, exist_ok=True)
    
    train_noisy_path = f'data/{noise_level}_percent_noise/' 
    train_clean_path = 'data/gt/'
    train_mask_path = 'data/mask/'
    train_noisy_files, train_clean_files, train_mask_files = [], [], []
    
    for file in os.listdir(train_clean_path):
        train_clean_files.append(train_clean_path+file)
        train_noisy_files.append(train_noisy_path+Path(Path(file).stem).stem+f'_{noise_level}%noise.nii.gz')
        train_mask_files.append(train_mask_path+Path(Path(file).stem).stem+f'_mask.nii.gz')
        
    val_noisy_files = train_noisy_files
    val_clean_files = train_clean_files
    val_mask_files = train_mask_files

    train(train_noisy_files, train_clean_files, train_mask_files, epochs, save_model_path, noise_level, val_noisy_files, val_clean_files, val_mask_files, resume_checkpoint=resume_ckpt)

if __name__ == '__main__':
    main()