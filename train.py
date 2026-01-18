import os
from pathlib import Path
from numpy import average
import torch
from torch.utils.data import DataLoader
from dataset import DenoiseDataset, ValDataset
from model.dncnn import DnCNN
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.metrics import peak_signal_noise_ratio as psnr
from pathlib import Path
from datetime import datetime
import numpy as np
def plot_loss(loss_list,noise_level,save_loss_dir):
    plt.plot(range(1,len(loss_list)+1),loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss of DnCNN on {noise_level}% noise')
    plt.savefig(f'{save_loss_dir}/{noise_level}p_noise_loss_{len(loss_list)}epochs.png')
    plt.close() #画完图后关闭画布，防止与下次绘图交叉影响

def plot_val(clean_slice, noisy_slice, denoised_slice,mask_slice, noise_level,epochs, save_val_dir):
    psnr_noisy = masked_psnr(clean_slice, noisy_slice,mask_slice, data_range=2.0)
    psnr_denoised = masked_psnr(clean_slice, denoised_slice,mask_slice, data_range=2.0)
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
    plt.close() #画完图后关闭画布，防止与下次绘图交叉影响
def masked_psnr(clean, denoised, mask, data_range=2.0):
    mse = ((clean - denoised) ** 2 * mask).sum() / (mask.sum() + 1e-8)
    return 10 * np.log10(data_range**2 / mse)


def train(train_noisy_files, train_clean_files, train_mask_files, num_epochs, save_model_path, noise_level,val_noisy_files, val_clean_files, val_mask_files):
    #设置保存路径
    folder_name = datetime.now().strftime("%Y%m%d%H%M")
    save_val_dir = f'result/val/{folder_name}/'
    Path(save_val_dir).mkdir(parents=True, exist_ok=True)
    save_loss_dir = f'result/loss/{folder_name}/'
    Path(save_loss_dir).mkdir(parents=True, exist_ok=True)
    #加载数据
    train_dataset = DenoiseDataset(train_noisy_files, train_clean_files, train_mask_files, patch_size=64)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataset = ValDataset(val_noisy_files, val_clean_files, val_mask_files)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    #初始化模型
    model = DnCNN(channels=1, num_layers=12, features=64).cuda()

    #损失函数和优化器
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_log = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        avg_loss = 0.0
        loop = tqdm(train_loader,leave=True, desc=f'Epoch {epoch+1}/{num_epochs}')
        for noisy,clean,mask in loop:
            noisy, clean, mask = noisy.cuda(), clean.cuda(), mask.cuda()
            optimizer.zero_grad()
            denoised = model(noisy)
            diff = (denoised - clean)**2
            loss = (diff * mask).sum() / (mask.sum() + 1e-8)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            avg_loss = total_loss / (loop.n + 1)
            
            loop.set_postfix(
                epoch=epoch+1,
                loss=loss.item(),
                avg_loss=avg_loss 
            )
        loss_log.append(avg_loss)
        if (epoch+1) % 10 == 0:
            plot_loss(loss_log, noise_level, save_loss_dir)
            print(f'Evaluate after epoch {epoch+1}...')
            model.eval()
            with torch.no_grad():
                    
                    for noisy, clean, mask in val_loader:
                        noisy, clean, mask = noisy.cuda(), clean.cuda(), mask.cuda()
                        denoised = model(noisy).cpu().numpy().squeeze()
                        clean = clean.cpu().numpy().squeeze()
                        noisy = noisy.cpu().numpy().squeeze()
                        mask = mask.cpu().numpy().squeeze()
                        # denoised = denoised * mask   #背景区域设为0
                        denoised = denoised * mask + (-1) * (1 - mask)  # 背景区域设置为-1（黑色）
                        plot_val(clean, noisy, denoised,mask, noise_level,epoch+1, save_val_dir)
                            

                            

    torch.save(model.state_dict(), f'{save_model_path}/model_{noise_level}%noise.pth')
    print(f'Model saved at {save_model_path}/model_{noise_level}p_noise.pth')

def main():
    folder_name = datetime.now().strftime("%Y%m%d%H%M")
    save_model_path = f'result/model/{folder_name}/'
    Path(save_model_path).mkdir(parents=True, exist_ok=True)
    epochs = 100
    noise_level = 4    #train不同模型时需要更改
    train_noisy_path = f'data/{noise_level}_percent_noise/'  #这里只填写路径，不需要文件名
    train_clean_path = 'data/gt/'
    train_mask_path = 'data/mask/'
    train_noisy_files = []
    train_clean_files = []
    train_mask_files = []
    for file in os.listdir(train_clean_path):
        train_clean_files.append(train_clean_path+file)
        train_noisy_files.append(train_noisy_path+Path(Path(file).stem).stem+f'_{noise_level}%noise.nii.gz')  #Path(file).stem获取文件名，Path(file).suffix获取文件后缀
        train_mask_files.append(train_mask_path+Path(Path(file).stem).stem+f'_mask.nii.gz')
    #这里暂时设置验证集和训练集一样，后续需要更改
    val_noisy_files = train_noisy_files
    val_clean_files = train_clean_files
    val_mask_files = train_mask_files

    train(train_noisy_files, train_clean_files, train_mask_files, epochs, save_model_path, noise_level, val_noisy_files, val_clean_files, val_mask_files)

if __name__ == '__main__':
    main()