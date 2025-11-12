import os
from pathlib import Path
from numpy import average
import torch
from torch.utils.data import DataLoader
from dataset import DenoiseDataset
from model.dncnn import DnCNN
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.metrics import peak_signal_noise_ratio as psnr


def plot_loss(loss_list,noise_level):
    plt.plot(range(1,len(loss_list)+1),loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss of DnCNN on {noise_level}% noise')
    plt.savefig(f'result/loss/{noise_level}%noise_loss_{len(loss_list)}epochs.png')
    plt.close() #画完图后关闭画布，防止与下次绘图交叉影响

def plot_val(clean_slice, noisy_slice, denoised_slice, noise_level,epochs):
    psnr_noisy = psnr(clean_slice, noisy_slice, data_range=clean_slice.max() - clean_slice.min())
    psnr_denoised = psnr(clean_slice, denoised_slice, data_range=clean_slice.max() - clean_slice.min())
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
    plt.savefig(f'result/val/{noise_level}%noise_{epochs}epochs.png')
    plt.close() #画完图后关闭画布，防止与下次绘图交叉影响

def train(train_noisy_files, train_clean_files, num_epochs, save_model_path, noise_level,val_noisy_files, val_clean_files):
    #加载数据
    train_dataset = DenoiseDataset(train_noisy_files, train_clean_files)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    #初始化模型
    model = DnCNN(channels=1, num_layers=10, features=64).cuda()

    #损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_log = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        avg_loss = 0.0
        loop = tqdm(train_loader,leave=True, desc=f'Epoch {epoch+1}/{num_epochs}')
        for noisy,clean in loop:
            noisy, clean = noisy.cuda(), clean.cuda()
            optimizer.zero_grad()
            denoised = model(noisy)
            loss = criterion(denoised, clean)
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
            plot_loss(loss_log, noise_level)
            print(f'Evaluate after epoch {epoch+1}...')
            model.eval()
            with torch.no_grad():
                if val_clean_files and val_noisy_files:
                    val_dataset = DenoiseDataset(val_noisy_files, val_clean_files)
                    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
                    for noisy, clean in val_loader:
                        noisy, clean = noisy.cuda(), clean.cuda()
                        denoised = model(noisy).cpu().numpy().squeeze()
                        clean = clean.cpu().numpy().squeeze()
                        noisy = noisy.cpu().numpy().squeeze()
                        slice = noisy.shape[2]//2
                        denoised_slice = denoised[...,slice]
                        clean_slice = clean[...,slice]
                        noisy_slice = noisy[...,slice]
                        plot_val(clean_slice, noisy_slice, denoised_slice, noise_level,epoch+1)
                            

                            

    torch.save(model.state_dict(), f'{save_model_path}/model_{noise_level}%noise.pth')
    print(f'Model saved at {save_model_path}/model_{noise_level}%noise.pth')

def main():
    
    save_model_path = 'result/model'
    epochs = 50
    noise_level = 2    #train不同模型时需要更改
    train_noisy_path = f'data/{noise_level}_percent_noise/'  #这里只填写路径，不需要文件名
    train_clean_path = 'data/gt/'
    train_noisy_files = []
    train_clean_files = []
    for file in os.listdir(train_clean_path):
        train_clean_files.append(train_clean_path+file)
        train_noisy_files.append(train_noisy_path+Path(Path(file).stem).stem+f'_{noise_level}%noise.nii.gz')  #Path(file).stem获取文件名，Path(file).suffix获取文件后缀
    #这里暂时设置验证集和训练集一样，后续需要更改
    val_noisy_files = train_noisy_files
    val_clean_files = train_clean_files
    train(train_noisy_files, train_clean_files, epochs, save_model_path, noise_level, val_noisy_files, val_clean_files)

if __name__ == '__main__':
    main()