import os
from pathlib import Path
from numpy import average
import torch
from torch.utils.data import DataLoader
from dataset import DenoiseDataset
from model.dncnn import DnCNN
import torch.nn as nn
from tqdm import tqdm

def train(train_noisy_files, train_clean_files, num_epochs, save_model_path, noise_level):
    #加载数据
    train_dataset = DenoiseDataset(train_noisy_files, train_clean_files)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    #初始化模型
    model = DnCNN(channels=1, num_layers=10, features=64).cuda()

    #损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
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
    torch.save(model.state_dict(), f'{save_model_path}/model_{noise_level}%noise.pth')






def main():
    
    save_model_path = 'model'
    epochs = 100
    noise_level = 2    #train不同模型时需要更改
    train_noisy_path = f'data/{noise_level}_percent_noise/'
    train_clean_path = 'data/gt/'
    train_noisy_files = []
    train_clean_files = []
    for file in os.listdir(train_clean_path):
        train_clean_files.append(train_clean_path+file)
        train_noisy_files.append(train_noisy_path+Path(Path(file).stem).stem+f'_{noise_level}%noise.nii.gz')  #Path(file).stem获取文件名，Path(file).suffix获取文件后缀
    train(train_noisy_files, train_clean_files, epochs, save_model_path, noise_level)

if __name__ == '__main__':
    main()