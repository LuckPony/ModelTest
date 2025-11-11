import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
class DenoiseDataset(Dataset):
    def __init__(self,noisy_files, clean_files, patch_size=32):
        self.noisy_files = noisy_files
        self.clean_files = clean_files
        self.patch = patch_size

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        # print("执行数据加载...")
        noisy = nib.load(self.noisy_files[idx]).get_fdata()  # pyright: ignore[reportPrivateImportUsage, reportAttributeAccessIssue]
        clean = nib.load(self.clean_files[idx]).get_fdata()  # pyright: ignore[reportPrivateImportUsage, reportAttributeAccessIssue]
        # print("noisy shape: ", noisy.shape)
        # print("clean shape: ", clean.shape)

        #由于图像是4维的，先忽略通道维度，逐方向运行
        noisy = noisy[..., 0]   #shape: (D,H,W)
        clean = clean[..., 0]   #shape: (D,H,W)
        

        #归一化
        # print("执行归一化...")
        noisy = (noisy - noisy.min()) / (noisy.max() - noisy.min())
        clean = (clean - clean.min()) / (clean.max() - clean.min())
        noisy = 2*noisy - 1
        clean = 2*clean - 1

        #转换成tensor
        noisy = torch.tensor(noisy, dtype=torch.float32).unsqueeze(0)#增加channel维度(batch维度会在DataLoader的时候新增一维)，保持Conv3D格式
        clean = torch.tensor(clean, dtype=torch.float32).unsqueeze(0)
        
        return noisy, clean