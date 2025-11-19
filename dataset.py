import random
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
class DenoiseDataset(Dataset):
    def __init__(self,noisy_files, clean_files, patch_size=32):
        self.samples = []  #保存(noisy_file_path, clean_file_path, direction_idx)
        self.patch = patch_size

        for n__file,c_file in zip(noisy_files,clean_files):
            noisy_img = nib.load(n__file).get_fdata()  # pyright: ignore[reportPrivateImportUsage, reportAttributeAccessIssue]
            clean_img = nib.load(c_file).get_fdata()  # pyright: ignore[reportPrivateImportUsage, reportAttributeAccessIssue]
            if noisy_img.ndim != 4:
                raise ValueError("Noisy image should have 4 dimensions")
            D,H,W,Dd = noisy_img.shape
            #将每个方向单独作为一个样本
            for d in range(Dd):
                self.samples.append((n__file,c_file,d))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # print("执行数据加载...")
        noisy_path, clean_path, direction_idx = self.samples[idx]
        noisy = nib.load(noisy_path).get_fdata()  # pyright: ignore[reportPrivateImportUsage, reportAttributeAccessIssue]
        clean = nib.load(clean_path).get_fdata()  # pyright: ignore[reportPrivateImportUsage, reportAttributeAccessIssue]

        #由于图像是4维的，取某一个方向，逐方向运行
        noisy = noisy[...,direction_idx]   #shape: (D,H,W)
        clean = clean[..., direction_idx]   #shape: (D,H,W)

        #使用 noisy 的 99th percentile 做全局缩放，保留噪声结构
        scale = np.percentile(noisy, 99)
        if scale < 1e-6:
            scale = 1.0
        noisy = noisy / scale
        clean = clean / scale
        # #归一化到[-1,1]
        # noisy = (noisy - noisy.min()) / (noisy.max() - noisy.min())
        # clean = (clean - clean.min()) / (clean.max() - clean.min())
        # noisy = 2*noisy - 1
        # clean = 2*clean - 1

        #随机裁剪3D patch
        D, H, W = noisy.shape
        p = self.patch
        if D < p or H < p or W < p:
            raise ValueError(f"Patch size {p} is larger than image size {D}x{H}x{W}")
        #随机选择起点
        dz = random.randint(0, D - p)
        dy = random.randint(0, H - p)
        dx = random.randint(0, W - p)
        noisy = noisy[dz:dz+p, dy:dy+p, dx:dx+p]
        clean = clean[dz:dz+p, dy:dy+p, dx:dx+p]

        #转换成tensor(1, D, H, W)
        noisy = torch.tensor(noisy, dtype=torch.float32).unsqueeze(0)#增加channel维度(batch维度会在DataLoader的时候新增一维)，保持Conv3D格式
        clean = torch.tensor(clean, dtype=torch.float32).unsqueeze(0)
        
        return noisy, clean