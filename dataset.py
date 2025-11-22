import random
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
class DenoiseDataset(Dataset):
    def __init__(self,noisy_files, clean_files, patch_size=64):
        self.noisy_img = []
        self.clean_img = []
        self.patch = patch_size
        #预加载所有数据
        for n__file,c_file in zip(noisy_files,clean_files):
            noisy_4d = nib.load(n__file).get_fdata()  #加载4D图像# pyright: ignore[reportPrivateImportUsage, reportAttributeAccessIssue]
            clean_4d = nib.load(c_file).get_fdata()  # pyright: ignore[reportPrivateImportUsage, reportAttributeAccessIssue]
            self.max = noisy_4d.max()
            self.min = noisy_4d.min()
            self.noisy_img.append(noisy_4d)
            self.clean_img.append(clean_4d)
            if noisy_4d.ndim != 4:
                raise ValueError("Noisy image should have 4 dimensions")
        #根据noisy图像构建对应的(slice,direction)对
        self.index_list = [] #(file_id, direction, slice_id)
        for f_id, img in enumerate(self.noisy_img):
            D, H, W, Dd = img.shape
            for d in range(Dd):
                for z in range(D):
                    self.index_list.append((f_id, d, z))

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        # print("执行数据加载...")
        file_id, direction, slice_id = self.index_list[idx]
        noisy_4d = self.noisy_img[file_id]
        clean_4d = self.clean_img[file_id]
        
        #由于图像是4维的，固定方向和切片，提取2D图像
        noisy = noisy_4d[slice_id, ..., direction]   #shape: D,(H,W),Dd
        clean = clean_4d[slice_id, ..., direction]   #shape: D,(H,W),Dd
        #归一化到[-1,1]
        noisy = (noisy - self.min) / (self.max - self.min)
        clean = (clean - self.min) / (self.max - self.min)
        noisy = 2*noisy - 1
        clean = 2*clean - 1

        #随机裁剪2D patch
        H, W = noisy.shape
        p = self.patch
        if H < p or W < p:
            raise ValueError(f"Patch size {p} is larger than image size {H}x{W}")
        #随机选择起点
        dy = random.randint(0, H - p)
        dx = random.randint(0, W - p)
        noisy = noisy[dy:dy+p, dx:dx+p]
        clean = clean[dy:dy+p, dx:dx+p]

        #转换成tensor(1,H, W)
        noisy = torch.tensor(noisy, dtype=torch.float32).unsqueeze(0)#增加channel维度(batch维度会在DataLoader的时候新增一维)，保持Conv3D格式
        clean = torch.tensor(clean, dtype=torch.float32).unsqueeze(0)
        
        return noisy, clean
    
class ValDataset(Dataset):
    def __init__(self, noisy_files, clean_files,patch=64):
        self.noisy_files = noisy_files
        self.clean_files = clean_files
        noisy_4d = nib.load(self.noisy_files[0]).get_fdata()  #pyright: ignore[reportPrivateImportUsage, reportAttributeAccessIssue]
        self.max = noisy_4d.max()
        self.min = noisy_4d.min()
        self.patch = patch
        
        
        
            
    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy_4d = nib.load(self.noisy_files[0]).get_fdata() #pyright: ignore[reportPrivateImportUsage, reportAttributeAccessIssue]
        clean_4d = nib.load(self.clean_files[0]).get_fdata() #pyright: ignore[reportPrivateImportUsage, reportAttributeAccessIssue]
        #验证只选取第一个方向的中间切片，不做patch
        direction = random.randint(0,noisy_4d.shape[3]-1)  #每次输出随机方向
        noisy = noisy_4d[noisy_4d.shape[0]//2, :, :, direction]
        clean = clean_4d[clean_4d.shape[0]//2, :, :, direction]

        #归一化到[-1,1]
        noisy = (noisy - self.min) / (self.max - self.min)
        clean = (clean - self.min) / (self.max - self.min)
        noisy = 2*noisy - 1
        clean = 2*clean - 1
        #随机裁剪2D patch
        H, W = noisy.shape
        p = self.patch
        if H < p or W < p:
            raise ValueError(f"Patch size {p} is larger than image size {H}x{W}")
        #随机选择起点
        dy = random.randint(0, H - p)
        dx = random.randint(0, W - p)
        noisy = noisy[dy:dy+p, dx:dx+p]
        clean = clean[dy:dy+p, dx:dx+p]
        #转换成tensor(1,H, W)
        noisy = torch.tensor(noisy, dtype=torch.float32).unsqueeze(0)#增加channel维度(batch维度会在DataLoader的时候新增一维)，保持Conv3D格式
        clean = torch.tensor(clean, dtype=torch.float32).unsqueeze(0)
        
        return noisy, clean