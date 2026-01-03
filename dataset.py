import json
import random
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib

import save_normalize
class DenoiseDataset(Dataset):
    def __init__(self,noisy_files, clean_files,mask_files, patch_size=64):
        self.noisy_img = []
        self.clean_img = []
        self.mask_img = []    #为每个noisy文件和clean文件配一个mask文件
        self.patch = patch_size
        #预加载所有数据
        for n__file,c_file,m_file in zip(noisy_files,clean_files,mask_files):
            noisy_4d = nib.load(n__file).get_fdata()  #加载4D图像# pyright: ignore[reportPrivateImportUsage, reportAttributeAccessIssue]
            clean_4d = nib.load(c_file).get_fdata()  # pyright: ignore[reportPrivateImportUsage, reportAttributeAccessIssue]
            mask_3d = nib.load(m_file).get_fdata()
            self.noisy_img.append(noisy_4d)
            self.clean_img.append(clean_4d)
            self.mask_img.append(mask_3d)
            if noisy_4d.ndim != 4:
                raise ValueError("Noisy image should have 4 dimensions")
            if mask_3d.ndim != 3:
                raise ValueError("Mask image should have 3 dimensions")
        #根据noisy图像构建对应的(slice,direction)对
        self.index_list = [] #(file_id, direction, slice_id)
        save_normalize.save_normalize(self.noisy_img, save_path="normalize/train_minmax.json")
        for f_id, img in enumerate(self.noisy_img):
            D, H, W, Dd = img.shape
            for d in range(Dd):
                for z in range(D):
                    self.index_list.append((f_id, d, z))
        #读取保存的归一化最大最小值,放在init中预防重复读取
        with open("normalize/train_minmax.json", "r") as f:
            self.stat_dict = json.load(f)

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        # print("执行数据加载...")
        file_id, direction, slice_id = self.index_list[idx]
        noisy_4d = self.noisy_img[file_id]
        clean_4d = self.clean_img[file_id]
        mask_3d = self.mask_img[file_id]
        
        #由于图像是4维的，固定方向和切片，提取2D图像
        noisy = noisy_4d[slice_id, ..., direction]   #shape: D,(H,W),Dd
        clean = clean_4d[slice_id, ..., direction]   #shape: D,(H,W),Dd
        mask = mask_3d[slice_id,...]
        stat_dict = self.stat_dict[str(file_id)]
        max = stat_dict["noisy"]["max"]
        min = stat_dict["noisy"]["min"]
        #归一化到[-1,1]
        noisy = (noisy - min) / (max - min)
        clean = (clean - min) / (max - min)
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
        mask = mask[dy:dy+p, dx:dx+p]

        #转换成tensor(1,H, W)
        noisy = torch.tensor(noisy, dtype=torch.float32).unsqueeze(0)#增加channel维度(batch维度会在DataLoader的时候新增一维)，保持Conv3D格式
        clean = torch.tensor(clean, dtype=torch.float32).unsqueeze(0)
        mask = (mask>0).astype(np.float32)
        mask = torch.tensor(mask).unsqueeze(0)
        
        return noisy, clean, mask
    
class ValDataset(Dataset):
    def __init__(self, noisy_files, clean_files,mask_files,patch=64):
        self.noisy_files = noisy_files
        self.clean_files = clean_files
        self.mask_files = mask_files
        noisy_4d = nib.load(self.noisy_files[0]).get_fdata()  # pyright: ignore[reportPrivateImportUsage, reportAttributeAccessIssue]
        self.max = noisy_4d.max()
        self.min = noisy_4d.min()
        self.patch = patch
      
    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy_4d = nib.load(self.noisy_files[0]).get_fdata() # pyright: ignore[reportPrivateImportUsage, reportAttributeAccessIssue]
        clean_4d = nib.load(self.clean_files[0]).get_fdata() # pyright: ignore[reportPrivateImportUsage, reportAttributeAccessIssue]
        mask_3d = nib.load(self.mask_files[0]).get_fdata()
        #验证只选取随机方向的中间切片，不做patch
        direction = random.randint(0,noisy_4d.shape[3]-1)  #每次输出随机方向
        noisy = noisy_4d[noisy_4d.shape[0]//2, :, :, direction]
        clean = clean_4d[clean_4d.shape[0]//2, :, :, direction]
        mask = mask_3d[mask_3d.shape[0]//2,...]

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
        mask = mask[dy:dy+p, dx:dx+p]
        #转换成tensor(1,H, W)
        noisy = torch.tensor(noisy, dtype=torch.float32).unsqueeze(0)#增加channel维度(batch维度会在DataLoader的时候新增一维)，保持Conv3D格式
        clean = torch.tensor(clean, dtype=torch.float32).unsqueeze(0)
        mask = (mask>0).astype(np.float32)
        mask = torch.tensor(mask).unsqueeze(0)
        return noisy, clean, mask
class TestDataset(Dataset):
    def __init__(self, noisy_files, clean_files, mask_files):
        self.noisy_img = []
        self.clean_img = []
        self.mask_img = []
        for n_file, c_file, m_file in zip(noisy_files, clean_files, mask_files):
            noisy_4d = nib.load(n_file).get_fdata()  # pyright: ignore[reportPrivateImportUsage, reportAttributeAccessIssue]
            clean_4d = nib.load(c_file).get_fdata()  # pyright: ignore[reportPrivateImportUsage, reportAttributeAccessIssue]
            mask_3d = nib.load(m_file).get_fdata()
            if noisy_4d.ndim != 4 or clean_4d.ndim != 4:
                raise ValueError(f"Noisy image {n_file} does not have 4 dimensions")
            if mask_3d.ndim != 3:
                raise ValueError(f"Mask image {m_file} does not have 3 dimensions")
            self.noisy_img.append(noisy_4d)
            self.clean_img.append(clean_4d)
            self.mask_img.append(mask_3d)
        self.index_list = []
        save_normalize.save_normalize(self.noisy_img, save_path="normalize/test_minmax.json")
        for f_id, img in enumerate(self.noisy_img):
            D, H, W, Dd = img.shape
            for d in range(Dd):
                for z in range (D):
                    self.index_list.append((f_id, z, d))
        #读取保存的归一化最大最小值，防止重复读取
        with open("normalize/test_minmax.json","r") as f:
            self.stat_dict = json.load(f)

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        f_id, z, d = self.index_list[idx]
        noisy_4d = self.noisy_img[f_id]
        clean_4d = self.clean_img[f_id]
        mask_3d = self.mask_img[f_id]
        # 转换4d图像为2D切片
        noisy = noisy_4d[z, :, :, d]
        clean = clean_4d[z, :, :, d]
        mask = mask_3d[z, :, :]
        stat_dict = self.stat_dict[str(f_id)]
        max = stat_dict["noisy"]["max"]
        min = stat_dict["noisy"]["min"]
        #进行归一化
        noisy = (noisy - min) / (max - min)
        clean = (clean - min) / (max - min)
        noisy = 2*noisy - 1
        clean = 2*clean - 1
        #转换为tensor [1,H,W]
        noisy = torch.tensor(noisy, dtype=torch.float32).unsqueeze(0)
        clean = torch.tensor(clean, dtype=torch.float32).unsqueeze(0)
        mask = (mask>0).astype(np.float32)
        mask = torch.tensor(mask).unsqueeze(0)
        return noisy, clean, mask, f_id, z, d

