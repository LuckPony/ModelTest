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
        self.mask_img = []    
        self.patch = patch_size
        
        # 预加载所有数据
        for n__file,c_file,m_file in zip(noisy_files,clean_files,mask_files):
            noisy_4d = nib.load(n__file).get_fdata()  
            clean_4d = nib.load(c_file).get_fdata()  
            mask_3d = nib.load(m_file).get_fdata()
            self.noisy_img.append(noisy_4d)
            self.clean_img.append(clean_4d)
            self.mask_img.append(mask_3d)
            if noisy_4d.ndim != 4:
                raise ValueError("Noisy image should have 4 dimensions")
            if mask_3d.ndim != 3:
                raise ValueError("Mask image should have 3 dimensions")
                
        # 构建对应的(file_id, direction, slice_id)索引对
        self.index_list = [] 
        save_normalize.save_normalize(self.noisy_img, save_path="normalize/train_minmax.json")
        for f_id, img in enumerate(self.noisy_img):
            D, H, W, Dd = img.shape
            for d in range(Dd):
                for z in range(D):
                    self.index_list.append((f_id, d, z))
                    
        with open("normalize/train_minmax.json", "r") as f:
            self.stat_dict = json.load(f)

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        file_id, direction, slice_id = self.index_list[idx]
        noisy_4d = self.noisy_img[file_id]
        clean_4d = self.clean_img[file_id]
        mask_3d = self.mask_img[file_id]
        
        D = noisy_4d.shape[0] # 获取深度(切片总数)
        
        # --- 2.5D 修改核心：安全获取前后切片 ---
        z_prev = max(0, slice_id - 1)      # 如果是第0层，上一层还是第0层
        z_next = min(D - 1, slice_id + 1)  # 如果是最后一层，下一层还是最后一层
        
        noisy_prev = noisy_4d[z_prev, ..., direction]
        noisy_curr = noisy_4d[slice_id, ..., direction]
        noisy_next = noisy_4d[z_next, ..., direction]
        
        # 将三层切片在通道维度堆叠 -> Shape: (3, H, W)
        noisy = np.stack([noisy_prev, noisy_curr, noisy_next], axis=0) 
        
        # clean 和 mask 只需要中心层 (目标层) -> Shape: (H, W)
        clean = clean_4d[slice_id, ..., direction]   
        mask = mask_3d[slice_id,...]
        
        stat_dict = self.stat_dict[str(file_id)]
        max_val = stat_dict["noisy"]["max"]
        min_val = stat_dict["noisy"]["min"]
        
        # 归一化到[-1,1] (numpy数组广播机制会自动处理 (3,H,W) 的 noisy)
        noisy = (noisy - min_val) / (max_val - min_val)
        clean = (clean - min_val) / (max_val - min_val)
        noisy = 2*noisy - 1
        clean = 2*clean - 1

        # 随机裁剪 2D patch
        _, H, W = noisy.shape # 注意现在noisy是(3, H, W)，所以解包时忽略通道维度
        p = self.patch
        if H < p or W < p:
            raise ValueError(f"Patch size {p} is larger than image size {H}x{W}")
            
        dy = random.randint(0, H - p)
        dx = random.randint(0, W - p)
        
        # 裁剪时保留所有通道维度 (对于noisy是3层一起裁)
        noisy = noisy[:, dy:dy+p, dx:dx+p] # Shape: (3, p, p)
        clean = clean[dy:dy+p, dx:dx+p]    # Shape: (p, p)
        mask = mask[dy:dy+p, dx:dx+p]      # Shape: (p, p)

        # 转换成tensor
        noisy = torch.tensor(noisy, dtype=torch.float32) # 本身已经是 (3, p, p)，不需要 unsqueeze
        clean = torch.tensor(clean, dtype=torch.float32).unsqueeze(0) # 变为 (1, p, p)
        mask = (mask>0).astype(np.float32)
        mask = torch.tensor(mask).unsqueeze(0) # 变为 (1, p, p)
        
        return noisy, clean, mask
    
class ValDataset(Dataset):
    def __init__(self, noisy_files, clean_files,mask_files,patch=64):
        self.noisy_files = noisy_files
        self.clean_files = clean_files
        self.mask_files = mask_files
        noisy_4d = nib.load(self.noisy_files[0]).get_fdata()  
        self.max = noisy_4d.max()
        self.min = noisy_4d.min()
        self.patch = patch
      
    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy_4d = nib.load(self.noisy_files[0]).get_fdata() 
        clean_4d = nib.load(self.clean_files[0]).get_fdata() 
        mask_3d = nib.load(self.mask_files[0]).get_fdata()
        
        D = noisy_4d.shape[0]
        direction = random.randint(0,noisy_4d.shape[3]-1)  
        
        z = D // 2 # 中心切片索引
        
        # --- 2.5D 修改核心 ---
        z_prev = max(0, z - 1)
        z_next = min(D - 1, z + 1)
        
        noisy = np.stack([noisy_4d[z_prev, :, :, direction],
                          noisy_4d[z, :, :, direction],
                          noisy_4d[z_next, :, :, direction]], axis=0) # Shape: (3, H, W)
        
        clean = clean_4d[z, :, :, direction] # Shape: (H, W)
        mask = mask_3d[z,...]

        noisy = (noisy - self.min) / (self.max - self.min)
        clean = (clean - self.min) / (self.max - self.min)
        noisy = 2*noisy - 1
        clean = 2*clean - 1
        
        _, H, W = noisy.shape
        p = self.patch
        dy = random.randint(0, H - p)
        dx = random.randint(0, W - p)
        
        noisy = noisy[:, dy:dy+p, dx:dx+p]
        clean = clean[dy:dy+p, dx:dx+p]
        mask = mask[dy:dy+p, dx:dx+p]
        
        noisy = torch.tensor(noisy, dtype=torch.float32)
        clean = torch.tensor(clean, dtype=torch.float32).unsqueeze(0)
        mask = (mask>0).astype(np.float32)
        mask = torch.tensor(mask).unsqueeze(0)
        
        return noisy, clean, mask

class TestDataset(Dataset):
    def __init__(self, noisy_files, clean_files, mask_files):
        # 初始化和预加载代码保持不变... (省略前面部分以节省空间，直接看__getitem__)
        self.noisy_img = []
        self.clean_img = []
        self.mask_img = []
        for n_file, c_file, m_file in zip(noisy_files, clean_files, mask_files):
            noisy_4d = nib.load(n_file).get_fdata()  
            clean_4d = nib.load(c_file).get_fdata()  
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
                    
        with open("normalize/test_minmax.json","r") as f:
            self.stat_dict = json.load(f)

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        f_id, z, d = self.index_list[idx]
        noisy_4d = self.noisy_img[f_id]
        clean_4d = self.clean_img[f_id]
        mask_3d = self.mask_img[f_id]
        
        D = noisy_4d.shape[0]
        
        # --- 2.5D 修改核心 ---
        z_prev = max(0, z - 1)
        z_next = min(D - 1, z + 1)
        
        noisy = np.stack([noisy_4d[z_prev, :, :, d],
                          noisy_4d[z, :, :, d],
                          noisy_4d[z_next, :, :, d]], axis=0) # Shape: (3, H, W)
                          
        clean = clean_4d[z, :, :, d] # Shape: (H, W)
        mask = mask_3d[z, :, :]
        
        stat_dict = self.stat_dict[str(f_id)]
        max_val = stat_dict["noisy"]["max"]
        min_val = stat_dict["noisy"]["min"]
        
        noisy = (noisy - min_val) / (max_val - min_val)
        clean = (clean - min_val) / (max_val - min_val)
        noisy = 2*noisy - 1
        clean = 2*clean - 1
        
        noisy = torch.tensor(noisy, dtype=torch.float32) # (3, H, W)
        clean = torch.tensor(clean, dtype=torch.float32).unsqueeze(0) # (1, H, W)
        mask = (mask>0).astype(np.float32)
        mask = torch.tensor(mask).unsqueeze(0) # (1, H, W)
        
        return noisy, clean, mask, f_id, z, d