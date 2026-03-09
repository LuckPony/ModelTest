import json
import random
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import save_normalize

class DenoiseDataset(Dataset):
    def __init__(self, noisy_files, clean_files, mask_files, patch_size=64):
        self.noisy_img = []
        self.clean_img = []
        self.mask_img = []    
        self.patch = patch_size
        
        # 预加载所有数据
        for n_file, c_file, m_file in zip(noisy_files, clean_files, mask_files):
            noisy_4d = nib.load(n_file).get_fdata()  
            clean_4d = nib.load(c_file).get_fdata()  
            mask_3d = nib.load(m_file).get_fdata()
            self.noisy_img.append(noisy_4d)
            self.clean_img.append(clean_4d)
            self.mask_img.append(mask_3d)
            if noisy_4d.ndim != 4:
                raise ValueError("Noisy image should have 4 dimensions")
                
        # 构建对应的(file_id, slice_id)索引对，抛弃 direction
        self.index_list = [] 
        save_normalize.save_normalize(self.noisy_img, save_path="normalize/train_minmax.json")
        for f_id, img in enumerate(self.noisy_img):
            D, H, W, Dd = img.shape
            # 解耦网络每次吃满 288 个方向，所以只需要遍历切片 D
            for z in range(D):
                self.index_list.append((f_id, z))
                    
        with open("normalize/train_minmax.json", "r") as f:
            self.stat_dict = json.load(f)

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        file_id, slice_id = self.index_list[idx]
        noisy_4d = self.noisy_img[file_id]
        clean_4d = self.clean_img[file_id]
        mask_3d = self.mask_img[file_id]
        
        # 提取切片上的所有 288 个方向，当前 shape: (H, W, 288)
        noisy = noisy_4d[slice_id, :, :, :]
        clean = clean_4d[slice_id, :, :, :]
        mask = mask_3d[slice_id, ...]
        
        # 转置为 PyTorch 需要的 (Channel, H, W) 格式 -> shape: (288, H, W)
        noisy = np.transpose(noisy, (2, 0, 1))
        clean = np.transpose(clean, (2, 0, 1))
        
        stat_dict = self.stat_dict[str(file_id)]
        max_val = stat_dict["noisy"]["max"]
        min_val = stat_dict["noisy"]["min"]
        
        # 归一化到[-1,1]
        noisy = (noisy - min_val) / (max_val - min_val)
        clean = (clean - min_val) / (max_val - min_val)
        noisy = 2*noisy - 1
        clean = 2*clean - 1

        # 随机裁剪 2D patch (保持通道维度 288 不变)
        _, H_img, W_img = noisy.shape
        p = self.patch
        if H_img < p or W_img < p:
            raise ValueError(f"Patch size {p} is larger than image size {H_img}x{W_img}")
        
        # 加入空背景过滤机制
        max_retries = 10
        for _ in range(max_retries):
            dy = random.randint(0, H_img - p)
            dx = random.randint(0, W_img - p)
            patch_mask = mask[dy:dy+p, dx:dx+p]
            if (patch_mask > 0).sum() > (p * p * 0.1): 
                break
                
        noisy = noisy[:, dy:dy+p, dx:dx+p] # Shape: (288, p, p)
        clean = clean[:, dy:dy+p, dx:dx+p] # Shape: (288, p, p)
        mask = patch_mask # Shape: (p, p)

        # 转换成 tensor，不需要再 unsqueeze(0) 因为已经有 288 这个通道维度了
        noisy = torch.tensor(noisy, dtype=torch.float32)
        clean = torch.tensor(clean, dtype=torch.float32)
        mask = (mask > 0).astype(np.float32)
        mask = torch.tensor(mask).unsqueeze(0) # Mask 变为 (1, p, p) 方便广播计算
        
        return noisy, clean, mask


class ValDataset(Dataset):
    def __init__(self, noisy_files, clean_files, mask_files, patch=64):
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
        z = D // 2 # 选取中间切片进行验证
        
        # 取中间切片的所有 288 个方向，Shape: (H, W, 288)
        noisy = noisy_4d[z, :, :, :]
        clean = clean_4d[z, :, :, :]
        mask = mask_3d[z, ...]

        # 转置为 (288, H, W)
        noisy = np.transpose(noisy, (2, 0, 1))
        clean = np.transpose(clean, (2, 0, 1))

        noisy = (noisy - self.min) / (self.max - self.min)
        clean = (clean - self.min) / (self.max - self.min)
        noisy = 2*noisy - 1
        clean = 2*clean - 1
        
        _, H_img, W_img = noisy.shape
        p = self.patch
        dy = random.randint(0, H_img - p)
        dx = random.randint(0, W_img - p)
        
        noisy = noisy[:, dy:dy+p, dx:dx+p]
        clean = clean[:, dy:dy+p, dx:dx+p]
        mask = mask[dy:dy+p, dx:dx+p]
        
        noisy = torch.tensor(noisy, dtype=torch.float32)
        clean = torch.tensor(clean, dtype=torch.float32)
        mask = (mask > 0).astype(np.float32)
        mask = torch.tensor(mask).unsqueeze(0)
        
        return noisy, clean, mask


class TestDataset(Dataset):
    def __init__(self, noisy_files, clean_files, mask_files):
        self.noisy_img = []
        self.clean_img = []
        self.mask_img = []
        for n_file, c_file, m_file in zip(noisy_files, clean_files, mask_files):
            noisy_4d = nib.load(n_file).get_fdata()  
            clean_4d = nib.load(c_file).get_fdata()  
            mask_3d = nib.load(m_file).get_fdata()
            self.noisy_img.append(noisy_4d)
            self.clean_img.append(clean_4d)
            self.mask_img.append(mask_3d)
            
        self.index_list = []
        save_normalize.save_normalize(self.noisy_img, save_path="normalize/test_minmax.json")
        for f_id, img in enumerate(self.noisy_img):
            D, H, W, Dd = img.shape
            # 测试集同样只遍历切片 D，不再遍历方向 Dd
            for z in range(D):
                self.index_list.append((f_id, z))
                    
        with open("normalize/test_minmax.json", "r") as f:
            self.stat_dict = json.load(f)

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        # 注意：这里只返回了 f_id 和 z，没有 d 了
        f_id, z = self.index_list[idx]
        noisy_4d = self.noisy_img[f_id]
        clean_4d = self.clean_img[f_id]
        mask_3d = self.mask_img[f_id]
        
        # 取出整个切片的所有方向 (H, W, 288)
        noisy = noisy_4d[z, :, :, :]
        clean = clean_4d[z, :, :, :]
        mask = mask_3d[z, :, :]
        
        # 转置为 (288, H, W)
        noisy = np.transpose(noisy, (2, 0, 1))
        clean = np.transpose(clean, (2, 0, 1))
        
        stat_dict = self.stat_dict[str(f_id)]
        max_val = stat_dict["noisy"]["max"]
        min_val = stat_dict["noisy"]["min"]
        
        noisy = (noisy - min_val) / (max_val - min_val)
        clean = (clean - min_val) / (max_val - min_val)
        noisy = 2*noisy - 1
        clean = 2*clean - 1
        
        noisy = torch.tensor(noisy, dtype=torch.float32) # (288, H, W)
        clean = torch.tensor(clean, dtype=torch.float32) # (288, H, W)
        mask = (mask > 0).astype(np.float32)
        mask = torch.tensor(mask).unsqueeze(0) # (1, H, W)
        
        # 返回值去掉了 d
        return noisy, clean, mask, f_id, z