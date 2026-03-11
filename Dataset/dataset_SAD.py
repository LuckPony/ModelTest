import json
import random
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import save_normalize

def get_bval_sort_indices(bval_path):
    """
    读取 bval 文件，返回按 b=0, b=1000, b=2000 排序的索引，
    以及用于后续还原原始顺序的逆向索引。
    """
    bvals = np.loadtxt(bval_path)
    
    # 考虑到扫描仪的微小误差，设置一定的宽容度
    b0_idx = np.where(bvals < 50)[0]
    b1000_idx = np.where((bvals > 900) & (bvals < 1100))[0]
    b2000_idx = np.where((bvals > 1900) & (bvals < 2100))[0]
    
    # 正向排序索引：将通道重排为 [b0, b1000, b2000]
    sorted_idx = np.concatenate([b0_idx, b1000_idx, b2000_idx])
    
    # 逆向还原索引：用于测试结束后把顺序变回原来的样子
    inverse_idx = np.argsort(sorted_idx)
    
    return sorted_idx, inverse_idx


class DenoiseDataset(Dataset):
    # 【修改】：增加了 bval_files 参数
    def __init__(self, noisy_files, clean_files, mask_files, bval_files, patch_size=64):
        self.noisy_img = []
        self.clean_img = []
        self.mask_img = []    
        self.patch = patch_size
        
        # 预加载所有数据并立即进行 b 值排序
        for n_file, c_file, m_file, b_file in zip(noisy_files, clean_files, mask_files, bval_files):
            noisy_4d = nib.load(n_file).get_fdata()  
            clean_4d = nib.load(c_file).get_fdata()  
            mask_3d = nib.load(m_file).get_fdata()
            
            if noisy_4d.ndim != 4:
                raise ValueError("Noisy image should have 4 dimensions")
                
            # 【核心逻辑】：读取排序索引，直接重排第四维度 (方向)
            sorted_idx, _ = get_bval_sort_indices(b_file)
            noisy_4d = noisy_4d[:, :, :, sorted_idx]
            clean_4d = clean_4d[:, :, :, sorted_idx]
            
            self.noisy_img.append(noisy_4d)
            self.clean_img.append(clean_4d)
            self.mask_img.append(mask_3d)
                
        # 构建对应的(file_id, slice_id)索引对
        self.index_list = [] 
        save_normalize.save_normalize(self.noisy_img, save_path="normalize/train_minmax.json")
        
        # 【修改】：加入扩增乘数，解决一个 Epoch 只有几个 batch 的“梯度饥荒”问题
        patches_per_slice = 50 
        
        for f_id, img in enumerate(self.noisy_img):
            D, H, W, Dd = img.shape
            for z in range(D):
                for _ in range(patches_per_slice):
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
        
        # 提取切片上的所有排好序的方向，当前 shape: (H, W, 198)
        noisy = noisy_4d[slice_id, :, :, :]
        clean = clean_4d[slice_id, :, :, :]
        mask = mask_3d[slice_id, ...]
        
        # 转置为 PyTorch 需要的 (Channel, H, W) 格式 -> shape: (198, H, W)
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

        # 随机裁剪 2D patch
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
                
        noisy = noisy[:, dy:dy+p, dx:dx+p] # Shape: (198, p, p)
        clean = clean[:, dy:dy+p, dx:dx+p] # Shape: (198, p, p)
        mask = patch_mask # Shape: (p, p)

        noisy = torch.tensor(noisy, dtype=torch.float32)
        clean = torch.tensor(clean, dtype=torch.float32)
        mask = (mask > 0).astype(np.float32)
        mask = torch.tensor(mask).unsqueeze(0) 
        
        return noisy, clean, mask


class ValDataset(Dataset):
    # 【修改】：增加了 bval_files 参数
    def __init__(self, noisy_files, clean_files, mask_files, bval_files, patch=64):
        self.noisy_files = noisy_files
        self.clean_files = clean_files
        self.mask_files = mask_files
        self.bval_files = bval_files
        
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
        
        # 【核心逻辑】：对验证集也要进行动态排序
        sorted_idx, _ = get_bval_sort_indices(self.bval_files[0])
        noisy_4d = noisy_4d[:, :, :, sorted_idx]
        clean_4d = clean_4d[:, :, :, sorted_idx]
        
        D = noisy_4d.shape[0]
        z = D // 2 # 选取中间切片进行验证
        
        noisy = noisy_4d[z, :, :, :]
        clean = clean_4d[z, :, :, :]
        mask = mask_3d[z, ...]

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
    # 【修改】：增加了 bval_files 参数
    def __init__(self, noisy_files, clean_files, mask_files, bval_files):
        self.noisy_img = []
        self.clean_img = []
        self.mask_img = []
        
        for n_file, c_file, m_file, b_file in zip(noisy_files, clean_files, mask_files, bval_files):
            noisy_4d = nib.load(n_file).get_fdata()  
            clean_4d = nib.load(c_file).get_fdata()  
            mask_3d = nib.load(m_file).get_fdata()
            
            # 【核心逻辑】：测试集同样进行排序
            sorted_idx, _ = get_bval_sort_indices(b_file)
            noisy_4d = noisy_4d[:, :, :, sorted_idx]
            clean_4d = clean_4d[:, :, :, sorted_idx]
            
            self.noisy_img.append(noisy_4d)
            self.clean_img.append(clean_4d)
            self.mask_img.append(mask_3d)
            
        self.index_list = []
        save_normalize.save_normalize(self.noisy_img, save_path="normalize/test_minmax.json")
        for f_id, img in enumerate(self.noisy_img):
            D, H, W, Dd = img.shape
            for z in range(D):
                self.index_list.append((f_id, z))
                    
        with open("normalize/test_minmax.json", "r") as f:
            self.stat_dict = json.load(f)

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        f_id, z = self.index_list[idx]
        noisy_4d = self.noisy_img[f_id]
        clean_4d = self.clean_img[f_id]
        mask_3d = self.mask_img[f_id]
        
        noisy = noisy_4d[z, :, :, :]
        clean = clean_4d[z, :, :, :]
        mask = mask_3d[z, :, :]
        
        noisy = np.transpose(noisy, (2, 0, 1))
        clean = np.transpose(clean, (2, 0, 1))
        
        stat_dict = self.stat_dict[str(f_id)]
        max_val = stat_dict["noisy"]["max"]
        min_val = stat_dict["noisy"]["min"]
        
        noisy = (noisy - min_val) / (max_val - min_val)
        clean = (clean - min_val) / (max_val - min_val)
        noisy = 2*noisy - 1
        clean = 2*clean - 1
        
        noisy = torch.tensor(noisy, dtype=torch.float32) 
        clean = torch.tensor(clean, dtype=torch.float32) 
        mask = (mask > 0).astype(np.float32)
        mask = torch.tensor(mask).unsqueeze(0) 
        
        return noisy, clean, mask, f_id, z