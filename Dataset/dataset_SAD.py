import json
import random

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

import save_normalize


def get_bval_sort_indices(bval_path):
    """
    返回将方向重新排序为 [b0, b1000, b2000] 的索引，
    以及用于恢复原始采集顺序的反向索引。
    """
    bvals = np.loadtxt(bval_path).reshape(-1)
    rounded_bvals = np.round(bvals / 100) * 100

    # 先进行四舍五入，因此像 995/1005 这样的相近数值仍会归入预期的区间。
    b0_idx = np.where(rounded_bvals == 0)[0]
    b1000_idx = np.where(rounded_bvals == 1000)[0]
    b2000_idx = np.where(rounded_bvals == 2000)[0]

    expected_counts = {"b0": 18, "b1000": 90, "b2000": 90}
    actual_counts = {"b0": len(b0_idx), "b1000": len(b1000_idx), "b2000": len(b2000_idx)}
    if actual_counts != expected_counts:
        raise ValueError(
            f"Unexpected shell counts in {bval_path}: {actual_counts}, expected {expected_counts}"
        )

    # 如果出现不受支持的 shell，应立即报错，而不是默默忽略它们。
    sorted_idx = np.concatenate([b0_idx, b1000_idx, b2000_idx])
    if len(sorted_idx) != len(bvals):
        used_mask = np.zeros(len(bvals), dtype=bool)
        used_mask[sorted_idx] = True
        unsupported_bvals = np.unique(rounded_bvals[~used_mask]).astype(int).tolist()
        raise ValueError(f"Unsupported b-values found in {bval_path}: {unsupported_bvals}")

    inverse_idx = np.argsort(sorted_idx)
    return sorted_idx, inverse_idx


class DenoiseDataset(Dataset):
    def __init__(self, noisy_files, clean_files, mask_files, bval_files, patch_size=64):
        self.noisy_img = []
        self.clean_img = []
        self.mask_img = []
        self.patch = patch_size

        for noisy_file, clean_file, mask_file, bval_file in zip(
            noisy_files, clean_files, mask_files, bval_files
        ):
            noisy_4d = nib.load(noisy_file).get_fdata()
            clean_4d = nib.load(clean_file).get_fdata()
            mask_3d = nib.load(mask_file).get_fdata()

            if noisy_4d.ndim != 4:
                raise ValueError("Noisy image should have 4 dimensions")

            sorted_idx, _ = get_bval_sort_indices(bval_file)
            self.noisy_img.append(noisy_4d[:, :, :, sorted_idx])
            self.clean_img.append(clean_4d[:, :, :, sorted_idx])
            self.mask_img.append(mask_3d)

        self.index_list = []
        save_normalize.save_normalize(self.noisy_img, save_path="normalize/train_minmax.json")

        # 从每个切片中重新采样几个随机片段，以确保一个 epoch 拥有足够的更新。
        patches_per_slice = 10
        for file_id, img in enumerate(self.noisy_img):
            depth = img.shape[0]
            for slice_id in range(depth):
                for _ in range(patches_per_slice):
                    self.index_list.append((file_id, slice_id))

        with open("normalize/train_minmax.json", "r") as f:
            self.stat_dict = json.load(f)

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        file_id, slice_id = self.index_list[idx]
        noisy_4d = self.noisy_img[file_id]
        clean_4d = self.clean_img[file_id]
        mask_3d = self.mask_img[file_id]

        # 【核心修改】：2.5D 切片抽取逻辑
        z_prev = max(0, slice_id - 1)
        z_next = min(noisy_4d.shape[0] - 1, slice_id + 1)
        
        # 提取 3 层作为输入，但目标 Ground Truth 只保留中心层
        noisy = noisy_4d[[z_prev, slice_id, z_next], :, :, :]  # Shape: (3, H, W, 198)
        clean = clean_4d[slice_id, :, :, :]                    # Shape: (H, W, 198)
        mask = mask_3d[slice_id, ...]                          # Shape: (H, W)

        # 转置为 PyTorch 所需的格式
        noisy = np.transpose(noisy, (0, 3, 1, 2))  # Shape: (3, 198, H, W)
        clean = np.transpose(clean, (2, 0, 1))     # Shape: (198, H, W)

        stat_dict = self.stat_dict[str(file_id)]
        max_val = stat_dict["noisy"]["max"]
        min_val = stat_dict["noisy"]["min"]

        noisy = 2 * ((noisy - min_val) / (max_val - min_val)) - 1
        clean = 2 * ((clean - min_val) / (max_val - min_val)) - 1

        _, _, height, width = noisy.shape
        patch = self.patch
        if height < patch or width < patch:
            raise ValueError(f"Patch size {patch} is larger than image size {height}x{width}")

        # 请重试几次，以免将大部分更新资源浪费在空的背景补丁上。
        for _ in range(10):
            dy = random.randint(0, height - patch)
            dx = random.randint(0, width - patch)
            patch_mask = mask[dy : dy + patch, dx : dx + patch]
            if (patch_mask > 0).sum() > (patch * patch * 0.1):
                break

        # 【核心修改】：裁剪时，noisy 有 4 个维度
        noisy = noisy[:, :, dy : dy + patch, dx : dx + patch]
        clean = clean[:, dy : dy + patch, dx : dx + patch]
        mask = patch_mask

        noisy = torch.tensor(noisy, dtype=torch.float32)
        clean = torch.tensor(clean, dtype=torch.float32)
        mask = torch.tensor((mask > 0).astype(np.float32)).unsqueeze(0)
        return noisy, clean, mask


class ValDataset(Dataset):
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

        sorted_idx, _ = get_bval_sort_indices(self.bval_files[0])
        noisy_4d = noisy_4d[:, :, :, sorted_idx]
        clean_4d = clean_4d[:, :, :, sorted_idx]

        z = noisy_4d.shape[0] // 2
        # 【核心修改】：验证集 2.5D 切片抽取
        z_prev = max(0, z - 1)
        z_next = min(noisy_4d.shape[0] - 1, z + 1)
        
        noisy = noisy_4d[[z_prev, z, z_next], :, :, :]
        clean = clean_4d[z, :, :, :]
        mask = mask_3d[z, ...]

        # 转置
        noisy = np.transpose(noisy, (0, 3, 1, 2))  # (3, 198, H, W)
        clean = np.transpose(clean, (2, 0, 1))     # (198, H, W)

        noisy = 2 * ((noisy - self.min) / (self.max - self.min)) - 1
        clean = 2 * ((clean - self.min) / (self.max - self.min)) - 1

        _, _, height, width = noisy.shape
        patch = self.patch
        dy = random.randint(0, height - patch)
        dx = random.randint(0, width - patch)

        # 裁剪
        noisy = noisy[:, :, dy : dy + patch, dx : dx + patch]
        clean = clean[:, dy : dy + patch, dx : dx + patch]
        mask = mask[dy : dy + patch, dx : dx + patch]

        noisy = torch.tensor(noisy, dtype=torch.float32)
        clean = torch.tensor(clean, dtype=torch.float32)
        mask = torch.tensor((mask > 0).astype(np.float32)).unsqueeze(0)
        return noisy, clean, mask


class TestDataset(Dataset):
    def __init__(self, noisy_files, clean_files, mask_files, bval_files):
        self.noisy_img = []
        self.clean_img = []
        self.mask_img = []

        for noisy_file, clean_file, mask_file, bval_file in zip(
            noisy_files, clean_files, mask_files, bval_files
        ):
            noisy_4d = nib.load(noisy_file).get_fdata()
            clean_4d = nib.load(clean_file).get_fdata()
            mask_3d = nib.load(mask_file).get_fdata()

            sorted_idx, _ = get_bval_sort_indices(bval_file)
            self.noisy_img.append(noisy_4d[:, :, :, sorted_idx])
            self.clean_img.append(clean_4d[:, :, :, sorted_idx])
            self.mask_img.append(mask_3d)

        self.index_list = []
        save_normalize.save_normalize(self.noisy_img, save_path="normalize/test_minmax.json")
        for file_id, img in enumerate(self.noisy_img):
            depth = img.shape[0]
            for slice_id in range(depth):
                self.index_list.append((file_id, slice_id))

        with open("normalize/test_minmax.json", "r") as f:
            self.stat_dict = json.load(f)

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        file_id, slice_id = self.index_list[idx]
        noisy_4d = self.noisy_img[file_id]
        clean_4d = self.clean_img[file_id]
        mask_3d = self.mask_img[file_id]

        # 【核心修改】：测试集 2.5D 切片抽取
        z_prev = max(0, slice_id - 1)
        z_next = min(noisy_4d.shape[0] - 1, slice_id + 1)
        
        noisy = noisy_4d[[z_prev, slice_id, z_next], :, :, :]
        clean = clean_4d[slice_id, :, :, :]
        mask = mask_3d[slice_id, :, :]

        noisy = np.transpose(noisy, (0, 3, 1, 2))  # (3, 198, H, W)
        clean = np.transpose(clean, (2, 0, 1))     # (198, H, W)

        stat_dict = self.stat_dict[str(file_id)]
        max_val = stat_dict["noisy"]["max"]
        min_val = stat_dict["noisy"]["min"]

        noisy = 2 * ((noisy - min_val) / (max_val - min_val)) - 1
        clean = 2 * ((clean - min_val) / (max_val - min_val)) - 1

        noisy = torch.tensor(noisy, dtype=torch.float32)
        clean = torch.tensor(clean, dtype=torch.float32)
        mask = torch.tensor((mask > 0).astype(np.float32)).unsqueeze(0)
        
        return noisy, clean, mask, file_id, slice_id