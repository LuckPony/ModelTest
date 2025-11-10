import torch
from torch.utils.data import Dataset
import numpy as np
class DenoiseDataset(Dataset):
    def __init__(self,noisy_files, clean_files, patch_size=32):
        self.noisy_files = noisy_files
        self.clean_files = clean_files
        self.patch = patch_size

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy = np.load(self.noisy_files[idx])
        clean = np.load(self.clean_files[idx])

        #归一化
        noisy = (noisy - noisy.min()) / (noisy.max() - noisy.min())
        clean = (clean - clean.min()) / (clean.max() - clean.min())
        noisy = 2*noisy - 1
        clean = 2*clean - 1

        #转换成tensor
        noisy = torch.tensor(noisy, dtype=torch.float32).unsqueeze(0)
        clean = torch.tensor(clean, dtype=torch.float32).unsqueeze(0)
        
        return noisy, clean