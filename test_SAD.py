import json
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# 【修改为导入解耦架构的网络和Dataset】
from Dataset.dataset_SAD import TestDataset
from model.dncnn_SAD import DnCNN_Decoupled

import nibabel as nib
from nibabel.loadsave import save
from nibabel.nifti1 import Nifti1Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from datetime import datetime

def unnormalize(x, max_val, min_val):
    x = (x + 1) / 2
    x = x * (max_val - min_val) + min_val
    return x

def masked_psnr_3d(clean, denoised, mask, data_range):
    # 【核心修改】：此时 clean 和 denoised 都是 (288, H, W)
    # mask 是 (1, H, W)，通过广播机制自动屏蔽 288 个通道的外围背景
    mse = ((clean - denoised) ** 2 * mask).sum() / (mask.sum() * clean.shape[0] + 1e-8)
    return 10 * np.log10(data_range ** 2 / mse)

def test(noisy_files, clean_files, mask_files, model_path, norm_path, save_path, noise_level):
    folder_data = Path(model_path).parent.name # 自动使用模型所在的文件夹名
    
    print("Loading data...")
    test_dataset = TestDataset(noisy_files, clean_files, mask_files)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    with open(norm_path, "r") as f:
        stat_dict = json.load(f)

    print("Loading model...")
    # ==========================================
    # 初始化解耦模型 (固定 288 个方向)
    # ==========================================
    model = DnCNN_Decoupled(num_directions=288, num_layers=12, features=64)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    denoised_volumes = {}  
    for f_id, clean_file in enumerate(clean_files):
        clean_img = nib.load(clean_file).get_fdata()   
        D, H, W, Dd = clean_img.shape
        denoised_volumes[f_id] = np.zeros((D, H, W, Dd))
    
    print("Testing...")
    # 【核心修改】：去掉了 d，因为一次预测一整层的 288 个方向
    for noisy, clean, mask, f_id, z in test_loader:
        noisy = noisy.to(device)
        clean = clean.to(device)
        mask = mask.to(device)
        
        with torch.no_grad():
            denoised = model(noisy).cpu().numpy().squeeze(0) # Shape: (288, H, W)
            clean = clean.cpu().numpy().squeeze(0)
            noisy = noisy.cpu().numpy().squeeze(0)
            mask = mask.cpu().numpy().squeeze(0) # Shape: (1, H, W)
        
        max_val = stat_dict[str(f_id.item())]["noisy"]['max']
        min_val = stat_dict[str(f_id.item())]["noisy"]['min']
        
        denoised = unnormalize(denoised, max_val, min_val)
        clean = unnormalize(clean, max_val, min_val)
        
        # 应用 mask，背景清零
        denoised = denoised * mask
        
        # 评估整层的指标 (288通道联动计算)
        psnr_val = masked_psnr_3d(clean, denoised, mask, data_range=max_val-min_val)
        
        # SSIM 需要指定通道轴 channel_axis=0，因为它现在是 (288, H, W)
        ssim_val = ssim(clean, denoised, data_range=max_val - min_val, channel_axis=0)
        
        rmse_val = (np.sqrt(np.mean(((clean - denoised) * mask) ** 2))) / (max_val - min_val) 
        
        print(f"File {f_id.item()} Slice {z.item()} -- PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f}, RMSE={rmse_val:.4f}")
        
        # 【核心修改】：转置为 NIfTI 格式 (H, W, 288) 并保存
        denoised_transposed = np.transpose(denoised, (1, 2, 0))
        denoised_volumes[f_id.item()][z.item(), :, :, :] = denoised_transposed

    print("Saving denoised volumes...")
    output_dir = os.path.join(save_path, folder_data)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for f_id, vol in denoised_volumes.items():
        out_file = os.path.join(output_dir, f"{f_id}_{noise_level}p_denoised.nii.gz")
        clean_image = nib.load(clean_files[f_id])  
        affine = clean_image.affine  
        save(Nifti1Image(vol, affine), out_file)  
        print("Saved:", out_file)

def main():
    noise_level = 6
    # 填入你刚才通过 train_decoupled.py 训练生成的最新文件夹名字
    data_folder = "20260309_decoupled" 
    
    test_noisy_path = "./data/test/noise/"
    test_clean_path = "./data/test/gt/"
    test_mask_path = './data/test/mask/'
    model_path = f"result/model/{data_folder}/model_{noise_level}%noise_final.pth"
    norm_path = "normalize/test_minmax.json"
    save_path = f"result/denoised"
    
    test_noisy_files, test_clean_files, test_mask_files = [], [], []
    
    for file in os.listdir(test_clean_path):
        test_clean_files.append(test_clean_path + file)
        test_noisy_files.append(test_noisy_path + Path(Path(file).stem).stem + f'_{noise_level}%noise.nii.gz')
        test_mask_files.append(test_mask_path + Path(Path(file).stem).stem + '_mask.nii.gz')
        
    test(test_noisy_files, test_clean_files, test_mask_files, model_path, norm_path, save_path, noise_level)

if __name__ == "__main__":
    main()