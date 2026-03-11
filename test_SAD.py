import json
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# 【核心修改 1：导入多壳层感知网络和逆向排序函数】
from Dataset.dataset_SAD import TestDataset, get_bval_sort_indices
from model.dncnn_SAD import ShellAwareDnCNN

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
    # 此时 clean 和 denoised 都是 (198, H, W)
    mse = ((clean - denoised) ** 2 * mask).sum() / (mask.sum() * clean.shape[0] + 1e-8)
    return 10 * np.log10(data_range ** 2 / mse)

# 【核心修改 2：传入 bval_files 参数】
def test(noisy_files, clean_files, mask_files, bval_files, model_path, norm_path, save_path, noise_level):
    folder_data = Path(model_path).parent.name 
    
    print("Loading data...")
    test_dataset = TestDataset(noisy_files, clean_files, mask_files, bval_files)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    with open(norm_path, "r") as f:
        stat_dict = json.load(f)

    print("Loading model...")
    # ==========================================
    # 【核心修改 3：初始化多壳层解耦模型 (总通道 198)】
    # ==========================================
    model = ShellAwareDnCNN(num_b0=18, num_b1000=90, num_b2000=90, num_layers=12)
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    denoised_volumes = {}  
    for f_id, clean_file in enumerate(clean_files):
        clean_img = nib.load(clean_file).get_fdata()   
        D, H, W, Dd = clean_img.shape # 此时 Dd 应该是 198
        denoised_volumes[f_id] = np.zeros((D, H, W, Dd))
    
    print("Testing...")
    for noisy, clean, mask, f_id, z in test_loader:
        noisy = noisy.to(device)
        clean = clean.to(device)
        mask = mask.to(device)
        
        with torch.no_grad():
            denoised = model(noisy).cpu().numpy().squeeze(0) # Shape: (198, H, W) 已经是排序好的
            clean = clean.cpu().numpy().squeeze(0)
            noisy = noisy.cpu().numpy().squeeze(0)
            mask = mask.cpu().numpy().squeeze(0) # Shape: (1, H, W)
        
        # 当前处理的文件 ID
        current_f_id = f_id.item()
        
        max_val = stat_dict[str(current_f_id)]["noisy"]['max']
        min_val = stat_dict[str(current_f_id)]["noisy"]['min']
        
        denoised = unnormalize(denoised, max_val, min_val)
        clean = unnormalize(clean, max_val, min_val)
        
        # 应用 mask，背景清零
        denoised = denoised * mask
        
        # 评估整层的指标 (排序状态下计算指标完全不受影响)
        psnr_val = masked_psnr_3d(clean, denoised, mask, data_range=max_val-min_val)
        ssim_val = ssim(clean, denoised, data_range=max_val - min_val, channel_axis=0)
        rmse_val = (np.sqrt(np.mean(((clean - denoised) * mask) ** 2))) / (max_val - min_val) 
        
        print(f"File {current_f_id} Slice {z.item()} -- PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f}, RMSE={rmse_val:.4f}")
        
        # ==========================================
        # 【核心修改 4：逆向还原通道顺序并保存】
        # ==========================================
        # 转置为 (H, W, 198)
        denoised_transposed = np.transpose(denoised, (1, 2, 0))
        
        # 获取当前被试的逆向还原索引
        current_bval_path = bval_files[current_f_id]
        _, inverse_idx = get_bval_sort_indices(current_bval_path)
        
        # 沿着最后一个维度 (方向通道) 进行洗牌还原
        denoised_restored = denoised_transposed[:, :, inverse_idx]
        
        # 保存进体积字典
        denoised_volumes[current_f_id][z.item(), :, :, :] = denoised_restored

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
    # 填入你训练生成的最新文件夹名字
    data_folder = "20260309_decoupled" 
    
    test_noisy_path = "./data/test/noise/"
    test_clean_path = "./data/test/gt/"
    test_mask_path = './data/test/mask/'
    test_bval_path = './data/test/bvals/' # 【新增测试集 bval 路径】
    
    model_path = f"result/model/{data_folder}/model_{noise_level}%noise_final.pth"
    norm_path = "normalize/test_minmax.json"
    save_path = f"result/denoised"
    
    test_noisy_files, test_clean_files, test_mask_files, test_bval_files = [], [], [], []
    
    for file in os.listdir(test_clean_path):
        test_clean_files.append(test_clean_path + file)
        test_noisy_files.append(test_noisy_path + Path(Path(file).stem).stem + f'_{noise_level}%noise.nii.gz')
        test_mask_files.append(test_mask_path + Path(Path(file).stem).stem + '_mask.nii.gz')
        # 【新增 bval 文件收集】
        test_bval_files.append(test_bval_path + Path(Path(file).stem).stem + '.bval')
        
    test(test_noisy_files, test_clean_files, test_mask_files, test_bval_files, model_path, norm_path, save_path, noise_level)

if __name__ == "__main__":
    main()