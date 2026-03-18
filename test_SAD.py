import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nibabel.loadsave import save
from nibabel.nifti1 import Nifti1Image
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader

from Dataset.dataset_SAD import TestDataset, get_bval_sort_indices
# 【核心修改 1】：导入新的 AS_Alternating_2p5D 网络
from model.dncnn_SAD import AS_Alternating_2p5D


def unnormalize(x, max_val, min_val):
    x = (x + 1) / 2
    return x * (max_val - min_val) + min_val


def masked_psnr_3d(clean, denoised, mask, data_range):
    mse = ((clean - denoised) ** 2 * mask).sum() / (mask.sum() * clean.shape[0] + 1e-8)
    return 10 * np.log10(data_range**2 / mse)


def masked_nrmse_3d(clean, denoised, mask, data_range):
    mse = ((clean - denoised) ** 2 * mask).sum() / (mask.sum() * clean.shape[0] + 1e-8)
    return np.sqrt(mse) / (data_range + 1e-8)


def test(noisy_files, clean_files, mask_files, bval_files, model_path, norm_path, save_path, noise_level):
    folder_data = Path(model_path).parent.name

    print("Loading data...")
    test_dataset = TestDataset(noisy_files, clean_files, mask_files, bval_files)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    with open(norm_path, "r") as f:
        stat_dict = json.load(f)

    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ==========================================
    # 【核心修改 2】：实例化 AS-2.5D 网络
    # ==========================================
    model = AS_Alternating_2p5D(num_blocks=3, hidden_spatial=32)
    
    # 将检查点映射到当前活动设备上，以便仅使用 CPU 的评估也能正常工作。
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    denoised_volumes = {}
    for file_id, clean_file in enumerate(clean_files):
        clean_img = nib.load(clean_file).get_fdata()
        depth, height, width, directions = clean_img.shape
        denoised_volumes[file_id] = np.zeros((depth, height, width, directions))

    print("Testing...")
    for noisy, clean, mask, file_id, z in test_loader:
        # noisy shape: [1, 3, 198, H, W]
        # clean shape: [1, 198, H, W]
        noisy = noisy.to(device)
        clean = clean.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            # 模型输出 shape: [1, 198, H, W] -> squeeze(0) -> [198, H, W]
            denoised = model(noisy).cpu().numpy().squeeze(0)
            clean = clean.cpu().numpy().squeeze(0)
            mask = mask.cpu().numpy().squeeze(0)

        current_file_id = file_id.item()
        max_val = stat_dict[str(current_file_id)]["noisy"]["max"]
        min_val = stat_dict[str(current_file_id)]["noisy"]["min"]

        denoised = unnormalize(denoised, max_val, min_val)
        clean = unnormalize(clean, max_val, min_val)
        denoised = denoised * mask

        # 在计算 SSIM/RMSE 之前对两个输入信号进行掩码处理，以便计算出的指标与掩码后的 PSNR 区域相匹配。
        clean_eval = clean * mask
        denoised_eval = denoised * mask
        psnr_val = masked_psnr_3d(clean, denoised, mask, data_range=max_val - min_val)
        ssim_val = ssim(clean_eval, denoised_eval, data_range=max_val - min_val, channel_axis=0)
        rmse_val = masked_nrmse_3d(clean, denoised, mask, data_range=max_val - min_val)
        
        print(
            f"File {current_file_id} Slice {z.item()} -- "
            f"PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f}, RMSE={rmse_val:.4f}"
        )

        # 在将 4D 卷写回磁盘之前，恢复原始的获取顺序。
        denoised_transposed = np.transpose(denoised, (1, 2, 0)) # -> [H, W, 198]
        _, inverse_idx = get_bval_sort_indices(bval_files[current_file_id])
        denoised_restored = denoised_transposed[:, :, inverse_idx]
        denoised_volumes[current_file_id][z.item(), :, :, :] = denoised_restored

    print("Saving denoised volumes...")
    output_dir = os.path.join(save_path, folder_data)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for file_id, volume in denoised_volumes.items():
        out_file = os.path.join(output_dir, f"{file_id}_{noise_level}p_denoised.nii.gz")
        clean_image = nib.load(clean_files[file_id])
        save(Nifti1Image(volume, clean_image.affine), out_file)
        print("Saved:", out_file)


def main():
    noise_level = 2
    # 【注意】：记得把你刚刚跑出来的最新 AS2p5D 的权重文件夹名字填到这里！
    data_folder = "202603091234_AS2p5D_2p" # 替换为你自己的文件夹名

    test_noisy_path = "./data/test/noise/"
    test_clean_path = "./data/test/gt/"
    test_mask_path = "./data/test/mask/"
    test_bval_path = "./data/test/bvals/"

    model_path = f"result/model/{data_folder}/model_{noise_level}%noise_final.pth"
    norm_path = "normalize/test_minmax.json"
    save_path = "result/denoised"

    test_noisy_files = []
    test_clean_files = []
    test_mask_files = []
    test_bval_files = []
    for file in os.listdir(test_clean_path):
        stem = Path(Path(file).stem).stem
        test_clean_files.append(test_clean_path + file)
        test_noisy_files.append(test_noisy_path + stem + f"_{noise_level}%noise.nii.gz")
        test_mask_files.append(test_mask_path + stem + "_mask.nii.gz")
        test_bval_files.append(test_bval_path + stem + "_bval")

    test(
        test_noisy_files,
        test_clean_files,
        test_mask_files,
        test_bval_files,
        model_path,
        norm_path,
        save_path,
        noise_level,
    )


if __name__ == "__main__":
    main()