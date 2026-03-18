import json
import os
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nibabel.loadsave import save
from nibabel.nifti1 import Nifti1Image
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader

from dataset import TestDataset
from model.dncnn import DnCNN


def unnormalize(x, max_val, min_val):
    x = (x + 1) / 2
    return x * (max_val - min_val) + min_val


def masked_psnr(clean, denoised, mask, data_range):
    mse = ((clean - denoised) ** 2 * mask).sum() / (mask.sum() + 1e-8)
    return 10 * np.log10(data_range**2 / mse)


def masked_nrmse(clean, denoised, mask, data_range):
    mse = ((clean - denoised) ** 2 * mask).sum() / (mask.sum() + 1e-8)
    return np.sqrt(mse) / (data_range + 1e-8)


def test(noisy_files, clean_files, mask_files, model_path, norm_path, save_path, noise_level):
    folder_data = datetime.now().strftime("%Y%m%d%H%M") + f"_{noise_level}p"

    print("Loading data...")
    test_dataset = TestDataset(noisy_files, clean_files, mask_files)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    with open(norm_path, "r") as f:
        stat_dict = json.load(f)

    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DnCNN(channels=1, num_layers=12, features=64)
    # Map checkpoints onto the active device so CPU-only evaluation also works.
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    denoised_volumes = {}
    for file_id, clean_file in enumerate(clean_files):
        clean_img = nib.load(clean_file).get_fdata()
        depth, height, width, directions = clean_img.shape
        denoised_volumes[file_id] = np.zeros((depth, height, width, directions))

    print("Testing...")
    for noisy, clean, mask, file_id, z, d in test_loader:
        noisy = noisy.to(device)
        clean = clean.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            denoised = model(noisy).cpu().numpy().squeeze()
            clean = clean.cpu().numpy().squeeze()
            mask = mask.cpu().numpy().squeeze()

        current_file_id = file_id.item()
        max_val = stat_dict[str(current_file_id)]["noisy"]["max"]
        min_val = stat_dict[str(current_file_id)]["noisy"]["min"]

        denoised = unnormalize(denoised, max_val, min_val)
        clean = unnormalize(clean, max_val, min_val)
        denoised = denoised * mask

        denoised_volumes[current_file_id][z.item(), :, :, d.item()] = denoised

        # Mask both inputs before SSIM/RMSE so the reported metrics match the masked PSNR region.
        clean_eval = clean * mask
        denoised_eval = denoised * mask
        psnr_val = masked_psnr(clean, denoised, mask, data_range=max_val - min_val)
        ssim_val = ssim(clean_eval, denoised_eval, data_range=max_val - min_val)
        rmse_val = masked_nrmse(clean, denoised, mask, data_range=max_val - min_val)
        print(
            f"File {current_file_id} Slice {z.item()} Dir {d.item()} -- "
            f"PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f}, RMSE={rmse_val:.4f}"
        )

    print("Saving denoised volumes...")
    output_dir = os.path.join(save_path, folder_data)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for file_id, volume in denoised_volumes.items():
        out_file = os.path.join(output_dir, f"{file_id}_{noise_level}p_denoised.nii.gz")
        clean_image = nib.load(clean_files[file_id])
        save(Nifti1Image(volume, clean_image.affine), out_file)
        print("Saved:", out_file)


def main():
    noise_level = 6
    data_folder = 202601182245
    test_noisy_path = "./data/test/noise/"
    test_clean_path = "./data/test/gt/"
    test_mask_path = "./data/test/mask/"
    model_path = f"result/model/{data_folder}/model_{noise_level}%noise.pth"
    norm_path = "normalize/test_minmax.json"
    save_path = "result/denoised"

    test_noisy_files = []
    test_clean_files = []
    test_mask_files = []
    for file in os.listdir(test_clean_path):
        stem = Path(Path(file).stem).stem
        test_clean_files.append(test_clean_path + file)
        test_noisy_files.append(test_noisy_path + stem + f"_{noise_level}%noise.nii.gz")
        test_mask_files.append(test_mask_path + stem + "_mask.nii.gz")

    test(test_noisy_files, test_clean_files, test_mask_files, model_path, norm_path, save_path, noise_level)


if __name__ == "__main__":
    main()
