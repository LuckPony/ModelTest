import json
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from model.dncnn import DnCNN
from dataset import TestDataset
import nibabel as nib
from nibabel.loadsave import load, save
from nibabel.nifti1 import Nifti1Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from datetime import datetime
def unnormalize(x,max,min):
    x = (x+1) / 2
    x = x * (max - min) + min
    return x
def masked_psnr(clean, denoised, mask, data_range):
    mse = ((clean - denoised) ** 2 * mask).sum() / (mask.sum() + 1e-8)
    return 10 * np.log10(data_range ** 2 / mse)
def test(noisy_files, clean_files, mask_files, model_path,norm_path,save_path,noise_level):
    folder_data = datetime.now().strftime("%Y%m%d%H%M")
    # 加载数据
    print("Loading data...")
    test_dataset = TestDataset(noisy_files, clean_files, mask_files)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    #读取测试数据集归一化时的参数
    with open(norm_path, "r") as f:
        stat_dict = json.load(f)
    

    #载入模型并配置模型参数
    print("Loading model...")
    model = DnCNN(channels=1, num_layers=12, features=64)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #初始化4D volume以保证按照2D切片最终可以再拼接成4D图像
    denoised_volumes = {}  #key: file_id → 4D array { [D,W,H,Dd],[...],... }
    for f_id,clean_file in enumerate(clean_files):
        clean_img = nib.load(clean_file).get_fdata()   # 使用nib.load替代load  # pyright: ignore[reportPrivateImportUsage, reportAttributeAccessIssue]
        D,W,H,Dd = clean_img.shape
        denoised_volumes[f_id] = np.zeros((D,W,H,Dd))
    
    #模型预测
    print("Testing...")
    for noisy,clean,mask, f_id, z, d in test_loader:
        noisy = noisy.to(device)
        clean = clean.to(device)
        mask = mask.to(device)
        
        #开始预测
        
        with torch.no_grad():
            denoised = model(noisy).cpu().numpy().squeeze()
            clean = clean.cpu().numpy().squeeze()
            noisy = noisy.cpu().numpy().squeeze()
            mask = mask.cpu().numpy().squeeze()
        
        #反归一化
        max = stat_dict[str(f_id.item())]["noisy"]['max']
        min = stat_dict[str(f_id.item())]["noisy"]['min']
        denoised = unnormalize(denoised, max, min)
        clean = unnormalize(clean, max, min)
        noisy = unnormalize(noisy, max, min)
        denoised = denoised * mask
        #保存进初始化的4D volume
        denoised_volumes[f_id.item()][z.item(),:,:,d.item()] = denoised

        #进行结果评估
        psnr_val = masked_psnr(clean, denoised, mask, data_range=max-min)
        ssim_val = ssim(clean,denoised,data_range = max - min)
        rmse_val = (np.sqrt(np.mean((clean - denoised) ** 2)))/(max - min) #计算归一化之后的RMSE
        print(f"File {f_id.item()} Slice {z.item()} Dir {d.item()} -- PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f}, RMSE={rmse_val:.4f}")
        
        #保存最终的去噪结果
    print("Saving denoised volumes...")
    output_dir = os.path.join(save_path, folder_data)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for f_id, vol in denoised_volumes.items():
        out_file = os.path.join(output_dir, f"{f_id}_{noise_level}p_denoised.nii.gz")
        clean_image = nib.load(clean_files[f_id])  # 使用nib.load替代load  # pyright: ignore[reportPrivateImportUsage]
        affine = clean_image.affine  # pyright: ignore[reportAttributeAccessIssue]
        save(Nifti1Image(vol, affine), out_file)  # 使用显式导入的save和Nifti1Image函数
        print("Saved:", out_file)


def main():
    noise_level = 2
    test_noisy_path = "data/2_percent_noise/"
    test_clean_path = "data/gt/"
    test_mask_path = 'data/mask/'
    model_path = f"result/model/model_{noise_level}%noise.pth"
    norm_path = "normalize/test_minmax.json"
    save_path = f"result/denoised"
    test_noisy_files = []
    test_clean_files = []
    test_mask_files = []
    for file in os.listdir(test_clean_path):
        test_clean_files.append(test_clean_path + file)
        test_noisy_files.append(test_noisy_path + Path(Path(file).stem).stem+f'_{noise_level}%noise.nii.gz')
        test_mask_files.append(test_mask_path+Path(Path(file).stem).stem+f'_mask.nii.gz')
    test(test_noisy_files, test_clean_files, test_mask_files, model_path, norm_path,save_path,noise_level)

if __name__ == "__main__":
    main()