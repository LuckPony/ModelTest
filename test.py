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

def unnormalize(x,max,min):
    x = (x+1) / 2
    x = x * (max - min) + min
    return x

def test(noisy_files, clean_files, model_path,norm_path,save_path,noise_level):
    # 加载数据
    print("Loading data...")
    test_dataset = TestDataset(noisy_files, clean_files)
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
    for noisy,clean, f_id, z, d in test_loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        #开始预测
        
        with torch.no_grad():
            denoised = model(noisy).cpu().numpy().squeeze()
            clean = clean.cpu().numpy().squeeze()
            noisy = noisy.cpu().numpy().squeeze()
        
        #反归一化
        max = stat_dict[str(f_id.item())]["noisy"]['max']
        min = stat_dict[str(f_id.item())]["noisy"]['min']
        denoised = unnormalize(denoised, max, min)
        clean = unnormalize(clean, max, min)
        noisy = unnormalize(noisy, max, min)

        #保存进初始化的4D volume
        denoised_volumes[f_id.item()][z.item(),:,:,d.item()] = denoised

        #进行结果评估
        psnr_val = psnr(clean,denoised,data_range = max - min)
        ssim_val = ssim(clean,denoised,data_range = max - min)
        rmse_val = np.sqrt(np.mean((clean - denoised) ** 2))
        print(f"File {f_id.item()} Slice {z.item()} Dir {d.item()} -- PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f}, RMSE={rmse_val:.4f}")
        
        #保存最终的去噪结果
    print("Saving denoised volumes...")
    for f_id, vol in denoised_volumes.items():
        out_file = os.path.join(save_path, f"{f_id}_{noise_level}p_denoised.nii.gz")
        clean_image = nib.load(clean_files[f_id])  # 使用nib.load替代load
        affine = clean_image.affine
        save(Nifti1Image(vol, affine), out_file)  # 使用显式导入的save和Nifti1Image函数
        print("Saved:", out_file)


def main():
    noise_level = 2
    test_noisy_path = "data/2_percent_noise/"
    test_clean_path = "data/gt/"
    model_path = f"result/model/model_{noise_level}%noise.pth"
    norm_path = "normalize/test_minmax.json"
    save_path = f"result/denoised"
    test_noisy_files = []
    test_clean_files = []
    for file in os.listdir(test_clean_path):
        test_clean_files.append(test_clean_path + file)
        test_noisy_files.append(test_noisy_path + Path(Path(file).stem).stem+f'_{noise_level}%noise.nii.gz')
    test(test_noisy_files, test_clean_files, model_path, norm_path,save_path,noise_level)

if __name__ == "__main__":
    main()