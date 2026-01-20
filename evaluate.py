import os
import json
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch
import lpips
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import dipy.reconst.dti as dti
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs

class ShellWiseEvaluator:
    def __init__(self, use_lpips=True):
        self.results = []
        self.use_lpips = use_lpips
        
        if self.use_lpips:
            try:
                self.lpips_loss = lpips.LPIPS(net='alex')
                if torch.cuda.is_available():
                    self.lpips_loss.cuda()
                print(">> LPIPS 模型加载成功")
            except:
                self.use_lpips = False
                print(">> LPIPS 加载失败，将跳过")

    def load_nii(self, path, data_max=None, data_min=None):
        """
        加载 NIfTI 并归一化到 0~1 (适合 PSNR/SSIM)
        """
        print(f"Reading: {os.path.basename(path)} ...")
        nii = nib.load(path)
        data = nii.get_fdata().astype(np.float32)
        
        if data_max is not None and data_min is not None:
            data = (data - data_min) / (data_max - data_min + 1e-8)
        
        data = np.clip(data, 0, 1.0)
        return data

    def parse_bvals(self, bval_path):
        """解析 bvals 文件"""
        print(f"Reading bvals: {bval_path}")
        raw_bvals = np.loadtxt(bval_path)
        shell_indices = {}
        rounded_bvals = np.round(raw_bvals / 100) * 100
        unique_b = np.unique(rounded_bvals)
        for b in unique_b:
            indices = np.where(rounded_bvals == b)[0]
            shell_indices[f"b{int(b)}"] = indices
            print(f"   -> Found shell b{int(b)}: {len(indices)} directions")
        return shell_indices

    def calc_metrics_for_subset(self, gt, den, mask, indices):
        """计算指定 Shell 的平均指标 (带 Mask)"""
        psnr_list, ssim_list = [], []
        
        for i in indices:
            vol_gt = gt[..., i]
            vol_den = den[..., i]
            vol_mask = mask[..., i] if mask.ndim == 4 else mask
            
            # 应用 Mask (背景置0)
            vol_gt_masked = vol_gt * vol_mask
            vol_den_masked = vol_den * vol_mask
            
            # 计算 PSNR (Masked MSE)
            mse = ((vol_gt_masked - vol_den_masked) ** 2 * vol_mask).sum() / (vol_mask.sum() + 1e-8)
            p = 10 * np.log10(1.0 / mse) 
            
            # 计算 SSIM 
            s = ssim(vol_gt_masked, vol_den_masked, data_range=1.0)
            
            psnr_list.append(p)
            ssim_list.append(s)
            
        return np.mean(psnr_list), np.mean(ssim_list)

    def calc_lpips_for_subset(self, gt, den, indices):
        """计算 LPIPS，数据需要归一化到 [-1, 1]"""
        if not self.use_lpips or len(indices) == 0:
            return np.nan
            
        idx = indices[0] 
        z_mid = gt.shape[2] // 2
        
        slice_gt = gt[:, :, z_mid, idx]
        slice_den = den[:, :, z_mid, idx]
        
        # 映射: [0, 1] -> [-1, 1]
        t_gt = torch.tensor(slice_gt).float().unsqueeze(0).unsqueeze(0).repeat(1,3,1,1) * 2 - 1
        t_den = torch.tensor(slice_den).float().unsqueeze(0).unsqueeze(0).repeat(1,3,1,1) * 2 - 1
        
        if torch.cuda.is_available():
            t_gt, t_den = t_gt.cuda(), t_den.cuda()
            
        with torch.no_grad():
            score = self.lpips_loss(t_gt, t_den)
        return score.item()

    def visualize_shell(self, gt, noisy, den, indices, shell_name, save_dir):
        if len(indices) == 0: return
        idx = indices[min(1, len(indices)-1)]
        z_mid = gt.shape[2] // 2
        
        s_gt = gt[:, :, z_mid, idx]
        s_noisy = noisy[:, :, z_mid, idx]
        s_den = den[:, :, z_mid, idx]
        s_res = np.abs(s_den - s_gt)
        
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        v_max = np.percentile(s_gt, 99.5) 
        
        axs[0].imshow(np.rot90(s_noisy), cmap='gray', vmin=0, vmax=v_max)
        axs[0].set_title(f"Noisy ({shell_name})")
        axs[1].imshow(np.rot90(s_den), cmap='gray', vmin=0, vmax=v_max)
        axs[1].set_title(f"Denoised ({shell_name})")
        axs[2].imshow(np.rot90(s_gt), cmap='gray', vmin=0, vmax=v_max)
        axs[2].set_title("Ground Truth")
        
        res_vmax = v_max * 0.15
        im = axs[3].imshow(np.rot90(s_res), cmap='jet', vmin=0, vmax=res_vmax)
        axs[3].set_title("Residual Map")
        plt.colorbar(im, ax=axs[3], fraction=0.046, pad=0.04)
        for ax in axs: ax.axis('off')
        
        fname = f"Visual_{shell_name}_idx{idx}.png"
        plt.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches='tight')
        plt.close()

    def calculate_angular_error(self, gt, den, mask, bval_path, bvec_path):
        """
        计算白质区域的角度误差
        """
        try:
            bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
            gtab = gradient_table(bvals, bvecs=bvecs)
            
            tenmodel = dti.TensorModel(gtab)
            # 全脑 mask 拟合
            tenfit_gt = tenmodel.fit(gt, mask=mask > 0)
            tenfit_den = tenmodel.fit(den, mask=mask > 0) # 这里传入 den (或 noisy)
            
            v1_gt = tenfit_gt.evecs[..., 0]
            v1_den = tenfit_den.evecs[..., 0]
            
            # 生成白质 Mask
            fa_gt = tenfit_gt.fa  
            wm_mask = (mask > 0) & (fa_gt > 0.2)
            
            if np.sum(wm_mask) == 0:
                print("!! Error: No White Matter found (FA > 0.2).")
                return np.nan

            dot_prod = np.sum(v1_gt * v1_den, axis=-1)
            dot_prod = np.clip(dot_prod, -1.0, 1.0)
            angles = np.degrees(np.arccos(np.abs(dot_prod)))
            
            mean_ae_wm = np.mean(angles[wm_mask])
            return mean_ae_wm

        except Exception as e:
            print(f"!! Failed: {e}")
            return np.nan

    def run(self, file_dict, output_dir):
        # 1. 加载归一化参数
        json_path = file_dict['json_path']
        with open(json_path, 'r') as f:
            stats = json.load(f)
        
        subject_key = '0' if '0' in stats else 0
        max_val = stats[subject_key]['noisy']['max']
        min_val = stats[subject_key]['noisy']['min']
        
        print(f"Using Norm Params -> Min: {min_val}, Max: {max_val}")

        # 加载数据
        gt = self.load_nii(file_dict['gt'], max_val, min_val)
        noisy = self.load_nii(file_dict['noisy'], max_val, min_val)
        den = self.load_nii(file_dict['denoised'], max_val, min_val)
        
        print(f"Reading Mask: {os.path.basename(file_dict['mask'])} ...")
        mask = nib.load(file_dict['mask']).get_fdata()
        mask = np.where(mask > 0.5, 1.0, 0.0)

        # 解析 bvals
        shells = self.parse_bvals(file_dict['bval'])

        #计算 DTI 角度误差 (AE) - 同时计算 Noisy 和 Denoised
        print("\n>> [1/2] 计算 Noisy 的 AE (Baseline)...")
        ae_score_noisy = self.calculate_angular_error(
            gt, noisy, mask, # 注意这里传的是 noisy
            file_dict['bval'], 
            file_dict['bvec']
        )
        print(f"   Noisy AE (Baseline): {ae_score_noisy:.4f}°")

        print("\n>> [2/2] 计算 Denoised 的 AE...")
        ae_score_denoised = self.calculate_angular_error(
            gt, den, mask, # 注意这里传的是 den
            file_dict['bval'], 
            file_dict['bvec']
        )
        print(f"   Denoised AE:         {ae_score_denoised:.4f}°")

        # 分层评估 (PSNR/SSIM/LPIPS)
        denoised_metrics = [] # 存去噪结果
        noisy_metrics = []    # 存噪声原始结果 (Baseline)

        print("\n>> 开始分层评估...")
        for shell_name, indices in shells.items():
            print(f"Processing {shell_name} ({len(indices)} directions)...")
            
            # --- A. 计算 Denoised 指标 ---
            m_psnr_d, m_ssim_d = self.calc_metrics_for_subset(gt, den, mask, indices)
            lpips_d = self.calc_lpips_for_subset(gt, den, indices)
            
            denoised_metrics.append({
                "Shell": shell_name,
                "Num_Directions": len(indices),
                "PSNR": round(m_psnr_d, 4),
                "SSIM": round(m_ssim_d, 4),
                "LPIPS": round(lpips_d, 4),
                "AE": "-"
            })

            # --- B. 计算 Noisy 指标 (Baseline) ---
            m_psnr_n, m_ssim_n = self.calc_metrics_for_subset(gt, noisy, mask, indices)
            lpips_n = self.calc_lpips_for_subset(gt, noisy, indices)

            noisy_metrics.append({
                "Shell": shell_name,
                "Num_Directions": len(indices),
                "PSNR": round(m_psnr_n, 4),
                "SSIM": round(m_ssim_n, 4),
                "LPIPS": round(lpips_n, 4),
                "AE": "-"
            })

            # 可视化 (只画 Denoised 对比图)
            self.visualize_shell(gt, noisy, den, indices, shell_name, output_dir)
        
        # --- 保存去噪结果 ---
        total_psnr_d = np.mean([d['PSNR'] for d in denoised_metrics])
        total_ssim_d = np.mean([d['SSIM'] for d in denoised_metrics])
        denoised_metrics.append({
            "Shell": "Global Average",
            "Num_Directions": gt.shape[-1],
            "PSNR": round(total_psnr_d, 4),
            "SSIM": round(total_ssim_d, 4),
            "LPIPS": "-",
            "AE": round(ae_score_denoised, 4)
        })
        
        df_den = pd.DataFrame(denoised_metrics)
        save_path_den = os.path.join(output_dir, "Shell_Wise_Metrics.xlsx")
        df_den.to_excel(save_path_den, index=False)
        
        # --- 保存 Noisyp评估结果 ---
        total_psnr_n = np.mean([d['PSNR'] for d in noisy_metrics])
        total_ssim_n = np.mean([d['SSIM'] for d in noisy_metrics])
        noisy_metrics.append({
            "Shell": "Global Average",
            "Num_Directions": gt.shape[-1],
            "PSNR": round(total_psnr_n, 4),
            "SSIM": round(total_ssim_n, 4),
            "LPIPS": "-",
            "AE": round(ae_score_noisy, 4)
        })

        df_noisy = pd.DataFrame(noisy_metrics)
        save_path_noisy = os.path.join(output_dir, "Noisy_Metrics.xlsx")
        df_noisy.to_excel(save_path_noisy, index=False)

        print(f"\n评估完成！")
        print(f"1. 去噪结果已保存至: {save_path_den}")
        print(f"2. 噪声基准已保存至: {save_path_noisy}")

if __name__ == "__main__":
    percentage = 6
    folder = datetime.now().strftime("%Y%m%d%H%M") + f'_{percentage}p'
    sub_id = "sub-105923"
    data_folder = "202601201929_6p"
    
    files = {
        'noisy':    f'./data/test/noise/{sub_id}__dwi_{percentage}%noise.nii.gz',
        'denoised': f'./result/denoised/{data_folder}/0_{percentage}p_denoised.nii.gz',
        'gt':       f'./data/test/gt/{sub_id}__dwi.nii.gz',
        'mask':     f'./data/test/mask/{sub_id}__dwi_mask.nii.gz', 
        'bval':     f'./data/test/bvals/{sub_id}__dwi_bvals',
        'bvec':     f'./data/test/bvecs/{sub_id}__dwi_bvecs',
        'json_path': './normalize/test_minmax.json'
    }
    
    output = f'./result/evaluate/{folder}/'
    Path(output).mkdir(parents=True, exist_ok=True)
    
    evaluator = ShellWiseEvaluator(use_lpips=True)
    evaluator.run(files, output)