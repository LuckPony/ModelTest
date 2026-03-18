import json
import os
from datetime import datetime
from pathlib import Path

import dipy.reconst.dti as dti
import lpips
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from skimage.metrics import structural_similarity as ssim


class ShellWiseEvaluator:
    def __init__(self, use_lpips=True):
        self.results = []
        self.use_lpips = use_lpips

        if self.use_lpips:
            try:
                self.lpips_loss = lpips.LPIPS(net="alex")
                if torch.cuda.is_available():
                    self.lpips_loss.cuda()
                print(">> LPIPS model loaded")
            except Exception:
                self.use_lpips = False
                print(">> Failed to load LPIPS, skipping it")

    def load_nii(self, path):
        print(f"Reading: {os.path.basename(path)} ...")
        return nib.load(path).get_fdata().astype(np.float32)

    def normalize_for_metrics(self, data, data_max, data_min):
        # Image metrics are evaluated in [0, 1], but DTI fitting should stay in raw signal space.
        data = (data - data_min) / (data_max - data_min + 1e-8)
        return np.clip(data, 0.0, 1.0)

    def parse_bvals(self, bval_path):
        print(f"Reading bvals: {bval_path}")
        raw_bvals = np.loadtxt(bval_path)
        rounded_bvals = np.round(raw_bvals / 100) * 100
        shell_indices = {}

        for bval in np.unique(rounded_bvals):
            indices = np.where(rounded_bvals == bval)[0]
            shell_indices[f"b{int(bval)}"] = indices
            print(f"   -> Found shell b{int(bval)}: {len(indices)} directions")

        return shell_indices

    def calc_metrics_for_subset(self, gt, den, mask, indices):
        psnr_list, ssim_list = [], []

        for idx in indices:
            vol_gt = gt[..., idx]
            vol_den = den[..., idx]
            vol_mask = mask[..., idx] if mask.ndim == 4 else mask

            vol_gt_masked = vol_gt * vol_mask
            vol_den_masked = vol_den * vol_mask

            mse = ((vol_gt_masked - vol_den_masked) ** 2 * vol_mask).sum() / (vol_mask.sum() + 1e-8)
            psnr_val = np.inf if mse <= 1e-12 else 10 * np.log10(1.0 / mse)
            ssim_val = ssim(vol_gt_masked, vol_den_masked, data_range=1.0)

            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)

        return np.mean(psnr_list), np.mean(ssim_list)

    def calc_lpips_for_subset(self, gt, den, indices):
        if not self.use_lpips or len(indices) == 0:
            return np.nan

        idx = indices[0]
        z_mid = gt.shape[2] // 2

        slice_gt = gt[:, :, z_mid, idx]
        slice_den = den[:, :, z_mid, idx]

        t_gt = torch.tensor(slice_gt).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1) * 2 - 1
        t_den = torch.tensor(slice_den).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1) * 2 - 1

        if torch.cuda.is_available():
            t_gt = t_gt.cuda()
            t_den = t_den.cuda()

        with torch.no_grad():
            score = self.lpips_loss(t_gt, t_den)
        return score.item()

    def visualize_shell(self, gt, noisy, den, indices, shell_name, save_dir):
        if len(indices) == 0:
            return

        idx = indices[min(1, len(indices) - 1)]
        z_mid = gt.shape[2] // 2

        s_gt = gt[:, :, z_mid, idx]
        s_noisy = noisy[:, :, z_mid, idx]
        s_den = den[:, :, z_mid, idx]
        s_res = np.abs(s_den - s_gt)

        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        v_max = np.percentile(s_gt, 99.5)

        axs[0].imshow(np.rot90(s_noisy), cmap="gray", vmin=0, vmax=v_max)
        axs[0].set_title(f"Noisy ({shell_name})")
        axs[1].imshow(np.rot90(s_den), cmap="gray", vmin=0, vmax=v_max)
        axs[1].set_title(f"Denoised ({shell_name})")
        axs[2].imshow(np.rot90(s_gt), cmap="gray", vmin=0, vmax=v_max)
        axs[2].set_title("Ground Truth")

        res_vmax = v_max * 0.15
        im = axs[3].imshow(np.rot90(s_res), cmap="jet", vmin=0, vmax=res_vmax)
        axs[3].set_title("Residual Map")
        plt.colorbar(im, ax=axs[3], fraction=0.046, pad=0.04)
        for ax in axs:
            ax.axis("off")

        plt.savefig(os.path.join(save_dir, f"Visual_{shell_name}_idx{idx}.png"), dpi=300, bbox_inches="tight")
        plt.close()

    def calculate_angular_error(self, gt, den, mask, bval_path, bvec_path):
        try:
            bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
            gtab = gradient_table(bvals, bvecs=bvecs)

            tenmodel = dti.TensorModel(gtab)
            tenfit_gt = tenmodel.fit(gt, mask=mask > 0)
            tenfit_den = tenmodel.fit(den, mask=mask > 0)

            v1_gt = tenfit_gt.evecs[..., 0]
            v1_den = tenfit_den.evecs[..., 0]

            fa_gt = tenfit_gt.fa
            wm_mask = (mask > 0) & (fa_gt > 0.2)

            if np.sum(wm_mask) == 0:
                print("!! Error: No white matter found (FA > 0.2).")
                return np.nan

            dot_prod = np.sum(v1_gt * v1_den, axis=-1)
            dot_prod = np.clip(dot_prod, -1.0, 1.0)
            angles = np.degrees(np.arccos(np.abs(dot_prod)))
            return np.mean(angles[wm_mask])
        except Exception as exc:
            print(f"!! Failed: {exc}")
            return np.nan

    def run(self, file_dict, output_dir):
        with open(file_dict["json_path"], "r") as f:
            stats = json.load(f)

        subject_key = "0" if "0" in stats else next(iter(stats))
        max_val = stats[subject_key]["noisy"]["max"]
        min_val = stats[subject_key]["noisy"]["min"]
        print(f"Using Norm Params -> Min: {min_val}, Max: {max_val}")

        # Keep two copies of the data:
        # 1) raw signal for AE / tensor fitting
        # 2) normalized signal for PSNR / SSIM / LPIPS
        gt_raw = self.load_nii(file_dict["gt"])
        noisy_raw = self.load_nii(file_dict["noisy"])
        den_raw = self.load_nii(file_dict["denoised"])

        gt = self.normalize_for_metrics(gt_raw, max_val, min_val)
        noisy = self.normalize_for_metrics(noisy_raw, max_val, min_val)
        den = self.normalize_for_metrics(den_raw, max_val, min_val)

        print(f"Reading Mask: {os.path.basename(file_dict['mask'])} ...")
        mask = nib.load(file_dict["mask"]).get_fdata()
        mask = np.where(mask > 0.5, 1.0, 0.0)

        shells = self.parse_bvals(file_dict["bval"])

        print("\n>> [1/2] Calculating Noisy AE (Baseline)...")
        ae_score_noisy = self.calculate_angular_error(
            gt_raw,
            noisy_raw,
            mask,
            file_dict["bval"],
            file_dict["bvec"],
        )
        print(f"   Noisy AE (Baseline): {ae_score_noisy:.4f} deg")

        print("\n>> [2/2] Calculating Denoised AE...")
        ae_score_denoised = self.calculate_angular_error(
            gt_raw,
            den_raw,
            mask,
            file_dict["bval"],
            file_dict["bvec"],
        )
        print(f"   Denoised AE:         {ae_score_denoised:.4f} deg")

        denoised_metrics = []
        noisy_metrics = []

        print("\n>> Starting shell-wise evaluation...")
        for shell_name, indices in shells.items():
            print(f"Processing {shell_name} ({len(indices)} directions)...")

            m_psnr_d, m_ssim_d = self.calc_metrics_for_subset(gt, den, mask, indices)
            lpips_d = self.calc_lpips_for_subset(gt, den, indices)
            denoised_metrics.append(
                {
                    "Shell": shell_name,
                    "Num_Directions": len(indices),
                    "PSNR": round(m_psnr_d, 4),
                    "SSIM": round(m_ssim_d, 4),
                    "LPIPS": round(lpips_d, 4),
                    "AE": "-",
                }
            )

            m_psnr_n, m_ssim_n = self.calc_metrics_for_subset(gt, noisy, mask, indices)
            lpips_n = self.calc_lpips_for_subset(gt, noisy, indices)
            noisy_metrics.append(
                {
                    "Shell": shell_name,
                    "Num_Directions": len(indices),
                    "PSNR": round(m_psnr_n, 4),
                    "SSIM": round(m_ssim_n, 4),
                    "LPIPS": round(lpips_n, 4),
                    "AE": "-",
                }
            )

            self.visualize_shell(gt, noisy, den, indices, shell_name, output_dir)

        total_psnr_d = np.mean([item["PSNR"] for item in denoised_metrics])
        total_ssim_d = np.mean([item["SSIM"] for item in denoised_metrics])
        denoised_metrics.append(
            {
                "Shell": "Global Average",
                "Num_Directions": gt.shape[-1],
                "PSNR": round(total_psnr_d, 4),
                "SSIM": round(total_ssim_d, 4),
                "LPIPS": "-",
                "AE": round(ae_score_denoised, 4),
            }
        )

        total_psnr_n = np.mean([item["PSNR"] for item in noisy_metrics])
        total_ssim_n = np.mean([item["SSIM"] for item in noisy_metrics])
        noisy_metrics.append(
            {
                "Shell": "Global Average",
                "Num_Directions": gt.shape[-1],
                "PSNR": round(total_psnr_n, 4),
                "SSIM": round(total_ssim_n, 4),
                "LPIPS": "-",
                "AE": round(ae_score_noisy, 4),
            }
        )

        save_path_den = os.path.join(output_dir, "Shell_Wise_Metrics.xlsx")
        pd.DataFrame(denoised_metrics).to_excel(save_path_den, index=False)

        save_path_noisy = os.path.join(output_dir, "Noisy_Metrics.xlsx")
        pd.DataFrame(noisy_metrics).to_excel(save_path_noisy, index=False)

        print("\nEvaluation complete.")
        print(f"1. Denoised metrics saved to: {save_path_den}")
        print(f"2. Noisy baseline saved to: {save_path_noisy}")


if __name__ == "__main__":
    percentage = 4
    folder = datetime.now().strftime("%Y%m%d%H%M") + f"_{percentage}p"
    sub_id = "sub-105923__dwi_filtered"
    data_folder = "202603111527_4p"

    files = {
        "noisy": f"./data/test/noise/{sub_id}_{percentage}%noise.nii.gz",
        "denoised": f"./result/denoised/{data_folder}/0_{percentage}p_denoised.nii.gz",
        "gt": f"./data/test/gt/{sub_id}.nii.gz",
        "mask": f"./data/test/mask/{sub_id}_mask.nii.gz",
        "bval": f"./data/test/bvals/{sub_id}_bval",
        "bvec": f"./data/test/bvecs/{sub_id}_bvec",
        "json_path": "./normalize/test_minmax.json",
    }

    output = f"./result/evaluate/{folder}/"
    Path(output).mkdir(parents=True, exist_ok=True)

    evaluator = ShellWiseEvaluator(use_lpips=True)
    evaluator.run(files, output)
