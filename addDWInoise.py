import nibabel as nib
import numpy as np
import os
from tqdm import tqdm


def add_rician_noise(input_file, output_file, noise_percentage):
    """
    给nii文件添加Rician噪声并保存 (基于归一化百分比 σ)
    参数:
    input_file (str): 输入的nii文件路径
    output_file (str): 输出的加噪nii文件路径
    noise_percentage (float): 噪声百分比 (0-100)，例如 2 表示 σ=0.02
    """
    # 加载nii文件
    print("加载输入图像...")
    img = nib.load(input_file)
    data = img.get_fdata()

    # 保存原始范围
    data_max = np.max(data)
    if data_max == 0:
        raise ValueError("输入图像的最大值为0，无法归一化")

    # 归一化到 [0,1]
    print("归一化数据...")
    data_norm = data / data_max

    # 获取数据形状
    shape = data_norm.shape
    if len(shape) > 3:
        data_reshaped = data_norm.reshape(-1, shape[-1])
    else:
        data_reshaped = data_norm.reshape(-1, 1)

    # 噪声标准差（相对于归一化后的数据）
    noise_std = noise_percentage / 100.0

    # 生成复高斯噪声
    print("生成复高斯噪声...")
    noise_real = np.random.normal(0, noise_std, data_reshaped.shape)
    noise_imag = np.random.normal(0, noise_std, data_reshaped.shape)

    # 添加噪声（带进度条）
    noisy_data = np.zeros_like(data_reshaped)
    print("添加Rician噪声...")
    for i in tqdm(range(data_reshaped.shape[0]), desc=f"添加 {noise_percentage}% 噪声"):
        noisy_data[i] = np.sqrt((data_reshaped[i] + noise_real[i])**2 + noise_imag[i]**2)

    # 恢复原始形状
    noisy_data = noisy_data.reshape(shape)

    # 反归一化回原始强度范围
    print("反归一化数据...")
    noisy_data = noisy_data * data_max

    # 创建新的nii图像
    print("创建新的nii.gz图像...")
    noisy_data = noisy_data.astype(np.float32)
    noisy_img = nib.Nifti1Image(noisy_data, img.affine, img.header)

    # 保存加噪后的图像（带进度条）
    print("保存加噪后的图像...")
    nib.save(noisy_img, output_file)

    print(f"✅ 成功添加 {noise_percentage}% 的Rician噪声")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"噪声标准差 (归一化尺度): {noise_std}")


def main():
    percentage = 6 # 添加噪声（σ=0.0？）
    file_index = 105923
    input_dir = f"./data/test/gt/sub-{file_index}__dwi.nii.gz"
    output_dir = f"./data/test/noise/sub-{file_index}__dwi_{percentage}%noise.nii.gz"

    # 检查输入文件是否存在
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"输入文件不存在: {input_dir}")

    # 检查噪声百分比是否有效
    if percentage < 0 or percentage > 100:
        raise ValueError("噪声百分比必须在0-100之间")

    # 添加噪声并保存
    add_rician_noise(input_dir, output_dir, percentage)


if __name__ == "__main__":
    main()