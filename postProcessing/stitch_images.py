import os
import cv2
import numpy as np
from PIL import Image
import re

def sort_key(filename):
    """提取文件名中的数字用于排序"""
    match = re.search(r'(\d+)epochs', filename)
    return int(match.group(1)) if match else 0

def stitch_validation_images(val_dir, output_path, num_images=5, scale_factor=0.5):
    """
    将验证图片按横向拼接成一张大图，包含三行：noisy、denoised、gt
    
    Args:
        val_dir: 验证图片目录路径
        output_path: 输出图片路径
        num_images: 要拼接的图片数量
        scale_factor: 缩放因子，用于缩小图像
    """
    # 获取所有验证图片文件
    image_files = [f for f in os.listdir(val_dir) if f.endswith('.png')]
    
    # 按照epoch数排序
    image_files.sort(key=sort_key)
    
    # 只取前num_images张图片
    image_files = image_files[:num_images]
    
    if len(image_files) < num_images:
        print(f"警告：只找到{len(image_files)}张图片，少于所需的{num_images}张")
    
    # 读取第一张图片获取原始尺寸信息
    first_img_path = os.path.join(val_dir, image_files[0])
    first_img = cv2.imread(first_img_path)
    if first_img is None:
        raise ValueError(f"无法读取图片: {first_img_path}")
    
    # 获取原始尺寸
    orig_height, orig_width = first_img.shape[:2]
    
    # 计算缩放后的尺寸
    resized_first_img = cv2.resize(first_img, None, fx=scale_factor, fy=scale_factor)
    img_height, img_width = resized_first_img.shape[:2]
    
    # 原始图片包含三个水平排列的子图，每个子图的宽度是总宽度的1/3
    orig_sub_img_width = orig_width // 3
    sub_img_width = img_width // 3
    
    # 创建一个足够大的空白画布（三行，每行包含num_images个子图）
    # 确保画布尺寸计算正确
    canvas_height = img_height * 3
    canvas_width = sub_img_width * len(image_files)
    stitched_img = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    
    print(f"画布尺寸: {canvas_height}x{canvas_width}")
    print(f"子图尺寸: {img_height}x{sub_img_width}")
    
    # 处理每张图片，提取三个子图并放置到对应的行中
    for col, img_file in enumerate(image_files):
        img_path = os.path.join(val_dir, img_file)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"警告：无法读取图片 {img_path}，跳过")
            continue
            
        # 缩放图片
        img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)
        
        # 验证缩放后图像尺寸是否一致
        if img.shape[0] != img_height or img.shape[1] != img_width:
            print(f"警告：图像 {img_file} 尺寸不一致，调整为统一尺寸")
            img = cv2.resize(img, (img_width, img_height))
            
        # 提取三个子图（noisy、denoised、gt）
        noisy_sub = img[:, 0:sub_img_width, :]
        denoised_sub = img[:, sub_img_width:2*sub_img_width, :]
        gt_sub = img[:, 2*sub_img_width:3*sub_img_width, :]
        
        # 放置到画布上对应的行中
        stitched_img[0:img_height, col*sub_img_width:(col+1)*sub_img_width] = noisy_sub
        stitched_img[img_height:2*img_height, col*sub_img_width:(col+1)*sub_img_width] = denoised_sub
        stitched_img[2*img_height:3*img_height, col*sub_img_width:(col+1)*sub_img_width] = gt_sub
        
        # 提取PSNR值并添加到denoised行上
        match = re.search(r'PSNR=([0-9.]+)', img_file)
        if match:
            psnr_val = match.group(1)
            # 调整文本位置，使其显示在正确的行上
            cv2.putText(stitched_img, f'{psnr_val}dB', 
                       (col*sub_img_width + 10, img_height*2 - 30),  # 调整位置到denoised行
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7 * scale_factor, (255, 255, 255), 2)
    
    # 保存拼接后的图片
    cv2.imwrite(output_path, stitched_img)
    print(f"已保存拼接图片到: {output_path}")
    print(f"共拼接了 {len(image_files)} 组图片")
    print(f"缩放因子: {scale_factor}")

def main():
    # 设置路径
    folder_name = '202601012328'  # 设置文件夹名称
    val_dir = f'result/val/{folder_name}'  # 验证图片目录
    output_dir = f'result/val/{folder_name}'
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置输出文件名
    output_path = os.path.join(output_dir, 'validation_progress_stitched.png')
    
    # 拼接图片，使用0.3的缩放因子使图像更小
    stitch_validation_images(val_dir, output_path, num_images=10, scale_factor=0.3)

if __name__ == '__main__':
    main()