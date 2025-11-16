import os
import cv2
import numpy as np
from PIL import Image
import re

def sort_key(filename):
    """提取文件名中的数字用于排序"""
    match = re.search(r'(\d+)epochs', filename)
    return int(match.group(1)) if match else 0

def stitch_validation_images(val_dir, output_path, num_images=5):
    """
    将验证图片按垂直顺序拼接成一张大图
    
    Args:
        val_dir: 验证图片目录路径
        output_path: 输出图片路径
        num_images: 要拼接的图片数量
    """
    # 获取所有验证图片文件
    image_files = [f for f in os.listdir(val_dir) if f.endswith('.png')]
    
    # 按照epoch数排序
    image_files.sort(key=sort_key)
    
    # 只取前num_images张图片
    image_files = image_files[:num_images]
    
    if len(image_files) < num_images:
        print(f"警告：只找到{len(image_files)}张图片，少于所需的{num_images}张")
    
    # 读取第一张图片获取尺寸信息
    first_img_path = os.path.join(val_dir, image_files[0])
    first_img = cv2.imread(first_img_path)
    if first_img is None:
        raise ValueError(f"无法读取图片: {first_img_path}")
    
    img_height, img_width = first_img.shape[:2]
    
    # 创建一个足够大的空白画布（垂直排列）
    stitched_img = np.zeros((img_height * len(image_files), img_width, 3), dtype=np.uint8)
    
    # 逐个读取图片并放置到画布上
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(val_dir, img_file)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"警告：无法读取图片 {img_path}，跳过")
            continue
            
        # 调整图片大小以匹配画布宽度（如果需要）
        if img.shape[1] != img_width:
            img = cv2.resize(img, (img_width, img_height))
            
        # 放置到画布上
        stitched_img[i*img_height:(i+1)*img_height, :img_width] = img
        
        # 在图片左侧添加epoch标签
        epoch_num = sort_key(img_file)
        cv2.putText(stitched_img, f'Epoch {epoch_num}', 
                   (10, i*img_height + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 255, 255), 2)
    
    # 保存拼接后的图片
    cv2.imwrite(output_path, stitched_img)
    print(f"已保存拼接图片到: {output_path}")
    print(f"共拼接了 {len(image_files)} 张图片")

def main():
    # 设置路径
    val_dir = 'result/val'  # 验证图片目录
    output_dir = 'result/val'
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置输出文件名
    output_path = os.path.join(output_dir, 'validation_progress_stitched.png')
    
    # 拼接图片
    stitch_validation_images(val_dir, output_path, num_images=10)

if __name__ == '__main__':
    main()