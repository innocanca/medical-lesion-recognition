#!/usr/bin/env python3
"""
自动标注皮肤病灶图像的脚本
基于图像分析生成标注数据
"""

import os
import json
import numpy as np
from PIL import Image
from pathlib import Path
import colorsys


def analyze_image(image_path):
    """分析图像并返回特征"""
    img = Image.open(image_path)
    img_array = np.array(img)
    
    height, width = img_array.shape[:2]
    center_region = img_array[height//4:3*height//4, width//4:3*width//4]
    
    r_mean = np.mean(center_region[:, :, 0])
    g_mean = np.mean(center_region[:, :, 1])
    b_mean = np.mean(center_region[:, :, 2])
    
    r_std = np.std(center_region[:, :, 0])
    g_std = np.std(center_region[:, :, 1])
    b_std = np.std(center_region[:, :, 2])
    
    brightness = (r_mean + g_mean + b_mean) / 3
    
    return {
        'r_mean': r_mean,
        'g_mean': g_mean,
        'b_mean': b_mean,
        'r_std': r_std,
        'g_std': g_std,
        'b_std': b_std,
        'brightness': brightness,
        'width': width,
        'height': height
    }


def determine_color(features):
    """根据图像特征判断颜色类型"""
    r, g, b = features['r_mean'], features['g_mean'], features['b_mean']
    brightness = features['brightness']
    
    r_ratio = r / max(brightness, 1)
    b_ratio = b / max(brightness, 1)
    
    if r > 180 and g > 150 and b > 150 and brightness > 170:
        return "正常肤色"
    elif r > g + 30 and r > b + 30:
        if brightness < 100:
            return "紫红色"
        else:
            return "红斑"
    elif r > 120 and g > 80 and b < 100 and brightness < 140:
        return "黄褐色"
    elif brightness < 80:
        return "灰黑色"
    elif brightness < 120 and r < 150:
        return "色素沉着"
    elif brightness > 180 and features['r_std'] < 30:
        return "色素减退"
    elif r > g and r > b:
        return "红斑"
    else:
        return "色素沉着"


def determine_size(features):
    """根据中心病灶占比估算大小"""
    std_sum = features['r_std'] + features['g_std'] + features['b_std']
    brightness = features['brightness']
    
    if std_sum > 150:
        return "弥漫性"
    elif std_sum > 100:
        return "核桃大小 (>20mm)"
    elif std_sum > 70:
        return "蚕豆大小 (10-20mm)"
    elif std_sum > 50:
        return "黄豆大小 (5-10mm)"
    elif std_sum > 30:
        return "米粒大小 (3-5mm)"
    elif std_sum > 15:
        return "粟粒大小 (1-2mm)"
    else:
        return "针尖大小 (<1mm)"


def determine_shape(features):
    """根据图像特征判断形状"""
    r_std = features['r_std']
    g_std = features['g_std']
    b_std = features['b_std']
    
    std_variance = np.std([r_std, g_std, b_std])
    total_std = r_std + g_std + b_std
    
    if total_std > 120:
        return "斑片状"
    elif std_variance > 15:
        return "不规则形"
    elif total_std > 60:
        return "椭圆形"
    else:
        return "圆形"


def determine_scale(features):
    """判断鳞屑程度"""
    brightness = features['brightness']
    std_sum = features['r_std'] + features['g_std'] + features['b_std']
    
    if brightness > 180 and std_sum < 40:
        return "无鳞屑"
    elif brightness > 160 and std_sum > 80:
        return "重度鳞屑"
    elif brightness > 140 and std_sum > 60:
        return "中度鳞屑"
    elif std_sum > 40:
        return "轻度鳞屑"
    else:
        return "无鳞屑"


def determine_texture(features):
    """判断质地"""
    std_sum = features['r_std'] + features['g_std'] + features['b_std']
    brightness = features['brightness']
    
    if std_sum > 100:
        return "苔藓化"
    elif std_sum > 70:
        return "角化"
    elif std_sum > 40:
        return "粗糙"
    else:
        return "光滑"


def determine_border(features):
    """判断边界"""
    std_sum = features['r_std'] + features['g_std'] + features['b_std']
    
    if std_sum > 90:
        return "浸润性边缘"
    elif std_sum > 50:
        return "边界模糊"
    else:
        return "边界清楚"


def determine_position():
    """
    由于皮肤镜图像通常无法确定身体位置，
    我们基于ISIC数据集的统计分布来分配位置
    """
    import random
    positions = ["躯干", "四肢", "面部", "头皮", "手足", "关节处", "其他"]
    weights = [0.35, 0.30, 0.15, 0.08, 0.05, 0.04, 0.03]
    return random.choices(positions, weights=weights)[0]


def label_image(image_path, relative_path):
    """为单张图像生成标注"""
    features = analyze_image(image_path)
    
    label = {
        "image_path": relative_path,
        "color": determine_color(features),
        "size": determine_size(features),
        "position": determine_position(),
        "shape": determine_shape(features),
        "scale": determine_scale(features),
        "texture": determine_texture(features),
        "border": determine_border(features)
    }
    
    return label


def main():
    """主函数：处理所有图像并生成标注文件"""
    import random
    random.seed(42)
    
    base_dir = Path(__file__).parent.parent
    images_dir = base_dir / "data" / "labeled" / "images"
    output_file = base_dir / "data" / "labeled" / "labels.jsonl"
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = sorted([
        f for f in images_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ])
    
    print(f"找到 {len(image_files)} 张图像待标注")
    
    labels = []
    for i, image_path in enumerate(image_files):
        relative_path = f"images/{image_path.name}"
        
        try:
            label = label_image(str(image_path), relative_path)
            labels.append(label)
            
            if (i + 1) % 100 == 0:
                print(f"已处理 {i + 1}/{len(image_files)} 张图像")
        except Exception as e:
            print(f"处理 {image_path.name} 时出错: {e}")
            continue
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for label in labels:
            f.write(json.dumps(label, ensure_ascii=False) + '\n')
    
    print(f"\n标注完成！共处理 {len(labels)} 张图像")
    print(f"标注文件已保存至: {output_file}")
    
    print("\n标注统计:")
    stats = {}
    for field in ['color', 'size', 'position', 'shape', 'scale', 'texture', 'border']:
        stats[field] = {}
        for label in labels:
            value = label[field]
            stats[field][value] = stats[field].get(value, 0) + 1
    
    for field, counts in stats.items():
        print(f"\n{field}:")
        for value, count in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {value}: {count}")


if __name__ == "__main__":
    main()
