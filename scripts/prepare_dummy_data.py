#!/usr/bin/env python3
"""
生成 dummy 标注数据，用于快速验证训练流程
运行: python scripts/prepare_dummy_data.py
"""

import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "labeled"
IMAGES_DIR = DATA_DIR / "images"

# 示例标注（与 config 中的术语对应）
SAMPLES = [
    {"image_path": "images/lesion_001.jpg", "color": "红斑", "size": "米粒大小 (3-5mm)", "position": "躯干", "shape": "斑片状", "scale": "轻度鳞屑", "texture": "粗糙", "border": "边界清楚"},
    {"image_path": "images/lesion_002.jpg", "color": "色素沉着", "size": "黄豆大小 (5-10mm)", "position": "面部", "shape": "圆形", "scale": "无鳞屑", "texture": "光滑", "border": "边界清楚"},
    {"image_path": "images/lesion_003.jpg", "color": "黄褐色", "size": "蚕豆大小 (10-20mm)", "position": "四肢", "shape": "椭圆形", "scale": "中度鳞屑", "texture": "角化", "border": "边界模糊"},
]


def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for i, sample in enumerate(SAMPLES):
        # 生成 224x224 的随机图像
        img = Image.new("RGB", (224, 224), color=(180 + i * 20, 100, 100))
        img.save(IMAGES_DIR / f"lesion_{i+1:03d}.jpg")

    labels_path = DATA_DIR / "labels.jsonl"
    with open(labels_path, "w", encoding="utf-8") as f:
        for s in SAMPLES:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"已生成 {len(SAMPLES)} 条 dummy 数据:")
    print(f"  图像目录: {IMAGES_DIR}")
    print(f"  标注文件: {labels_path}")
    print("运行训练: python training/train.py --epochs 3")


if __name__ == "__main__":
    main()
