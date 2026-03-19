#!/usr/bin/env python3
"""
病灶多标签分类训练脚本
使用标注数据 (data/labeled/labels.jsonl) 训练模型
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

# 添加项目根目录
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_labels(labels_path: str) -> List[dict]:
    """加载 JSONL 标注"""
    data = []
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser(description="病灶识别模型训练")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(ROOT / "data" / "labeled"),
        help="标注数据目录",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "models" / "checkpoints"),
        help="模型输出目录",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="训练轮数",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="批次大小",
    )
    args = parser.parse_args()

    labels_path = Path(args.data_dir) / "labels.jsonl"
    if not labels_path.exists():
        print(f"错误: 未找到标注文件 {labels_path}")
        print("请按照 data/README.md 格式准备标注数据")
        sys.exit(1)

    data = load_labels(str(labels_path))
    print(f"加载了 {len(data)} 条标注")

    # 检查图像路径
    data_dir = Path(args.data_dir)
    valid = 0
    for item in data:
        img_path = data_dir / item.get("image_path", "")
        if img_path.exists():
            valid += 1
    print(f"有效图像: {valid}/{len(data)}")

    # TODO: 接入实际训练流程
    # 1. 构建多任务/多标签数据集
    # 2. 使用 ResNet/ViT 等 backbone + 多输出头
    # 3. 训练并保存最佳模型
    print("\n训练流程占位: 请根据实际 backbone 和标注格式实现 train_epoch() 等逻辑")
    print(f"输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
