#!/usr/bin/env python3
"""
病灶多任务分类训练脚本
使用标注数据 (data/labeled/labels.jsonl) 训练模型
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# 添加项目根目录
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.dataset import LesionDataset, load_terminology_from_yaml, TASKS
from training.model import MultiTaskLesionModel


def load_labels(labels_path: str) -> List[dict]:
    """加载 JSONL 标注"""
    data = []
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def collate_fn(batch):
    """自定义 collate，处理 dict 形式的 label"""
    images = torch.stack([b[0] for b in batch])
    labels = {task: torch.stack([b[1][task] for b in batch]) for task in TASKS if task in batch[0][1]}
    return images, labels


def train_epoch(model, loader, optim, device, num_classes):
    model.train()
    total_loss = 0.0
    criterions = {task: nn.CrossEntropyLoss(ignore_index=-1) for task in TASKS}

    for images, labels in loader:
        images = images.to(device)
        optim.zero_grad()
        logits = model(images)

        loss = 0.0
        for task in TASKS:
            if task not in labels:
                continue
            tgt = labels[task].to(device)
            mask = tgt >= 0
            if mask.any():
                loss = loss + criterions[task](logits[task], tgt)

        loss.backward()
        optim.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, device, terminology):
    model.eval()
    correct = {task: 0 for task in TASKS}
    total = {task: 0 for task in TASKS}

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        for task in TASKS:
            if task not in labels:
                continue
            tgt = labels[task].to(device)
            mask = tgt >= 0
            if mask.any():
                pred = logits[task].argmax(1)
                correct[task] += (pred[mask] == tgt[mask]).sum().item()
                total[task] += mask.sum().item()

    acc = {t: correct[t] / total[t] if total[t] > 0 else 0.0 for t in TASKS}
    return acc


def main():
    parser = argparse.ArgumentParser(description="病灶识别模型训练")
    parser.add_argument("--data-dir", type=str, default=str(ROOT / "data" / "labeled"), help="标注数据目录")
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "models" / "checkpoints"), help="模型输出目录")
    parser.add_argument("--config", type=str, default=str(ROOT / "config" / "config.yaml"), help="配置文件路径")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=16, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备")
    parser.add_argument("--no-pretrained", action="store_true", help="不使用预训练权重（网络问题时使用）")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    labels_path = Path(args.data_dir) / "labels.jsonl"
    if not labels_path.exists():
        print(f"错误: 未找到标注文件 {labels_path}")
        print("请按照 data/README.md 格式准备标注数据，参考 data/labeled/labels.example.jsonl")
        sys.exit(1)

    terminology = load_terminology_from_yaml(args.config)
    num_classes = {task: len(opts) for task, opts in terminology.items()}
    dataset = LesionDataset(
        data_dir=args.data_dir,
        labels_path=str(labels_path),
        terminology=terminology,
        image_size=224,
        is_train=True,
    )

    n_val = int(len(dataset) * args.val_ratio)
    n_train = len(dataset) - n_val
    if n_train < 1 or n_val < 1:
        print("错误: 数据量过少，至少需要 2 条以上才能划分 train/val")
        sys.exit(1)
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    device = torch.device(args.device)
    use_pretrained = not args.no_pretrained
    model = MultiTaskLesionModel(num_classes=num_classes, pretrained=use_pretrained).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    print(f"训练样本: {n_train}, 验证样本: {n_val}")
    print(f"设备: {device}")
    print("-" * 50)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optim, device, num_classes)
        scheduler.step()
        acc = evaluate(model, val_loader, device, terminology)
        mean_acc = sum(acc.values()) / max(1, sum(1 for v in acc.values() if v > 0))
        if mean_acc > best_acc:
            best_acc = mean_acc
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "terminology": terminology,
                "num_classes": num_classes,
            }
            torch.save(ckpt, output_dir / "best_model.pt")
            print(f"Epoch {epoch:3d} | loss={loss:.4f} | val_acc={mean_acc:.4f} | 已保存最佳模型")

        if epoch % 5 == 0:
            print(f"  各任务准确率: {', '.join(f'{k}={v:.2f}' for k, v in acc.items() if v > 0)}")

    print("-" * 50)
    print(f"训练完成，最佳验证准确率: {best_acc:.4f}")
    print(f"模型已保存至: {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
