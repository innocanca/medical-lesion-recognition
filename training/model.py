"""
多任务病灶分类模型
ResNet  backbone + 各维度独立分类头
"""

import os
from typing import Dict, List

import torch
import torch.nn as nn
from torchvision import models


class MultiTaskLesionModel(nn.Module):
    """
    多任务分类模型
    backbone: ResNet18 (可改为 ResNet50 等)
    每个任务一个独立的全连接头
    """

    def __init__(self, num_classes: Dict[str, int], pretrained: bool = True):
        super().__init__()
        if pretrained:
            try:
                self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                print("已加载 ImageNet 预训练权重")
            except (AttributeError, Exception) as e:
                print(f"警告: 无法加载预训练权重 ({e})")
                print("使用随机初始化权重继续训练...")
                self.backbone = models.resnet18(weights=None)
        else:
            self.backbone = models.resnet18(weights=None)
            print("使用随机初始化权重")
        self.backbone.fc = nn.Identity()
        feat_dim = 512

        self.heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num),
            )
            for task, num in num_classes.items()
        })

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.backbone(x)
        return {task: head(feat) for task, head in self.heads.items()}
