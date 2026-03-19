"""
病灶标注数据集
将 labels.jsonl 转为 PyTorch Dataset，支持多任务分类
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


# 各维度的默认类别列表（与 config 对应）
DEFAULT_TERMINOLOGY = {
    "color": ["正常肤色", "红斑", "色素沉着", "色素减退", "紫红色", "黄褐色", "灰黑色", "其他"],
    "size": ["针尖大小 (<1mm)", "粟粒大小 (1-2mm)", "米粒大小 (3-5mm)", "黄豆大小 (5-10mm)", "蚕豆大小 (10-20mm)", "核桃大小 (>20mm)", "弥漫性"],
    "position": ["面部", "头皮", "躯干", "四肢", "手足", "关节处", "黏膜", "其他"],
    "shape": ["圆形", "椭圆形", "不规则形", "斑片状", "条索状", "网状", "靶形", "其他"],
    "scale": ["无鳞屑", "轻度鳞屑", "中度鳞屑", "重度鳞屑"],
    "texture": ["光滑", "粗糙", "角化", "苔藓化", "其他"],
    "border": ["边界清楚", "边界模糊", "浸润性边缘", "其他"],
}

TASKS = ["color", "size", "position", "shape", "scale", "texture", "border"]


def load_terminology_from_yaml(config_path: str) -> Dict[str, List[str]]:
    """从 config 加载术语，若失败则用默认"""
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        term = cfg.get("terminology", {})
        return {
            "color": term.get("colors", DEFAULT_TERMINOLOGY["color"]),
            "size": term.get("size_categories", DEFAULT_TERMINOLOGY["size"]),
            "position": term.get("positions", DEFAULT_TERMINOLOGY["position"]),
            "shape": term.get("shapes", DEFAULT_TERMINOLOGY["shape"]),
            "scale": term.get("scale_presence", DEFAULT_TERMINOLOGY["scale"]),
            "texture": term.get("textures", DEFAULT_TERMINOLOGY["texture"]),
            "border": term.get("borders", DEFAULT_TERMINOLOGY["border"]),
        }
    except Exception:
        return DEFAULT_TERMINOLOGY


class LesionDataset(Dataset):
    """
    病灶多任务分类数据集
    每个样本输出：image, {task: label_index}
    """

    def __init__(
        self,
        data_dir: str,
        labels_path: str,
        terminology: Dict[str, List[str]],
        image_size: int = 224,
        is_train: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.terminology = terminology
        self.image_size = image_size
        self.is_train = is_train

        # 构建 label -> index 映射
        self.task_to_idx: Dict[str, Dict[str, int]] = {}
        for task, options in terminology.items():
            self.task_to_idx[task] = {opt: i for i, opt in enumerate(options)}

        self.samples: List[dict] = []
        with open(labels_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))

        # 基础 transform
        if is_train:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(),
                T.RandomAffine(degrees=10, translate=(0.1, 0.1)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        item = self.samples[idx]
        img_path = self.data_dir / item["image_path"]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        labels = {}
        for task in TASKS:
            val = item.get(task)
            idx_map = self.task_to_idx.get(task, {})
            if val in idx_map:
                labels[task] = idx_map[val]
            else:
                labels[task] = -1  # 缺失时忽略该任务
        return image, {k: torch.tensor(v, dtype=torch.long) for k, v in labels.items()}
