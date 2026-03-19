"""
病灶识别推理模块
支持加载训练好的模型进行预测，无模型时返回占位结果供联调
"""

from pathlib import Path
from typing import Optional

import yaml
from PIL import Image

from .schema import LesionDescription


class LesionPredictor:
    """病灶识别预测器"""

    def __init__(self, config_path: str = "config/config.yaml", root_dir: Optional[Path] = None):
        self.root = root_dir or Path(config_path).resolve().parent.parent
        self.config = self._load_config(config_path)
        self.model = None
        self.terminology = None
        self.model_loaded = False
        self._load_model()

    def _load_config(self, config_path: str) -> dict:
        """加载配置"""
        path = Path(config_path)
        if not path.exists():
            return {"model": {"device": "cpu", "image_size": 224}}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_model(self) -> None:
        """加载模型，若不存在则保持 None"""
        cfg = self.config.get("model", {})
        checkpoint_path = cfg.get("checkpoint_path", "")
        if not checkpoint_path:
            return
        path = Path(checkpoint_path)
        if not path.is_absolute():
            path = self.root / path
        if not path.exists():
            return
        try:
            import torch

            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                ckpt = torch.load(path, map_location="cpu")
            num_classes = ckpt.get("num_classes")
            terminology = ckpt.get("terminology")
            if not num_classes or not terminology:
                return

            from training.model import MultiTaskLesionModel
            self.model = MultiTaskLesionModel(num_classes=num_classes, pretrained=False)
            self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
            device = cfg.get("device", "cpu")
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
            self.model = self.model.to(device)
            self.model.eval()
            self.terminology = terminology
            self.model_loaded = True
        except Exception as e:
            # 静默失败，保持占位模式
            pass

    def _preprocess(self, image: Image.Image):
        """图像预处理，返回 tensor"""
        import torch
        from torchvision import transforms as T

        size = self.config.get("model", {}).get("image_size", 224)
        img = image.convert("RGB").resize((size, size))
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        return transform(img).unsqueeze(0)

    def predict(self, image: Image.Image) -> LesionDescription:
        """对单张图像进行病灶识别"""
        if self.model_loaded and self.model is not None:
            return self._model_predict(image)
        return self._placeholder_predict()

    def _model_predict(self, image: Image.Image) -> LesionDescription:
        """模型推理"""
        import torch

        cfg = self.config.get("model", {})
        device = next(self.model.parameters()).device
        x = self._preprocess(image).to(device)

        with torch.no_grad():
            logits = self.model(x)

        preds = {}
        for task, logit in logits.items():
            probs = logit.softmax(1)[0]
            idx = logit.argmax(1).item()
            opts = self.terminology.get(task, [])
            preds[task] = (opts[idx] if 0 <= idx < len(opts) else opts[0], probs[idx].item())

        # 构建综合描述
        parts = []
        for k in ["color", "size", "position", "shape", "scale"]:
            if k in preds:
                parts.append(preds[k][0])
        summary = "，".join(parts) if parts else "见各维度描述"
        confidence = sum(p[1] for p in preds.values()) / max(1, len(preds))

        return LesionDescription(
            color=preds.get("color", (None, 0))[0],
            size=preds.get("size", (None, 0))[0],
            position=preds.get("position", (None, 0))[0],
            shape=preds.get("shape", (None, 0))[0],
            scale=preds.get("scale", (None, 0))[0],
            texture=preds.get("texture", (None, 0))[0],
            border=preds.get("border", (None, 0))[0],
            confidence=round(confidence, 4),
            summary=summary,
        )

    def _placeholder_predict(self) -> LesionDescription:
        """占位预测，用于无模型时的联调"""
        terminology = self.config.get("terminology", {})
        colors = terminology.get("colors", ["红斑"])
        sizes = terminology.get("size_categories", ["米粒大小 (3-5mm)"])
        positions = terminology.get("positions", ["躯干"])
        shapes = terminology.get("shapes", ["斑片状"])
        scales = terminology.get("scale_presence", ["轻度鳞屑"])
        textures = terminology.get("textures", ["粗糙"])
        borders = terminology.get("borders", ["边界清楚"])

        return LesionDescription(
            color=colors[1] if len(colors) > 1 else colors[0],
            size=sizes[2] if len(sizes) > 2 else sizes[0],
            position=positions[2] if len(positions) > 2 else positions[0],
            shape=shapes[3] if len(shapes) > 3 else shapes[0],
            scale=scales[1] if len(scales) > 1 else scales[0],
            texture=textures[1] if len(textures) > 1 else textures[0],
            border=borders[0],
            confidence=0.0,
            summary="【占位结果】请使用标注数据训练模型后，将 checkpoint 放到 models/checkpoints/ 以启用真实推理。"
        )
