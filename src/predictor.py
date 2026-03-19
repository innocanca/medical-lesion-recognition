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

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.model = None
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
        checkpoint = cfg.get("checkpoint_path", "")
        if not checkpoint or not Path(checkpoint).exists():
            return
        try:
            import torch
            # 预留：加载实际模型
            # self.model = load_your_model(checkpoint)
            self.model_loaded = True
        except Exception:
            pass

    def _preprocess(self, image: Image.Image) -> Image.Image:
        """图像预处理"""
        size = self.config.get("model", {}).get("image_size", 224)
        return image.convert("RGB").resize((size, size))

    def predict(self, image: Image.Image) -> LesionDescription:
        """
        对单张图像进行病灶识别
        """
        _ = self._preprocess(image)  # 预处理

        if self.model_loaded and self.model is not None:
            # 实际模型推理
            return self._model_predict(image)
        else:
            # 无模型时返回占位结果，便于 API 联调
            return self._placeholder_predict()

    def _model_predict(self, image: Image.Image) -> LesionDescription:
        """模型推理（待接入实际模型）"""
        # TODO: 调用 self.model 进行推理
        return self._placeholder_predict()

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
            confidence=0.0,  # 占位时置信度为 0
            summary="【占位结果】请使用标注数据训练模型后，将 checkpoint 放到 models/checkpoints/ 以启用真实推理。"
        )
