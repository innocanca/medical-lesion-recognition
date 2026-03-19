"""
医学规范术语输出 schema
用于标准化描述病灶的颜色、大小、位置、形状等多维度特征
"""

from pydantic import BaseModel, Field
from typing import Optional


class LesionDescription(BaseModel):
    """病灶多维度医学规范描述"""

    # 颜色 (Color)
    color: Optional[str] = Field(None, description="颜色描述，如红斑、色素沉着等")

    # 大小 (Size)
    size: Optional[str] = Field(None, description="大小分类，如粟粒大小、蚕豆大小等")

    # 位置 (Position)
    position: Optional[str] = Field(None, description="病灶位置，如面部、躯干等")

    # 形状 (Shape)
    shape: Optional[str] = Field(None, description="形状描述，如圆形、斑片状等")

    # 鳞屑 (Scale)
    scale: Optional[str] = Field(None, description="鳞屑程度，无/轻度/中度/重度")

    # 质地 (Texture)
    texture: Optional[str] = Field(None, description="质地，如光滑、粗糙、角化等")

    # 边界 (Border)
    border: Optional[str] = Field(None, description="边界清晰度")

    # 置信度 (0-1)
    confidence: Optional[float] = Field(None, description="整体预测置信度")

    # 完整文字描述（辅助医生快速阅读）
    summary: Optional[str] = Field(None, description="综合描述摘要")


class AnalysisResponse(BaseModel):
    """API 分析响应"""

    success: bool = True
    lesion_description: LesionDescription = Field(..., description="病灶医学规范描述")
    model_version: Optional[str] = Field(None, description="使用的模型版本")


class HealthResponse(BaseModel):
    """健康检查响应"""

    status: str = "ok"
    model_loaded: bool = False
