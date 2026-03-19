"""
病灶识别 API 服务
提供 RESTful 接口供医生/系统调用
"""

import io
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# 添加项目根目录到路径
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.predictor import LesionPredictor
from src.schema import AnalysisResponse, LesionDescription, HealthResponse

app = FastAPI(
    title="医疗病灶识别 API",
    description="基于图像的病灶医学规范描述识别，辅助医生诊断。支持颜色、大小、位置、形状、鳞屑等多维度描述。",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局预测器
predictor: LesionPredictor = None


@app.on_event("startup")
async def startup():
    global predictor
    root = Path(__file__).resolve().parent.parent
    config_path = root / "config" / "config.yaml"
    predictor = LesionPredictor(str(config_path))


@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查"""
    return HealthResponse(
        status="ok",
        model_loaded=predictor.model_loaded if predictor else False,
    )


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_lesion(image: UploadFile = File(...)):
    """
    上传病灶图像，返回医学规范术语描述
    
    支持格式: JPEG, PNG
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "请上传图片文件 (JPEG/PNG)")

    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(400, f"无法解析图像: {str(e)}")

    description = predictor.predict(img)
    return AnalysisResponse(
        success=True,
        lesion_description=description,
        model_version="placeholder" if not predictor.model_loaded else "v1",
    )


@app.post("/analyze/base64")
async def analyze_base64(body: dict):
    """
    通过 base64 图像数据进行分析（可选接口）
    body: {"image": "base64_encoded_string"}
    """
    import base64
    b64 = body.get("image")
    if not b64:
        raise HTTPException(400, "缺少 image 字段")
    try:
        data = base64.b64decode(b64)
        img = Image.open(io.BytesIO(data))
    except Exception as e:
        raise HTTPException(400, f"无效的 base64 图像: {str(e)}")
    description = predictor.predict(img)
    return AnalysisResponse(
        success=True,
        lesion_description=description,
        model_version="placeholder" if not predictor.model_loaded else "v1",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
