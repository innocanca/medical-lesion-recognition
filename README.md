# 医疗病灶识别系统 (Medical Lesion Recognition)

基于图像的病灶医学规范术语识别，辅助医生进行诊断。系统对患处进行多维度描述，包括**颜色、大小、位置、形状、鳞屑**等，输出符合医学规范的标准化术语。

## 功能特性

- **多维度描述**：颜色、大小、位置、形状、鳞屑、质地、边界等
- **RESTful API**：以接口形式调用，便于集成到现有诊疗系统
- **可扩展术语**：通过配置文件自定义医学术语
- **标注数据支持**：支持自定义标注数据进行模型训练

## 快速开始

### 1. 环境准备

```bash
cd medical-lesion-recognition
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 启动 API 服务

```bash
# 方式一：直接运行
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# 方式二：Docker
docker-compose up -d
```

### 3. 调用接口

**健康检查**
```bash
curl http://localhost:8000/health
```

**图像分析**
```bash
curl -X POST -F "image=@your_lesion_image.jpg" http://localhost:8000/analyze
```

**API 文档**  
启动后访问: http://localhost:8000/docs

## 项目结构

```
medical-lesion-recognition/
├── api/                 # FastAPI 服务
│   └── main.py
├── config/              # 配置文件
│   └── config.yaml      # 医学术语与模型配置
├── data/                # 数据目录
│   ├── raw/             # 原始图像
│   └── labeled/         # 标注数据（含 labels.jsonl）
├── models/              # 模型
│   └── checkpoints/     # 训练 checkpoint
├── src/                 # 核心逻辑
│   ├── schema.py        # 输出 schema
│   └── predictor.py     # 推理逻辑
├── training/            # 训练脚本
│   └── train.py
├── scripts/             # 脚本
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 提供标注数据

请按 `data/README.md` 准备标注数据：

1. 将图像放入 `data/labeled/images/`
2. 创建 `data/labeled/labels.jsonl`，每行一条 JSON：
   ```json
   {"image_path": "images/xxx.jpg", "color": "红斑", "size": "米粒大小 (3-5mm)", "position": "躯干", "shape": "斑片状", "scale": "轻度鳞屑", "texture": "粗糙", "border": "边界清楚"}
   ```

## 训练模型

准备完标注数据后执行：

```bash
python training/train.py --data-dir data/labeled --epochs 10
```

训练完成后将最佳模型保存至 `models/checkpoints/`，API 将自动加载。

## 医学术语配置

在 `config/config.yaml` 的 `terminology` 中可扩展或修改颜色、大小、位置、形状等选项，保持与标注数据一致。

## 上传到 GitHub

```bash
cd medical-lesion-recognition
git init
git add .
git commit -m "Initial commit: Medical Lesion Recognition API"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/medical-lesion-recognition.git
git push -u origin main
```

## License

MIT
