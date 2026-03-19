# 医疗病灶识别系统 (Medical Lesion Recognition)

基于图像的病灶医学规范术语识别，辅助医生进行诊断。系统对患处进行多维度描述，包括**颜色、大小、位置、形状、鳞屑**等，输出符合医学规范的标准化术语。

> 📖 **详细文档**：参见 [docs/使用指南.md](docs/使用指南.md) —— 包含完整的训练步骤与 API 调用说明。

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
pip install -r requirements-train.txt  # 训练依赖
```

### 2. 启动服务

**启动 API 服务**
```bash
# 方式一：直接运行
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# 方式二：Docker
docker-compose up -d
```

**启动前端页面**
```bash
cd web && python -m http.server 8080
```

### 3. 访问地址

| 服务 | 地址 |
|------|------|
| 前端页面 | http://localhost:8080 |
| API 服务 | http://localhost:8000 |
| API 文档 (Swagger) | http://localhost:8000/docs |

### 4. 调用接口

**健康检查**
```bash
curl http://localhost:8000/health
```

**图像分析（文件上传）**
```bash
curl -X POST -F "image=@your_lesion_image.jpg" http://localhost:8000/analyze
```

**图像分析（Base64）**
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_string"}' \
  http://localhost:8000/analyze/base64
```

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
├── scripts/             # 工具脚本
│   └── auto_label.py    # 自动标注脚本
├── web/                 # 前端页面
│   └── index.html       # 病灶识别 Web 界面
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

### 1. 安装训练依赖

```bash
pip install -r requirements.txt
pip install -r requirements-train.txt
```

### 2. 准备标注数据

- 将图像放入 `data/labeled/images/`
- 创建 `data/labeled/labels.jsonl`，每行一条 JSON（参考 `data/labeled/labels.example.jsonl`）：

```json
{"image_path": "images/xxx.jpg", "color": "红斑", "size": "米粒大小 (3-5mm)", "position": "躯干", "shape": "斑片状", "scale": "轻度鳞屑", "texture": "粗糙", "border": "边界清楚"}
```

**自动标注（可选）**：如果有大量图片需要初步标注，可使用自动标注脚本：
```bash
python scripts/auto_label.py
```

### 3. 执行训练

```bash
# 标准训练（使用 ImageNet 预训练权重）
python training/train.py --data-dir data/labeled --epochs 20 --batch-size 16

# 如遇网络问题无法下载预训练权重，使用此命令
python training/train.py --data-dir data/labeled --epochs 20 --batch-size 16 --no-pretrained
```

常用参数：
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--epochs` | 训练轮数 | 20 |
| `--batch-size` | 批次大小 | 16 |
| `--lr` | 学习率 | 1e-4 |
| `--val-ratio` | 验证集比例 | 0.2 |
| `--device` | 设备 (cuda/cpu) | auto |
| `--no-pretrained` | 不使用预训练权重 | false |

### 4. 使用训练好的模型

训练完成后，最佳模型会保存至 `models/checkpoints/best_model.pt`。重启 API 服务后会自动加载该模型。

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
