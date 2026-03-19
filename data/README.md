# 数据目录说明

## 目录结构

```
data/
├── raw/          # 原始未标注图像
├── labeled/      # 标注后的数据
│   ├── images/   # 标注对应的图像文件
│   └── labels.jsonl  # 标注文件（JSONL 格式）
└── README.md     # 本文件
```

## 标注格式 (labels.jsonl)

每行一个 JSON 对象，格式如下：

```json
{
  "image_path": "images/lesion_001.jpg",
  "color": "红斑",
  "size": "米粒大小 (3-5mm)",
  "position": "躯干",
  "shape": "斑片状",
  "scale": "轻度鳞屑",
  "texture": "粗糙",
  "border": "边界清楚",
  "diagnosis": "银屑病"
}
```

### 字段说明

| 字段 | 必填 | 说明 |
|------|------|------|
| image_path | 是 | 相对于 `data/labeled/` 的图像路径 |
| color | 是 | 颜色，需在 config 的 terminology.colors 中 |
| size | 是 | 大小分类 |
| position | 是 | 位置 |
| shape | 是 | 形状 |
| scale | 是 | 鳞屑程度 |
| texture | 否 | 质地 |
| border | 否 | 边界 |
| diagnosis | 否 | 诊断（可选，用于辅助训练） |

### 术语参考

请参考 `config/config.yaml` 中的 `terminology` 部分，保持标注术语与配置一致。
如需扩展，可同时修改 config 和本说明。
