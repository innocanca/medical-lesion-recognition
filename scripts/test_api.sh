#!/bin/bash
# 测试 API（需要先启动服务）

# 健康检查
curl -s http://localhost:8000/health | python -m json.tool

# 使用测试图像分析（如有）
# curl -X POST -F "image=@test.jpg" http://localhost:8000/analyze | python -m json.tool
