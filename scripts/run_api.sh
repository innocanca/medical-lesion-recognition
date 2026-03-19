#!/bin/bash
# 启动 API 服务

cd "$(dirname "$0")/.." || exit 1
exec python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
