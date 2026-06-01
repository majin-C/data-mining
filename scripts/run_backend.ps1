# 启动 FastAPI 后端服务。
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$env:HF_ENDPOINT = "https://hf-mirror.com"
conda run -n paper-embed-rec uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload

