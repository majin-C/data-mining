# 创建独立 Conda 环境，避免把课程设计依赖安装到 base 环境。
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$EnvName = "paper-embed-rec"

Set-Location $ProjectRoot

# 先检查环境是否已经存在，存在则复用，避免重复下载大量依赖。
$existing = conda env list | Select-String -Pattern "^\s*$EnvName\s"
if (-not $existing) {
    # environment.yml 保持 ASCII，避免 Windows 控制台把 YAML 按 GBK 解码时报错。
    conda env create -f environment.yml
}

# 安装 GPU 版 PyTorch。RTX 5070 Laptop 的驱动支持 CUDA 12.8，因此使用 cu128 wheel。
# 这里显式安装 torch / torchvision / torchaudio，避免 PyPI 默认 CPU 版覆盖 GPU 版。
conda run -n $EnvName python -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 明确写入 Hugging Face 镜像站，后续模型下载脚本和后端都会读取这个变量。
Write-Host "Conda 环境已准备完成。运行前请执行：conda activate $EnvName"
Write-Host "模型下载将使用 HF_ENDPOINT=https://hf-mirror.com"
