# 启动 Vue 前端开发服务器。
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location (Join-Path $ProjectRoot "frontend")

# 依次尝试当前 Conda 环境和项目常用环境，找到第一个真实存在的 npm.cmd。
# 这里不直接对 $env:CONDA_PREFIX 调 Join-Path，因为用户可能在 base 环境、未激活环境，
# 或者某些终端里该变量为空；先过滤空路径可以避免出现 “LiteralPath 为空值” 的报错。
$NodeCandidates = @(
    $env:CONDA_PREFIX,
    "D:\Pythonproject\HelloAgent\paper-embed-rec"
) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }

$NodeRoot = $null
$Npm = $null
foreach ($Candidate in $NodeCandidates) {
    $CandidateNpm = [System.IO.Path]::Combine($Candidate, "npm.cmd")
    if (Test-Path -LiteralPath $CandidateNpm) {
        $NodeRoot = $Candidate
        $Npm = $CandidateNpm
        break
    }
}

# 如果两个候选路径都没有 npm.cmd，直接抛出清晰错误，避免后续 & $Npm 变成空命令。
if ([string]::IsNullOrWhiteSpace($Npm)) {
    throw "未找到 npm.cmd，请先激活 paper-embed-rec 环境，或检查 D:\Pythonproject\HelloAgent\paper-embed-rec 是否存在。"
}

$env:PATH = "$NodeRoot;$env:PATH"

& $Npm install --registry=https://registry.npmmirror.com --cache (Join-Path $ProjectRoot ".npm-cache") --prefer-online
& $Npm run dev
