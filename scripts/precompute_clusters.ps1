$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = "D:\Pythonproject\HelloAgent\paper-embed-rec\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "未找到 Python：$Python"
}

Push-Location $ProjectRoot
try {
    & $Python -m backend.app.cli precompute-clusters @args
}
finally {
    Pop-Location
}
