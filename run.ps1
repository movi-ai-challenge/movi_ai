# ============================================================
# MOVI AI - 실행 (Windows)
#
#   .\run.ps1 fds     # 이상거래 탐지 API  :8000
#   .\run.ps1 voice   # 음성 분석 API      :8001
#   .\run.ps1 test    # pytest
#   .\run.ps1 shell   # venv python REPL
# ============================================================
param(
    [Parameter(Position = 0)][string]$Target,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if ($Target -notin @("fds", "voice", "test", "shell")) {
    Write-Host "사용법: .\run.ps1 {fds|voice|test|shell}"
    Write-Host "  fds     이상거래 탐지 API  :8000"
    Write-Host "  voice   음성 분석 API      :8001"
    Write-Host "  test    pytest"
    Write-Host "  shell   venv python REPL"
    exit 1
}

# .venv 가 표준. .venv-win 은 예전 세팅에 대한 fallback.
$py = $null
foreach ($d in @(".venv", ".venv-win")) {
    $candidate = Join-Path $PSScriptRoot "$d\Scripts\python.exe"
    if (Test-Path $candidate) { $py = $candidate; break }
}
if (-not $py) {
    Write-Error "가상환경이 없습니다. 먼저 .\setup.ps1 을 실행하세요."
    exit 1
}

$env:PYTHONUTF8 = "1"

switch ($Target) {
    "fds"   { & $py -m uvicorn src.fraud_detection.api:app --reload --port 8000 --env-file .env }
    "voice" { & $py -m uvicorn src.voice_analysis.api:app --reload --port 8001 --env-file .env }
    "test"  { & $py -m pytest -q @Rest }
    "shell" { & $py }
}
