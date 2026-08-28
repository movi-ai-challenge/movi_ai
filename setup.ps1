# ============================================================
# MOVI AI - 개발환경 세팅 (Windows)
#
#   .\setup.ps1          # .venv 생성 + 의존성 설치
#   .\setup.ps1 -Force   # 기존 .venv 를 지우고 새로 생성
#
# 실행정책 오류가 나면:
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1
# ============================================================
param([switch]$Force)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# ---- 1. Python 찾기 (3.11 또는 3.12) ----------------------
$py = $null
foreach ($v in @("3.12", "3.11")) {
    try {
        $p = (& py "-$v" -c "import sys; print(sys.executable)" 2>$null)
        if ($LASTEXITCODE -eq 0 -and $p) { $py = $p; break }
    } catch {}
}
if (-not $py) {
    try {
        $ver = (& python -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null)
        if ($ver -eq "3.11" -or $ver -eq "3.12") {
            $py = (& python -c "import sys; print(sys.executable)")
        }
    } catch {}
}
if (-not $py) {
    Write-Error "Python 3.11 또는 3.12 가 필요합니다. https://www.python.org/downloads/"
    exit 1
}
Write-Host "==> Python: $py"

# ---- 2. 기존 venv 검사 -------------------------------------
$winPython = ".venv\Scripts\python.exe"
if ((Test-Path ".venv") -and -not (Test-Path $winPython)) {
    if ($Force) {
        Write-Host "==> 이 플랫폼에서 쓸 수 없는 .venv 를 삭제합니다"
        Remove-Item -Recurse -Force ".venv"
    } else {
        Write-Error @"
.venv 가 있지만 Windows 용이 아닙니다 (다른 OS/머신에서 생성됨).
  .\setup.ps1 -Force   로 다시 실행하면 삭제 후 재생성합니다.
"@
        exit 1
    }
} elseif ($Force -and (Test-Path ".venv")) {
    Remove-Item -Recurse -Force ".venv"
}

# ---- 3. venv 생성 + 설치 -----------------------------------
if (-not (Test-Path ".venv")) {
    Write-Host "==> .venv 생성"
    & $py -m venv .venv
}
Write-Host "==> pip 업그레이드"
& $winPython -m pip install --upgrade --quiet pip setuptools wheel
Write-Host "==> requirements.txt 설치 (수 분 소요)"
& $winPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Error "의존성 설치 실패"; exit 1 }

# ---- 4. .env -----------------------------------------------
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "==> .env 를 생성했습니다. OPENAI_API_KEY 등 실제 값을 채우세요."
}

# ---- 5. Jupyter 커널 ---------------------------------------
& $winPython -m ipykernel install --user --name movi-ai --display-name "Python (movi_ai)" 2>$null | Out-Null

Write-Host ""
Write-Host "완료. 실행:"
Write-Host "  .\run.ps1 fds     # 이상거래 탐지 API  :8000"
Write-Host "  .\run.ps1 voice   # 음성 분석 API      :8001"
Write-Host "  .\run.ps1 test    # 테스트"
