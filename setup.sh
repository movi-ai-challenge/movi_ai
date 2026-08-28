#!/usr/bin/env bash
# ============================================================
# MOVI AI - 개발환경 세팅 (macOS / Linux)
#
#   bash setup.sh          # .venv 생성 + 의존성 설치
#   bash setup.sh --force  # 기존 .venv 를 지우고 새로 생성
# ============================================================
set -euo pipefail
cd "$(dirname "$0")"

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

# ---- 1. Python 찾기 (3.11 또는 3.12) ----------------------
PY=""
for c in python3.12 python3.11 python3; do
  if command -v "$c" >/dev/null 2>&1; then
    v=$("$c" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
    case "$v" in 3.11|3.12) PY="$c"; break;; esac
  fi
done
if [ -z "$PY" ]; then
  echo "ERROR: Python 3.11 또는 3.12 가 필요합니다."
  echo "  brew install python@3.12   또는   https://www.python.org/downloads/"
  exit 1
fi
echo "==> Python: $PY ($("$PY" --version))"

# ---- 2. 기존 venv 검사 -------------------------------------
if [ -d .venv ] && ! ./.venv/bin/python -c "pass" >/dev/null 2>&1; then
  if [ "$FORCE" = "1" ]; then
    echo "==> 이 플랫폼에서 쓸 수 없는 .venv 를 삭제합니다"
    rm -rf .venv
  else
    echo "ERROR: .venv 가 있지만 macOS/Linux 용이 아닙니다 (다른 OS/머신에서 생성됨)."
    echo "       bash setup.sh --force  로 다시 실행하면 삭제 후 재생성합니다."
    exit 1
  fi
fi
[ "$FORCE" = "1" ] && rm -rf .venv

# ---- 3. venv 생성 + 설치 -----------------------------------
[ -d .venv ] || { echo "==> .venv 생성"; "$PY" -m venv .venv; }
echo "==> pip 업그레이드"
./.venv/bin/python -m pip install --upgrade --quiet pip setuptools wheel
echo "==> requirements.txt 설치 (수 분 소요)"
./.venv/bin/python -m pip install -r requirements.txt

# ---- 4. .env -----------------------------------------------
if [ ! -f .env ]; then
  cp .env.example .env
  echo "==> .env 를 생성했습니다. OPENAI_API_KEY 등 실제 값을 채우세요."
fi

# ---- 5. Jupyter 커널 ---------------------------------------
./.venv/bin/python -m ipykernel install --user --name movi-ai \
  --display-name "Python (movi_ai)" >/dev/null 2>&1 || true

echo
echo "완료. 실행:"
echo "  bash run.sh fds     # 이상거래 탐지 API  :8000"
echo "  bash run.sh voice   # 음성 분석 API      :8001"
echo "  bash run.sh test    # 테스트"
