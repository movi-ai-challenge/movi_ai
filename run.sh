#!/usr/bin/env bash
# ============================================================
# MOVI AI - 실행 (macOS / Linux)
#
#   bash run.sh fds     # 이상거래 탐지 API  :8000
#   bash run.sh voice   # 음성 분석 API      :8001
#   bash run.sh test    # pytest
#   bash run.sh shell   # venv python REPL
# ============================================================
set -euo pipefail
cd "$(dirname "$0")"

TARGET="${1:-}"
case "$TARGET" in
  fds|voice|test|shell) ;;
  *)
    echo "사용법: bash run.sh {fds|voice|test|shell}"
    echo "  fds     이상거래 탐지 API  :8000"
    echo "  voice   음성 분석 API      :8001"
    echo "  test    pytest"
    echo "  shell   venv python REPL"
    exit 1
    ;;
esac

PY=./.venv/bin/python
if ! "$PY" -c "pass" >/dev/null 2>&1; then
  echo "ERROR: .venv 가 없거나 이 플랫폼에서 쓸 수 없습니다."
  echo "       먼저 'bash setup.sh' 를 실행하세요."
  exit 1
fi

export PYTHONUTF8=1

case "$TARGET" in
  fds)
    exec "$PY" -m uvicorn src.fraud_detection.api:app --reload --port 8000 --env-file .env
    ;;
  voice)
    exec "$PY" -m uvicorn src.voice_analysis.api:app --reload --port 8001 --env-file .env
    ;;
  test)
    shift
    exec "$PY" -m pytest -q "$@"
    ;;
  shell)
    exec "$PY"
    ;;
esac
