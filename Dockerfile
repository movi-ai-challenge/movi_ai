# ============================================================
# MOVI AI - FDS / Voice Analysis API
#
# 하나의 이미지로 두 서비스를 띄운다. APP_MODULE / PORT 로 구분.
#   FDS   : src.fraud_detection.api:app  (8000)
#   Voice : src.voice_analysis.api:app   (8001)
#
# linux/amd64, linux/arm64 (Graviton) 모두 동작한다.
# ============================================================

# ---------- 1단계: 의존성 설치 ----------
FROM python:3.12-slim AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt


# ---------- 2단계: 런타임 ----------
FROM python:3.12-slim

LABEL org.opencontainers.image.title="movi-ai"

# 백엔드(movi_backend)와 동일한 uid/gid 규칙
RUN groupadd -g 10001 movi \
    && useradd -u 10001 -g movi -M -s /usr/sbin/nologin movi

WORKDIR /app

COPY --from=build /opt/venv /opt/venv

# 런타임에 필요한 것만. data/ 는 .dockerignore 로 제외됨
COPY --chown=movi:movi src/    ./src/
COPY --chown=movi:movi models/ ./models/

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    TZ=Asia/Seoul \
    APP_MODULE=src.fraud_detection.api:app \
    PORT=8000 \
    UVICORN_WORKERS=1

EXPOSE 8000

# curl/wget 을 설치하지 않기 위해 python 으로 확인한다
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import os,sys,urllib.request;\
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ['PORT']+'/health',timeout=4).status==200 else 1)" \
  || exit 1

USER movi

CMD ["sh", "-c", "exec uvicorn \"$APP_MODULE\" --host 0.0.0.0 --port \"$PORT\" --workers \"$UVICORN_WORKERS\""]
