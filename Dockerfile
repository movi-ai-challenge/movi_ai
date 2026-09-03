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

# UVICORN_WORKERS 를 2 로 둔다. 요청 하나가 STT+GPT 로 15~20초 걸리는데
# 워커가 하나면 그 사이 들어온 WebSocket 업그레이드가 타임아웃된다.
# 동기 호출은 스레드풀로 넘겨 두었지만, 워커가 늘면 한쪽이 막혀도 다른 쪽이 받는다.
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    TZ=Asia/Seoul \
    APP_MODULE=src.fraud_detection.api:app \
    PORT=8000 \
    UVICORN_WORKERS=2 \
    ROOT_PATH=

EXPOSE 8000

# curl/wget 을 설치하지 않기 위해 python 으로 확인한다
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import os,sys,urllib.request;\
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ['PORT']+'/health',timeout=4).status==200 else 1)" \
  || exit 1

USER movi

# --root-path: 리버스 프록시가 접두사를 붙여 노출할 때 필요하다.
# nginx 가 /ai/voice/ 로 받아 접두사를 떼고 넘기므로, 앱이 접두사를 모르면
# /docs 가 참조하는 openapi.json 과 Try it out 요청 경로가 백엔드로 잘못 간다.
# 비어 있으면 (로컬/컨테이너 직접 호출) 아무 영향이 없다.
CMD ["sh", "-c", "exec uvicorn \"$APP_MODULE\" --host 0.0.0.0 --port \"$PORT\" --workers \"$UVICORN_WORKERS\" --root-path \"$ROOT_PATH\""]
