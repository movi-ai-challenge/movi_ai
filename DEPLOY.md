# MOVI AI — EC2 배포

백엔드(`movi_backend`)와 같은 EC2, 같은 컨벤션(Docker Hub + GitHub Actions)으로 배포한다.

```
develop 푸시
  → test (pytest)
  → Docker Hub 이미지 푸시
  → SSH 로 EC2 배포 (healthcheck 실패 시 이전 컨테이너로 롤백)
```

컨테이너는 2개, 이미지는 1개다. `APP_MODULE` / `PORT` 로 구분한다.

| 컨테이너 | APP_MODULE | 포트 |
|---|---|---|
| `movi-ai-fds` | `src.fraud_detection.api:app` | 8000 |
| `movi-ai-voice` | `src.voice_analysis.api:app` | 8001 |

---

## 1. 백엔드 연동 — 반드시 확인

**백엔드도 컨테이너이므로 `localhost:8000` 으로는 AI 에 닿지 않는다.**
컨테이너 안의 `localhost` 는 그 컨테이너 자신이다. 실제로 확인한 결과:

```
백엔드 컨테이너 -> http://localhost:8000    실패
백엔드 컨테이너 -> http://172.17.0.1:8000   실패 (127.0.0.1 바인딩이라 게이트웨이에 안 열림)
백엔드 컨테이너 -> http://movi-ai-fds:8000  200 OK
```

그래서 **공용 네트워크 `movi-net`** 에 양쪽을 붙이고 컨테이너 이름으로 호출한다.

### (1) 백엔드 워크플로 — 적용 완료

`movi_backend/.github/workflows/deploy-develop.yml` 에 반영했다.
`--network` 로 교체하지 않고 `docker network connect` 로 **추가 연결**하므로
기존 bridge 연결과 publish 포트에 영향이 없다.

```bash
# docker pull 다음
sudo docker network create movi-net 2>/dev/null || true

# docker run 성공 직후
sudo docker network connect movi-net "$CONTAINER_NAME" 2>/dev/null || true
```

### (2) `DEV_APPLICATION` 시크릿 — 직접 수정 필요

`application.yml` 의 기본값이 **FDS·Voice 둘 다 `http://localhost:8000`** 이고
`client-type` 기본값이 **`mock`** 이라, 그대로 두면 AI 서버를 호출하지 않는다.

`client-type` 의 유효값은 **`http`** 다. 백엔드 코드에서
`@ConditionalOnProperty(havingValue = "http")` 로 실제 클라이언트가 등록되고,
그 외의 값은 전부 Mock 으로 떨어진다.

```yaml
movi:
  fds:
    client-type: http
    base-url: http://movi-ai-fds:8000
  voice:
    client-type: http
    base-url: http://movi-ai-voice:8001   # 8000 아님
```

---

## 2. GitHub Secrets

`movi_ai` 저장소에 등록한다. 백엔드와 이름이 겹치는 것은 값이 같아야 한다.

| 시크릿 | 설명 |
|---|---|
| `AI_DOCKER_IMAGE_NAME` | 예: `<dockerhub계정>/movi-ai` |
| `AI_ENV_FILE` | 아래 3장의 `.env` 내용 전체 |
| `AI_GCP_SERVICE_ACCOUNT` | GCP 서비스 계정 JSON 전체 (STT/TTS 안 쓰면 빈 값) |
| `DOCKER_USERNAME` | 백엔드와 동일 |
| `DOCKER_HUB_TOKEN` | 백엔드와 동일 |
| `EC2_HOST` | 백엔드와 동일 |
| `EC2_SSH_PRIVATE_KEY` | 백엔드와 동일 |

---

## 3. `AI_ENV_FILE` 작성

`docker --env-file` 로 주입된다. **따옴표를 벗기지 않는다** — 실제 확인 결과:

```
QUOTED="abc123"   ->  값이 '"abc123"' (따옴표 포함)
PLAIN=abc123      ->  값이 'abc123'
```

`OPENAI_API_KEY="sk-..."` 처럼 쓰면 키에 따옴표가 붙어 인증이 실패한다.
**따옴표 없이** 쓴다.

```
BACKEND_BASE_URL=https://moviback.duckdns.org
VOICE_COMMAND_ENDPOINT=/api/v1/voice/command
BACKEND_TIMEOUT=5

FDS_MODEL_PATH=models/electronic/isolation_forest.joblib
FDS_THRESHOLD=0.44611697

OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_TIMEOUT_SECONDS=5
OPENAI_MAX_RETRIES=0

VOICE_DEBUG=false

# STT/TTS 를 쓸 때만
GOOGLE_CLOUD_PROJECT=<프로젝트 ID>
GOOGLE_APPLICATION_CREDENTIALS=/app/secrets/gcp.json
GOOGLE_STT_LOCATION=us
GOOGLE_STT_MODEL=chirp_3
```

`#` 주석과 빈 줄은 허용된다.

---

## 4. EC2 최초 1회 준비

```bash
docker network create movi-net
```

이후 백엔드를 재배포하면 워크플로가 자동으로 `movi-net` 에 연결한다.

---

## 5. 수동 배포 / 확인

EC2 에서 직접 띄울 때:

```bash
cd /home/ubuntu/movi-ai
docker compose -f docker-compose.prod.yml --env-file .env up -d
```

상태 확인:

```bash
docker ps --filter name=movi-ai
curl -s localhost:8000/health
curl -s localhost:8001/health
docker logs --tail 50 movi-ai-voice
```

`8000` / `8001` 은 `127.0.0.1` 에만 바인딩되어 있어 외부에서 접근할 수 없다.
SSH 로 들어가야 위 curl 이 동작한다. 보안그룹에 8000/8001 을 열 필요가 없다.

---

## 6. 롤백

배포 스크립트가 자동 처리한다. 실제로 검증한 동작:

```
잘못된 이미지 배포 → healthcheck unhealthy 감지
  → 새 컨테이너 제거 → 이전 컨테이너 복구 → 서비스 정상 응답
  → 스크립트 exit 1 → Actions 실패 표시
```

수동 롤백:

```bash
docker pull <이미지>:<이전 커밋 SHA>
# 또는 남아있는 롤백 컨테이너를 되돌린다
docker rename movi-ai-fds-rollback movi-ai-fds && docker start movi-ai-fds
```

---

## 7. 이미지에 대해

- 크기 **824MB** (numpy/scipy/scikit-learn 이 대부분)
- `python:3.12-slim` 기반, non-root `uid 10001` (백엔드와 동일 규칙)
- `linux/amd64`, `linux/arm64`(Graviton) 모두 동작 — 런타임 의존성 전부 양쪽 휠 확인함
- `data/`(648MB), `.venv`, `google-cloud-sdk`, `.env` 는 `.dockerignore` 로 제외
- 런타임에 `matplotlib` / `jupyter` / `sounddevice` 를 넣지 않는다.
  API 가 실제로 import 하지 않는 것을 확인했고, `sounddevice` 는 Linux 에서
  시스템 `libportaudio2` 를 추가로 요구한다. 이들은 `requirements-dev.txt` 로 분리했다.

---

## 8. 알려진 이슈

| 이슈 | 내용 |
|---|---|
| API 규격 불일치 | `docs/ai-api-contract.md` 는 `POST /internal/v1/voice/analyze` (multipart audio) 를 요구하지만 현재 `voice_analysis/api.py` 의 엔드포인트와 다르다. 배포와 별개로 맞춰야 한다. |
| 카드 모델 | `models/card/isolation_forest.joblib` 이 0바이트. FDS API 는 electronic 모델만 쓰므로 기동에는 지장 없다. |
| `test_voice_backend_flow` | 2차 백엔드 호출이 mock 되지 않아 실서버 401 을 받는다. CI 에서는 `-k "not backend_flow"` 로 제외한다. |
| `.env` 가 git 에 커밋됨 | OpenAI 키가 저장소 히스토리에 남아있다. 키 재발급 필요. |
