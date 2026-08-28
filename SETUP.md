# MOVI AI 개발환경 세팅

macOS / Windows 양쪽 모두 지원합니다. 명령어만 다르고 결과는 동일합니다.

---

## 0. 사전 준비

| 항목 | 내용 |
|---|---|
| Python | **3.11 또는 3.12** (3.13 이상은 핀 고정된 휠이 없어 실패합니다) |
| Git | 저장소 클론용 |

- macOS: `brew install python@3.12`
- Windows: <https://www.python.org/downloads/> 설치 시 **"Add python.exe to PATH"** 체크

---

## 1. 세팅 (최초 1회)

### macOS / Linux
```bash
bash setup.sh
```

### Windows (PowerShell)
```powershell
.\setup.ps1
```

> 실행정책 오류(`... cannot be loaded because running scripts is disabled`)가 나면:
> ```powershell
> powershell -ExecutionPolicy Bypass -File .\setup.ps1
> ```

스크립트가 하는 일:

1. Python 3.11/3.12 탐색
2. `.venv` 생성
3. `requirements.txt` 설치
4. `.env` 가 없으면 `.env.example` 복사
5. Jupyter 커널 `Python (movi_ai)` 등록

### `.venv` 가 이미 있다는 에러가 나면

저장소에 **다른 개발자의 macOS `.venv` 가 통째로 커밋되어 있습니다**
(`/Users/jungda-eun/Documents/MOVI/.venv` 경로가 박혀 있어 다른 머신에서는 동작하지 않습니다).

지우고 새로 만들면 됩니다:

```bash
bash setup.sh --force
```
```powershell
.\setup.ps1 -Force
```

---

## 2. `.env` 채우기

`.env.example` 에 전체 항목이 설명되어 있습니다. 최소한 아래는 채워야 합니다.

```
OPENAI_API_KEY=sk-...            # 음성 요구사항 분석
GOOGLE_CLOUD_PROJECT=...         # STT 사용 시 필수 (없으면 STT 동작 안 함)
GOOGLE_APPLICATION_CREDENTIALS=... # 서비스 계정 JSON 경로
```

> **주의**: 애플리케이션 코드에는 `load_dotenv` 가 없습니다.
> `.env` 는 `uvicorn --env-file .env` 로만 주입되며, 이는 run 스크립트에 이미 포함돼 있습니다.
> (pytest 는 루트 `conftest.py` 가 처리합니다.)

---

## 3. 실행

| 목적 | macOS / Linux | Windows |
|---|---|---|
| 이상거래 탐지 API (:8000) | `bash run.sh fds` | `.\run.ps1 fds` |
| 음성 분석 API (:8001) | `bash run.sh voice` | `.\run.ps1 voice` |
| 테스트 | `bash run.sh test` | `.\run.ps1 test` |
| venv 파이썬 REPL | `bash run.sh shell` | `.\run.ps1 shell` |

Swagger UI: <http://127.0.0.1:8000/docs> / <http://127.0.0.1:8001/docs>

pytest 인자는 그대로 전달됩니다:
```bash
bash run.sh test -k "not backend_flow"
```

---

## 4. 알려진 이슈

| 이슈 | 내용 |
|---|---|
| `models/card/isolation_forest.joblib` | **0바이트**. 학습된 적이 없습니다. FDS API 는 electronic 모델만 쓰므로 실행에는 지장 없습니다. 필요하면 `train_iforest.py` 로 재학습하세요. |
| `notebooks/data_analysis.ipynb` | 0바이트 (빈 placeholder) |
| `test_voice_backend_flow.py` 실패 | 2차 백엔드 호출이 mock 처리되지 않아 실서버로 나가 `401 AUTH_4010` 을 받습니다. 환경 문제가 아니라 백엔드 인증 토큰이 필요한 통합 테스트입니다. |
| `.env` 가 git 에 커밋됨 | OpenAI 키가 저장소에 그대로 있습니다. 키 재발급 + `git rm --cached .env` 권장. |
| 번들된 `google-cloud-sdk/`, `google-cloud-cli-darwin-arm.tar.gz` | **macOS ARM 전용**입니다. Windows 에서는 gcloud CLI 를 별도 설치하세요. |

---

## 5. 버전을 고정한 이유

`models/electronic/isolation_forest.joblib` 은 `scikit-learn 1.9.0` / `numpy 2.4.6` 으로 학습됐습니다.
버전이 다르면 로드 시 `InconsistentVersionWarning` 이나 실패가 발생하므로
`requirements.txt` 의 ML 스택 버전은 **바꾸지 마세요**.

모든 핀은 macOS(arm64/x86_64) · Windows, Python 3.11/3.12 휠이 존재하는 것으로 확인했습니다.
