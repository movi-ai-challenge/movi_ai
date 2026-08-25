---

# MOVI AI — 음성 요구사항 분석 & 이상거래 탐지

## 1. 프로젝트 개요

MOVI의 AI 파트는 크게 두 가지 기능을 담당한다.

1. **음성 기반 금융 요구사항 분석**
2. **이상거래 탐지(Fraud Detection)**

사용자의 음성을 금융 서비스에서 사용할 수 있는 구조화된 요청으로 변환하고, 실제 거래 실행 전 해당 거래의 이상 여부를 판단하여 Spring Backend에 전달하는 것을 목표로 한다.

### 전체 서비스 흐름

```text
사용자 음성
    ↓
Google STT V2
    ↓
Wake Word 감지
    ↓
Intent / Entity 분석
    ↓
누락 정보 확인 및 보완
    ↓
Requirement JSON
    ↓
Spring Backend
    ↓
거래 정보 + 과거 거래 이력
    ↓
Python FDS
    ↓
Isolation Forest
    ↓
Anomaly Score
    ↓
Fraud Detection
    ↓
Spring Backend
```

향후 FDS는 다음 구조로 확장한다.

```text
Isolation Forest Score
        +
Rule Engine Score
        ↓
Final Risk Score
        ↓
LOW / MEDIUM / HIGH
```

---

# 2. 기술 스택

| 구분           | 기술                             |
| ------------ | ------------------------------ |
| Backend      | Spring                         |
| Frontend     | React                          |
| AI / Feature | Python                         |
| STT          | Google Cloud Speech-to-Text V2 |
| 요구사항 분석      | GPT Structured Output          |
| FDS          | scikit-learn Isolation Forest  |
| AI API       | FastAPI                        |
| 데이터 처리       | Pandas / NumPy                 |
| 전처리          | scikit-learn                   |
| 모델 저장        | Joblib                         |
| Dataset      | AIHub 금융 이상거래 데이터              |

---

# 3. 프로젝트 구조

```text
MOVI/
│
├── data/
│   ├── Train/
│   │   ├── TL_전자금융공동망/
│   │   └── TL_카드거래/
│   │
│   └── Validation/
│       ├── VL_전자금융공동망/
│       └── VL_카드거래/
│
├── models/
│   └── electronic/
│       ├── isolation_forest.joblib
│       └── threshold.json
│
├── reports/
│   └── electronic_metrics.json
│
├── src/
│   ├── fraud_detection/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── data_loader.py
│   │   ├── feature_engineering.py
│   │   ├── train_iforest.py
│   │   ├── evaluate.py
│   │   ├── inference.py
│   │   └── test_dummy_transaction.py
│   │
│   └── voice_analysis/
│       ├── __init__.py
│       ├── config.py
│       ├── stt_stream_service.py
│       ├── wake_word_detector.py
│       ├── schemas.py
│       ├── requirement_analyzer.py
│       ├── requirement_validator.py
│       ├── conversation_context.py
│       ├── follow_up_entity_parser.py
│       └── voice_pipeline.py
│
└── requirements.txt
```

FDS API 연동 단계에서는 다음 모듈을 추가할 예정이다.

```text
fraud_detection/
├── api.py
└── schemas.py
```

---

# 4. 이상거래 탐지

## 4.1 목표

단순히 전체 금융 거래에서 드문 거래를 찾는 것뿐 아니라,

> **현재 거래가 해당 사용자의 과거 거래 패턴과 비교했을 때 얼마나 이례적인지**

판단하는 것을 목표로 한다.

현재는 AIHub **전자금융공동망 데이터**를 우선 사용하여 개발하고 있다.

---

# 5. `config.py`

## 역할

데이터 경로, 모델 경로, Target, 학습 설정 등을 중앙에서 관리한다.

```text
Dataset Path
Model Path
Threshold Path
Report Path
Target Column
Isolation Forest Parameter
Random State
Training Sample Size
```

### 주요 고려사항

* 파일별 경로 하드코딩 제거
* 전자금융 / 카드 모델 설정 분리
* Target Leakage 방지
* 학습 환경 재현 가능성 확보

다음 컬럼은 모델 Feature에서 제외한다.

```text
이상거래여부
이상거래유형
이상거래설명
```

계좌번호 등의 ID는 숫자의 크기 자체를 학습하지 않고 Historical Feature 계산을 위한 Key로 사용한다.

---

# 6. `data_loader.py`

## 역할

AIHub의 여러 CSV를 자동 탐색하여 Train / Validation 데이터를 로드한다.

### 구현 기능

```text
✅ CSV 자동 탐색
✅ 여러 CSV 병합
✅ Train / Validation 분리
✅ Dataset별 Schema 검증
✅ 일부 파일 테스트 로딩
✅ Chunk Loader
✅ 시간순 거래 정렬
```

Historical Feature는 과거 거래 순서에 의존하기 때문에 전자금융 데이터는 다음 순서로 정렬한다.

```text
거래일자
    ↓
거래시간대
```

이를 통해 현재 거래의 Feature 계산에 미래 거래가 포함되는 것을 방지한다.

---

# 7. `feature_engineering.py`

## 역할

원본 거래 데이터를 Isolation Forest가 학습할 수 있는 Feature로 변환한다.

현재 전자금융 기준 **18개 Engineered Feature**를 생성한다.

## 기본 거래 Feature

| Feature      | 의미                   |
| ------------ | -------------------- |
| `log_amount` | 로그 변환 거래금액           |
| `hour_sin`   | 거래시간 Cyclic Encoding |
| `hour_cos`   | 거래시간 Cyclic Encoding |
| `is_night`   | 심야 거래 여부             |
| `weekday`    | 거래 요일                |
| `is_weekend` | 주말 거래 여부             |
| `same_bank`  | 출금/입금 금융회사 동일 여부     |

## Historical Feature

### `amount_ratio`

현재 거래금액이 사용자의 과거 평균 거래금액 대비 얼마나 큰지를 나타낸다.

```text
현재 거래금액
──────────────
과거 평균 거래금액
```

현재 거래는 평균 계산에서 제외한다.

---

### `amount_zscore`

현재 거래금액이 사용자의 과거 거래금액 분포에서 얼마나 벗어났는지 나타낸다.

```text
현재 거래금액 - 과거 평균
──────────────────────
과거 표준편차
```

---

### `new_recipient`

현재 수취인이 해당 출금계좌에서 처음 등장한 수취인인지 판단한다.

```text
처음 송금
→ 1

기존 송금 이력 존재
→ 0
```

---

### `unusual_medium`

사용자가 이전에 사용하지 않았던 거래 매체인지 판단한다.

```text
기존 매체
→ 0

처음 등장한 매체
→ 1
```

단, 계좌 자체의 첫 거래는 비교할 이력이 없으므로 `0`으로 처리한다.

---

### `historical_transaction_count`

현재 거래 이전까지 해당 출금계좌가 수행한 전체 거래 횟수이다.

```text
첫 거래   → 0
두 번째   → 1
세 번째   → 2
```

---

### `same_day_transaction_count`

같은 날짜에 현재 거래 이전까지 발생한 거래 횟수이다.

짧은 기간 반복 송금 패턴을 파악하기 위한 Feature이다.

---

### `same_time_bucket_count`

같은 날짜 및 동일 시간대에 발생한 이전 거래 횟수이다.

```text
동일 계좌
+
동일 날짜
+
동일 시간대
```

기준으로 반복 거래를 탐지한다.

---

# 8. 전처리

## Numeric Feature

숫자형 Feature의 결측값은 Train 데이터 기준 Median으로 처리한다.

```text
Numeric
   ↓
SimpleImputer(strategy="median")
```

초기 거래에서는 과거 데이터가 존재하지 않아 `amount_ratio`, `amount_zscore` 등이 `NaN`일 수 있으므로 의도된 처리이다.

## Categorical Feature

다음 Feature는 One-Hot Encoding 한다.

```text
출금금융회사일련번호
입금금융회사일련번호
자금구분
매체구분
```

```python
OneHotEncoder(
    handle_unknown="ignore"
)
```

Validation 및 서비스 환경에서 Train에 없었던 값이 등장하더라도 오류가 발생하지 않도록 구성하였다.

Sparse Matrix와 `float32`를 사용하여 메모리 사용량을 줄인다.

---

# 9. `train_iforest.py`

## 역할

정상거래 패턴을 기준으로 Isolation Forest를 학습한다.

Historical Feature 추가 이후 학습 순서를 다음과 같이 변경하였다.

```text
Train 거래 전체
    ↓
시간순 정렬
    ↓
Historical Feature Engineering
    ↓
정상거래 추출
    ↓
Random Sampling
    ↓
Preprocessor Fit
    ↓
Isolation Forest
    ↓
Model Bundle 저장
```

### 중요한 변경

기존에는:

```text
정상거래 추출
→ Sampling
→ Historical Feature
```

순서였으나, 이 경우 거래 History가 제거되면서 `amount_ratio`, `new_recipient` 등이 왜곡될 수 있었다.

따라서 **전체 거래 기준으로 Historical Feature를 먼저 계산한 후 정상거래만 학습에 사용하도록 변경**하였다.

---

# 10. Isolation Forest

현재 모델은 정상 거래 패턴을 학습한다.

```python
IsolationForest(
    contamination="auto",
    random_state=42,
    n_jobs=-1
)
```

Target인 `이상거래여부`는 모델 학습 Feature로 사용하지 않는다.

Validation 단계에서만 실제 Fraud Label과 비교한다.

---

# 11. Model Bundle

모델만 별도로 저장하지 않고 다음 객체를 함께 저장한다.

```python
{
    "model": ...,
    "preprocessor": ...,
    "dataset_type": ...,
    "feature_names": ...,
    "training_score_summary": ...,
    "environment": ...
}
```

저장 위치:

```text
models/electronic/isolation_forest.joblib
```

이를 통해 서비스에서도 학습 당시와 동일한 Preprocessor를 사용한다.

---

# 12. `evaluate.py`

## 역할

실제 AIHub Validation Label을 이용하여 모델의 탐지 성능과 Threshold를 결정한다.

```text
Validation 전체
    ↓
시간순 정렬
    ↓
Historical Feature
    ↓
Train Preprocessor
    ↓
Isolation Forest
    ↓
Anomaly Score
    ↓
실제 Label 비교
    ↓
Threshold 탐색
```

Isolation Forest 원본 Score는 낮을수록 이상하므로 서비스에서는 다음과 같이 변환한다.

```python
anomaly_score = -model.score_samples(X)
```

따라서 MOVI에서는:

```text
Anomaly Score ↑
=
위험도 ↑
```

로 사용한다.

---

# 13. 전체 Validation 평가 결과

전자금융공동망 Validation 전체 데이터:

```text
전체 거래 : 492,678
정상 거래 : 490,770
이상 거래 :   1,908
```

Best F1 Threshold:

```text
0.446117
```

## 현재 Isolation Forest 성능

| Metric              |        결과 |
| ------------------- | --------: |
| Precision           |  `0.0926` |
| Recall              |  `0.1981` |
| F1                  |  `0.1262` |
| Average Precision   |  `0.0581` |
| PR-AUC              |  `0.0579` |
| ROC-AUC             |  `0.8848` |
| False Positive Rate | `0.00755` |
| False Negative Rate | `0.80189` |

Confusion Matrix:

```text
                 Pred Normal   Pred Fraud

Actual Normal        487,065        3,705
Actual Fraud           1,530          378
```

### 결과 해석

정상/이상 거래의 평균 Anomaly Score:

```text
Normal Mean : 0.3737
Fraud Mean  : 0.4176
```

Fraud 거래의 Score가 전반적으로 높게 나타나므로 **이상거래 Ranking 자체는 유효**하다.

ROC-AUC 역시 약 `0.885`로 정상/이상 거래의 상대적 순위 분리는 확인하였다.

하지만 현재 Threshold에서는 Recall이 약 `19.8%`로 낮아 Isolation Forest 단독으로 최종 Fraud 여부를 결정하기에는 한계가 있다.

따라서 최종 FDS에서는:

```text
Isolation Forest
+
Rule Engine
+
Risk Score
```

구조를 사용할 예정이다.

---

# 14. Dummy 이상거래 테스트

실제 모델 동작을 확인하기 위해 사용자 과거 거래 History를 구성하고 정상 거래와 의도적으로 만든 이상거래를 비교하였다.

## 정상 거래

```text
평소 금액과 유사
기존 수취인
기존 거래 매체
주간 거래
동일 금융회사
```

결과:

```text
score_samples = -0.412090
```

## 이상 거래

```text
평소 약 10만원
→ 1,000만원 송금

신규 수취인
심야 거래
새로운 거래 매체
타행 거래
```

생성된 주요 Feature:

```text
amount_ratio    = 99.29
amount_zscore   = 758.71
is_night        = 1
same_bank       = 0
new_recipient   = 1
unusual_medium  = 1
```

결과:

```text
score_samples = -0.482914
```

즉:

```text
이상거래 Score < 정상거래 Score
```

가 확인되었다.

서비스용 Anomaly Score 기준으로 변환하면:

```text
Normal  = 0.412090
Fraud   = 0.482914
```

Validation Threshold `0.446117` 기준으로 Dummy 이상거래는 Fraud 영역에 위치한다.

---

# 15. 현재 FDS 한계

## 1. Isolation Forest 단독 Recall

현재 Best F1 Threshold 기준 실제 Fraud의 약 20%만 탐지한다.

따라서 Isolation Forest는 최종 결정 모델보다는:

> **정상 패턴에서 얼마나 벗어났는지 나타내는 Anomaly Score 생성기**

역할로 사용하는 것이 적합하다.

## 2. Score 분포 중첩

정상과 이상 거래 Score 평균에는 차이가 있지만 두 분포가 일부 겹친다.

Threshold를 낮추면 Recall은 증가하지만 False Positive가 크게 증가한다.

## 3. Train → Validation History 연결

현재 Validation History는 Validation 내부 거래를 기준으로 생성한다.

실서비스에서는 이전 기간의 사용자 History까지 연결해야 하므로 향후 Historical State를 개선할 예정이다.

## 4. 극단값

`amount_ratio`, `amount_zscore`에서 큰 극단값이 존재한다.

향후 로그 변환 및 Robust/Clip 전략을 비교하여 모델 성능 변화를 검증할 예정이다.

---

# 16. 음성 기반 요구사항 분석

## 처리 흐름

```text
사용자 음성
    ↓
Google STT V2
    ↓
Streaming Transcript
    ↓
Wake Word "모비야"
    ↓
GPT Requirement Analyzer
    ↓
Intent + Entity
    ↓
Validator
    ↓
누락 정보 확인
    ↓
Follow-up
    ↓
Requirement Complete
```

---

# 17. `stt_stream_service.py`

Google Cloud Speech-to-Text V2를 사용하여 실시간 마이크 음성을 텍스트로 변환한다.

구현 범위:

```text
✅ Streaming STT
✅ PCM Audio 처리
✅ Interim Transcript
✅ Final Transcript
✅ 실제 마이크 입력 테스트
```

최종 요구사항 분석에는 Final Transcript만 전달한다.

---

# 18. `wake_word_detector.py`

호출어:

```text
모비야
```

가 포함된 경우에만 실제 금융 명령을 분석한다.

예:

```text
모비야 김민수한테 오만원 보내줘
```

↓

```text
김민수한테 오만원 보내줘
```

일반 대화가 금융 명령으로 처리되는 것을 방지하고 불필요한 GPT 호출을 줄인다.

---

# 19. `requirement_analyzer.py`

GPT Structured Output을 사용하여 음성 명령에서 다음 정보를 추출한다.

```text
Intent
Entity
Missing Fields
```

예:

```text
김민수한테 오만원 보내줘
```

↓

```json
{
  "intent": "transfer_money",
  "entities": {
    "recipient_name": "김민수",
    "recipient_bank": null,
    "recipient_account": null,
    "amount": 50000
  }
}
```

---

# 20. `requirement_validator.py`

LLM이 반환한 결과를 그대로 실행하지 않고 코드 규칙을 이용하여 필수 Entity를 다시 검증한다.

계좌이체 기준:

```text
recipient_name
recipient_bank
recipient_account
amount
```

역할을 다음과 같이 분리하였다.

```text
GPT
→ 자연어 이해

Validator
→ 금융 요청 실행 가능 여부 검증
```

---

# 21. Multi-turn 요구사항 분석

누락된 정보가 있는 경우 해당 정보만 다시 요청한다.

```text
사용자
"김민수한테 오만원 보내줘"

↓

recipient_name = 김민수
amount = 50000

↓

"받는 분의 은행을 말씀해주세요."

↓

사용자
"국민은행"

↓

Context 병합
```

관련 모듈:

```text
conversation_context.py
follow_up_entity_parser.py
voice_pipeline.py
```

---

# 22. 현재 지원 Intent

| 기능       | Intent           |
| -------- | ---------------- |
| 화면 읽기    | `read_screen`    |
| 계좌 이체    | `transfer_money` |
| 거래내역 조회  | `check_history`  |
| 가입 적금 조회 | `check_savings`  |
| 확인       | `confirm`        |
| 거절       | `deny`           |
| 취소       | `cancel`         |
| 미지원 명령   | `unknown`        |

---

# 23. 현재 Voice Pipeline 완료 범위

```text
✅ Google STT V2
✅ Streaming STT
✅ Wake Word
✅ GPT Intent 분석
✅ Entity 추출
✅ 금액 구조화
✅ Requirement Validator
✅ Missing Field 판단
✅ Multi-turn Context
✅ Follow-up Entity 분석
✅ Requirement JSON
```

상태값:

```text
ignored
awaiting_command
need_more_info
ready
unsupported
error
```

---

# 24. Backend 연동 구조

중간 통합에서는 Python FDS를 FastAPI 서비스로 노출하고 Spring Backend가 호출하는 구조를 사용한다.

```text
React
    ↓
Spring Backend
    ↓
POST /api/v1/fraud/detect
    ↓
Python FastAPI
    ↓
FDS Model
    ↓
JSON Response
    ↓
Spring
```

FDS Request는:

```text
현재 거래
+
해당 계좌의 과거 거래 History
```

를 전달하는 것을 기준으로 설계하였다.

예정 Response:

```json
{
  "transaction_id": "tx-001",
  "anomaly_score": 0.482914,
  "threshold": 0.446117,
  "is_fraud": true,
  "risk_level": "HIGH",
  "model": "isolation_forest"
}
```

현재 API 요청/응답 명세까지 정의하였으며 FastAPI ↔ Spring 실제 통합 테스트를 진행할 예정이다.

---

# 25. 현재 개발 상태

## 음성 요구사항 분석

```text
✅ 실시간 STT
✅ Wake Word
✅ Intent 분석
✅ Entity 추출
✅ Missing Entity 검사
✅ Multi-turn 보완
✅ Requirement JSON 생성
```

## 이상거래 탐지

```text
✅ AIHub 데이터 로더
✅ 시간순 데이터 정렬
✅ Feature Engineering
✅ Historical Feature
✅ amount_ratio
✅ amount_zscore
✅ new_recipient
✅ unusual_medium
✅ historical_transaction_count
✅ same_day_transaction_count
✅ same_time_bucket_count
✅ One-Hot Encoding
✅ Sparse Matrix
✅ Isolation Forest 학습
✅ Model Bundle 저장
✅ Dummy 이상거래 테스트
✅ 전체 Validation 평가
✅ Threshold 탐색
✅ Evaluation Metrics 저장
```

## Backend 연동

```text
✅ FastAPI 연동 구조 설계
✅ Request / Response JSON 명세
✅ Spring ↔ Python API 명세
⬜ FastAPI 실제 Endpoint 구현 및 실행 검증
⬜ Spring 실제 호출 테스트
```

---

# 26. 2주차 개발 계획

현재 상태에서는 **모델을 계속 세부 튜닝하기보다 FDS 완성 → Backend 통합까지 연결하는 것**을 우선하는 것이 좋다.

|  우선순위  | 작업                            | 세부 내용                                | 완료 기준                       |
| :----: | ----------------------------- | ------------------------------------ | --------------------------- |
| **P0** | FastAPI FDS API 구현            | `/health`, `/api/v1/fraud/detect` 구현 | curl 요청 → 실제 모델 결과 반환       |
| **P0** | Spring 중간 통합                  | Spring → FastAPI HTTP 호출             | Spring에서 FDS JSON 수신        |
| **P0** | History 전달                    | Spring에서 사용자 최근 거래 조회 후 FDS 전달       | Historical Feature 정상 계산    |
| **P1** | Rule Engine                   | 고액·심야·신규 수취인·비정상 매체·반복거래 규칙          | Rule Score 산출               |
| **P1** | Final Risk Score              | IF Score + Rule Score 결합             | 0~100 또는 LOW/MEDIUM/HIGH 반환 |
| **P1** | Train → Validation History 연결 | 평가 시 이전 거래 State 유지                  | Historical Feature 정확도 개선   |
| **P2** | Feature 안정화 실험                | amount_ratio log 변환, z-score 극단값 처리  | Baseline 대비 지표 비교           |
| **P2** | Threshold 정책 개선               | F1 외 Recall 중심 Threshold 비교          | FDS 정책 Threshold 확정         |
| **P2** | Dummy Scenario 확대             | 정상~강한 이상 5단계 테스트                     | 위험도 단계별 증가 확인               |
| **P2** | 음성 → FDS 연결                   | `transfer_money` 완료 후 FDS 호출         | 음성 이체 요청 → 이상거래 결과          |
| **P3** | API 예외처리                      | 모델 실패, history 부족, schema 오류         | 명확한 HTTP Error 반환           |
| **P3** | 통합 테스트                        | React → Spring → Python 전체 흐름        | End-to-End 성공               |

---

## 2주차 TODO

### 1. Backend 중간 통합 — 우선 진행

```text
[ ] FastAPI 설치 및 API Server 구성
[ ] GET /health 구현
[ ] POST /api/v1/fraud/detect 구현
[ ] 서버 시작 시 Model Bundle 1회 Load
[ ] threshold.json Load
[ ] Spring Request → AIHub Feature 형식 변환
[ ] History + Current Transaction Feature Engineering
[ ] 현재 거래 마지막 Row만 추론
[ ] JSON Response 반환
[ ] curl/Postman 테스트
[ ] Spring WebClient 연동 테스트
```

### 2. Rule Engine

```text
[ ] 고액 거래 Rule
[ ] 평소 금액 대비 급증 Rule
[ ] 심야 거래 Rule
[ ] 신규 수취인 Rule
[ ] 신규 거래 매체 Rule
[ ] 동일 날짜 반복 거래 Rule
[ ] 동일 시간대 반복 거래 Rule
[ ] Rule별 Score 정의
[ ] Rule 발생 이유(reason) 반환
```

예정 형태:

```json
{
  "rule_score": 55,
  "triggered_rules": [
    "HIGH_AMOUNT_RATIO",
    "NEW_RECIPIENT",
    "NIGHT_TRANSACTION"
  ]
}
```

### 3. Risk Score

```text
[ ] Isolation Forest Score 정규화
[ ] Rule Score 정규화
[ ] IF / Rule 가중치 결정
[ ] Final Risk Score 계산
[ ] LOW / MEDIUM / HIGH 정책 정의
[ ] Backend Response 반영
```

예정:

```text
Isolation Forest
     ↓
Model Risk

Rule Engine
     ↓
Rule Risk

Model Risk + Rule Risk
     ↓
Final Risk Score
```

### 4. FDS 모델 개선

```text
[ ] Train History → Validation History 연결
[ ] 현재 Baseline Metric 보존
[ ] amount_ratio 분포 재확인
[ ] log_amount_ratio 실험
[ ] amount_zscore 극단값 분석
[ ] Robust / Clip 후보 실험
[ ] 동일 Validation 기준 재평가
[ ] PR-AUC / Recall / F1 비교
```

현재 비교 기준:

```text
ROC-AUC : 0.8848
PR-AUC  : 0.0579
F1      : 0.1262
Recall  : 0.1981
```

### 5. Voice ↔ FDS 연결

```text
[ ] VoicePipeline status=ready 확인
[ ] transfer_money Entity → Spring Request Mapping
[ ] 실제 계좌 정보 조회
[ ] 거래 History 조회
[ ] FDS 호출
[ ] Fraud Result 수신
[ ] 사용자 확인 단계 연결
```

최종 목표:

```text
"모비야 김민수한테 오만원 보내줘"
        ↓
Intent / Entity
        ↓
누락 정보 보완
        ↓
Spring
        ↓
거래 History 조회
        ↓
FDS
        ↓
Risk Score
        ↓
거래 진행 / 추가 확인
```

---

# 27. 현재 기준 다음 개발 순서

```text
1. FastAPI 구현
        ↓
2. Spring ↔ Python 중간 통합
        ↓
3. Rule Engine
        ↓
4. Risk Score
        ↓
5. Historical Feature 개선
        ↓
6. 재학습 / 재평가
        ↓
7. Voice → FDS 연결
        ↓
8. React → Spring → Python E2E 테스트
```
