# MOVI AI — 음성 요구사항 분석 & 이상거래 탐지

> **AI / Python 파트 개발 문서**  
> 사용자의 음성 명령을 금융 서비스용 구조화 요청으로 변환하고, 실제 거래 전 이상거래 위험도를 산출하여 Spring Backend에 제공한다.

---

## 1. 프로젝트 개요

MOVI AI 파트는 크게 두 영역을 담당한다.

| 영역 | 역할 |
|---|---|
| **Voice AI** | STT 결과에서 Wake Word, Intent, Entity를 추출하고 Backend 연동용 요청으로 변환 |
| **Fraud Detection System (FDS)** | 현재 거래와 과거 거래 이력을 이용해 이상거래 위험도를 계산 |

팀 전체 기술 스택은 다음과 같다.

| 구분 | 기술 |
|---|---|
| Backend | Spring |
| Frontend | React |
| AI / Feature | Python |
| STT | Google Cloud Speech-to-Text V2 |
| 요구사항 분석 | GPT Structured Output |
| FDS Model | scikit-learn Isolation Forest |
| Rule 기반 탐지 | Python Rule Engine |
| AI API | FastAPI |
| 데이터 처리 | Pandas / NumPy |
| 전처리 | scikit-learn |
| 모델 저장 | Joblib |
| Dataset | AIHub 금융 이상거래 데이터 |

---

# 2. 전체 서비스 구조

## 2.1 최종 서비스 흐름

```text
사용자 음성
    ↓
Google STT V2
    ↓
Wake Word "모비야"
    ↓
Intent / Entity 분석
    ↓
Python Voice API
    ↓
Spring Backend
    ├─ 필수값 / 빈 값 판단
    ├─ 추가 정보 요청
    ├─ 사용자 / 계좌 / 적금 / 거래내역 DB 조회
    └─ 실제 금융 기능 처리
    ↓
거래 정보 + 동일 출금계좌의 과거 거래 History
    ↓
Python FDS API
    ├─ Historical Feature Engineering
    ├─ Isolation Forest
    ├─ Rule Engine
    └─ Final Risk Score
    ↓
Spring Backend
    ↓
React / TTS
```

### 역할 분리 원칙

```text
Python
├─ STT
├─ Wake Word
├─ Intent / Entity 추출
├─ Follow-up Entity 추출
├─ Backend Request Mapping
├─ Isolation Forest
├─ Rule Engine
└─ Risk Score

Spring
├─ 사용자 인증
├─ 빈 값 / 필수값 판단
├─ requested_field 결정
├─ 추가 질문 결정
├─ DB 조회
├─ 실제 이체 처리
└─ Python API 호출
```

Python에서는 금융 요청의 필수값을 최종 판단하지 않는다.  
추출 가능한 Entity는 `null`을 포함하여 그대로 Backend로 전달하고, Backend가 실제 서비스 상태와 사용자 DB 정보를 기준으로 다음 동작을 결정한다.

---

# 3. 현재 프로젝트 구조

```text
MOVI/
├── data/
│   ├── Train/
│   │   ├── TL_전자금융공동망/
│   │   └── TL_카드거래/
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
│   │   ├── schemas.py
│   │   ├── transaction_mapper.py
│   │   ├── inference_feature_builder.py
│   │   ├── inference_service.py
│   │   ├── fraud_service.py
│   │   ├── rule_engine.py
│   │   ├── risk_score.py
│   │   ├── api.py
│   │   └── test_*.py
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
│       ├── voice_pipeline.py
│       ├── request_mapper.py
│       ├── backend_client.py
│       ├── api_schemas.py
│       ├── voice_service.py
│       ├── api.py
│       └── test_*.py
│
└── requirements.txt
```

> `requirement_validator.py`, `conversation_context.py`는 초기 Python 중심 Multi-turn 구조에서 구현되었으며 현재 서비스 연동에서는 필수값 판단 책임이 Backend로 이동하였다. 기존 코드는 로컬 검증 및 fallback 용도로 유지할 수 있다.

---

# 4. 이상거래 탐지 FDS

## 4.1 목표

단순히 전체 데이터에서 드문 거래를 찾는 것이 아니라,

> **현재 거래가 해당 사용자의 과거 거래 패턴과 비교했을 때 얼마나 이례적인지**

판단하는 것을 목표로 한다.

현재 FDS는 다음 세 결과를 결합한다.

```text
Historical Features
        ↓
Isolation Forest
        ↓
Model Risk Score
        ┐
        ├── Final Risk Score → LOW / MEDIUM / HIGH
        ┘
Rule Engine
        ↓
Rule Score
```

---

## 4.2 데이터

AIHub 금융 이상거래 데이터 중 **전자금융공동망 데이터**를 우선 사용하였다.

### Validation 전체 규모

```text
전체 거래 : 492,678건
정상 거래 : 490,770건
이상 거래 :   1,908건
```

Historical Feature가 거래 순서에 의존하므로 다음 기준으로 시간순 정렬한다.

```text
거래일자
   ↓
거래시간대
```

현재 거래의 Feature 계산에 미래 거래가 포함되지 않도록 하기 위한 처리이다.

---

# 5. 데이터 로딩 — `data_loader.py`

## 역할

- Train / Validation CSV 자동 탐색
- 여러 CSV 병합
- Dataset Schema 검증
- Chunk 단위 로딩 지원
- 거래 시간순 정렬
- 전자금융 / 카드 데이터 구조 분리

Historical Feature는 Sampling 이후가 아니라 **전체 거래 이력을 먼저 기준으로 계산**한다.

```text
전체 Train 거래
    ↓
시간순 정렬
    ↓
Historical Feature 생성
    ↓
정상 거래만 추출
    ↓
Sampling
    ↓
모델 학습
```

Sampling을 먼저 하면 과거 거래 일부가 제거되어 `amount_ratio`, `new_recipient`, 누적 거래 횟수 등이 왜곡될 수 있기 때문이다.

---

# 6. Feature Engineering — `feature_engineering.py`

현재 전자금융공동망 기준 다음 Feature를 사용한다.

## 6.1 기본 거래 Feature

| Feature | 의미 |
|---|---|
| `log_amount` | 로그 변환 거래금액 |
| `hour_sin` | 시간 Cyclic Encoding |
| `hour_cos` | 시간 Cyclic Encoding |
| `is_night` | 심야 거래 여부 |
| `weekday` | 거래 요일 |
| `is_weekend` | 주말 여부 |
| `same_bank` | 출금 / 입금 금융회사 동일 여부 |

## 6.2 범주형 Feature

다음 값은 코드값 자체의 크기에 의미가 없으므로 One-Hot Encoding 한다.

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

실서비스에서 학습 시 없던 범주가 들어와도 추론 오류가 발생하지 않도록 한다.

## 6.3 Historical Feature

| Feature | 의미 |
|---|---|
| `amount_ratio` | 현재 금액 / 과거 평균 금액 |
| `amount_zscore` | 과거 금액 분포 기준 현재 금액 Z-score |
| `new_recipient` | 해당 출금계좌에서 처음 등장한 수취인 |
| `unusual_medium` | 사용자가 과거 사용하지 않았던 거래 매체 |
| `historical_transaction_count` | 현재 거래 이전 누적 거래 수 |
| `same_day_transaction_count` | 같은 날짜의 이전 거래 수 |
| `same_time_bucket_count` | 같은 날짜 + 같은 시간대의 이전 거래 수 |

현재 거래 자체는 과거 통계 계산에서 제외한다.

---

# 7. 전처리

## Numeric

```text
Numeric Feature
    ↓
SimpleImputer(strategy="median")
```

첫 거래처럼 과거 이력이 없는 경우 `amount_ratio`, `amount_zscore`가 `NaN`이 되는 것은 정상이며, 학습 데이터 기준 Median으로 처리한다.

## Categorical

```text
Categorical Feature
    ↓
OneHotEncoder(handle_unknown="ignore")
```

전처리 결과는 Sparse Matrix + `float32`를 사용하여 메모리 사용량을 줄인다.

---

# 8. Isolation Forest — `train_iforest.py`

Isolation Forest는 정상 거래 패턴을 학습한다.

```python
IsolationForest(
    contamination="auto",
    random_state=42,
    n_jobs=-1
)
```

Target 관련 컬럼은 모델 Feature로 사용하지 않는다.

```text
이상거래여부
이상거래유형
이상거래설명
```

Label은 Validation 평가 및 Threshold 선택에만 사용한다.

---

# 9. Model Bundle

학습된 모델과 Preprocessor를 함께 저장한다.

```python
{
    "model": model,
    "preprocessor": preprocessor,
    "dataset_type": dataset_type,
    "feature_names": feature_names,
    "training_score_summary": training_score_summary,
    "environment": environment
}
```

저장 위치:

```text
models/electronic/isolation_forest.joblib
```

추론에서는 학습 당시 저장한 동일 Preprocessor에 대해 **`transform()`만 사용**한다.

```text
Train     → fit / fit_transform
Inference → transform only
```

---

# 10. Validation 평가 — `evaluate.py`

## 평가 흐름

```text
Validation 전체
    ↓
시간순 정렬
    ↓
Historical Feature Engineering
    ↓
Train Preprocessor Transform
    ↓
Isolation Forest
    ↓
Anomaly Score
    ↓
실제 Fraud Label 비교
    ↓
Threshold 탐색
```

Isolation Forest의 `score_samples()`는 낮을수록 이상치에 가깝다.

MOVI에서는 사용자와 Backend가 직관적으로 이해할 수 있도록 방향을 반전한다.

```python
anomaly_score = -model.score_samples(X)
```

따라서:

```text
Anomaly Score ↑
=
이상거래 위험 ↑
```

---

# 11. Baseline 성능

Best F1 Threshold:

```text
0.446117
```

| Metric | 결과 |
|---|---:|
| Precision | 0.0926 |
| Recall | 0.1981 |
| F1 | 0.1262 |
| Average Precision | 0.0581 |
| PR-AUC | 0.0579 |
| ROC-AUC | 0.8848 |
| False Positive Rate | 0.00755 |
| False Negative Rate | 0.80189 |

### Confusion Matrix

```text
                 Pred Normal   Pred Fraud
Actual Normal        487,065        3,705
Actual Fraud           1,530          378
```

### Score 평균

```text
Normal Mean : 0.3737
Fraud Mean  : 0.4176
```

ROC-AUC 약 `0.8848`로 거래 Ranking 능력은 유효하지만, Best F1 Threshold 기준 Recall은 약 `19.8%`로 낮다.

따라서 Isolation Forest는 최종 판정기보다:

> **정상 패턴에서 얼마나 벗어났는지를 표현하는 Model Risk 신호**

로 사용하고 Rule Engine과 결합하였다.

---

# 12. 실서비스 거래 입력 Mapping

## `schemas.py`

Spring에서 Python으로 전달할 거래 API DTO를 정의한다.

```json
{
  "current_transaction": {
    "transaction_id": "tx-current",
    "sender_account": "100001",
    "receiver_account": "999999",
    "sender_bank": "KB",
    "receiver_bank": "SH",
    "transaction_type": "transfer",
    "amount": 10000000,
    "transaction_datetime": "2026-08-26T00:30:00",
    "medium": "WEB"
  },
  "history": []
}
```

## `transaction_mapper.py`

Spring DTO를 AIHub 학습 구조로 변환한다.

```text
sender_account        → 출금계좌일련번호
receiver_account      → 입금계좌일련번호
sender_bank           → 출금금융회사일련번호
receiver_bank         → 입금금융회사일련번호
amount                → 거래금액
transaction_datetime  → 거래일자 + 거래시간대
medium                → 매체구분
```

실제 시간은 학습 데이터와 동일한 3시간 Bucket으로 변환한다.

| 실제 시각 | 거래시간대 |
|---|---:|
| 00:00 ~ 02:59 | 0 |
| 03:00 ~ 05:59 | 3 |
| 06:00 ~ 08:59 | 6 |
| 09:00 ~ 11:59 | 9 |
| 12:00 ~ 14:59 | 12 |
| 15:00 ~ 17:59 | 15 |
| 18:00 ~ 20:59 | 18 |
| 21:00 ~ 23:59 | 21 |

---

# 13. Rule Engine — `rule_engine.py`

Isolation Forest가 놓칠 수 있는 명시적인 위험 패턴을 보완한다.

| Rule | 1차 조건 | Score |
|---|---|---:|
| `HIGH_AMOUNT_RATIO` | 평소 대비 5배 이상 | +25 |
| `EXTREME_AMOUNT_ZSCORE` | Z-score 절댓값 5 이상 | +20 |
| `NIGHT_TRANSACTION` | 심야 거래 | +15 |
| `NEW_RECIPIENT` | 신규 수취인 | +15 |
| `UNUSUAL_MEDIUM` | 신규 거래 매체 | +15 |
| `CROSS_BANK` | 타행 거래 | +5 |
| `REPEATED_SAME_DAY` | 당일 이전 거래 3건 이상 | +10 |
| `REPEATED_TIME_BUCKET` | 동일 시간대 이전 거래 2건 이상 | +10 |

Rule Score는 최대 100점으로 제한한다.

### 단위 테스트

```text
정상 거래
→ rule_score = 0
→ triggered_rules = []

고위험 거래
→ rule_score = 95
→ 6개 Rule Trigger
```

> 현재 Threshold와 가중치는 1차 서비스 정책값이다. 추후 Validation 기반으로 Rule별 Precision / Recall을 확인하여 조정한다.

---

# 14. Final Risk Score — `risk_score.py`

Isolation Forest와 Rule Engine 결과를 0~100 점수로 통합한다.

## 14.1 Model Risk Score

Isolation Forest의 `anomaly_score`를 단순히 ×100 하지 않고 Validation 분포를 이용해 Piecewise Linear Mapping 한다.

| anomaly_score | model_risk_score |
|---:|---:|
| 0.373701 | 0 |
| 0.419761 | 40 |
| 0.446117 | 70 |
| 0.472755 | 90 |
| 0.528859 | 100 |

중간 값은 선형 보간한다.

## 14.2 최종 가중치

현재 1차 정책:

```text
Model Risk 40%
+
Rule Risk  60%
=
Final Risk Score
```

```text
Final Risk Score = Model Risk × 0.4 + Rule Score × 0.6
```

## 14.3 Risk Level

| Final Risk Score | Level |
|---:|---|
| 0 ~ 39 | `LOW` |
| 40 ~ 69 | `MEDIUM` |
| 70 ~ 100 | `HIGH` |

### 단위 테스트 결과

```text
Normal
Model Risk : 14.15
Rule Score : 0
Final      : 5.66
Level      : LOW

Medium
Model Risk : 51.65
Rule Score : 40
Final      : 44.66
Level      : MEDIUM

High
Model Risk : 66.46
Rule Score : 95
Final      : 83.58
Level      : HIGH
```

---

# 15. FDS FastAPI — `fraud_detection/api.py`

## Endpoint

```text
GET  /health
POST /api/v1/fraud/detect
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

Health Check:

```json
{
  "status": "ok",
  "service": "fraud-detection",
  "model_loaded": true,
  "model": "isolation_forest",
  "rule_engine_loaded": true,
  "risk_score_enabled": true,
  "threshold": 0.446117
}
```

## 실제 API 검증 결과

고위험 Dummy 거래:

```text
과거 거래
50,000원
70,000원
100,000원

현재 거래
10,000,000원
+ 신규 수취인
+ 타행
+ 신규 매체
+ 심야
```

Feature:

```text
amount_ratio      = 136.36
amount_zscore     = 394.45
is_night          = 1
same_bank         = 0
new_recipient     = 1
unusual_medium    = 1
```

실제 API Response:

```json
{
  "transaction_id": "tx-current",
  "anomaly_score": 0.443008,
  "threshold": 0.446117,
  "is_anomaly": false,
  "model": "isolation_forest",
  "rule_score": 95,
  "final_risk_score": 83.58,
  "risk_level": "HIGH",
  "triggered_rules": [
    "HIGH_AMOUNT_RATIO",
    "EXTREME_AMOUNT_ZSCORE",
    "NIGHT_TRANSACTION",
    "NEW_RECIPIENT",
    "UNUSUAL_MEDIUM",
    "CROSS_BANK"
  ]
}
```

이 사례에서 Isolation Forest 단독 결과는 Threshold 직전으로 `is_anomaly=false`였지만, Rule Engine이 명시적 위험 패턴을 포착하여 최종 Risk Level은 `HIGH`로 판단하였다.

```text
Isolation Forest → 정상 영역
Rule Engine       → 95
Final Risk Score  → 83.58
Risk Level        → HIGH
```

Isolation Forest 단독 탐지의 Recall 한계를 Rule Engine이 보완하는 구조가 실제 API 수준에서 확인되었다.

---

# 16. 음성 요구사항 분석

## 16.1 처리 흐름

```text
사용자 음성
    ↓
Google STT V2
    ↓
Streaming Transcript
    ↓
Final Transcript
    ↓
Wake Word "모비야"
    ↓
GPT Requirement Analyzer
    ↓
Intent / Entity
    ↓
Python Voice API
    ↓
Spring Backend
```

기존에는 Python의 `RequirementValidator`, `ConversationContext`가 누락값을 판단했으나 현재 연동 구조에서는 책임을 Backend로 이동하였다.

```text
Python
→ 언어 이해 / Entity 추출

Backend
→ 빈 값 / 필수값 판단
→ requested_field 결정
→ 추가 질문 결정
```

---

# 17. Streaming STT — `stt_stream_service.py`

Google Cloud Speech-to-Text V2를 이용해 실제 마이크 입력을 실시간으로 변환한다.

### 구현 완료

- Streaming STT
- PCM Audio 처리
- Interim Transcript
- Final Transcript
- 실제 마이크 입력 검증

Requirement 분석에는 Final Transcript만 사용한다.

---

# 18. Wake Word — `wake_word_detector.py`

호출어:

```text
모비야
```

예:

```text
"모비야 김민수한테 오만원 보내줘"
                ↓
"김민수한테 오만원 보내줘"
```

Wake Word가 없는 일반 대화를 금융 명령으로 처리하지 않는다.

---

# 19. Requirement Analyzer — `requirement_analyzer.py`

GPT Structured Output으로 Intent / Entity를 추출한다.

예:

```text
김민수한테 오만원 보내줘
```

```json
{
  "intent": "transfer_money",
  "entities": {
    "recipient_name": "김민수",
    "recipient_bank": null,
    "recipient_account": null,
    "amount": 50000,
    "source_bank": null,
    "source_account": null
  }
}
```

현재 주요 Intent:

| 기능 | Intent |
|---|---|
| 화면 읽기 | `read_screen` |
| 계좌 이체 | `transfer_money` |
| 거래내역 조회 | `check_history` |
| 가입 적금 조회 | `check_savings` |
| 확인 | `confirm` |
| 거절 | `deny` |
| 취소 | `cancel` |
| 미지원 | `unknown` |

---

# 20. Follow-up Entity 분석

Backend가 추가 정보가 필요하다고 판단하면 다음 형태를 반환하는 구조이다.

```json
{
  "request_id": "req-123",
  "status": "need_more_info",
  "requested_field": "recipient_bank",
  "message": "받는 분의 은행을 말씀해주세요."
}
```

Python은 `requested_field`를 기준으로 다음 음성을 분석한다.

```text
Backend
requested_field = recipient_bank
        ↓
사용자
"국민은행이야"
        ↓
FollowUpEntityParser
        ↓
recipient_bank = 국민은행
```

기존 Entity에 새로운 값을 병합한 뒤 동일 `request_id`로 다시 Backend에 전달한다.

---

# 21. Backend Request Mapping — `request_mapper.py`

Python은 Intent별 분석 결과를 Backend 공통 요청 구조로 변환한다.

```json
{
  "request_id": "req-123",
  "intent": "transfer_money",
  "transcript": "김민수한테 오만원 보내줘",
  "payload": {
    "recipient_name": "김민수",
    "recipient_bank": null,
    "recipient_account": null,
    "amount": 50000,
    "source_bank": null,
    "source_account": null
  }
}
```

현재 Backend 전달 Intent:

```text
transfer_money
check_savings
check_history
```

빈 Entity는 추측하지 않고 `null` 상태로 전달한다.

---

# 22. Voice Backend Client — `backend_client.py`

Spring Backend로 JSON을 전달하는 HTTP Client를 구현하였다.

### 처리 항목

- POST Request
- Timeout
- Connection Error
- HTTP 4xx / 5xx
- Backend JSON Response
- 환경변수 기반 Base URL / Endpoint 관리

현재 연동 대상:

```text
https://moviback.duckdns.org
```

제안 Endpoint:

```text
POST /api/v1/voice/command
```

실제 요청은 Backend 인증 미구현 상태에서 `401`까지 확인하였으며, Python Client 자체 구현은 완료한 상태이다.

---

# 23. Voice Multi-turn

하나의 금융 요청에는 동일 `request_id`를 유지한다.

```text
"김민수한테 오만원 보내줘"
        ↓
request_id = req-123
        ↓
Backend
recipient_bank 요청
        ↓
"국민은행"
        ↓
request_id = req-123 유지
        ↓
Backend
recipient_account 요청
        ↓
...
```

Mock Backend 테스트를 통해 다음 항목을 검증하였다.

```text
최초 Intent / Entity
        ↓
requested_field 저장
        ↓
Follow-up Entity 추출
        ↓
기존 Entity 갱신
        ↓
동일 request_id 유지
```

---

# 24. Voice FastAPI

현재 Voice AI도 별도의 FastAPI로 제공한다.

```text
GET  /health
POST /api/v1/voice/analyze
POST /api/v1/voice/follow-up
```

로컬 테스트:

```text
http://127.0.0.1:8001/docs
```

## 실제 Swagger 검증

### 계좌 이체

Input:

```json
{
  "transcript": "모비야 김민수한테 오만원 보내줘"
}
```

Result:

```json
{
  "status": "analyzed",
  "intent": "transfer_money",
  "transcript": "김민수한테 오만원 보내줘",
  "entities": {
    "recipient_name": "김민수",
    "recipient_bank": null,
    "recipient_account": null,
    "amount": 50000
  }
}
```

### 적금 조회

```text
"모비야 현재 가입한 적금 알려줘"
→ check_savings
```

### 거래내역 조회

```text
"모비야 최근 거래내역 알려줘"
→ check_history
```

### Follow-up

```text
requested_field = recipient_bank
"국민은행이야"
→ recipient_bank = 국민은행
```

모두 HTTP `200` 기준으로 정상 검증하였다.

---

# 25. Python API 현황

개발 중 로컬에서는 Voice / FDS API를 분리하여 테스트한다.

| API | Local Port | Endpoint |
|---|---:|---|
| Voice AI | `8001` | `/api/v1/voice/analyze` |
| Voice Follow-up | `8001` | `/api/v1/voice/follow-up` |
| FDS | `8000` | `/api/v1/fraud/detect` |

최종 배포에서는 하나의 FastAPI App으로 통합할 수 있다.

```text
MOVI AI FastAPI
├─ /api/v1/voice/analyze
├─ /api/v1/voice/follow-up
└─ /api/v1/fraud/detect
```

---

# 26. Backend 팀 연동 규격

## 26.1 Voice

### Python → Spring

```text
POST /api/v1/voice/command
```

```json
{
  "request_id": "req-123",
  "intent": "transfer_money",
  "transcript": "김민수한테 오만원 보내줘",
  "payload": {
    "recipient_name": "김민수",
    "recipient_bank": null,
    "recipient_account": null,
    "amount": 50000
  }
}
```

### 추가 정보 필요 시 Spring → Python

```json
{
  "request_id": "req-123",
  "status": "need_more_info",
  "requested_field": "recipient_bank",
  "message": "받는 분의 은행을 말씀해주세요."
}
```

Spring 담당:

- 빈 값 및 필수값 검증
- `requested_field` 선택
- 사용자 인증
- 계좌 / 적금 / 거래내역 DB 조회
- 실제 이체 처리

---

## 26.2 FDS

### Spring → Python

```text
POST /api/v1/fraud/detect
```

```json
{
  "current_transaction": {
    "transaction_id": "tx-001",
    "sender_account": "...",
    "receiver_account": "...",
    "sender_bank": "...",
    "receiver_bank": "...",
    "transaction_type": "transfer",
    "amount": 500000,
    "transaction_datetime": "2026-08-26T15:30:00",
    "medium": "MOBILE"
  },
  "history": []
}
```

`history`에는 현재 거래보다 이전의 **동일 출금계좌 거래 이력**을 전달한다.

### Python → Spring

```json
{
  "transaction_id": "tx-001",
  "anomaly_score": 0.443008,
  "threshold": 0.446117,
  "is_anomaly": false,
  "model": "isolation_forest",
  "rule_score": 95,
  "final_risk_score": 83.58,
  "risk_level": "HIGH",
  "triggered_rules": [
    "HIGH_AMOUNT_RATIO",
    "NEW_RECIPIENT",
    "NIGHT_TRANSACTION"
  ]
}
```

---

# 27. 현재 개발 완료 현황

## Voice AI

- [x] Google STT V2 설정
- [x] Streaming STT
- [x] 실제 마이크 입력
- [x] Interim / Final Transcript
- [x] Wake Word
- [x] GPT Intent 분석
- [x] Entity 추출
- [x] Follow-up Entity 분석
- [x] Request Mapping
- [x] Backend HTTP Client
- [x] request_id Multi-turn 구조
- [x] Backend `requested_field` 기반 Entity 갱신
- [x] Voice FastAPI
- [x] `/health`
- [x] `/docs`
- [x] `transfer_money` Swagger 검증
- [x] `check_savings` Swagger 검증
- [x] `check_history` Swagger 검증
- [x] Follow-up Swagger 검증
- [ ] 실제 Spring 인증 연동
- [ ] Backend 조회 결과 → TTS

## FDS

- [x] AIHub 데이터 Loader
- [x] Train / Validation 로딩
- [x] 시간순 정렬
- [x] 기본 Feature Engineering
- [x] Historical Feature Engineering
- [x] One-Hot Encoding
- [x] Sparse Matrix / float32
- [x] Isolation Forest 학습
- [x] Model Bundle 저장
- [x] Validation 전체 평가
- [x] Threshold 탐색
- [x] Dummy 정상 / 이상거래 테스트
- [x] Spring DTO Schema
- [x] Spring → AIHub Mapping
- [x] 저장 Model / Preprocessor 실제 추론
- [x] FDS FastAPI
- [x] `/health`
- [x] `/docs`
- [x] 실제 `POST /api/v1/fraud/detect`
- [x] Rule Engine
- [x] Rule Engine 단위 테스트
- [x] Rule Engine API 통합
- [x] Model Risk 정규화
- [x] Final Risk Score
- [x] LOW / MEDIUM / HIGH
- [x] Final Risk Score API 통합
- [ ] 5단계 Risk Scenario 검증
- [ ] Train → Validation History 연결 개선
- [ ] Historical Feature 극단값 안정화
- [ ] 재학습 / 재평가

---

# 28. 현재 한계 및 개선 예정

## Isolation Forest 단독 Recall

현재 Best F1 Threshold 기준 Recall 약 `19.8%`로 단독 최종 판정에는 한계가 있다.

따라서:

```text
Isolation Forest
→ 분포 기반 이상도

Rule Engine
→ 명시적 위험 패턴

Final Risk Score
→ 서비스용 통합 판단
```

구조를 사용한다.

## Train → Validation History 연결

현재 Validation Historical Feature는 Validation 내부 이력을 중심으로 계산한다.

향후:

```text
Train History
    +
Validation History
```

를 연결하여 실제 서비스 환경과 가까운 방식으로 평가할 예정이다.

## Historical Feature 극단값

실제 테스트에서 다음과 같은 값이 관찰되었다.

```text
amount_ratio  > 100
amount_zscore > 300
```

향후:

- `log_amount_ratio`
- Clip
- Robust Scaling / Robust 처리

등을 비교한다.

## Rule / Risk Score 정책

현재 Rule Weight와 Model 40% / Rule 60% 비율은 **1차 서비스 정책**이다.

추후 Validation Fraud Label을 이용하여:

- Rule별 Precision / Recall
- Rule 조합별 탐지율
- Model / Rule 가중치
- LOW / MEDIUM / HIGH 경계

를 재조정한다.

---

# 29. 남은 2주차 AI/Python Task

| 우선순위 | 작업 | 완료 기준 |
|---|---|---|
| P0 | 5단계 Risk Scenario 검증 | 위험 조건 증가에 따라 Risk Score 증가 |
| P1 | Train → Validation History 연결 | Historical Feature 평가 현실성 개선 |
| P1 | `amount_ratio` 안정화 | 기존 Baseline과 비교 |
| P1 | `amount_zscore` 극단값 처리 | Clip / Robust 비교 |
| P1 | FDS 재학습 / 재평가 | ROC-AUC / PR-AUC / Recall / F1 비교 |
| P1 | Rule Weight 검증 | Validation 기반 Rule 정책 조정 |
| P1 | Model / Rule 가중치 검증 | Final Risk 정책 보정 |
| P2 | Backend Response → TTS | 적금 / 거래내역 음성 출력 |
| P2 | Voice → FDS 서비스 흐름 연결 | 이체 요청에서 FDS 결과까지 연결 |
| P3 | API Error Response 정리 | 4xx / 5xx 응답 규격 정리 |
| 대기 | 실제 Spring Voice 연동 | Backend 인증 구현 후 검증 |
| 대기 | Spring → FDS 실제 호출 | Backend에서 Python API 호출 |
| 최종 | React → Spring → Python E2E | 대표 시나리오 전체 성공 |

---

# 30. 다음 개발 순서

```text
현재 완료
├─ Streaming STT
├─ Voice Intent / Entity
├─ Voice FastAPI
├─ Backend Mapping / Client
├─ Isolation Forest
├─ FDS FastAPI
├─ Rule Engine
└─ Final Risk Score
        ↓
1. 5단계 Risk Scenario 검증
        ↓
2. Train → Validation History 연결
        ↓
3. Historical Feature 안정화
        ↓
4. 모델 재학습 / 재평가
        ↓
5. Rule / Risk 정책 보정
        ↓
6. Backend Response → TTS
        ↓
7. Backend 준비 후 실제 API 연동
        ↓
8. React → Spring → Python End-to-End Test
```

---

# 31. 실행 예시

## FDS API

```bash
uvicorn src.fraud_detection.api:app --reload --port 8000
```

```text
Swagger
http://127.0.0.1:8000/docs
```

## Voice API

```bash
uvicorn src.voice_analysis.api:app --reload --port 8001
```

```text
Swagger
http://127.0.0.1:8001/docs
```

## 환경변수

OpenAI API Key 등 Secret은 코드에 하드코딩하지 않는다.

```env
OPENAI_API_KEY=...
BACKEND_BASE_URL=https://moviback.duckdns.org
VOICE_COMMAND_ENDPOINT=/api/v1/voice/command
```

`.env`는 Git에 포함하지 않는다.

```gitignore
.env
```

---

# 32. 핵심 설계 요약

### Voice AI

```text
음성 이해는 Python
금융 비즈니스 판단은 Spring
```

### FDS

```text
Isolation Forest
= 사용자 패턴에서 벗어난 정도

Rule Engine
= 설명 가능한 명시적 위험 조건

Final Risk Score
= 서비스에서 사용할 통합 위험도
```

### Integration

```text
React
  ↓
Spring
  ├─ Voice Command 관리
  ├─ DB / 금융 기능
  └─ Python AI API 호출
          ↓
      MOVI AI
      ├─ Voice Analysis
      └─ Fraud Detection
```

AI 모델과 서비스 비즈니스 로직을 분리하여 각 파트가 독립적으로 개발·테스트할 수 있도록 구성한다.
