# MOVI AI - 음성 요구사항 분석 및 이상거래 탐지

## 1. 프로젝트 개요

MOVI AI 파트는 다음 두 가지 기능을 담당한다.

1. 음성 기반 금융 요구사항 분석
2. 이상거래 탐지(Fraud Detection)

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

향후 FDS는 다음 구조로 확장할 예정이다.

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

## 2. 기술 스택

| 구분 | 기술 |
| --- | --- |
| Backend | Spring |
| Frontend | React |
| AI / Feature | Python |
| STT | Google Cloud Speech-to-Text V2 |
| 요구사항 분석 | GPT Structured Output |
| FDS | scikit-learn Isolation Forest |
| AI API | FastAPI |
| 데이터 처리 | Pandas / NumPy |
| 전처리 | scikit-learn |
| 모델 저장 | Joblib |
| Dataset | AIHub 금융 이상거래 데이터 |

---

## 3. 프로젝트 구조

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

Backend 연동 단계에서 다음 파일을 추가할 예정이다.

```text
src/fraud_detection/
├── api.py
└── schemas.py
```

---

# 이상거래 탐지

## 4. FDS 개발 목표

단순히 전체 거래에서 드문 거래를 탐지하는 것이 아니라 현재 거래가 해당 사용자의 과거 거래 패턴과 비교했을 때 얼마나 이례적인지 판단하는 것을 목표로 한다.

현재는 AIHub 전자금융공동망 데이터를 기준으로 Isolation Forest 기반 모델을 개발하였다.

---

## 5. `config.py`

### 역할

프로젝트 전반의 경로 및 모델 설정을 중앙에서 관리한다.

주요 관리 항목은 다음과 같다.

```text
Dataset Path
Model Path
Threshold Path
Report Path
Target Column
Isolation Forest Parameters
Random State
Training Sample Size
```

### 주요 고려사항

- 파일별 경로 하드코딩 제거
- 전자금융 / 카드 데이터 설정 분리
- Target Leakage 방지
- 모델 학습 재현성 확보

다음 컬럼은 모델 Feature로 사용하지 않는다.

```text
이상거래여부
이상거래유형
이상거래설명
```

계좌번호와 같은 ID 값은 숫자 자체를 Feature로 넣지 않고 사용자별 Historical Feature를 계산하기 위한 Key로 사용한다.

---

## 6. `data_loader.py`

### 역할

AIHub의 여러 CSV 파일을 자동 탐색하여 Train / Validation 데이터를 로드한다.

### 구현 기능

- CSV 자동 탐색
- 여러 CSV 병합
- Train / Validation 데이터 분리
- Dataset Schema 검증
- 일부 파일 및 일부 행 테스트 로딩
- Chunk Loader
- 거래 시간순 정렬

Historical Feature는 거래 순서에 의존하기 때문에 전자금융 데이터는 다음 순서로 정렬한다.

```text
거래일자
    ↓
거래시간대
```

현재 거래의 Feature 계산에 미래 거래 정보가 포함되는 것을 방지하기 위한 처리이다.

---

## 7. `feature_engineering.py`

### 역할

원본 거래 데이터를 Isolation Forest가 학습할 수 있는 Feature로 변환한다.

현재 전자금융공동망 기준 18개의 Engineered Feature를 생성한다.

### 기본 거래 Feature

| Feature | 의미 |
| --- | --- |
| `log_amount` | 로그 변환 거래금액 |
| `hour_sin` | 거래시간 Cyclic Encoding |
| `hour_cos` | 거래시간 Cyclic Encoding |
| `is_night` | 심야 거래 여부 |
| `weekday` | 거래 요일 |
| `is_weekend` | 주말 거래 여부 |
| `same_bank` | 출금 / 입금 금융회사 동일 여부 |

### 범주형 Feature

다음 값은 숫자 크기에 의미가 없는 코드값이므로 One-Hot Encoding을 적용한다.

```text
출금금융회사일련번호
입금금융회사일련번호
자금구분
매체구분
```

### Historical Feature

#### `amount_ratio`

현재 거래금액이 사용자 과거 평균 거래금액의 몇 배인지를 나타낸다.

```text
현재 거래금액
──────────────
과거 평균 거래금액
```

현재 거래 자체는 과거 평균 계산에서 제외한다.

#### `amount_zscore`

현재 거래금액이 사용자 과거 거래금액 분포에서 얼마나 벗어났는지 나타낸다.

```text
현재 거래금액 - 과거 평균
──────────────────────
과거 표준편차
```

#### `new_recipient`

현재 수취인이 해당 출금계좌에서 처음 등장한 수취인인지 확인한다.

```text
신규 수취인
→ 1

기존 거래 이력 존재
→ 0
```

#### `unusual_medium`

사용자가 과거에 사용하지 않았던 거래 매체인지 확인한다.

```text
신규 거래 매체
→ 1

기존 거래 매체
→ 0
```

계좌 자체의 첫 거래는 비교 가능한 과거 매체 정보가 없으므로 `0`으로 처리한다.

#### `historical_transaction_count`

현재 거래 이전까지 해당 출금계좌에서 발생한 전체 거래 횟수이다.

```text
첫 거래
→ 0

두 번째 거래
→ 1

세 번째 거래
→ 2
```

#### `same_day_transaction_count`

같은 날짜에 현재 거래 이전까지 발생한 거래 횟수이다.

반복 송금이나 단기간 다수 거래 패턴을 판단하기 위해 사용한다.

#### `same_time_bucket_count`

같은 날짜와 동일 시간대에 발생한 이전 거래 횟수이다.

```text
동일 출금계좌
+
동일 날짜
+
동일 시간대
```

기준으로 반복 거래를 탐지한다.

---

## 8. 전처리

### Numeric Feature

숫자형 Feature의 결측값은 Train 데이터의 Median으로 처리한다.

```text
Numeric Feature
    ↓
SimpleImputer(strategy="median")
```

초기 거래의 경우 과거 거래가 존재하지 않아 `amount_ratio`, `amount_zscore`가 `NaN`일 수 있다.

이는 정상적인 상황이며 Median Imputation을 통해 처리한다.

### Categorical Feature

범주형 Feature는 One-Hot Encoding을 적용한다.

```python
OneHotEncoder(
    handle_unknown="ignore"
)
```

Validation 또는 실제 서비스에서 Train 데이터에 존재하지 않았던 범주가 등장하더라도 오류가 발생하지 않도록 처리하였다.

One-Hot Encoding 이후에는 Sparse Matrix와 `float32`를 사용하여 메모리 사용량을 줄인다.

---

## 9. `train_iforest.py`

### 역할

정상거래 패턴을 기준으로 Isolation Forest를 학습하고 Model Bundle을 저장한다.

### 학습 흐름

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
Isolation Forest 학습
    ↓
Model Bundle 저장
```

Historical Feature 추가 전에는 정상거래를 먼저 Sampling한 뒤 Feature Engineering을 수행하였다.

이 방식은 일부 거래 이력이 제거되어 `amount_ratio`, `new_recipient`, 거래 횟수 등의 값이 왜곡될 수 있다.

따라서 현재는 전체 거래를 기준으로 Historical Feature를 먼저 생성한 뒤 정상거래만 학습 데이터로 사용한다.

---

## 10. Isolation Forest

현재 모델은 정상거래 패턴을 학습한다.

```python
IsolationForest(
    contamination="auto",
    random_state=42,
    n_jobs=-1
)
```

Target인 `이상거래여부`는 모델 Feature로 사용하지 않는다.

실제 Label은 Validation 평가 및 Threshold 결정 단계에서만 사용한다.

---

## 11. Model Bundle

모델과 전처리 객체를 하나의 Bundle로 저장한다.

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

저장 위치는 다음과 같다.

```text
models/electronic/isolation_forest.joblib
```

실제 서비스에서도 학습 당시 사용한 동일한 Preprocessor를 사용하기 위해 Model과 Preprocessor를 함께 저장한다.

---

## 12. `evaluate.py`

### 역할

AIHub Validation Label을 이용하여 모델 성능을 평가하고 이상거래 판정을 위한 Threshold를 결정한다.

### 평가 흐름

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
    ↓
평가지표 계산
```

Isolation Forest의 `score_samples()`는 값이 낮을수록 이상치에 가깝다.

MOVI에서는 위험도가 높을수록 Score가 증가하는 형태로 사용하기 위해 다음과 같이 변환한다.

```python
anomaly_score = -model.score_samples(X)
```

따라서 프로젝트에서는 다음 기준을 사용한다.

```text
Anomaly Score 증가
=
이상거래 위험 증가
```

---

## 13. 전체 Validation 평가 결과

전자금융공동망 전체 Validation 데이터 기준 결과이다.

```text
전체 거래 : 492,678건
정상 거래 : 490,770건
이상 거래 :   1,908건
```

Best F1 Threshold:

```text
0.446117
```

### 평가 결과

| Metric | 결과 |
| --- | ---: |
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

### Anomaly Score 비교

```text
Normal Mean : 0.3737
Fraud Mean  : 0.4176
```

실제 Fraud 거래의 Anomaly Score가 정상거래보다 전반적으로 높게 나타났다.

ROC-AUC는 약 `0.8848`로 정상 / 이상거래의 상대적인 Ranking은 유효한 것으로 확인하였다.

하지만 Best F1 Threshold 기준 Recall은 약 `19.8%`로 Isolation Forest 단독으로 최종 이상거래 여부를 결정하기에는 한계가 존재한다.

따라서 Isolation Forest는 최종 판정기보다 정상 패턴에서 얼마나 벗어났는지를 나타내는 Anomaly Score 생성기로 사용하고 Rule Engine과 결합할 예정이다.

---

## 14. Dummy 이상거래 테스트

실제 모델 동작을 확인하기 위해 특정 사용자의 과거 거래 이력을 구성한 뒤 정상 거래와 의도적으로 생성한 이상거래를 비교하였다.

### 정상 거래 조건

```text
평소와 유사한 거래금액
기존 수취인
기존 거래 매체
일반 시간대
동일 금융회사
```

결과:

```text
score_samples = -0.412090
anomaly_score = 0.412090
```

### 이상 거래 조건

```text
평소 약 100,000원 거래
    ↓
10,000,000원 송금

신규 수취인
심야 거래
신규 거래 매체
타행 거래
```

주요 생성 Feature:

```text
amount_ratio   = 99.29
amount_zscore  = 758.71
is_night       = 1
same_bank      = 0
new_recipient  = 1
unusual_medium = 1
```

결과:

```text
score_samples = -0.482914
anomaly_score = 0.482914
```

현재 Validation Threshold:

```text
0.446117
```

따라서 Dummy 이상거래는 Threshold보다 높은 Anomaly Score를 기록하여 Fraud 영역에 위치한다.

---

## 15. 현재 FDS 한계

### Isolation Forest 단독 탐지 성능

현재 Best F1 Threshold 기준 Recall이 약 `19.8%`로 실제 이상거래를 다수 놓친다.

Isolation Forest 단독 판정보다 Rule Engine과 결합하는 구조가 필요하다.

### 정상 / 이상 Score 분포 중첩

정상거래와 이상거래 평균 Score에는 차이가 있지만 일부 구간에서 Score 분포가 겹친다.

Threshold를 낮추면 Recall은 증가하지만 False Positive가 크게 증가한다.

### Train History와 Validation History 연결

현재 Validation의 Historical Feature는 Validation 내부의 거래 이력을 기준으로 생성한다.

향후 Train 기간의 거래 History와 Validation History를 연결하여 실제 서비스 환경과 가까운 형태로 개선할 예정이다.

### Historical Feature 극단값

`amount_ratio`, `amount_zscore`에서 큰 극단값이 존재한다.

향후 로그 변환, Clip 또는 Robust 처리에 따른 성능 변화를 비교할 예정이다.

---

# 음성 기반 요구사항 분석

## 16. 처리 흐름

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
Requirement Validator
    ↓
누락 정보 확인
    ↓
Follow-up
    ↓
Requirement Complete
```

---

## 17. `stt_stream_service.py`

Google Cloud Speech-to-Text V2를 이용하여 실시간 마이크 음성을 텍스트로 변환한다.

### 구현 완료

- Streaming STT
- PCM Audio 처리
- Interim Transcript
- Final Transcript
- 실제 마이크 입력 테스트

최종 요구사항 분석에는 Final Transcript만 전달한다.

---

## 18. `wake_word_detector.py`

호출어 `모비야`가 포함된 음성 명령만 금융 명령으로 처리한다.

예:

```text
모비야 김민수한테 오만원 보내줘
```

변환:

```text
김민수한테 오만원 보내줘
```

Wake Word가 없는 일반 대화가 금융 명령으로 처리되는 것을 방지하고 불필요한 GPT API 호출을 줄인다.

---

## 19. `requirement_analyzer.py`

GPT Structured Output을 이용하여 음성 명령에서 다음 정보를 추출한다.

```text
Intent
Entity
Missing Fields
```

예:

```text
김민수한테 오만원 보내줘
```

결과:

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

## 20. `requirement_validator.py`

LLM의 결과를 그대로 금융 기능 실행에 사용하지 않고 Python 코드에서 필수 Entity를 다시 검증한다.

계좌이체 기준 필수 Entity:

```text
recipient_name
recipient_bank
recipient_account
amount
```

역할을 다음과 같이 분리한다.

```text
GPT
→ 자연어 분석

Validator
→ 금융 요청 실행 가능 여부 검증
```

---

## 21. Multi-turn 요구사항 분석

사용자가 한 문장에서 모든 정보를 제공하지 않은 경우 누락된 정보만 추가로 요청한다.

```text
사용자
"김민수한테 오만원 보내줘"

    ↓

recipient_name = 김민수
amount = 50000
recipient_bank = null
recipient_account = null

    ↓

"받는 분의 은행을 말씀해주세요."

    ↓

사용자
"국민은행"

    ↓

기존 Context에 recipient_bank 병합
```

관련 모듈:

```text
conversation_context.py
follow_up_entity_parser.py
voice_pipeline.py
```

---

## 22. 현재 지원 Intent

| 기능 | Intent |
| --- | --- |
| 화면 읽기 | `read_screen` |
| 계좌 이체 | `transfer_money` |
| 거래내역 조회 | `check_history` |
| 가입 적금 조회 | `check_savings` |
| 확인 | `confirm` |
| 거절 | `deny` |
| 취소 | `cancel` |
| 미지원 명령 | `unknown` |

---

## 23. Voice Pipeline 완료 범위

- [x] Google STT V2 설정
- [x] Streaming STT
- [x] Wake Word 감지
- [x] GPT Intent 분석
- [x] Entity 추출
- [x] 금액 표현 구조화
- [x] Requirement Validator
- [x] Missing Field 판단
- [x] Multi-turn Context
- [x] Follow-up Entity 분석
- [x] Requirement JSON 생성

Voice Pipeline 상태값:

```text
ignored
awaiting_command
need_more_info
ready
unsupported
error
```

---

# Backend 연동

## 24. FDS Backend 연동 구조

중간 통합에서는 Python FDS를 FastAPI 서버로 제공하고 Spring Backend가 해당 API를 호출하도록 구성한다.

```text
React
    ↓
Spring Backend
    ↓
POST /api/v1/fraud/detect
    ↓
Python FastAPI
    ↓
Feature Engineering
    ↓
Isolation Forest
    ↓
Anomaly Score
    ↓
JSON Response
    ↓
Spring Backend
```

FDS Request에는 다음 정보를 포함한다.

```text
현재 거래
+
동일 출금계좌의 과거 거래 History
```

예상 Response:

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

현재 FastAPI Request / Response 구조 및 Spring 호출 API 명세까지 정의한 상태이다.

---

# 현재 개발 상태

## 25. 음성 요구사항 분석

- [x] Google STT V2
- [x] Streaming STT
- [x] Wake Word 감지
- [x] Intent 분석
- [x] Entity 추출
- [x] Missing Entity 검사
- [x] Multi-turn 보완
- [x] Requirement JSON 생성

## 26. 이상거래 탐지

- [x] AIHub 데이터 Loader
- [x] Train / Validation 데이터 로딩
- [x] 거래 시간순 정렬
- [x] 기본 Feature Engineering
- [x] Historical Feature Engineering
- [x] `amount_ratio`
- [x] `amount_zscore`
- [x] `new_recipient`
- [x] `unusual_medium`
- [x] `historical_transaction_count`
- [x] `same_day_transaction_count`
- [x] `same_time_bucket_count`
- [x] One-Hot Encoding
- [x] Sparse Matrix 처리
- [x] Isolation Forest 학습
- [x] Model Bundle 저장
- [x] Dummy 정상 / 이상거래 테스트
- [x] 전체 Validation 평가
- [x] Threshold 탐색
- [x] Evaluation Metrics 저장

## 27. Backend 연동

- [x] FastAPI 연동 구조 설계
- [x] Request / Response JSON 명세
- [x] Spring ↔ Python API 명세
- [ ] FastAPI Endpoint 구현
- [ ] FastAPI 서버 실행 검증
- [ ] Spring 실제 호출 테스트

---

# 2주차 개발 계획

## 28. 우선순위

| 우선순위 | 작업 | 완료 기준 |
| --- | --- | --- |
| P0 | FastAPI FDS API 구현 | 실제 모델 결과 JSON 반환 |
| P0 | Spring ↔ FastAPI 중간 통합 | Spring에서 FDS 결과 수신 |
| P0 | 거래 History 전달 | Historical Feature 정상 생성 |
| P1 | Rule Engine 구현 | Rule Score 생성 |
| P1 | Final Risk Score 구현 | LOW / MEDIUM / HIGH 반환 |
| P1 | Train → Validation History 연결 | Historical Feature 평가 정확도 개선 |
| P2 | Historical Feature 안정화 | Baseline 대비 성능 비교 |
| P2 | Threshold 정책 개선 | Recall / F1 기반 정책 비교 |
| P2 | Dummy Scenario 확대 | 위험도 단계별 Score 비교 |
| P2 | Voice → FDS 연결 | 음성 이체 요청에서 FDS 호출 |
| P3 | API 예외처리 | 오류 상황별 HTTP Response 처리 |
| P3 | End-to-End 통합 테스트 | React → Spring → Python 전체 연결 |

---

## 29. Backend 중간 통합

- [ ] FastAPI 설치 및 서버 구성
- [ ] `GET /health` 구현
- [ ] `POST /api/v1/fraud/detect` 구현
- [ ] 서버 시작 시 Model Bundle 1회 Load
- [ ] `threshold.json` Load
- [ ] Spring Request를 FDS Feature 형식으로 변환
- [ ] History + Current Transaction Feature Engineering
- [ ] 현재 거래에 대한 Isolation Forest 추론
- [ ] Anomaly Score 계산
- [ ] Threshold 비교
- [ ] Fraud 여부 JSON 반환
- [ ] curl / Postman 테스트
- [ ] Spring WebClient 호출 테스트

---

## 30. Rule Engine

다음 규칙을 기반으로 설명 가능한 이상거래 점수를 생성한다.

- [ ] 고액 거래 Rule
- [ ] 평소 거래금액 대비 급증 Rule
- [ ] 심야 거래 Rule
- [ ] 신규 수취인 Rule
- [ ] 신규 거래 매체 Rule
- [ ] 동일 날짜 반복 거래 Rule
- [ ] 동일 시간대 반복 거래 Rule
- [ ] Rule별 가중치 정의
- [ ] `rule_score` 계산
- [ ] Triggered Rule 반환

예상 결과:

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

---

## 31. Final Risk Score

Isolation Forest와 Rule Engine 결과를 결합하여 서비스에서 사용할 최종 위험도를 생성한다.

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
    ↓
LOW / MEDIUM / HIGH
```

TODO:

- [ ] Isolation Forest Score 정규화
- [ ] Rule Score 정규화
- [ ] Model / Rule 가중치 결정
- [ ] Final Risk Score 계산
- [ ] LOW / MEDIUM / HIGH 기준 정의
- [ ] FastAPI Response에 Risk Score 추가

---

## 32. FDS 모델 개선

현재 Baseline 비교 기준:

```text
ROC-AUC : 0.8848
PR-AUC  : 0.0579
F1      : 0.1262
Recall  : 0.1981
```

TODO:

- [ ] Train History → Validation History 연결
- [ ] 현재 Baseline Metric 보존
- [ ] `amount_ratio` 분포 재확인
- [ ] `log_amount_ratio` 실험
- [ ] `amount_zscore` 극단값 분석
- [ ] Clip / Robust 처리 실험
- [ ] 동일 Validation 기준 재평가
- [ ] Precision / Recall / F1 비교
- [ ] PR-AUC 비교
- [ ] 개선 Feature 최종 선택

---

## 33. Dummy Scenario 확대

현재 정상 / 강한 이상거래 2개 Case를 다음과 같이 확대한다.

```text
Case 1
평소와 동일한 거래

Case 2
고액 거래

Case 3
고액 + 신규 수취인

Case 4
고액 + 신규 수취인 + 심야 + 신규 매체

Case 5
고액 + 신규 수취인 + 심야 + 신규 매체 + 반복 거래
```

TODO:

- [ ] Case별 Dummy Transaction 생성
- [ ] Feature 생성 결과 확인
- [ ] Anomaly Score 비교
- [ ] Rule Score 비교
- [ ] Final Risk Score 비교
- [ ] 위험 조건 증가에 따라 Risk Score가 증가하는지 확인

---

## 34. Voice → FDS 연결

Voice Pipeline에서 계좌이체 요구사항이 완성되면 Spring을 통해 FDS를 호출한다.

```text
"모비야 김민수한테 오만원 보내줘"
        ↓
STT
        ↓
Intent / Entity 분석
        ↓
누락 정보 보완
        ↓
Requirement Complete
        ↓
Spring Backend
        ↓
거래 History 조회
        ↓
Python FDS
        ↓
Risk Score
        ↓
거래 진행 / 추가 확인
```

TODO:

- [ ] `VoicePipeline`의 `status=ready` 확인
- [ ] `transfer_money` Entity를 Backend Request로 변환
- [ ] 실제 계좌 정보 조회 연결
- [ ] 거래 History 조회
- [ ] FDS 호출
- [ ] Fraud Detection 결과 수신
- [ ] 사용자 확인 단계 연결

---

## 35. API 예외처리

다음 상황에 대한 예외처리를 추가한다.

- [ ] Model 파일 없음
- [ ] Threshold 파일 없음
- [ ] Request Schema 오류
- [ ] 거래 History 없음
- [ ] 필수 거래 정보 누락
- [ ] Feature Engineering 실패
- [ ] Model Inference 실패
- [ ] Spring ↔ FastAPI 통신 실패
- [ ] 명확한 HTTP Status Code 반환

---

## 36. 최종 통합 테스트

최종 목표 흐름:

```text
React
    ↓
사용자 음성
    ↓
Python Voice Analysis
    ↓
Requirement JSON
    ↓
Spring Backend
    ↓
거래 및 History 조회
    ↓
Python FDS FastAPI
    ↓
Isolation Forest
    +
Rule Engine
    ↓
Final Risk Score
    ↓
Spring Backend
    ↓
React
```

TODO:

- [ ] React → Spring 요청
- [ ] Spring → Python Voice / FDS 연동
- [ ] FDS Response 확인
- [ ] 위험 거래 처리 분기
- [ ] 정상 거래 처리 분기
- [ ] 오류 Response 테스트
- [ ] End-to-End Scenario 테스트

---

# 다음 개발 순서

```text
1. FastAPI FDS 구현
        ↓
2. Spring ↔ Python 중간 통합
        ↓
3. Rule Engine 구현
        ↓
4. Final Risk Score 구현
        ↓
5. Train → Validation History 연결
        ↓
6. Historical Feature 개선
        ↓
7. 모델 재학습 / 재평가
        ↓
8. Dummy Scenario 확대
        ↓
9. Voice → FDS 연결
        ↓
10. React → Spring → Python E2E 테스트
```
