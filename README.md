# MOVI - 이상거래 탐지 Baseline 개발 기록

## 1. 개발 개요

MOVI의 이상거래 탐지 기능은 거래 데이터를 기반으로 일반적인 거래 패턴에서 벗어난 거래를 탐지하고, 해당 거래의 위험도를 Backend에 전달하는 것을 목표로 한다.

현재 단계에서는 AIHub 금융 데이터를 이용하여 **Isolation Forest 기반 Baseline 이상거래 탐지 파이프라인**을 구축하였다.

### 개발 환경

* Language: Python
* IDE: VSCode
* 주요 Library

  * pandas
  * numpy
  * scikit-learn
  * scipy
  * joblib
* Dataset

  * AIHub 전자금융공동망 거래 데이터
  * AIHub 카드거래 데이터

### 전체 처리 흐름

```text
거래 데이터
    ↓
데이터 로딩
    ↓
Feature Engineering
    ↓
Isolation Forest 학습
    ↓
Validation 평가
    ↓
Threshold 결정
    ↓
실제 거래 입력
    ↓
Anomaly Score 계산
    ↓
이상거래 판정
    ↓
Risk Score 생성
    ↓
Backend 전달
```

---

# 2. 프로젝트 구조

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
│   ├── electronic/
│   │   ├── isolation_forest.joblib
│   │   └── threshold.json
│   │
│   └── card/
│
├── reports/
│   └── electronic_metrics.json
│
├── src/
│   └── fraud_detection/
│       ├── __init__.py
│       ├── config.py
│       ├── data_loader.py
│       ├── feature_engineering.py
│       ├── train_iforest.py
│       ├── evaluate.py
│       └── inference.py
│
└── requirements.txt
```

각 파일은 하나의 책임만 가지도록 분리하였다.

```text
config
   ↓
data_loader
   ↓
feature_engineering
   ↓
train_iforest
   ↓
evaluate
   ↓
inference
```

---

# 3. 구현 순서

## ① `config.py`

### 역할

프로젝트 전체에서 사용하는 **경로, 데이터 설정, 모델 설정, 학습 Hyperparameter를 중앙 관리**한다.

주요 관리 항목은 다음과 같다.

```text
데이터 위치
모델 저장 위치
평가 결과 저장 위치
Target 컬럼
Leakage 컬럼
Sampling 크기
Isolation Forest Parameter
```

### 주요 고려사항

#### 1. 경로 하드코딩 제거

각 파일에서 다음과 같은 방식으로 경로를 직접 작성하지 않는다.

```python
pd.read_csv(
    "../../../data/Train/..."
)
```

대신 `config.py`에서 프로젝트 루트를 기준으로 관리한다.

```python
PROJECT_ROOT = Path(__file__).resolve().parents[2]
```

이를 통해 실행 위치가 달라져도 동일한 데이터 경로를 사용할 수 있도록 구성하였다.

---

#### 2. 전자금융과 카드 모델 분리

두 데이터는 Feature 의미와 거래 구조가 다르기 때문에 하나의 모델로 처리하지 않는다.

```text
전자금융공동망
    ↓
Electronic Fraud Model

카드거래
    ↓
Card Fraud Model
```

따라서 설정 역시 다음처럼 분리하였다.

```python
ELECTRONIC_CONFIG
CARD_CONFIG
```

---

#### 3. Target Leakage 방지

다음 컬럼들은 학습 Feature에서 반드시 제외한다.

```text
이상거래여부
이상거래유형
이상거래설명
```

특히 `이상거래유형`, `이상거래설명`은 실제 이상거래 발생 이후 알 수 있는 정보이므로 모델에 포함될 경우 성능이 비정상적으로 높아지는 **Target Leakage**가 발생할 수 있다.

---

#### 4. ID 값은 Feature와 분리

다음 값은 숫자로 되어 있지만 실제 의미는 식별자이다.

```text
출금계좌일련번호
입금계좌일련번호
카드KEY
가맹점KEY
```

따라서 ID 숫자의 크기를 모델이 학습하지 않도록 직접 Feature로 사용하지 않는다.

향후 사용자 행동 기반 Feature를 만들기 위한 Grouping Key로 활용한다.

---

# 4. ② `data_loader.py`

## 역할

AIHub 데이터가 여러 CSV 파일로 나누어져 있기 때문에 해당 파일들을 자동 탐색하고 필요한 형태로 로딩한다.

주요 기능은 다음과 같다.

```text
CSV 자동 탐색
여러 CSV 병합
Train / Validation 선택
전자금융 / 카드 선택
CSV 컬럼 검증
Chunk Loading
```

---

## 주요 고려사항

### 1. 여러 CSV 자동 탐색

특정 CSV 이름을 직접 지정하지 않고:

```python
directory.rglob("*.csv")
```

를 이용하여 폴더 내부 CSV를 자동 탐색한다.

따라서 새로운 분기 데이터 파일이 추가되더라도 별도의 코드 수정 없이 처리할 수 있다.

---

### 2. CSV Schema 검증

여러 CSV를 단순히 `concat()`할 경우 파일마다 컬럼 구조가 다르면 의도하지 않은 `NaN` 컬럼이 발생할 수 있다.

따라서 첫 번째 CSV의 컬럼 구조를 기준으로 나머지 파일들의 Schema를 검증한다.

```text
CSV A
12 columns
     ↓
CSV B
12 columns
     ↓
동일 여부 확인
```

Schema가 다르면 즉시 오류를 발생시키도록 구현하였다.

---

### 3. 대용량 데이터 Chunk Loading

전자금융 Train 데이터는 수백만 건 규모이기 때문에 전체 데이터를 한 번에 RAM에 로드하는 방식을 피하였다.

```text
전체 데이터
약 400만 건

        ↓

100,000건
100,000건
100,000건
...
```

`chunksize` 기반 Iterator를 제공하여 이후 학습 과정에서 메모리 사용량을 줄일 수 있도록 구현하였다.

```python
iter_dataset_chunks(...)
```

---

### 4. 테스트 모드 제공

전체 데이터를 매번 로딩하면 개발 속도가 느려지므로 다음 옵션을 제공하였다.

```python
max_files=1
nrows_per_file=1000
```

이를 이용하면 실제 대규모 데이터를 처리하기 전에 일부 데이터만 빠르게 테스트할 수 있다.

---

# 5. ③ `feature_engineering.py`

## 역할

원본 거래 데이터를 Isolation Forest가 학습할 수 있는 형태의 Feature로 변환한다.

현재 Baseline에서는 **거래 자체의 특성을 기반으로 이상 여부를 판단하는 Feature v0**를 구현하였다.

---

# 전자금융 Feature

## 거래 금액

원본:

```text
거래금액
```

변환:

```text
log_amount
```

코드:

```python
log_amount = np.log1p(amount)
```

### 고려 이유

금융 거래 데이터는 다음처럼 금액 분포가 크게 치우칠 수 있다.

```text
10,000
50,000
100,000
1,000,000
100,000,000
```

금액을 그대로 사용할 경우 극단적으로 큰 거래값이 Feature 공간에 큰 영향을 줄 수 있다.

따라서 로그 변환을 통해 금액의 규모 차이는 유지하면서 분포의 왜도를 완화하였다.

---

# 거래 시간

원본:

```text
거래시간대
```

변환:

```text
hour_sin
hour_cos
```

시간은 일반적인 숫자와 달리 **순환 구조**를 가진다.

예를 들어:

```text
23시 ↔ 0시
```

는 실제로 가까운 시간이지만 숫자만 사용하면 매우 먼 값으로 표현된다.

따라서 Sin/Cos 기반 Cyclic Encoding을 사용하였다.

---

# 거래 날짜

원본:

```text
거래일자
```

변환:

```text
weekday
is_weekend
```

이를 통해 단순 날짜값 자체보다:

```text
월요일 거래
주말 거래
```

등 거래 패턴과 관련된 정보를 모델이 활용할 수 있도록 하였다.

---

# 동일 금융회사 여부

다음 두 Feature에서:

```text
출금금융회사일련번호
입금금융회사일련번호
```

새로운 Feature를 생성하였다.

```text
same_bank
```

```text
같은 금융회사 → 1
다른 금융회사 → 0
```

단순 코드값 이외에 거래 관계에 대한 의미 있는 정보를 추가하기 위한 Feature이다.

---

# 범주형 Feature

다음과 같은 값은 숫자로 되어 있더라도 순서나 크기에 의미가 없다.

```text
금융회사 코드
자금구분
매체구분
```

예를 들어 은행 코드:

```text
110
155
160
```

에서 `160 > 155`라는 관계에는 의미가 없다.

따라서 `OneHotEncoder`를 사용하여 범주형 데이터로 처리하였다.

---

# Sparse Matrix 사용

One-Hot Encoding 이후 Feature 수는 증가하지만 대부분의 값은 `0`이다.

```text
0 0 0 1 0 0 0 0 1 ...
```

따라서 Dense Matrix 대신 Sparse Matrix를 사용하여 메모리 사용량을 줄였다.

---

# Train / Validation 전처리 분리

Train:

```text
Feature Engineering
        ↓
Preprocessor.fit()
        ↓
transform()
```

Validation / Inference:

```text
Feature Engineering
        ↓
Train에서 학습한 Preprocessor
        ↓
transform()
```

Validation이나 실제 사용자 입력에 다시 `fit()`을 수행하지 않는다.

이는 학습과 추론 시 Feature 구조가 달라지는 것을 방지하기 위해 매우 중요하다.

---

# 6. ④ `train_iforest.py`

## 역할

정상 거래를 기반으로 Isolation Forest를 학습하고 모델을 저장한다.

전체 학습 과정:

```text
Train CSV
    ↓
Chunk Loading
    ↓
정상거래 추출
    ↓
Random Sampling
    ↓
Feature Engineering
    ↓
Isolation Forest
    ↓
Model 저장
```

---

# 정상거래만 학습

Train 데이터에는 정상거래와 이상거래가 모두 존재한다.

Baseline에서는:

```python
이상거래여부 == 0
```

인 데이터만 학습에 사용하였다.

목표는 Isolation Forest에게:

> 정상적인 금융 거래의 Feature 공간

을 학습시키는 것이다.

이후 정상적인 거래 패턴에서 크게 벗어난 거래를 이상치로 판단한다.

---

# 300,000건 Sampling

전체 정상거래 수백만 건을 모두 학습하지 않고 Baseline에서는:

```text
300,000건
```

을 Sampling하여 사용하였다.

이유는 다음과 같다.

* 초기 Baseline 개발 속도 확보
* 메모리 사용량 감소
* 모델 반복 실험 시간 감소
* Feature 수정 시 빠른 재학습

향후 최종 모델에서는 Sampling Size를 변경하여 성능 차이를 비교할 수 있다.

---

# Sampling 편향 방지

단순히:

```python
normal_df.head(300000)
```

을 사용하면 앞쪽 월 또는 특정 파일의 데이터만 학습될 가능성이 있다.

따라서 전체 Train 데이터를 순회하며 각 행에 Random Priority를 부여하였다.

```text
전체 정상거래
      ↓
Random Priority
      ↓
가장 작은 300,000개 유지
```

이를 통해 전체 Train 데이터 범위에서 균등한 Random Sample을 얻도록 설계하였다.

---

# Isolation Forest 설정

Baseline Parameter:

```python
IsolationForest(
    n_estimators=200,
    max_samples=4096,
    contamination="auto",
    n_jobs=-1,
    random_state=42
)
```

### `n_estimators`

Isolation Tree의 개수.

Tree 수가 많아질수록 모델 안정성이 증가하지만 학습 비용 역시 증가한다.

### `max_samples`

각 Tree를 생성할 때 사용하는 Sample 수.

### `contamination`

현재는 `"auto"`를 사용한다.

최종 이상거래 판정 Threshold는 Isolation Forest 기본값을 그대로 사용하지 않고 Validation 데이터를 기반으로 별도 결정한다.

### `random_state`

동일한 조건에서 실험 결과를 재현할 수 있도록 설정하였다.

---

# 모델 + Preprocessor 통합 저장

단순히 모델만 저장하지 않는다.

```python
{
    "model": model,
    "preprocessor": preprocessor,
    "dataset_type": ...,
    "feature_names": ...,
    "training_score_summary": ...,
    "environment": ...
}
```

형태의 Model Bundle을 생성하여 `joblib`으로 저장한다.

```text
models/
└── electronic/
    └── isolation_forest.joblib
```

### 고려 이유

실서비스에서도 반드시 **학습 당시 사용한 동일한 Encoder와 전처리 방법**을 사용해야 한다.

따라서 Model과 Preprocessor를 함께 저장하였다.

---

# 7. ⑤ `evaluate.py`

## 역할

학습한 Isolation Forest를 Validation 데이터에 적용하여 성능을 평가하고 실제 이상거래 판정을 위한 Threshold를 결정한다.

```text
Validation
    ↓
Feature Transform
    ↓
Isolation Forest
    ↓
Anomaly Score
    ↓
실제 Label과 비교
    ↓
Threshold 탐색
    ↓
평가지표 계산
```

---

# Anomaly Score 정의

scikit-learn의 Isolation Forest는 `score_samples()` 값이 낮을수록 이상치에 가깝다.

서비스에서는 반대로:

```text
점수가 높을수록 위험
```

인 것이 직관적이기 때문에 다음과 같이 정의하였다.

```python
anomaly_score = -model.score_samples(X)
```

따라서 전체 프로젝트에서는:

```text
Anomaly Score ↑
=
위험도 ↑
```

라는 규칙을 사용한다.

---

# Threshold 결정

모델의 기본 `predict()` 결과를 그대로 사용하지 않고 Validation Label을 이용하여 별도 Threshold를 결정하였다.

판정 규칙:

```python
if anomaly_score >= threshold:
    fraud = 1
else:
    fraud = 0
```

현재 Baseline에서는 **F1 Score가 최대가 되는 Threshold**를 선택한다.

---

# 평가 지표

금융 이상거래 데이터는 클래스 불균형이 매우 크기 때문에 Accuracy만으로 평가하면 안 된다.

주요 지표:

```text
Precision
Recall
F1 Score
Average Precision
PR-AUC
ROC-AUC
Confusion Matrix
```

---

## Precision

모델이 이상거래라고 판단한 것 중 실제 이상거래 비율.

```text
TP
─────────
TP + FP
```

Precision이 낮으면 정상 거래를 이상거래로 판단하는 **오탐**이 많다는 의미이다.

---

## Recall

실제 이상거래 중 모델이 탐지한 비율.

```text
TP
─────────
TP + FN
```

FDS에서는 실제 이상거래를 놓치는 `FN`이 중요하므로 Recall을 주요 지표로 확인한다.

---

## F1 Score

Precision과 Recall의 균형을 평가한다.

현재 Baseline Threshold 선택 기준으로 사용하였다.

---

## PR-AUC

클래스 불균형이 큰 Fraud Detection 문제에서 모델의 Precision/Recall 관계를 평가하기 위해 사용한다.

---

## Confusion Matrix

```text
                 Prediction

                 Normal   Fraud

Actual Normal      TN       FP
Actual Fraud       FN       TP
```

특히:

```text
FP
= 정상 거래를 이상거래로 오탐

FN
= 실제 이상거래를 놓침
```

을 중점적으로 확인한다.

---

# Threshold 저장

평가 결과 선택된 Threshold는 다음 위치에 저장한다.

```text
models/
└── electronic/
    └── threshold.json
```

예:

```json
{
    "dataset_type": "electronic",
    "threshold": 0.53,
    "selection_method": "max_f1_on_validation"
}
```

이를 실제 `inference.py`에서 사용한다.

---

# Metric 저장

최종 평가 결과는 다음 위치에 저장한다.

```text
reports/
└── electronic_metrics.json
```

이를 통해 향후 Feature Engineering 버전별 성능을 비교할 수 있다.

```text
Baseline
    ↓
Feature v1
    ↓
Feature v2
```

각 버전의 Precision / Recall / F1 등을 비교하여 실제 Feature 추가 효과를 검증할 수 있다.

---

# 8. ⑥ `inference.py`

## 역할

실제 거래 1건을 입력받아 저장된 모델을 사용해 이상거래 여부와 Risk Score를 반환한다.

```text
실제 거래 JSON
      ↓
Preprocessor
      ↓
Isolation Forest
      ↓
Anomaly Score
      ↓
Threshold
      ↓
Fraud 여부
      ↓
Risk Score
```

---

# 실제 입력 예시

```json
{
    "출금금융회사일련번호": 155,
    "입금금융회사일련번호": 160,
    "자금구분": 1,
    "거래금액": 5000000,
    "거래시간대": 3,
    "매체구분": 2,
    "거래일자": 20260812
}
```

---

# 결과 예시

```json
{
    "dataset_type": "electronic",
    "is_fraud": true,
    "risk_score": 81.35,
    "risk_level": "CRITICAL",
    "anomaly_score": 0.58342851,
    "threshold": 0.52137482,
    "decision_margin": 0.06205369
}
```

---

# 반환값 의미

| 값                 | 설명                             |
| ----------------- | ------------------------------ |
| `is_fraud`        | 최종 이상거래 판정                     |
| `anomaly_score`   | Isolation Forest 기반 이상치 점수     |
| `threshold`       | Validation에서 결정한 기준값           |
| `decision_margin` | Anomaly Score와 Threshold의 차이   |
| `risk_score`      | 서비스 표시용 0~100 위험도              |
| `risk_level`      | LOW / MEDIUM / HIGH / CRITICAL |

---

# Risk Score 주의사항

`risk_score`는 Fraud 발생 확률이 아니다.

예를 들어:

```text
risk_score = 85
```

라고 해서:

```text
85% 확률로 이상거래
```

라는 의미가 아니다.

현재 Risk Score는 Isolation Forest의 Anomaly Score와 Threshold 간 거리를 기반으로 만든 **서비스용 위험 지표**이다.

---

# Risk Level

현재 표시용 위험등급은 다음과 같다.

```text
0 ~ 30
LOW

30 ~ 50
MEDIUM

50 ~ 70
HIGH

70 ~ 100
CRITICAL
```

단, 실제 Fraud 판정은 Risk Level이 아니라:

```python
anomaly_score >= threshold
```

를 기준으로 수행한다.

---

# 모델 로딩 방식

실제 API Server에서는 요청마다 모델을 다시 읽으면 안 된다.

잘못된 구조:

```text
Request
  ↓
model load
  ↓
prediction

Request
  ↓
model load
  ↓
prediction
```

대신 서버 시작 시:

```python
detector = FraudDetector("electronic")
```

를 한 번 생성하고:

```text
Server Start
     ↓
Model Load 1회
     ↓
Request 1 → predict
Request 2 → predict
Request 3 → predict
```

형태로 사용한다.

이를 통해 추론 요청에 대한 불필요한 Disk I/O를 줄인다.

---

# 9. 현재 Baseline Feature

## 전자금융공동망

| 구분  | Feature      | 목적            |
| --- | ------------ | ------------- |
| 금액  | `log_amount` | 거래금액 분포 왜도 완화 |
| 시간  | `hour_sin`   | 거래시간 순환성 표현   |
| 시간  | `hour_cos`   | 거래시간 순환성 표현   |
| 날짜  | `weekday`    | 요일 거래 특성      |
| 날짜  | `is_weekend` | 주말 거래 여부      |
| 금융사 | `same_bank`  | 동일 금융회사 송금 여부 |
| 금융사 | 출금금융회사       | 금융회사 유형       |
| 금융사 | 입금금융회사       | 금융회사 유형       |
| 거래  | 자금구분         | 거래 유형         |
| 거래  | 매체구분         | 거래 매체         |

---

# 10. 현재 Baseline의 한계

현재 모델은 주로:

> **전체 거래 중 이 거래가 일반적인 거래 형태에서 벗어났는가?**

를 판단한다.

하지만 실제 금융 이상거래에서는:

> **이 사용자가 평소 하던 행동과 비교했을 때 이상한가?**

가 매우 중요하다.

예를 들어 다음 두 사용자를 생각할 수 있다.

```text
사용자 A
평균 거래금액: 50,000원

현재 거래:
5,000,000원
```

이는 평소보다 약 100배 큰 거래다.

반면:

```text
사용자 B
평균 거래금액: 4,000,000원

현재 거래:
5,000,000원
```

같은 `5,000,000원`이라도 두 사용자에게 위험 의미가 다르다.

현재 Baseline에서는 이러한 **개인별 행동 패턴**이 충분히 반영되지 않는다.

---

# 11. 다음 Feature Engineering 계획

다음 단계에서는 계좌번호를 Feature 자체로 넣는 대신 사용자별 과거 거래를 집계하는 Key로 사용한다.

## ① `amount_zscore`

```text
현재 거래금액이
사용자의 평소 거래금액에서
얼마나 벗어났는가
```

개념:

```text
현재 거래금액 - 사용자 평균 거래금액
────────────────────────────
사용자 거래금액 표준편차
```

---

## ② `recipient_seen_before`

현재 수취인에게 과거 송금한 적이 있는지 판단한다.

```text
과거 송금 존재
→ 1

처음 보는 수취인
→ 0
```

피싱이나 계좌 탈취 상황에서 처음 등장한 수취인 여부가 유용한 Feature가 될 수 있다.

---

## ③ `recipient_transaction_count`

사용자와 해당 수취인 간 과거 거래 횟수.

```text
A → B

과거 50회 거래

vs

A → C

과거 0회 거래
```

를 구분한다.

---

## ④ `user_hour_frequency`

사용자가 현재 시간대에 평소 얼마나 자주 거래했는지 계산한다.

예:

```text
사용자 A

09시 거래 30%
12시 거래 35%
18시 거래 20%
03시 거래 0.2%
```

03시 거래가 발생하면 사용자 개인 기준으로 이례적인 거래임을 모델이 판단할 수 있다.

---

# 12. Baseline 이후 모델 고도화 전략

향후 개발 순서는 다음과 같이 진행한다.

```text
Baseline
│
│ 거래 자체 Feature
│
├─ log_amount
├─ hour
├─ weekday
├─ same_bank
└─ 거래 유형
│
▼
Feature Engineering v1
│
│ 사용자 행동 Feature
│
├─ amount_zscore
├─ recipient_seen_before
├─ recipient_transaction_count
└─ user_hour_frequency
│
▼
Isolation Forest 재학습
│
▼
Validation 평가
│
▼
Baseline vs v1 비교
```

단순히 Feature를 많이 추가하는 것이 아니라:

```text
Feature 추가
    ↓
재학습
    ↓
Precision / Recall / F1 비교
    ↓
실제 개선 여부 판단
```

방식으로 실험한다.

---

# 13. 개발 과정에서 중요하게 고려한 부분

이번 Baseline 구현에서는 다음 원칙을 중점적으로 적용하였다.

### 1. 데이터 Leakage 방지

Label 및 이상거래 발생 이후 생성되는 정보를 학습 Feature에서 제거하였다.

### 2. Train / Validation 전처리 일관성

Train에서 학습된 Preprocessor를 Validation과 Inference에서 그대로 사용한다.

### 3. 대용량 데이터 처리

수백만 건의 금융 데이터를 한 번에 메모리에 올리지 않고 Chunk 단위로 처리한다.

### 4. 불균형 데이터 평가

Accuracy보다 Precision, Recall, F1, PR-AUC를 중심으로 평가한다.

### 5. Threshold 분리

Isolation Forest의 기본 판정보다 Validation 데이터를 기반으로 서비스용 Threshold를 별도로 관리한다.

### 6. 모델 재현성

`random_state`와 학습 환경 정보를 Model Bundle에 저장한다.

### 7. 학습과 추론 코드 재사용

Feature Engineering 로직을 별도 모듈로 분리하여 Train과 실제 추론 사이의 전처리 차이를 방지하였다.

### 8. 서비스 연동 고려

최종 결과를 Python 내부 값으로 끝내지 않고 Backend가 바로 사용할 수 있는 JSON 형태로 정의하였다.

---

# 14. 현재 개발 완료 범위

```text
Feature Engineering 설계

✅ 거래 금액 Feature
✅ 거래 시간 Feature
✅ 거래 일자 Feature
✅ 금융회사 Feature
✅ 거래 방식 Feature
✅ ID / Leakage 분리
✅ One-Hot Encoding
✅ 대용량 Sparse 처리
```

```text
Baseline 학습 파이프라인

✅ Train 데이터 Loader
✅ Validation 데이터 Loader
✅ Chunk Loading
✅ 정상거래 Sampling
✅ Isolation Forest 학습
✅ Model 저장
✅ Preprocessor 저장
✅ Validation 평가
✅ Threshold 결정
✅ 평가 Metric 저장
✅ 단일 거래 Inference
✅ Risk Score 생성
```

---

# 15. 현재 최종 파이프라인

```text
AIHub Train Dataset
        ↓
data_loader.py
        ↓
정상거래 Sampling
        ↓
feature_engineering.py
        ↓
train_iforest.py
        ↓
isolation_forest.joblib
        ↓

AIHub Validation Dataset
        ↓
evaluate.py
        ↓
threshold.json
electronic_metrics.json
        ↓

실제 거래
        ↓
inference.py
        ↓
Feature Engineering
        ↓
Isolation Forest
        ↓
Anomaly Score
        ↓
Threshold 비교
        ↓
Fraud Detection
        ↓
Risk Score
        ↓
Backend 전달
```

현재 단계에서는 **전자금융공동망 데이터 기준 Isolation Forest Baseline 학습 → 평가 → 실제 추론까지 전체 파이프라인 구현을 완료**하였다. 다음 단계에서는 사용자별 과거 거래 이력을 반영하는 행동 기반 Feature를 추가해 Baseline 대비 탐지 성능을 비교·고도화할 예정이다.
