from pathlib import Path

# ============================================================
# 1. 프로젝트 경로
# ============================================================

# 현재 파일:
# MOVI/src/fraud_detection/config.py
#
# parents[0] = fraud_detection
# parents[1] = src
# parents[2] = MOVI
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# 2. 데이터 경로
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"

TRAIN_DIR = DATA_DIR / "Train"
VALIDATION_DIR = DATA_DIR / "Validation"


# 전자금융공동망
ELECTRONIC_TRAIN_DIR = TRAIN_DIR / "TL_전자금융공동망"
ELECTRONIC_VALIDATION_DIR = VALIDATION_DIR / "VL_전자금융공동망"


# 카드거래
CARD_TRAIN_DIR = TRAIN_DIR / "TL_카드거래"
CARD_VALIDATION_DIR = VALIDATION_DIR / "VL_카드거래"


# ============================================================
# 3. 모델 저장 경로
# ============================================================

MODEL_DIR = PROJECT_ROOT / "models"

ELECTRONIC_MODEL_DIR = MODEL_DIR / "electronic"
CARD_MODEL_DIR = MODEL_DIR / "card"


ELECTRONIC_MODEL_PATH = (
    ELECTRONIC_MODEL_DIR / "isolation_forest.joblib"
)

CARD_MODEL_PATH = (
    CARD_MODEL_DIR / "isolation_forest.joblib"
)


# ============================================================
# 4. 평가 결과 저장 경로
# ============================================================

REPORT_DIR = PROJECT_ROOT / "reports"

ELECTRONIC_REPORT_PATH = (
    REPORT_DIR / "electronic_metrics.json"
)

CARD_REPORT_PATH = (
    REPORT_DIR / "card_metrics.json"
)


# ============================================================
# 5. Threshold 저장 경로
# ============================================================

ELECTRONIC_THRESHOLD_PATH = (
    ELECTRONIC_MODEL_DIR / "threshold.json"
)

CARD_THRESHOLD_PATH = (
    CARD_MODEL_DIR / "threshold.json"
)


# ============================================================
# 6. 공통 데이터 설정
# ============================================================

# 실제 AIHub CSV 확인 결과 UTF-8-SIG
CSV_ENCODING = "utf-8-sig"

TARGET_COLUMN = "이상거래여부"

# 모델 학습에 절대 사용하면 안 되는 컬럼
# Target Leakage 방지
LEAKAGE_COLUMNS = [
    "이상거래여부",
    "이상거래유형",
    "이상거래설명",
]


# ============================================================
# 7. 전자금융공동망 데이터 설정
# ============================================================

ELECTRONIC_CONFIG = {

    "name": "electronic",

    # --------------------------------------------------------
    # 데이터 위치
    # --------------------------------------------------------

    "train_dir": ELECTRONIC_TRAIN_DIR,
    "validation_dir": ELECTRONIC_VALIDATION_DIR,

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    "target": TARGET_COLUMN,

    # --------------------------------------------------------
    # Leakage
    # --------------------------------------------------------

    "leakage_columns": LEAKAGE_COLUMNS,

    # --------------------------------------------------------
    # ID 컬럼
    #
    # 숫자로 보이지만 실제로는 식별자이므로
    # 연속형 숫자 Feature로 사용하지 않음
    # --------------------------------------------------------

    "id_columns": [
        "출금계좌일련번호",
        "입금계좌일련번호",
    ],

    # --------------------------------------------------------
    # 기본 숫자형 컬럼
    # --------------------------------------------------------

    "numeric_columns": [
        "거래금액",
        "거래시간대",
        "거래일자",
    ],

    # --------------------------------------------------------
    # 범주형 컬럼
    # --------------------------------------------------------

    "categorical_columns": [
        "출금금융회사일련번호",
        "입금금융회사일련번호",
        "자금구분",
        "매체구분",
    ],

    # --------------------------------------------------------
    # 모델 저장
    # --------------------------------------------------------

    "model_path": ELECTRONIC_MODEL_PATH,
    "threshold_path": ELECTRONIC_THRESHOLD_PATH,
    "report_path": ELECTRONIC_REPORT_PATH,
}


# ============================================================
# 8. 카드거래 데이터 설정
# ============================================================

CARD_CONFIG = {

    "name": "card",

    # --------------------------------------------------------
    # 데이터 위치
    # --------------------------------------------------------

    "train_dir": CARD_TRAIN_DIR,
    "validation_dir": CARD_VALIDATION_DIR,

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    "target": TARGET_COLUMN,

    # --------------------------------------------------------
    # Leakage
    # --------------------------------------------------------

    "leakage_columns": LEAKAGE_COLUMNS,

    # --------------------------------------------------------
    # ID 컬럼
    # --------------------------------------------------------

    "id_columns": [
        "카드KEY",
        "가맹점KEY",
    ],

    # --------------------------------------------------------
    # 주요 숫자형 컬럼
    # --------------------------------------------------------

    "numeric_columns": [
        "통합승인금액",
        "승인시간대",
        "카드이용한도금액",
        "연령",
        "경과일수_최종이용일자",
        "전월_매출건수",
        "전월_매출금액",
        "할부가능개월수",
    ],

    # --------------------------------------------------------
    # 범주형 컬럼
    # --------------------------------------------------------

    "categorical_columns": [
        "개인법인구분코드_회원",
        "국내해외여부",
        "가맹점광역시도코드",
        "남녀구분코드",
        "승인거래코드",
        "승인발생경로코드",
        "가맹점승인업종코드",
        "가맹점누적매출금액_구간화",
        "개인법인구분코드_가맹점",
        "가맹점여부_신규",
        "인터넷판매여부",
        "가맹점상태코드",
        "가맹점형태구분코드",
        "로고구분코드",
        "일시불할부구분코드",
        "카드구분코드",
    ],

    # --------------------------------------------------------
    # 모델 저장
    # --------------------------------------------------------

    "model_path": CARD_MODEL_PATH,
    "threshold_path": CARD_THRESHOLD_PATH,
    "report_path": CARD_REPORT_PATH,
}


# ============================================================
# 9. 학습 설정
# ============================================================

RANDOM_STATE = 42

# Baseline에서는 전체 약 400만 건을 바로 학습하지 않고
# 정상거래 중 일부를 샘플링
TRAIN_SAMPLE_SIZE = 300_000


# ============================================================
# 10. Isolation Forest 설정
# ============================================================

ISOLATION_FOREST_PARAMS = {

    # 생성할 Isolation Tree 개수
    "n_estimators": 200,

    # 각 Tree가 학습할 최대 Sample
    "max_samples": 4096,

    # Threshold는 Validation에서 별도로 결정할 예정
    "contamination": "auto",

    # CPU 병렬 처리
    "n_jobs": -1,

    # 재현성
    "random_state": RANDOM_STATE,
}


# ============================================================
# 11. 출력 디렉토리 생성
# ============================================================

def create_output_directories():
    """
    모델과 평가 결과를 저장할 디렉토리가 없으면 생성한다.
    """

    directories = [
        MODEL_DIR,
        ELECTRONIC_MODEL_DIR,
        CARD_MODEL_DIR,
        REPORT_DIR,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

'''
if __name__ == "__main__":

    create_output_directories()

    print("PROJECT_ROOT")
    print(PROJECT_ROOT)

    print("\n전자금융 Train")
    print(ELECTRONIC_TRAIN_DIR)

    print("\n카드 Train")
    print(CARD_TRAIN_DIR)

    print("\n전자금융 모델 저장")
    print(ELECTRONIC_MODEL_PATH)
'''