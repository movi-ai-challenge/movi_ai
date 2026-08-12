from typing import Sequence

import numpy as np
import pandas as pd

from pandas.api.types import is_numeric_dtype

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# Import
# ============================================================

try:
    from .config import TARGET_COLUMN
except ImportError:
    from config import TARGET_COLUMN


# ============================================================
# 1. 공통 설정
# ============================================================

MISSING_CATEGORY = "__MISSING__"


# ============================================================
# 2. 전자금융공동망 Feature 정의
# ============================================================

ELECTRONIC_REQUIRED_COLUMNS = [
    "출금금융회사일련번호",
    "입금금융회사일련번호",
    "자금구분",
    "거래금액",
    "거래시간대",
    "매체구분",
    "거래일자",
]


ELECTRONIC_NUMERIC_FEATURES = [
    "log_amount",
    "hour_sin",
    "hour_cos",
    "weekday",
    "is_weekend",
    "same_bank",
]


ELECTRONIC_CATEGORICAL_FEATURES = [
    "출금금융회사일련번호",
    "입금금융회사일련번호",
    "자금구분",
    "매체구분",
]


# ============================================================
# 3. 카드거래 Feature 정의
# ============================================================

CARD_REQUIRED_COLUMNS = [
    "승인일자",
    "승인시간대",
    "통합승인금액",
    "카드이용한도금액",
    "연령",
    "경과일수_최종이용일자",
    "전월_매출건수",
    "전월_매출금액",
    "할부가능개월수",

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
]


CARD_NUMERIC_FEATURES = [
    "log_amount",
    "log_card_limit",
    "amount_limit_ratio",

    "hour_sin",
    "hour_cos",

    "weekday",
    "is_weekend",

    "age",
    "days_since_last_use",

    "log_prev_month_sales_count",
    "log_prev_month_sales_amount",

    "installment_available_months",
]


CARD_CATEGORICAL_FEATURES = [
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
]


# ============================================================
# 4. Dataset별 Feature 설정
# ============================================================

FEATURE_CONFIGS = {

    "electronic": {
        "required_columns": ELECTRONIC_REQUIRED_COLUMNS,
        "numeric_features": ELECTRONIC_NUMERIC_FEATURES,
        "categorical_features": ELECTRONIC_CATEGORICAL_FEATURES,
    },

    "card": {
        "required_columns": CARD_REQUIRED_COLUMNS,
        "numeric_features": CARD_NUMERIC_FEATURES,
        "categorical_features": CARD_CATEGORICAL_FEATURES,
    },
}


# ============================================================
# 5. Dataset Type 검증
# ============================================================

def get_feature_config(
    dataset_type: str,
) -> dict:
    """
    dataset_type에 해당하는 Feature 설정을 반환한다.
    """

    if dataset_type not in FEATURE_CONFIGS:

        raise ValueError(
            f"지원하지 않는 dataset_type입니다: {dataset_type}\n"
            f"사용 가능한 값: {list(FEATURE_CONFIGS.keys())}"
        )

    return FEATURE_CONFIGS[dataset_type]


# ============================================================
# 6. 필수 컬럼 검증
# ============================================================

def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Sequence[str],
    dataset_type: str,
) -> None:
    """
    Feature Engineering에 필요한 컬럼이
    DataFrame에 모두 존재하는지 검사한다.
    """

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"\n[{dataset_type}] Feature Engineering에 "
            f"필요한 컬럼이 없습니다.\n"
            f"누락 컬럼: {missing_columns}"
        )


# ============================================================
# 7. 숫자 변환 Helper
# ============================================================

def to_numeric(
    series: pd.Series,
) -> pd.Series:
    """
    숫자로 변환할 수 없는 값은 NaN으로 처리한다.
    """

    return pd.to_numeric(
        series,
        errors="coerce",
    )


# ============================================================
# 8. 범주형 변수 정규화
# ============================================================

def normalize_categorical(
    series: pd.Series,
) -> pd.Series:
    """
    범주형 Feature의 타입을 문자열로 통일한다.

    예:
        155 -> "155"
        2   -> "2"
        "00" -> "00"
        NaN -> "__MISSING__"

    카드 데이터에는
        C, Q, A, _, 00
    같은 문자열 코드도 존재하기 때문에
    문자열 코드는 그대로 유지한다.
    """

    # --------------------------------------------------------
    # 숫자형 Category
    # --------------------------------------------------------

    if is_numeric_dtype(series):

        numeric = pd.to_numeric(
            series,
            errors="coerce",
        )

        valid_values = numeric.dropna()

        # 1.0 / 2.0처럼 실제 의미는 정수 코드인 경우
        if (
            len(valid_values) == 0
            or np.all(
                np.isclose(
                    valid_values.to_numpy() % 1,
                    0,
                )
            )
        ):

            result = (
                numeric
                .round()
                .astype("Int64")
                .astype("string")
            )

        else:

            result = numeric.astype("string")

    # --------------------------------------------------------
    # 문자열 Category
    # --------------------------------------------------------

    else:

        result = (
            series
            .astype("string")
            .str.strip()
        )

    return result.fillna(
        MISSING_CATEGORY
    )


# ============================================================
# 9. YYYYMMDD 날짜 파싱
# ============================================================

def parse_yyyymmdd(
    series: pd.Series,
) -> pd.Series:
    """
    20240321 형태의 정수 날짜를 datetime으로 변환한다.
    """

    date_string = (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .round()
        .astype("Int64")
        .astype("string")
    )

    return pd.to_datetime(
        date_string,
        format="%Y%m%d",
        errors="coerce",
    )


# ============================================================
# 10. 시간 Cyclic Encoding
# ============================================================

def create_hour_features(
    series: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    시간값을 sin/cos 두 Feature로 변환한다.

    예:
        23시와 0시가 서로 가까운 시간이라는
        순환 구조를 모델에 표현하기 위함.
    """

    hour = to_numeric(series)

    hour_sin = np.sin(
        2 * np.pi * hour / 24
    )

    hour_cos = np.cos(
        2 * np.pi * hour / 24
    )

    return (
        hour_sin,
        hour_cos,
    )


# ============================================================
# 11. 전자금융공동망 Feature Engineering
# ============================================================

def engineer_electronic_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    전자금융공동망 원본 데이터를
    Isolation Forest용 Feature로 변환한다.

    원본
    ----------------------------------
    거래금액
    거래시간대
    거래일자
    출금금융회사
    입금금융회사
    자금구분
    매체구분

    ↓

    Feature
    ----------------------------------
    log_amount
    hour_sin
    hour_cos
    weekday
    is_weekend
    same_bank
    + categorical features
    """

    validate_required_columns(
        df,
        ELECTRONIC_REQUIRED_COLUMNS,
        "electronic",
    )

    features = pd.DataFrame(
        index=df.index
    )

    # ========================================================
    # 거래 금액
    # ========================================================

    amount = (
        to_numeric(df["거래금액"])
        .clip(lower=0)
    )

    # 금액 분포의 극단적인 왜도를 줄임
    features["log_amount"] = np.log1p(
        amount
    )


    # ========================================================
    # 거래 시간
    # ========================================================

    (
        features["hour_sin"],
        features["hour_cos"],
    ) = create_hour_features(
        df["거래시간대"]
    )


    # ========================================================
    # 거래 일자
    # ========================================================

    transaction_date = parse_yyyymmdd(
        df["거래일자"]
    )

    features["weekday"] = (
        transaction_date
        .dt
        .weekday
    )

    features["is_weekend"] = (
        transaction_date
        .dt
        .weekday
        .isin([5, 6])
        .astype("int8")
    )


    # ========================================================
    # 동일 금융회사 여부
    # ========================================================

    sender_bank = normalize_categorical(
        df["출금금융회사일련번호"]
    )

    receiver_bank = normalize_categorical(
        df["입금금융회사일련번호"]
    )

    features["same_bank"] = (
        sender_bank == receiver_bank
    ).astype("int8")


    # ========================================================
    # 범주형 Feature
    # ========================================================

    features["출금금융회사일련번호"] = sender_bank

    features["입금금융회사일련번호"] = receiver_bank

    features["자금구분"] = normalize_categorical(
        df["자금구분"]
    )

    features["매체구분"] = normalize_categorical(
        df["매체구분"]
    )


    # ========================================================
    # 무한대 방지
    # ========================================================

    features.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    return features


# ============================================================
# 12. 카드거래 Feature Engineering
# ============================================================

def engineer_card_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    카드거래 데이터를 Isolation Forest용 Feature로 변환한다.
    """

    validate_required_columns(
        df,
        CARD_REQUIRED_COLUMNS,
        "card",
    )

    features = pd.DataFrame(
        index=df.index
    )


    # ========================================================
    # 거래 금액
    # ========================================================

    amount = (
        to_numeric(df["통합승인금액"])
        .clip(lower=0)
    )

    card_limit = (
        to_numeric(df["카드이용한도금액"])
        .clip(lower=0)
    )

    features["log_amount"] = np.log1p(
        amount
    )

    features["log_card_limit"] = np.log1p(
        card_limit
    )


    # 카드 한도 대비 승인금액 비율
    #
    # 한도가 0인 경우 Division by Zero 방지를 위해 NaN 처리
    features["amount_limit_ratio"] = (
        amount
        /
        card_limit.replace(
            0,
            np.nan,
        )
    )


    # ========================================================
    # 승인 시간
    # ========================================================

    (
        features["hour_sin"],
        features["hour_cos"],
    ) = create_hour_features(
        df["승인시간대"]
    )


    # ========================================================
    # 승인 일자
    # ========================================================

    approval_date = parse_yyyymmdd(
        df["승인일자"]
    )

    features["weekday"] = (
        approval_date
        .dt
        .weekday
    )

    features["is_weekend"] = (
        approval_date
        .dt
        .weekday
        .isin([5, 6])
        .astype("int8")
    )


    # ========================================================
    # 회원 특성
    # ========================================================

    features["age"] = to_numeric(
        df["연령"]
    )


    # ========================================================
    # 최근 카드 이용 패턴
    # ========================================================

    features["days_since_last_use"] = (
        to_numeric(
            df["경과일수_최종이용일자"]
        )
    )


    # ========================================================
    # 가맹점 전월 매출
    # ========================================================

    previous_sales_count = (
        to_numeric(
            df["전월_매출건수"]
        )
        .clip(lower=0)
    )

    previous_sales_amount = (
        to_numeric(
            df["전월_매출금액"]
        )
        .clip(lower=0)
    )

    features[
        "log_prev_month_sales_count"
    ] = np.log1p(
        previous_sales_count
    )

    features[
        "log_prev_month_sales_amount"
    ] = np.log1p(
        previous_sales_amount
    )


    # ========================================================
    # 할부 가능 개월 수
    # ========================================================

    features[
        "installment_available_months"
    ] = to_numeric(
        df["할부가능개월수"]
    )


    # ========================================================
    # 범주형 Feature
    # ========================================================

    for column in CARD_CATEGORICAL_FEATURES:

        features[column] = (
            normalize_categorical(
                df[column]
            )
        )


    # ========================================================
    # 무한대 값 처리
    # ========================================================

    features.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    return features


# ============================================================
# 13. 공통 Feature Engineering 함수
# ============================================================

def engineer_features(
    df: pd.DataFrame,
    dataset_type: str,
) -> pd.DataFrame:
    """
    dataset_type에 따라 적절한 Feature Engineering 수행.
    """

    if dataset_type == "electronic":

        return engineer_electronic_features(
            df
        )

    elif dataset_type == "card":

        return engineer_card_features(
            df
        )

    else:

        raise ValueError(
            f"지원하지 않는 dataset_type입니다: "
            f"{dataset_type}"
        )


# ============================================================
# 14. Target 추출
# ============================================================

def extract_target(
    df: pd.DataFrame,
) -> pd.Series:
    """
    이상거래여부 Target을 추출한다.

    Feature Engineering과 Target을 명확하게 분리하여
    Target Leakage를 방지한다.
    """

    if TARGET_COLUMN not in df.columns:

        raise ValueError(
            f"Target 컬럼이 없습니다: "
            f"{TARGET_COLUMN}"
        )

    target = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce",
    )

    if target.isna().any():

        raise ValueError(
            "Target에 NaN 또는 숫자로 변환할 수 없는 "
            "값이 존재합니다."
        )

    target = target.astype("int8")

    unique_values = set(
        target.unique()
    )

    if not unique_values.issubset(
        {0, 1}
    ):

        raise ValueError(
            "이상거래여부는 0 또는 1이어야 합니다.\n"
            f"현재 값: {unique_values}"
        )

    return target


# ============================================================
# 15. Preprocessor 생성
# ============================================================

def build_preprocessor(
    dataset_type: str,
) -> ColumnTransformer:
    """
    숫자형 / 범주형 Feature를
    Isolation Forest 입력 형태로 변환하는 Preprocessor 생성.

    Numeric
        ↓
    NaN → Median

    Categorical
        ↓
    One-Hot Encoding
    """

    config = get_feature_config(
        dataset_type
    )

    numeric_features = config[
        "numeric_features"
    ]

    categorical_features = config[
        "categorical_features"
    ]


    # ========================================================
    # Numeric Pipeline
    # ========================================================

    numeric_transformer = SimpleImputer(
        strategy="median"
    )


    # ========================================================
    # Categorical Pipeline
    # ========================================================

    categorical_transformer = OneHotEncoder(

        # Validation / 실제 서비스에서
        # Train에 없던 범주가 등장해도 에러를 내지 않음
        handle_unknown="ignore",

        # 대용량 데이터이므로 Dense Matrix 대신 Sparse Matrix
        sparse_output=True,

        # 메모리 사용량 감소
        dtype=np.float32,
    )


    # ========================================================
    # Column Transformer
    # ========================================================

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "num",
                numeric_transformer,
                numeric_features,
            ),

            (
                "cat",
                categorical_transformer,
                categorical_features,
            ),
        ],

        # 지정하지 않은 컬럼 제거
        remainder="drop",

        # 가능하면 Sparse Matrix 유지
        sparse_threshold=1.0,
    )

    return preprocessor


# ============================================================
# 16. Train용 Fit + Transform
# ============================================================

def fit_transform_features(
    df: pd.DataFrame,
    dataset_type: str,
):
    """
    Train 데이터용 함수.

    1. Feature Engineering
    2. Preprocessor 생성
    3. fit
    4. transform

    Returns
    -------
    X
        Isolation Forest 입력 Matrix

    preprocessor
        학습된 전처리 객체
    """

    engineered_df = engineer_features(
        df,
        dataset_type,
    )

    preprocessor = build_preprocessor(
        dataset_type
    )

    X = preprocessor.fit_transform(
        engineered_df
    )

    # IsolationForest 효율을 위해 float32
    X = X.astype(
        np.float32
    )

    return (
        X,
        preprocessor,
    )


# ============================================================
# 17. Validation / Inference용 Transform
# ============================================================

def transform_features(
    df: pd.DataFrame,
    dataset_type: str,
    preprocessor,
):
    """
    이미 Train에서 학습된 Preprocessor를 사용한다.

    주의:
        Validation / Inference에서는
        절대로 fit_transform()을 호출하지 않는다.

        반드시 transform()만 사용한다.
    """

    engineered_df = engineer_features(
        df,
        dataset_type,
    )

    X = preprocessor.transform(
        engineered_df
    )

    X = X.astype(
        np.float32
    )

    return X


# ============================================================
# 18. 최종 Feature 이름 확인
# ============================================================

def get_transformed_feature_names(
    preprocessor,
) -> list[str]:
    """
    One-Hot Encoding 후 실제 모델에 입력되는
    Feature 이름 목록을 반환한다.
    """

    return (
        preprocessor
        .get_feature_names_out()
        .tolist()
    )


# ============================================================
# 19. Feature 정보 출력
# ============================================================

def print_feature_summary(
    raw_df: pd.DataFrame,
    engineered_df: pd.DataFrame,
    dataset_type: str,
) -> None:
    """
    Feature Engineering 결과 확인용.
    """

    print()
    print("=" * 70)
    print(f"FEATURE SUMMARY : {dataset_type}")
    print("=" * 70)

    print(
        f"Raw Columns        : "
        f"{len(raw_df.columns)}"
    )

    print(
        f"Engineered Columns : "
        f"{len(engineered_df.columns)}"
    )

    print()

    print("[Engineered Features]")

    for column in engineered_df.columns:

        print(
            f"- {column:<35} "
            f"{engineered_df[column].dtype}"
        )

    print()
    print("[Missing Values]")

    missing = (
        engineered_df
        .isna()
        .sum()
    )

    missing = missing[
        missing > 0
    ]

    if len(missing) == 0:

        print("없음")

    else:

        print(missing)

    print("=" * 70)


# ============================================================
# 20. 테스트
# ============================================================

if __name__ == "__main__":

    try:
        from .data_loader import load_dataset

    except ImportError:
        from data_loader import load_dataset


    # ========================================================
    # 전자금융 테스트
    # ========================================================

    print()
    print("전자금융 Feature Engineering 테스트")
    print()


    electronic_df = load_dataset(
        dataset_type="electronic",
        split="train",

        # 전체 데이터가 아닌 테스트용
        max_files=1,
        nrows_per_file=1_000,
    )


    electronic_features = (
        engineer_electronic_features(
            electronic_df
        )
    )


    print_feature_summary(
        electronic_df,
        electronic_features,
        "electronic",
    )


    X_electronic, electronic_preprocessor = (
        fit_transform_features(
            electronic_df,
            "electronic",
        )
    )


    print()
    print(
        "[전자금융 최종 Matrix]"
    )

    print(
        "Shape:",
        X_electronic.shape,
    )

    print(
        "Type:",
        type(X_electronic),
    )

    print(
        "Dtype:",
        X_electronic.dtype,
    )


    # ========================================================
    # 실제 변환된 Feature 개수
    # ========================================================

    feature_names = (
        get_transformed_feature_names(
            electronic_preprocessor
        )
    )


    print()
    print(
        f"최종 Feature 개수: "
        f"{len(feature_names)}"
    )


    print()
    print(
        "첫 20개 Feature:"
    )

    for name in feature_names[:20]:

        print(
            "-",
            name,
        )