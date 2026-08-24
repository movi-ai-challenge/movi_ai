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
    "출금계좌일련번호",
    "입금계좌일련번호",

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

    "amount_ratio",
    "amount_zscore",

    "hour_sin",
    "hour_cos",
    "is_night",

    "weekday",
    "is_weekend",

    "same_bank",

    "new_recipient",
    "unusual_medium",

    "historical_transaction_count",
    "same_day_transaction_count",
    "same_time_bucket_count",
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
    """

    if is_numeric_dtype(series):

        numeric = pd.to_numeric(
            series,
            errors="coerce",
        )

        valid_values = numeric.dropna()

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
    20240321 형태의 날짜를 datetime으로 변환한다.
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
# 11. 전자금융 Historical Feature
# ============================================================

def create_electronic_historical_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    과거 거래 이력을 기반으로 Historical Feature 생성.

    생성 Feature
    ----------------------------------
    amount_ratio
    amount_zscore
    new_recipient
    unusual_medium
    historical_transaction_count
    same_day_transaction_count
    same_time_bucket_count
    """

    history = pd.DataFrame(
        index=df.index
    )

    amount = (
        to_numeric(df["거래금액"])
        .clip(lower=0)
    )

    sender = normalize_categorical(
        df["출금계좌일련번호"]
    )

    receiver = normalize_categorical(
        df["입금계좌일련번호"]
    )

    medium = normalize_categorical(
        df["매체구분"]
    )


    # ========================================================
    # 사용자별 과거 평균 거래금액
    # ========================================================

    historical_mean = (
        amount
        .groupby(sender)
        .transform(
            lambda x:
                x.expanding()
                .mean()
                .shift(1)
        )
    )


    # ========================================================
    # 사용자별 과거 표준편차
    # ========================================================

    historical_std = (
        amount
        .groupby(sender)
        .transform(
            lambda x:
                x.expanding()
                .std()
                .shift(1)
        )
    )


    # ========================================================
    # amount_ratio
    # ========================================================

    history["amount_ratio"] = (
        amount
        /
        historical_mean.replace(
            0,
            np.nan,
        )
    )


    # ========================================================
    # amount_zscore
    # ========================================================

    history["amount_zscore"] = (
        amount - historical_mean
    ) / historical_std.replace(
        0,
        np.nan,
    )


    # ========================================================
    # 신규 수취인
    # ========================================================

    pair_count = (
        pd.DataFrame(
            {
                "sender": sender,
                "receiver": receiver,
            },
            index=df.index,
        )
        .groupby(
            [
                "sender",
                "receiver",
            ]
        )
        .cumcount()
    )

    history["new_recipient"] = (
        pair_count == 0
    ).astype("int8")


    # ========================================================
    # 평소 사용하지 않던 매체
    # ========================================================

    sender_medium_count = (
        pd.DataFrame(
            {
                "sender": sender,
                "medium": medium,
            },
            index=df.index,
        )
        .groupby(
            [
                "sender",
                "medium",
            ]
        )
        .cumcount()
    )

    sender_total_count = (
        sender
        .groupby(sender)
        .cumcount()
    )

    history["unusual_medium"] = (
        (sender_total_count > 0)
        &
        (sender_medium_count == 0)
    ).astype("int8")


    # ========================================================
    # 전체 과거 거래 횟수
    # ========================================================

    history[
        "historical_transaction_count"
    ] = (
        sender
        .groupby(sender)
        .cumcount()
        .astype("int32")
    )


    # ========================================================
    # 같은 날짜 내 이전 거래 횟수
    # ========================================================

    transaction_date = (
        pd.to_numeric(
            df["거래일자"],
            errors="coerce",
        )
    )

    same_day_count = (
        pd.DataFrame(
            {
                "sender": sender,
                "date": transaction_date,
            },
            index=df.index,
        )
        .groupby(
            [
                "sender",
                "date",
            ]
        )
        .cumcount()
    )

    history[
        "same_day_transaction_count"
    ] = (
        same_day_count
        .astype("int32")
    )


    # ========================================================
    # 같은 날짜 + 같은 시간대 이전 거래 횟수
    # ========================================================

    transaction_hour = (
        pd.to_numeric(
            df["거래시간대"],
            errors="coerce",
        )
    )

    same_time_bucket_count = (
        pd.DataFrame(
            {
                "sender": sender,
                "date": transaction_date,
                "hour": transaction_hour,
            },
            index=df.index,
        )
        .groupby(
            [
                "sender",
                "date",
                "hour",
            ]
        )
        .cumcount()
    )

    history[
        "same_time_bucket_count"
    ] = (
        same_time_bucket_count
        .astype("int32")
    )


    # ========================================================
    # 무한대 방지
    # ========================================================

    history.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    return history


# ============================================================
# 12. 전자금융 Feature Engineering
# ============================================================

def engineer_electronic_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    validate_required_columns(
        df,
        ELECTRONIC_REQUIRED_COLUMNS,
        "electronic",
    )

    features = pd.DataFrame(
        index=df.index
    )


    # 거래 금액
    amount = (
        to_numeric(df["거래금액"])
        .clip(lower=0)
    )

    features["log_amount"] = np.log1p(
        amount
    )


    # 거래 시간
    (
        features["hour_sin"],
        features["hour_cos"],
    ) = create_hour_features(
        df["거래시간대"]
    )

    hour = to_numeric(
        df["거래시간대"]
    )

    features["is_night"] = (
        hour.isin(
            [0, 3]
        )
    ).astype("int8")


    # 거래 일자
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


    # 동일 금융회사 여부
    sender_bank = normalize_categorical(
        df["출금금융회사일련번호"]
    )

    receiver_bank = normalize_categorical(
        df["입금금융회사일련번호"]
    )

    features["same_bank"] = (
        sender_bank == receiver_bank
    ).astype("int8")


    # 범주형
    features[
        "출금금융회사일련번호"
    ] = sender_bank

    features[
        "입금금융회사일련번호"
    ] = receiver_bank

    features["자금구분"] = (
        normalize_categorical(
            df["자금구분"]
        )
    )

    features["매체구분"] = (
        normalize_categorical(
            df["매체구분"]
        )
    )


    # Historical Feature
    historical_features = (
        create_electronic_historical_features(
            df
        )
    )

    for column in historical_features.columns:
        features[column] = (
            historical_features[column]
        )


    features.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    return features


# ============================================================
# 13. 카드거래 Feature Engineering
# ============================================================

def engineer_card_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    validate_required_columns(
        df,
        CARD_REQUIRED_COLUMNS,
        "card",
    )

    features = pd.DataFrame(
        index=df.index
    )


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

    features["log_card_limit"] = (
        np.log1p(
            card_limit
        )
    )

    features["amount_limit_ratio"] = (
        amount
        /
        card_limit.replace(
            0,
            np.nan,
        )
    )


    (
        features["hour_sin"],
        features["hour_cos"],
    ) = create_hour_features(
        df["승인시간대"]
    )


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


    features["age"] = to_numeric(
        df["연령"]
    )


    features[
        "days_since_last_use"
    ] = to_numeric(
        df["경과일수_최종이용일자"]
    )


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


    features[
        "installment_available_months"
    ] = to_numeric(
        df["할부가능개월수"]
    )


    for column in CARD_CATEGORICAL_FEATURES:
        features[column] = (
            normalize_categorical(
                df[column]
            )
        )


    features.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    return features


# ============================================================
# 14. 공통 Feature Engineering 함수
# ============================================================

def engineer_features(
    df: pd.DataFrame,
    dataset_type: str,
) -> pd.DataFrame:

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
# 15. Target 추출
# ============================================================

def extract_target(
    df: pd.DataFrame,
) -> pd.Series:

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

    target = target.astype(
        "int8"
    )

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
# 16. Preprocessor 생성
# ============================================================

def build_preprocessor(
    dataset_type: str,
) -> ColumnTransformer:

    config = get_feature_config(
        dataset_type
    )

    numeric_features = config[
        "numeric_features"
    ]

    categorical_features = config[
        "categorical_features"
    ]


    numeric_transformer = SimpleImputer(
        strategy="median"
    )


    categorical_transformer = (
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
            dtype=np.float32,
        )
    )


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

        remainder="drop",
        sparse_threshold=1.0,
    )

    return preprocessor


# ============================================================
# 17. Train용 Fit + Transform
# ============================================================

def fit_transform_features(
    df: pd.DataFrame,
    dataset_type: str,
):

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

    X = X.astype(
        np.float32
    )

    return (
        X,
        preprocessor,
    )


# ============================================================
# 18. Validation / Inference용 Transform
# ============================================================

def transform_features(
    df: pd.DataFrame,
    dataset_type: str,
    preprocessor,
):

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
# 19. 최종 Feature 이름
# ============================================================

def get_transformed_feature_names(
    preprocessor,
) -> list[str]:

    return (
        preprocessor
        .get_feature_names_out()
        .tolist()
    )


# ============================================================
# 20. Feature 정보 출력
# ============================================================

def print_feature_summary(
    raw_df: pd.DataFrame,
    engineered_df: pd.DataFrame,
    dataset_type: str,
) -> None:

    print()
    print("=" * 70)
    print(
        f"FEATURE SUMMARY : "
        f"{dataset_type}"
    )
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
# 21. 테스트
# ============================================================

if __name__ == "__main__":

    try:
        from .data_loader import load_dataset
    except ImportError:
        from data_loader import load_dataset


    print()
    print(
        "전자금융 Feature Engineering 테스트"
    )
    print()


    electronic_df = load_dataset(
        dataset_type="electronic",
        split="train",
        max_files=1,
        nrows_per_file=50_000,
    )


    electronic_features = (
        engineer_electronic_features(
            electronic_df
        )
    )


    # ========================================================
    # Historical Feature 확인
    # ========================================================

    print()
    print("[Historical Features]")

    columns = [
        "log_amount",
        "amount_ratio",
        "amount_zscore",
        "is_night",
        "new_recipient",
        "unusual_medium",
        "historical_transaction_count",
        "same_day_transaction_count",
        "same_time_bucket_count",
    ]

    print(
        electronic_features[
            columns
        ].head(20)
    )


    # ========================================================
    # Feature Summary
    # ========================================================

    print_feature_summary(
        electronic_df,
        electronic_features,
        "electronic",
    )


    # ========================================================
    # Preprocessor 테스트
    # ========================================================

    (
        X_electronic,
        electronic_preprocessor,
    ) = fit_transform_features(
        electronic_df,
        "electronic",
    )


    print()
    print("[전자금융 최종 Matrix]")

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
    # 실제 Feature 이름
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


    # ========================================================
    # Historical Feature 통계
    # ========================================================

    print()
    print("[Historical Feature 통계]")

    print(
        electronic_features[
            [
                "amount_ratio",
                "amount_zscore",
                "new_recipient",
                "unusual_medium",
                "historical_transaction_count",
                "same_day_transaction_count",
                "same_time_bucket_count",
            ]
        ].describe()
    )


    # ========================================================
    # 신규 수취인
    # ========================================================

    print()
    print("[신규 수취인 분포]")

    print(
        electronic_features[
            "new_recipient"
        ]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )


    # ========================================================
    # 과거 누적 거래
    # ========================================================

    print()
    print("[과거 누적 거래 횟수 분포]")

    print(
        electronic_features[
            "historical_transaction_count"
        ]
        .value_counts()
        .sort_index()
        .head(10)
    )


    # ========================================================
    # 같은 날짜 거래
    # ========================================================

    print()
    print("[같은 날짜 거래 횟수 분포]")

    print(
        electronic_features[
            "same_day_transaction_count"
        ]
        .value_counts()
        .sort_index()
        .head(10)
    )


    # ========================================================
    # 같은 시간대 거래
    # ========================================================

    print()
    print("[같은 시간대 거래 횟수 분포]")

    print(
        electronic_features[
            "same_time_bucket_count"
        ]
        .value_counts()
        .sort_index()
        .head(10)
    )


    # ========================================================
    # 최대값
    # ========================================================

    print()
    print("[최대 거래 횟수]")

    print(
        "Max Historical Count:",
        electronic_features[
            "historical_transaction_count"
        ].max()
    )

    print(
        "Max Same Day Count:",
        electronic_features[
            "same_day_transaction_count"
        ].max()
    )

    print(
        "Max Same Time Bucket Count:",
        electronic_features[
            "same_time_bucket_count"
        ].max()
    )


    # ========================================================
    # 거래시간대
    # ========================================================

    print()
    print("[거래시간대 분포]")

    print(
        electronic_df[
            "거래시간대"
        ]
        .value_counts()
        .sort_index()
    )


    # ========================================================
    # 반복거래 상위값
    # ========================================================

    print()
    print(
        "[Same Day Transaction Count 상위 값]"
    )

    print(
        electronic_features[
            "same_day_transaction_count"
        ]
        .value_counts()
        .sort_index()
        .tail(10)
    )


    print()
    print(
        "[Same Time Bucket Count 상위 값]"
    )

    print(
        electronic_features[
            "same_time_bucket_count"
        ]
        .value_counts()
        .sort_index()
        .tail(10)
    )


    # ========================================================
    # 비정상 매체
    # ========================================================

    print()
    print("[비정상 매체 사용 분포]")

    print(
        electronic_features[
            "unusual_medium"
        ]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )


    # ========================================================
    # 신규 수취인 + 비정상 매체
    # ========================================================

    print()
    print(
        "[신규 수취인 + 비정상 매체 조합]"
    )

    print(
        pd.crosstab(
            electronic_features[
                "new_recipient"
            ],
            electronic_features[
                "unusual_medium"
            ],
            rownames=[
                "new_recipient"
            ],
            colnames=[
                "unusual_medium"
            ],
        )
    )