from __future__ import annotations

from typing import Callable

import pandas as pd

from .schemas import FraudDetectionRequest
from .transaction_mapper import (
    request_to_dataframe,
    find_current_transaction_index,
)


# ============================================================
# Type
# ============================================================

FeatureEngineer = Callable[
    [pd.DataFrame],
    pd.DataFrame,
]


# ============================================================
# Inference Feature Builder
# ============================================================

def build_inference_features(
    request: FraudDetectionRequest,
    feature_engineer: FeatureEngineer,
) -> pd.DataFrame:
    """
    Spring에서 전달받은

        현재 거래
        +
        과거 거래 History

    를 기존 Feature Engineering에 연결하여
    현재 거래 1건의 Feature만 반환한다.

    Parameters
    ----------
    request:
        Spring → FDS API Request

    feature_engineer:
        기존 feature_engineering.py의
        Feature 생성 함수.

        입력:
            AIHub 스타일 Raw DataFrame

        출력:
            Feature Engineering이 적용된 DataFrame

    Returns
    -------
    pd.DataFrame

        현재 거래에 해당하는 Feature 1행.
    """

    # ========================================================
    # 1. API Request → AIHub Raw DataFrame
    # ========================================================

    raw_df = request_to_dataframe(
        request
    )


    if raw_df.empty:

        raise ValueError(
            "Inference에 사용할 거래 데이터가 없습니다."
        )


    # ========================================================
    # 2. 현재 Transaction 위치 확인
    # ========================================================

    current_transaction_id = (
        request
        .current_transaction
        .transaction_id
    )


    current_index = (
        find_current_transaction_index(
            raw_df,
            current_transaction_id,
        )
    )


    # ========================================================
    # 3. 현재 거래 식별 정보 보존
    #
    # Feature Engineering 이후에도
    # 현재 거래를 정확히 찾아야 하므로
    # 내부용 식별 컬럼을 하나 사용한다.
    # ========================================================

    raw_df = raw_df.copy()

    raw_df[
        "_is_current_transaction"
    ] = False

    raw_df.loc[
        current_index,
        "_is_current_transaction"
    ] = True


    # ========================================================
    # 4. Feature Engineering
    #
    # 중요:
    # History + Current를 함께 넣어야 한다.
    #
    # current만 따로 Feature Engineering하면
    # amount_ratio
    # amount_zscore
    # new_recipient
    # unusual_medium
    # transaction_count
    # 등이 정상 계산되지 않는다.
    # ========================================================

    feature_df = feature_engineer(
        raw_df.copy()
    )


    if feature_df.empty:

        raise ValueError(
            "Feature Engineering 결과가 비어 있습니다."
        )


    # ========================================================
    # 5. 현재 거래 Feature만 추출
    # ========================================================

    if (
        "_is_current_transaction"
        not in feature_df.columns
    ):

        raise ValueError(
            "Feature Engineering 과정에서 "
            "'_is_current_transaction' 컬럼이 제거되었습니다."
        )


    current_feature_df = feature_df[
        feature_df[
            "_is_current_transaction"
        ]
    ].copy()


    if len(current_feature_df) != 1:

        raise ValueError(
            "현재 거래 Feature를 정확히 "
            f"1건 찾지 못했습니다. "
            f"count={len(current_feature_df)}"
        )


    # ========================================================
    # 6. API / 디버깅 전용 컬럼 제거
    # ========================================================

    drop_columns = [
        "_is_current_transaction",
        "_transaction_id",
        "_transaction_datetime",
    ]


    existing_drop_columns = [
        column
        for column in drop_columns
        if column in current_feature_df.columns
    ]


    current_feature_df = (
        current_feature_df.drop(
            columns=existing_drop_columns
        )
    )


    return current_feature_df.reset_index(
        drop=True
    )