from __future__ import annotations

from typing import Any

import pandas as pd

from .schemas import (
    TransactionData,
    FraudDetectionRequest,
)


# ============================================================
# 시간대 Bucket 변환
# ============================================================

def convert_hour_to_time_bucket(
    hour: int,
) -> int:
    """
    실제 거래 시각(hour)을
    AIHub 거래시간대 Bucket으로 변환한다.

    AIHub 거래시간대:
        0, 3, 6, 9, 12, 15, 18, 21

    Example
    -------
    00:30 -> 0
    02:10 -> 0
    03:00 -> 3
    14:20 -> 12
    23:10 -> 21
    """

    if not 0 <= hour <= 23:
        raise ValueError(
            f"유효하지 않은 hour 값입니다: {hour}"
        )

    return (hour // 3) * 3


# ============================================================
# 단일 거래 변환
# ============================================================

def transaction_to_aihub_row(
    transaction: TransactionData,
) -> dict[str, Any]:
    """
    Spring API의 TransactionData를
    기존 FDS Feature Engineering에서 사용하는
    AIHub 스타일 컬럼 구조로 변환한다.
    """

    transaction_datetime = (
        transaction.transaction_datetime
    )

    transaction_date = int(
        transaction_datetime.strftime("%Y%m%d")
    )

    time_bucket = convert_hour_to_time_bucket(
        transaction_datetime.hour
    )


    return {
        # ----------------------------------------------------
        # 계좌
        # ----------------------------------------------------

        "출금계좌일련번호": (
            transaction.sender_account
        ),

        "입금계좌일련번호": (
            transaction.receiver_account
        ),

        # ----------------------------------------------------
        # 금융기관
        # ----------------------------------------------------

        "출금금융회사일련번호": (
            transaction.sender_bank
        ),

        "입금금융회사일련번호": (
            transaction.receiver_bank
        ),

        # ----------------------------------------------------
        # 자금구분
        #
        # 현재 API의 transaction_type을
        # 기존 자금구분 컬럼에 연결
        # ----------------------------------------------------

        "자금구분": (
            transaction.transaction_type
        ),

        # ----------------------------------------------------
        # 금액
        # ----------------------------------------------------

        "거래금액": (
            float(transaction.amount)
        ),

        # ----------------------------------------------------
        # 시간
        # ----------------------------------------------------

        "거래시간대": (
            time_bucket
        ),

        # ----------------------------------------------------
        # 거래 매체
        # ----------------------------------------------------

        "매체구분": (
            transaction.medium
        ),

        # ----------------------------------------------------
        # 거래일자
        # ----------------------------------------------------

        "거래일자": (
            transaction_date
        ),

        # ----------------------------------------------------
        # API용 추가 컬럼
        #
        # 모델 Feature에는 사용하지 않아도 되지만
        # 추적/디버깅 용도로 보존
        # ----------------------------------------------------

        "_transaction_id": (
            transaction.transaction_id
        ),

        "_transaction_datetime": (
            transaction_datetime
        ),
    }


# ============================================================
# API Request → DataFrame
# ============================================================

def request_to_dataframe(
    request: FraudDetectionRequest,
) -> pd.DataFrame:
    """
    FraudDetectionRequest를 Feature Engineering용
    DataFrame으로 변환한다.

    순서:
        history
        ↓
        current_transaction

    반드시 현재 거래가 가장 마지막 Row가 되도록 한다.
    """

    rows: list[dict[str, Any]] = []


    # ========================================================
    # 1. History
    # ========================================================

    for transaction in request.history:

        rows.append(
            transaction_to_aihub_row(
                transaction
            )
        )


    # ========================================================
    # 2. Current Transaction
    # ========================================================

    rows.append(
        transaction_to_aihub_row(
            request.current_transaction
        )
    )


    dataframe = pd.DataFrame(
        rows
    )


    # ========================================================
    # 시간 순 정렬
    # ========================================================

    dataframe = dataframe.sort_values(
        by=[
            "_transaction_datetime",
        ],
        ascending=True,
        kind="stable",
    ).reset_index(
        drop=True
    )


    return dataframe


# ============================================================
# 현재 거래 Index
# ============================================================

def find_current_transaction_index(
    dataframe: pd.DataFrame,
    transaction_id: str | None,
) -> int:
    """
    변환된 DataFrame에서 현재 거래 위치를 찾는다.

    transaction_id가 존재하면 ID 기준으로 찾고,
    없으면 마지막 Row를 현재 거래로 간주한다.
    """

    if dataframe.empty:

        raise ValueError(
            "거래 DataFrame이 비어 있습니다."
        )


    if transaction_id:

        matched = dataframe.index[
            dataframe["_transaction_id"]
            == transaction_id
        ].tolist()


        if matched:

            return matched[-1]


    # transaction_id가 없거나 찾지 못한 경우
    # 시간순 마지막 거래를 현재 거래로 판단
    return dataframe.index[-1]