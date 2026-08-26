from __future__ import annotations

import pandas as pd

from .schemas import (
    TransactionData,
    FraudDetectionRequest,
)

from .inference_feature_builder import (
    build_inference_features,
)


# ============================================================
# 테스트용 Fake Feature Engineering
# ============================================================

def fake_feature_engineer(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    # 테스트용 단순 Feature
    result["log_amount_test"] = (
        result["거래금액"]
    )

    result[
        "history_count_test"
    ] = range(
        len(result)
    )

    return result


# ============================================================
# Test
# ============================================================

def test_builder():

    request = FraudDetectionRequest(

        current_transaction=TransactionData(

            transaction_id="tx-current",

            sender_account="sender-001",

            receiver_account="receiver-new",

            sender_bank="KB",

            receiver_bank="SH",

            transaction_type="transfer",

            amount=10_000_000,

            transaction_datetime=(
                "2026-08-26T00:30:00"
            ),

            medium="MOBILE",
        ),

        history=[

            TransactionData(

                transaction_id="tx-history-1",

                sender_account="sender-001",

                receiver_account="receiver-001",

                sender_bank="KB",

                receiver_bank="KB",

                transaction_type="transfer",

                amount=50_000,

                transaction_datetime=(
                    "2026-08-25T14:00:00"
                ),

                medium="MOBILE",
            ),

            TransactionData(

                transaction_id="tx-history-2",

                sender_account="sender-001",

                receiver_account="receiver-002",

                sender_bank="KB",

                receiver_bank="KB",

                transaction_type="transfer",

                amount=100_000,

                transaction_datetime=(
                    "2026-08-25T18:00:00"
                ),

                medium="MOBILE",
            ),
        ],
    )


    feature_df = build_inference_features(

        request=request,

        feature_engineer=(
            fake_feature_engineer
        ),
    )


    print()
    print("=" * 80)
    print("CURRENT TRANSACTION FEATURES")
    print("=" * 80)

    print(
        feature_df.to_string(
            index=False
        )
    )


    # 현재 거래 한 행만 존재해야 함
    assert len(feature_df) == 1


    # 현재 거래 금액 확인
    assert (
        feature_df.iloc[0][
            "거래금액"
        ]
        ==
        10_000_000
    )


    # history 2건 뒤 현재 거래이므로 2
    assert (
        feature_df.iloc[0][
            "history_count_test"
        ]
        ==
        2
    )


    print()
    print(
        "[PASS] Inference Feature Builder"
    )


if __name__ == "__main__":

    test_builder()