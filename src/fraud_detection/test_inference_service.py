from __future__ import annotations

import numpy as np
import pandas as pd

from .schemas import (
    TransactionData,
    FraudDetectionRequest,
)

from .inference_service import (
    detect_fraud,
)


# ============================================================
# Fake Feature Engineering
# ============================================================

def fake_feature_engineer(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    result[
        "log_amount"
    ] = np.log1p(
        result["거래금액"]
    )

    return result


# ============================================================
# Fake Preprocessor
# ============================================================

class FakePreprocessor:

    def transform(
        self,
        df,
    ):

        return df[
            [
                "log_amount",
            ]
        ].to_numpy()


# ============================================================
# Fake Isolation Forest
# ============================================================

class FakeIsolationForest:

    def score_samples(
        self,
        X,
    ):

        # 테스트에서는 위험 거래라고 가정
        return np.array(
            [
                -0.482914,
            ]
        )


# ============================================================
# Test
# ============================================================

def test_inference_service():

    # --------------------------------------------------------
    # Request
    # --------------------------------------------------------

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

                transaction_id="history-1",

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

                transaction_id="history-2",

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


    # --------------------------------------------------------
    # Fake Bundle
    # --------------------------------------------------------

    bundle = {

        "model": (
            FakeIsolationForest()
        ),

        "preprocessor": (
            FakePreprocessor()
        ),

        "feature_names": [
            "log_amount",
        ],

        "threshold": (
            0.446117
        ),
    }


    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    result = detect_fraud(

        request=request,

        bundle=bundle,

        feature_engineer=(
            fake_feature_engineer
        ),
    )


    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FDS INFERENCE RESULT")
    print("=" * 70)

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )


    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    assert (
        result["transaction_id"]
        ==
        "tx-current"
    )

    assert (
        abs(
            result["anomaly_score"]
            -
            0.482914
        )
        <
        1e-6
    )

    assert (
        result["is_anomaly"]
        is True
    )


    print()
    print(
        "[PASS] FDS Inference Service"
    )


if __name__ == "__main__":

    test_inference_service()