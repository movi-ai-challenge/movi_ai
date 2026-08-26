from __future__ import annotations

import json

from .schemas import (
    TransactionData,
    FraudDetectionRequest,
)

from .fraud_service import (
    FraudDetectionService,
    FraudDetectionServiceError,
)


# ============================================================
# Test Request
# ============================================================

def create_test_request() -> FraudDetectionRequest:

    return FraudDetectionRequest(

        # ====================================================
        # 현재 거래
        #
        # 일부러 위험도가 높을 수 있는 거래를 구성
        # ====================================================

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

            medium="WEB",
        ),


        # ====================================================
        # 과거 거래
        # ====================================================

        history=[

            TransactionData(

                transaction_id="history-001",

                sender_account="sender-001",

                receiver_account="receiver-001",

                sender_bank="KB",

                receiver_bank="KB",

                transaction_type="transfer",

                amount=50_000,

                transaction_datetime=(
                    "2026-08-24T14:00:00"
                ),

                medium="MOBILE",
            ),


            TransactionData(

                transaction_id="history-002",

                sender_account="sender-001",

                receiver_account="receiver-001",

                sender_bank="KB",

                receiver_bank="KB",

                transaction_type="transfer",

                amount=70_000,

                transaction_datetime=(
                    "2026-08-24T18:00:00"
                ),

                medium="MOBILE",
            ),


            TransactionData(

                transaction_id="history-003",

                sender_account="sender-001",

                receiver_account="receiver-002",

                sender_bank="KB",

                receiver_bank="KB",

                transaction_type="transfer",

                amount=100_000,

                transaction_datetime=(
                    "2026-08-25T10:00:00"
                ),

                medium="MOBILE",
            ),


            TransactionData(

                transaction_id="history-004",

                sender_account="sender-001",

                receiver_account="receiver-002",

                sender_bank="KB",

                receiver_bank="KB",

                transaction_type="transfer",

                amount=80_000,

                transaction_datetime=(
                    "2026-08-25T15:00:00"
                ),

                medium="MOBILE",
            ),

        ],
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)
    print("REAL ISOLATION FOREST TEST")
    print("=" * 70)


    try:

        # ====================================================
        # Service Load
        # ====================================================

        service = (
            FraudDetectionService()
        )


        print()
        print(
            "[MODEL]",
            service.model_path,
        )

        print(
            "[DATASET TYPE]",
            service.dataset_type,
        )

        print(
            "[THRESHOLD]",
            service.threshold,
        )


        # ====================================================
        # Request
        # ====================================================

        request = (
            create_test_request()
        )


        # ====================================================
        # Detection
        # ====================================================

        result = service.detect(
            request
        )


        # ====================================================
        # Output
        # ====================================================

        print()
        print("=" * 70)
        print("RESULT")
        print("=" * 70)


        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            )
        )


        print()
        print(
            "[PASS] Real Isolation Forest Inference"
        )


    except FraudDetectionServiceError as error:

        print()
        print("=" * 70)
        print("FDS ERROR")
        print("=" * 70)

        print(
            error
        )


if __name__ == "__main__":

    main()