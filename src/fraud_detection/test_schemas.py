from __future__ import annotations

import json

from .schemas import (
    TransactionData,
    FraudDetectionRequest,
    FraudDetectionResponse,
)


def test_request_schema():

    request = FraudDetectionRequest(

        current_transaction=TransactionData(

            transaction_id="tx-001",

            sender_account="sender-001",

            receiver_account="receiver-999",

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

                transaction_id="tx-history-001",

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

                transaction_id="tx-history-002",

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


    print()
    print("=" * 70)
    print("FDS REQUEST")
    print("=" * 70)

    print(
        json.dumps(
            request.model_dump(
                mode="json"
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


def test_response_schema():

    response = FraudDetectionResponse(

        transaction_id="tx-001",

        anomaly_score=0.482914,

        threshold=0.446117,

        is_anomaly=True,
    )


    print()
    print("=" * 70)
    print("FDS RESPONSE")
    print("=" * 70)

    print(
        json.dumps(
            response.model_dump(
                mode="json"
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":

    test_request_schema()

    test_response_schema()