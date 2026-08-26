from __future__ import annotations

from .schemas import (
    TransactionData,
    FraudDetectionRequest,
)

from .transaction_mapper import (
    convert_hour_to_time_bucket,
    request_to_dataframe,
    find_current_transaction_index,
)


def test_time_bucket():

    assert convert_hour_to_time_bucket(0) == 0
    assert convert_hour_to_time_bucket(2) == 0

    assert convert_hour_to_time_bucket(3) == 3
    assert convert_hour_to_time_bucket(5) == 3

    assert convert_hour_to_time_bucket(14) == 12

    assert convert_hour_to_time_bucket(18) == 18

    assert convert_hour_to_time_bucket(23) == 21

    print(
        "[PASS] 시간대 Bucket 변환"
    )


def test_request_mapper():

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
                    "2026-08-25T14:20:00"
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
                    "2026-08-25T18:40:00"
                ),

                medium="MOBILE",
            ),
        ],
    )


    dataframe = request_to_dataframe(
        request
    )


    print()
    print("=" * 80)
    print("MAPPED DATAFRAME")
    print("=" * 80)

    print(
        dataframe.to_string(
            index=False
        )
    )


    current_index = (
        find_current_transaction_index(
            dataframe,
            request.current_transaction.transaction_id,
        )
    )


    print()
    print(
        "Current Transaction Index:",
        current_index,
    )


    current_row = (
        dataframe.loc[
            current_index
        ]
    )


    assert (
        current_row[
            "출금계좌일련번호"
        ]
        ==
        "sender-001"
    )

    assert (
        current_row[
            "입금계좌일련번호"
        ]
        ==
        "receiver-new"
    )

    assert (
        current_row[
            "거래금액"
        ]
        ==
        10_000_000
    )

    assert (
        current_row[
            "거래시간대"
        ]
        ==
        0
    )


    print()
    print(
        "[PASS] Transaction Mapping"
    )


if __name__ == "__main__":

    test_time_bucket()

    test_request_mapper()