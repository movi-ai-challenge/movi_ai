from src.voice_analysis.request_mapper import (
    build_backend_request,
)


def test_transfer():

    result = build_backend_request(

        request_id="req-test-transfer",

        transcript=(
            "김민수한테 오만원 보내줘"
        ),

        intent="transfer_money",

        entities={
            "recipient_name": "김민수",
            "recipient_bank": None,
            "recipient_account": None,
            "amount": 50000,
            "source_bank": None,
            "source_account": None,
        },
    )

    print()
    print("=" * 60)
    print("TRANSFER")
    print("=" * 60)

    print(result)


def test_savings():

    result = build_backend_request(

        request_id="req-test-saving",

        transcript=(
            "가입한 적금 알려줘"
        ),

        intent="check_savings",

        entities={},
    )

    print()
    print("=" * 60)
    print("SAVINGS")
    print("=" * 60)

    print(result)


def test_history():

    result = build_backend_request(

        request_id="req-test-history",

        transcript=(
            "최근 거래내역 알려줘"
        ),

        intent="check_history",

        entities={
            "date_from": None,
            "date_to": None,
            "limit": 10,
        },
    )

    print()
    print("=" * 60)
    print("HISTORY")
    print("=" * 60)

    print(result)


if __name__ == "__main__":

    test_transfer()

    test_savings()

    test_history()