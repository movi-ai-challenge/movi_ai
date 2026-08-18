from .schemas import (
    RequirementAnalysis,
    RequirementEntities,
)
from .requirement_validator import RequirementValidator


def print_result(
    title: str,
    analysis: RequirementAnalysis,
):
    print("=" * 70)
    print(title)
    print(
        analysis.model_dump_json(
            indent=2,
            ensure_ascii=False,
        )
    )


def main():

    validator = RequirementValidator()

    # --------------------------------------------------------
    # Case 1
    # 송금 정보 일부 누락
    # --------------------------------------------------------

    case1 = RequirementAnalysis(
        intent="transfer_money",
        entities=RequirementEntities(
            recipient_name="김민수",
            recipient_bank="국민은행",
            amount=50000,
        ),

        # GPT가 실수로 누락 필드가 없다고 했다고 가정
        missing_fields=[],

        original_text=(
            "김민수한테 국민은행으로 "
            "오만원 보내줘"
        ),
    )

    validated1 = validator.validate(case1)

    print_result(
        "CASE 1 - 계좌번호 누락",
        validated1,
    )

    print(
        "Complete:",
        validator.is_complete(validated1),
    )


    # --------------------------------------------------------
    # Case 2
    # 모든 필드 존재
    # --------------------------------------------------------

    case2 = RequirementAnalysis(
        intent="transfer_money",
        entities=RequirementEntities(
            recipient_name="김민수",
            recipient_bank="국민은행",
            recipient_account="1234567890",
            amount=50000,
        ),
        missing_fields=[
            "recipient_account",
        ],
        original_text=(
            "김민수 국민은행 "
            "1234567890 계좌로 "
            "오만원 보내줘"
        ),
    )

    validated2 = validator.validate(case2)

    print_result(
        "CASE 2 - 모든 정보 확보",
        validated2,
    )

    print(
        "Complete:",
        validator.is_complete(validated2),
    )


    # --------------------------------------------------------
    # Case 3
    # 거래내역
    # --------------------------------------------------------

    case3 = RequirementAnalysis(
        intent="check_history",
        entities=RequirementEntities(
            date_from="2026-08-18",
            date_to="2026-08-18",
        ),
        missing_fields=[
            "bank",
        ],
        original_text="어제 거래내역 알려줘",
    )

    validated3 = validator.validate(case3)

    print_result(
        "CASE 3 - 거래내역",
        validated3,
    )

    print(
        "Complete:",
        validator.is_complete(validated3),
    )


if __name__ == "__main__":
    main()