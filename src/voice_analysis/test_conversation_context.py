from .schemas import (
    RequirementAnalysis,
    RequirementEntities,
)

from .requirement_validator import (
    RequirementValidator,
)

from .conversation_context import (
    ConversationContext,
)


def print_state(context):
    requirement = context.get_current()

    print(
        requirement.model_dump_json(
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "Complete:",
        context.is_complete()
    )

    print(
        "Next Question:",
        context.get_follow_up_question()
    )

    print("-" * 70)


def main():

    validator = RequirementValidator()

    context = ConversationContext(
        validator=validator
    )

    # ========================================================
    # 최초 사용자 명령
    # ========================================================

    requirement = RequirementAnalysis(
        intent="transfer_money",

        entities=RequirementEntities(
            recipient_name="김민수",
            amount=50000,
        ),

        missing_fields=[],

        original_text=(
            "김민수한테 오만원 보내줘"
        ),
    )

    print("=" * 70)
    print("1. 최초 요구사항")
    print("=" * 70)

    context.start(requirement)

    print_state(context)

    # ========================================================
    # 사용자: "국민은행"
    # ========================================================

    print()
    print("=" * 70)
    print("2. 사용자 추가 응답: 국민은행")
    print("=" * 70)

    context.apply_follow_up(
        "국민은행"
    )

    print_state(context)

    # ========================================================
    # 사용자: 계좌번호
    # ========================================================

    print()
    print("=" * 70)
    print("3. 사용자 추가 응답: 1234567890")
    print("=" * 70)

    context.apply_follow_up(
        "1234567890"
    )

    print_state(context)


if __name__ == "__main__":
    main()