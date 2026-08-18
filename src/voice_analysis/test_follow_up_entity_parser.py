from .follow_up_entity_parser import FollowUpEntityParser


def main():

    parser = FollowUpEntityParser()

    test_cases = [
        (
            "recipient_bank",
            "아 국민은행으로 해줘",
        ),
        (
            "recipient_account",
            "계좌번호는 123-456-7890이야",
        ),
        (
            "amount",
            "십만원",
        ),
        (
            "recipient_name",
            "김민수야",
        ),
    ]

    for field_name, user_text in test_cases:

        print("=" * 70)
        print("FIELD :", field_name)
        print("INPUT :", user_text)

        result = parser.parse(
            field_name=field_name,
            user_text=user_text,
        )

        print(
            result.model_dump_json(
                indent=2,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()