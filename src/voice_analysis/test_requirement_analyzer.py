from .requirement_analyzer import RequirementAnalyzer


def main():

    analyzer = RequirementAnalyzer()

    test_cases = [
        "김민수한테 국민은행으로 오만원 보내줘",
        "박지현에게 십만원 송금해줘",
        "어제 거래내역 알려줘",
        "현재 가입한 적금 알려줘",
        "화면 읽어줘",
        "응 진행해줘",
        "아니야",
        "취소해줘",
        "오늘 날씨 알려줘",
    ]

    for command in test_cases:

        print("=" * 70)
        print("INPUT:", command)

        result = analyzer.analyze(command)

        print(
            result.model_dump_json(
                indent=2,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()