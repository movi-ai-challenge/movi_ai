from .voice_pipeline import VoicePipeline
import json


def print_result(title: str, result: dict):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )


def main():

    pipeline = VoicePipeline()

    # ========================================================
    # 1. 계좌 이체 테스트
    # ========================================================

    result = pipeline.process_text(
        "모비야 김민수한테 오만원 보내줘"
    )

    print_result(
        "1-1. 최초 송금 명령",
        result,
    )

    result = pipeline.process_follow_up(
        "국민은행으로 해줘"
    )

    print_result(
        "1-2. 은행 추가 입력",
        result,
    )

    result = pipeline.process_follow_up(
        "계좌번호는 123-456-7890이야"
    )

    print_result(
        "1-3. 계좌번호 추가 입력",
        result,
    )

    # ========================================================
    # 2. 화면 읽기 테스트
    # ========================================================

    pipeline.reset()

    result = pipeline.process_text(
        "모비야 화면 읽어줘"
    )

    print_result(
        "2. 화면 읽기",
        result,
    )

    # ========================================================
    # 3. 거래내역 조회 테스트
    # ========================================================

    pipeline.reset()

    result = pipeline.process_text(
        "모비야 어제 거래내역 알려줘"
    )

    print_result(
        "3. 거래내역 조회",
        result,
    )

    # ========================================================
    # 4. 적금 조회 테스트
    # ========================================================

    pipeline.reset()

    result = pipeline.process_text(
        "모비야 현재 가입한 적금 알려줘"
    )

    print_result(
        "4. 적금 조회",
        result,
    )

    # ========================================================
    # 5. 지원하지 않는 명령
    # ========================================================

    pipeline.reset()

    result = pipeline.process_text(
        "모비야 오늘 날씨 알려줘"
    )

    print_result(
        "5. 지원하지 않는 명령",
        result,
    )

    # ========================================================
    # 6. Wake Word 없음
    # ========================================================

    pipeline.reset()

    result = pipeline.process_text(
        "김민수한테 오만원 보내줘"
    )

    print_result(
        "6. Wake Word 없음",
        result,
    )

if __name__ == "__main__":
    main()