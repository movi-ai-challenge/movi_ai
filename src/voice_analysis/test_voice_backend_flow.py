from unittest.mock import patch

from .voice_pipeline import VoicePipeline


def test_transfer_flow():

    pipeline = VoicePipeline()


    # ========================================================
    # 1차 Backend Response
    # 은행 필요
    # ========================================================

    first_backend_response = {

        "request_id": "req-test",

        "status": "need_more_info",

        "requested_field": "recipient_bank",

        "message": (
            "받는 분의 은행을 말씀해주세요."
        ),

        "data": None,
    }


    with patch(
        "src.voice_analysis.voice_pipeline.send_voice_command",
        return_value=first_backend_response,
    ):

        result1 = pipeline.process_text(
            "모비야 김민수한테 오만원 보내줘"
        )


    print()
    print("=" * 60)
    print("1차 요청")
    print("=" * 60)

    print(result1)


    request_id = pipeline.request_id


    # ========================================================
    # 2차 Backend Response
    # 계좌번호 필요
    # ========================================================

    second_backend_response = {

        "request_id": request_id,

        "status": "need_more_info",

        "requested_field": "recipient_account",

        "message": (
            "받는 분의 계좌번호를 말씀해주세요."
        ),

        "data": None,
    }


    with patch(
        "src.voice_analysis.voice_pipeline.send_voice_command",
        return_value=second_backend_response,
    ):

        result2 = pipeline.process_follow_up(
            "국민은행이야"
        )


    print()
    print("=" * 60)
    print("2차 요청")
    print("=" * 60)

    print(result2)


    # ========================================================
    # 검증
    # ========================================================

    assert (
        result1["request_id"]
        ==
        result2["request_id"]
    )

    assert (
        result2["entities"][
            "recipient_bank"
        ]
        ==
        "국민은행"
    )

    assert (
        result2["requested_field"]
        ==
        "recipient_account"
    )


    print()
    print("Multi-turn 테스트 성공")


if __name__ == "__main__":

    test_transfer_flow()