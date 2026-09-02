import re
from dataclasses import dataclass


@dataclass
class WakeWordResult:
    """
    Wake Word 감지 결과.

    activated:
        호출어가 감지되었는지 여부

    command:
        호출어를 제거한 실제 사용자 명령

    original_text:
        STT에서 전달받은 원본 텍스트
    """

    activated: bool
    command: str
    original_text: str


class WakeWordDetector:
    """
    '모비야' 호출어 감지 및 제거.

    Google STT의 Final Transcript를 받아
    실제 AI 분석이 필요한 명령 부분만 추출한다.
    """

    def __init__(
        self,
        wake_words: list[str] | None = None,
    ):
        """
        표기 변형을 함께 받는다.

        STT 는 짧은 호출어를 자주 흘린다. 실측에서 모델별로 이렇게 갈렸다.
            long    -> 모비야   (정확)
            chirp_3 -> 모비아
            short   -> 모기야

        한 글자 차이로 호출을 못 알아들으면 사용자는 이유를 알 수 없이 계속
        다시 부르게 된다. 반대로 너무 넓히면 잡담이 명령으로 새므로, 실제로
        관측된 변형과 사용자가 영문으로 부르는 경우까지만 넣는다.

        호출어는 '듣기 시작'이 아니라 '명령 구간의 시작'을 정할 뿐이고, 이체는
        뒤에서 다시 확인을 받는다. 조금 넓게 잡아도 바로 돈이 움직이지 않는다.
        """
        self.wake_words = wake_words or [
            "모비야",
            "모비 야",
            "모비아",
            "무비야",
            "모피야",
            "movi야",
            "Movi야",
        ]

    def detect(
        self,
        text: str,
    ) -> WakeWordResult:
        """
        Wake Word를 감지하고 실제 명령어를 반환한다.

        Example
        -------
        입력:
            모비야 김민수한테 오만원 보내줘

        출력:
            activated=True
            command="김민수한테 오만원 보내줘"
        """

        if not text:
            return WakeWordResult(
                activated=False,
                command="",
                original_text="",
            )

        original_text = text.strip()

        # STT가 생성한 연속 공백 정리
        cleaned_text = re.sub(
            r"\s+",
            " ",
            original_text,
        ).strip()

        for wake_word in self.wake_words:

            # 호출어 위치 검색
            index = cleaned_text.find(wake_word)

            if index == -1:
                continue

            # 호출어 이후 내용만 실제 명령으로 사용
            command_start = index + len(wake_word)

            command = cleaned_text[
                command_start:
            ].strip()

            # 호출어 바로 뒤에 쉼표 등이 들어올 수 있음
            command = re.sub(
                r"^[,\s.!?]+",
                "",
                command,
            ).strip()

            return WakeWordResult(
                activated=True,
                command=command,
                original_text=original_text,
            )

        return WakeWordResult(
            activated=False,
            command="",
            original_text=original_text,
        )


# ============================================================
# Simple Test
# ============================================================

if __name__ == "__main__":

    detector = WakeWordDetector()

    test_cases = [
        "모비야 김민수한테 오만원 보내줘",
        "모비야 화면 읽어줘",
        "모비야 어제 거래내역 알려줘",
        "모비 야 가입한 적금 알려줘",
        "김민수한테 오만원 보내줘",
        "오늘 날씨 좋다",
        "",
    ]

    for text in test_cases:

        result = detector.detect(text)

        print("-" * 60)
        print("Original  :", repr(text))
        print("Activated :", result.activated)
        print("Command   :", repr(result.command))