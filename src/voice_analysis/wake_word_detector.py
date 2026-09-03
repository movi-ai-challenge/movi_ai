import re
from dataclasses import dataclass

# 한글 음절을 자모로 쪼갤 때 쓰는 값. 유니코드가 정한 배치라 바뀌지 않는다.
_HANGUL_BASE = 0xAC00
_HANGUL_LAST = 0xD7A3
_JUNGSEONG_COUNT = 21
_JONGSEONG_COUNT = 28


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

        목록에 없는 변형까지 받으려면 detect() 의 자모 비교를 참고한다. 관측된
        변형을 하나씩 적어 두는 방식만으로는 새 오인식이 나올 때마다 사용자가
        먼저 겪게 된다.
        """
        self.wake_words = wake_words or [
            "모비야",
            "모비 야",
            "모비아",
            "무비야",
            "모피야",
            "모기야",
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

        fuzzy = self._detect_by_jamo(cleaned_text, original_text)

        if fuzzy is not None:
            return fuzzy

        return WakeWordResult(
            activated=False,
            command="",
            original_text=original_text,
        )

    # --------------------------------------------------------
    # 목록에 없는 오인식까지 받는다
    # --------------------------------------------------------

    def _detect_by_jamo(
        self,
        cleaned_text: str,
        original_text: str,
    ) -> WakeWordResult | None:
        """
        자모 하나 차이까지 호출어로 인정한다.

        STT 는 짧은 호출어에서 자음 하나를 자주 흘린다. 실측한 변형(모기야·모비아
        ·무비야·모피야)은 모두 "모비야" 와 <b>자모 9개 중 딱 하나</b>만 달랐다.
        표기를 하나씩 적어 두는 대신 이 규칙으로 받으면 아직 못 본 변형도 걸린다.

        두 글자 이상 다르면 받지 않는다. 그쯤 되면 사용자가 부른 말이 맞는지
        알 수 없고, 잡담이 명령으로 새기 시작한다.

        호출어를 넓게 잡아도 바로 돈이 움직이지는 않는다 — 이체는 뒤에서 확인을
        한 번 더 받는다.
        """

        target = self._to_jamo("모비야")

        # 공백을 건너뛰고 한글 음절만 모은다. "모비 야" 처럼 띄어 적어도 잡는다.
        syllables = [
            (index, char)
            for index, char in enumerate(cleaned_text)
            if _HANGUL_BASE <= ord(char) <= _HANGUL_LAST
        ]

        for start in range(len(syllables) - 2):
            window = syllables[start:start + 3]
            candidate = "".join(char for _, char in window)

            if self._jamo_distance(self._to_jamo(candidate), target) > 1:
                continue

            command_start = window[-1][0] + 1
            command = cleaned_text[command_start:].strip()
            command = re.sub(r"^[,\s.!?]+", "", command).strip()

            return WakeWordResult(
                activated=True,
                command=command,
                original_text=original_text,
            )

        return None

    def _to_jamo(self, text: str) -> list[int]:
        """한글 음절을 초성·중성·종성 번호로 편다. 한글이 아니면 건너뛴다."""

        jamo: list[int] = []

        for char in text:
            code = ord(char) - _HANGUL_BASE

            if not 0 <= code <= _HANGUL_LAST - _HANGUL_BASE:
                continue

            jamo.append(code // (_JUNGSEONG_COUNT * _JONGSEONG_COUNT))
            jamo.append((code % (_JUNGSEONG_COUNT * _JONGSEONG_COUNT)) // _JONGSEONG_COUNT)
            jamo.append(code % _JONGSEONG_COUNT)

        return jamo

    def _jamo_distance(self, left: list[int], right: list[int]) -> int:
        """자리별로 다른 자모의 개수. 길이가 다르면 비교 대상이 아니다."""

        if len(left) != len(right):
            # 한글이 아닌 글자가 섞여 자모 수가 안 맞는 경우다. 호출어로 보지 않는다.
            return len(right) + 1

        return sum(1 for a, b in zip(left, right) if a != b)


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