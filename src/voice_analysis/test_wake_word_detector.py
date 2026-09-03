from .wake_word_detector import WakeWordDetector


detector = WakeWordDetector()


# ============================================================
# STT 가 호출어를 흘리는 경우
#
# 짧은 호출어는 자음 하나가 자주 바뀐다. 한 글자 차이로 못 알아들으면
# 사용자는 이유를 모른 채 계속 다시 부르게 된다.
# ============================================================

def test_정확한_호출어를_잡는다():
    result = detector.detect("모비야 주혁에게 오만원 보내줘")

    assert result.activated is True
    assert result.command == "주혁에게 오만원 보내줘"


def test_자음이_하나_바뀐_오인식도_잡는다():
    # 실측: short 모델은 "모기야", chirp_3 는 "모비아" 로 흘렸다.
    for heard in ("모기야", "모비아", "무비야", "모피야"):
        result = detector.detect(f"{heard} 주혁에게 오만원 보내줘")

        assert result.activated is True, heard
        assert result.command == "주혁에게 오만원 보내줘", heard


def test_띄어_적어도_잡는다():
    result = detector.detect("모비 야 주혁에게 오만원 보내줘")

    assert result.activated is True
    assert result.command == "주혁에게 오만원 보내줘"


def test_호출어_앞에_말이_붙어도_잡는다():
    # "야 모비야 …" 처럼 부르는 사람이 많다.
    result = detector.detect("야 모비야 주혁에게 오만원 보내줘")

    assert result.activated is True
    assert result.command == "주혁에게 오만원 보내줘"


def test_호출어_뒤의_쉼표를_명령에서_뗀다():
    result = detector.detect("모비야, 잔액 알려줘")

    assert result.command == "잔액 알려줘"


# ============================================================
# 넓히되 새지 않게
# ============================================================

def test_호출어가_없으면_명령으로_치지_않는다():
    result = detector.detect("주혁에게 오만원 보내줘")

    assert result.activated is False
    assert result.command == ""


def test_두_글자_이상_다르면_받지_않는다():
    # 여기까지 받으면 잡담이 명령으로 샌다.
    for heard in ("고구마", "도시락", "오늘은"):
        assert detector.detect(f"{heard} 주혁에게 보내줘").activated is False, heard


def test_빈_발화는_활성화하지_않는다():
    for text in ("", "   "):
        result = detector.detect(text)

        assert result.activated is False
        assert result.command == ""


def test_명령이_비어_있어도_호출은_인정한다():
    # 부르기만 하고 말을 잇지 않은 경우다. 호출은 인정하고 명령은 비운다.
    result = detector.detect("모비야")

    assert result.activated is True
    assert result.command == ""
