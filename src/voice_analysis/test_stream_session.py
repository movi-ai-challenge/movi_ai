"""
스트리밍 세션의 호출어·누적 처리 검증.

Google STT 를 실제로 부르지 않는다. 인식 결과 dict 만 흘려 넣어, 화면에 무엇이
보이고 언제 명령으로 인정되는지를 확인한다.
"""

from .stream_session import StreamSession


def interim(text: str) -> dict:
    return {"type": "interim", "text": text, "stability": 0.8}


def final(text: str) -> dict:
    return {"type": "final", "text": text, "confidence": 0.94}


def test_호출어_전에는_명령으로_인정하지_않는다():
    session = StreamSession()

    result = session.consume(interim("오늘 날씨 좋다"))

    assert result["activated"] is False
    assert result["command"] == ""


def test_호출어를_만나면_뒤의_말을_명령으로_삼는다():
    session = StreamSession()

    result = session.consume(final("모비야 김민수한테 오만원 보내줘"))

    assert result["activated"] is True
    assert result["command"] == "김민수한테 오만원 보내줘"


def test_호출어와_명령이_나뉘어_들어와도_찾는다():
    """
    "모비야"까지 확정된 뒤 명령이 따로 들어오는 것이 실제로 흔하다.
    조각 하나만 보면 호출어를 놓친다.
    """
    session = StreamSession()

    first = session.consume(final("모비야"))
    assert first["activated"] is True

    second = session.consume(interim("오만원 보내줘"))

    assert second["activated"] is True
    assert second["command"] == "오만원 보내줘"


def test_한번_활성화되면_유지된다():
    session = StreamSession()
    session.consume(final("모비야 김민수한테"))

    result = session.consume(interim("오만원 보내줘"))

    assert result["activated"] is True


def test_화면에_보여줄_문장은_확정분과_진행분을_합친다():
    """실시간 표시용 값이다. 확정된 문장 뒤에 지금 말하는 중인 조각이 붙는다."""
    session = StreamSession()
    session.consume(final("모비야 김민수한테"))

    result = session.consume(interim("오만원"))

    assert result["fullText"] == "모비야 김민수한테 오만원"


def test_확정된_문장은_사라지지_않는다():
    session = StreamSession()
    session.consume(final("모비야 김민수한테"))
    session.consume(interim("오만"))

    result = session.consume(final("오만원 보내줘"))

    assert result["fullText"] == "모비야 김민수한테 오만원 보내줘"


def test_STT_가_흘린_호출어_변형도_받는다():
    """
    실측에서 모델별로 모비야 / 모비아 / 모기야 로 갈렸다. 한 글자 차이로
    호출을 못 알아들으면 사용자는 이유 없이 계속 다시 부르게 된다.
    """
    for heard in ["모비야", "모비아", "무비야", "모비 야", "movi야"]:
        session = StreamSession()

        result = session.consume(final(f"{heard} 오만원 보내줘"))

        assert result["activated"] is True, heard
        assert result["command"] == "오만원 보내줘", heard


def test_비슷하지_않은_말은_호출어로_보지_않는다():
    session = StreamSession()

    result = session.consume(final("오늘 모임 있어"))

    assert result["activated"] is False
