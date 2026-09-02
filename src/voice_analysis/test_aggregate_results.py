"""한 응답에 담긴 결과를 어떻게 묶는지 검증."""

from dataclasses import dataclass

from .stream_session import aggregate_results


@dataclass
class FakeAlternative:
    transcript: str
    confidence: float = 0.9


@dataclass
class FakeResult:
    alternatives: list
    is_final: bool = False
    stability: float = 0.8


def result(text: str, is_final: bool = False) -> FakeResult:
    return FakeResult(alternatives=[FakeAlternative(text)], is_final=is_final)


def test_확정_전_조각은_하나로_합친다():
    """
    조각마다 따로 내보내면 받는 쪽이 마지막 것만 남겨 앞부분을 잃는다.
    실제로 화면 문장이 뒤로 가는 증상이 났다.
    """
    messages = aggregate_results([result("모비야 김민수한테"), result("오만원")])

    assert len(messages) == 1
    assert messages[0]["type"] == "interim"
    assert messages[0]["text"] == "모비야 김민수한테 오만원"


def test_확정된_결과는_각각_내보낸다():
    """확정 구간은 받는 쪽이 순서대로 누적해야 하므로 합치지 않는다."""
    messages = aggregate_results([
        result("모비야", is_final=True),
        result("김민수한테 오만원", is_final=True),
    ])

    assert [m["text"] for m in messages] == ["모비야", "김민수한테 오만원"]
    assert all(m["type"] == "final" for m in messages)


def test_확정과_진행이_섞이면_확정이_먼저다():
    messages = aggregate_results([
        result("모비야", is_final=True),
        result("김민수"),
    ])

    assert [(m["type"], m["text"]) for m in messages] == [
        ("final", "모비야"),
        ("interim", "김민수"),
    ]


def test_빈_문장과_대안없는_결과는_버린다():
    messages = aggregate_results([
        FakeResult(alternatives=[]),
        result("   "),
        result("오만원"),
    ])

    assert len(messages) == 1
    assert messages[0]["text"] == "오만원"


def test_결과가_없으면_아무것도_내보내지_않는다():
    assert aggregate_results([]) == []
