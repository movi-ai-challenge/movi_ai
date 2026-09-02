"""
스트리밍 음성 인식 세션.

WebSocket 으로 들어오는 오디오 조각을 Google STT 로 흘려보내고, 인식 결과를
되돌려 준다. 배치 경로({@code POST /internal/v1/voice/analyze})와 달리 말이
끝나기를 기다리지 않는다.

호출어 처리
----------
Google STT 는 오디오를 계속 받아야 문장 맥락을 유지한다. 그래서 호출어를
"듣기 시작하는 스위치"로 쓰지 않고, 흐르는 인식 결과에서 '모비야'를 찾는다.
호출어가 나오기 전 발화는 명령으로 취급하지 않는다.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from .wake_word_detector import WakeWordDetector, WakeWordResult


# 오디오 큐가 이만큼 쌓이면 보내는 쪽이 너무 빠른 것이다. 무한정 쌓으면
# 메모리를 먹고 인식 지연도 함께 늘어난다.
MAX_PENDING_CHUNKS = 200


@dataclass
class StreamState:
    """한 세션이 지금까지 무엇을 들었는지."""

    activated: bool = False
    command: str = ""
    last_interim: str = ""
    finals: list[str] = field(default_factory=list)

    def full_text(self) -> str:
        return " ".join(self.finals).strip()


class AudioStream:
    """
    WebSocket 수신과 Google STT 사이를 잇는 큐.

    STTStreamService.recognize 는 AsyncIterator[bytes] 를 받는다. WebSocket 은
    이벤트로 도착하므로 그대로는 맞물리지 않아, 큐를 사이에 두고 반복자로 감싼다.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue(
            maxsize=MAX_PENDING_CHUNKS,
        )
        self._closed = False

    async def push(self, chunk: bytes) -> None:
        if self._closed:
            return
        await self._queue.put(chunk)

    async def close(self) -> None:
        """더 보낼 오디오가 없음을 알린다. 반복자가 끝나고 STT 세션이 닫힌다."""
        if self._closed:
            return
        self._closed = True
        await self._queue.put(None)

    async def __aiter__(self) -> AsyncIterator[bytes]:
        while True:
            chunk = await self._queue.get()
            if chunk is None:
                return
            yield chunk


class StreamSession:
    """
    인식 결과를 화면에 보여줄 수 있는 형태로 다듬는다.

    Google 은 확정 전 추정치(interim)를 계속 고쳐 보낸다. 그대로 흘리면 글자가
    붙었다 지워졌다 하므로, 확정된 문장과 진행 중인 문장을 나눠서 넘긴다.
    """

    def __init__(self, detector: WakeWordDetector | None = None) -> None:
        self.state = StreamState()
        self._detector = detector or WakeWordDetector()

    def consume(self, result: dict) -> dict:
        """
        STT 결과 하나를 받아 클라이언트로 보낼 메시지를 만든다.

        호출어 이전 발화는 command 에 넣지 않는다. "모비야"라고 부르기 전에
        오간 잡담이 이체 명령으로 해석되면 안 된다.
        """
        text = (result.get("text") or "").strip()
        is_final = result.get("type") == "final"

        # 확정된 문장은 누적분에 넣는다. 그때 진행분은 비워야 한다 — 같은 문장을
        # 누적분과 진행분에 모두 두면 화면에 두 번 나온다.
        if is_final:
            self.state.finals.append(text)
            self.state.last_interim = ""
            pending = ""
        else:
            self.state.last_interim = text
            pending = text

        full_text = self._provisional_text(pending)
        wake = self._detect(full_text)

        if wake.activated:
            self.state.activated = True
            self.state.command = wake.command

        return {
            "type": result.get("type"),
            "text": text,
            "activated": self.state.activated,
            "command": self.state.command,
            "fullText": full_text,
            "confidence": result.get("confidence"),
            "stability": result.get("stability"),
        }

    def _provisional_text(self, interim: str) -> str:
        """확정된 문장 + 진행 중인 문장. 화면에 그대로 보여줄 값이다."""
        parts = [self.state.full_text(), interim]
        return " ".join(part for part in parts if part).strip()

    def _detect(self, text: str) -> WakeWordResult:
        """
        누적 문장 전체로 판정한다. 호출어와 명령이 서로 다른 조각에 나뉘어
        들어오면("모비야" / "오만원 보내줘") 조각 하나만 봐서는 찾지 못한다.
        """
        return self._detector.detect(text)
