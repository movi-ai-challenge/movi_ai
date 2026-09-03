"""
계약 엔드포인트(POST /internal/v1/voice/analyze) 통합 테스트.

Google STT 와 OpenAI 는 mock 으로 대체한다.
검증 대상은 계약 준수 여부다 — 응답 필드, intent/entity 매핑,
오류 코드와 HTTP status, requestId/voiceSessionId 반향.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from . import api as api_module
from .schemas import RequirementAnalysis, RequirementEntities
from .stt_batch_service import SttError, SttResult


client = TestClient(api_module.app)


def _post(
    *,
    transcript="김민수한테 오만원 보내줘",
    stt_confidence=0.93,
    analysis=None,
    data=None,
    stt_error=None,
):
    """
    STT 를 mock 한 뒤 계약 엔드포인트를 호출한다.
    """

    class FakeStt:
        def transcribe(self, audio_bytes):
            if stt_error is not None:
                raise stt_error
            return SttResult(
                transcript=transcript,
                confidence=stt_confidence,
            )

    payload = {
        "requestId": "voice-123",
        "voiceSessionId": "15",
    }
    if data:
        payload.update(data)

    with patch.object(
        api_module, "_get_stt_service", lambda: FakeStt()
    ):
        if analysis is None:
            return client.post(
                "/internal/v1/voice/analyze",
                files={"audio": ("a.wav", b"RIFFfake", "audio/wav")},
                data=payload,
            )

        with patch.object(
            api_module.voice_service,
            "analyze_command",
            lambda text: analysis,
        ):
            return client.post(
                "/internal/v1/voice/analyze",
                files={"audio": ("a.wav", b"RIFFfake", "audio/wav")},
                data=payload,
            )


def _analysis(intent, confidence=0.96, **entities):
    return RequirementAnalysis(
        intent=intent,
        intent_confidence=confidence,
        entities=RequirementEntities(**entities),
        original_text="테스트",
    )


# ============================================================
# 정상 응답
# ============================================================

def test_transfer_returns_contract_shape():

    response = _post(
        analysis=_analysis(
            "transfer_money",
            recipient_name="김민수",
            amount=50000,
        )
    )

    assert response.status_code == 200
    body = response.json()

    # 백엔드 VoiceAnalysisResponseValidator 가 요구하는 필드
    assert body["requestId"] == "voice-123"
    assert body["voiceSessionId"] == 15
    assert body["transcript"]
    assert body["intent"] == "TRANSFER"
    assert 0.0 <= body["sttConfidence"] <= 1.0
    assert 0.0 <= body["intentConfidence"] <= 1.0
    assert body["entities"] is not None
    assert body["entityConfidences"] is not None
    assert body["detectedMissingEntities"] == []
    assert isinstance(body["processingMs"], int)
    assert body["processingMs"] >= 0


def test_entities_use_contract_keys():

    body = _post(
        analysis=_analysis(
            "transfer_money",
            recipient_name="김민수",
            amount=50000,
        )
    ).json()

    assert set(body["entities"]) == {
        "amount",
        "recipient",
        "sourceAccountAlias",
        "bankName",
        "startDate",
        "endDate",
    }
    assert body["entities"]["recipient"] == "김민수"
    assert body["entities"]["amount"] == 50000


def test_missing_amount_is_reported():

    body = _post(
        analysis=_analysis(
            "transfer_money",
            recipient_name="김민수",
        )
    ).json()

    assert body["detectedMissingEntities"] == ["AMOUNT"]


def test_unsupported_intent_becomes_unknown():

    body = _post(
        analysis=_analysis("check_savings")
    ).json()

    assert body["intent"] == "UNKNOWN"


# ============================================================
# 재질문 답변 (expectedSlots)
# ============================================================

def test_follow_up_keeps_expected_intent():
    """
    계약 2.5: expectedIntent=TRANSFER, expectedSlots=['AMOUNT'] 문맥에서
    "오만 원" 만 들어와도 intent 는 TRANSFER 를 유지해야 한다.
    """

    class FakeParsed:
        field_name = "amount"
        value = 50000
        success = True

    with patch.object(
        api_module.voice_service,
        "parse_follow_up",
        lambda **kwargs: FakeParsed(),
    ):
        body = _post(
            transcript="오만 원",
            data={
                "expectedIntent": "TRANSFER",
                "expectedSlots": '["AMOUNT"]',
            },
        ).json()

    assert body["intent"] == "TRANSFER"
    assert body["entities"]["amount"] == 50000

    # AI 는 이전 수취인을 다시 채우지 않는다 (계약 2.5)
    assert body["entities"]["recipient"] is None
    assert body["detectedMissingEntities"] == ["RECIPIENT"]


def test_malformed_expected_slots_falls_back_to_full_analysis():
    """
    백엔드가 형식을 조금 바꿔도 음성 명령 전체가 실패하면 안 된다.
    """

    response = _post(
        analysis=_analysis("transfer_money", amount=50000, recipient_name="김민수"),
        data={"expectedSlots": "이건JSON이아님"},
    )

    assert response.status_code == 200
    assert response.json()["intent"] == "TRANSFER"


# ============================================================
# 오류 (계약 2.7)
# ============================================================

def test_empty_transcript_returns_422():

    response = _post(
        stt_error=SttError("EMPTY_TRANSCRIPT", "인식 실패")
    )

    assert response.status_code == 422
    error = response.json()["detail"]["error"]
    assert error["code"] == "EMPTY_TRANSCRIPT"
    assert error["retryable"] is True


def test_stt_provider_error_returns_502():

    response = _post(
        stt_error=SttError("STT_PROVIDER_ERROR", "google 실패")
    )

    assert response.status_code == 502
    assert response.json()["detail"]["error"]["code"] == "STT_PROVIDER_ERROR"


def test_audio_too_long_returns_400_and_is_not_retryable():

    response = _post(
        stt_error=SttError("AUDIO_TOO_LONG", "너무 김")
    )

    assert response.status_code == 400
    assert response.json()["detail"]["error"]["retryable"] is False


def test_error_body_echoes_request_id():

    response = _post(
        stt_error=SttError("EMPTY_TRANSCRIPT", "인식 실패")
    )

    assert response.json()["detail"]["requestId"] == "voice-123"


def test_stream_without_final_returns_retryable_error():
    class FakeStreamService:
        async def recognize(self, audio_stream):
            async for _ in audio_stream:
                pass
            if False:
                yield None

    with patch.object(
        api_module,
        "_get_stt_stream_service",
        lambda: FakeStreamService(),
    ):
        with client.websocket_connect(
            "/internal/v1/voice/stream?voiceSessionId=15"
        ) as websocket:
            websocket.send_text("EOS")
            message = websocket.receive_json()

    assert message == {
        "type": "error",
        "code": "NO_FINAL_RESULT",
        "message": "음성을 최종 문장으로 확정하지 못했습니다.",
        "retryable": True,
    }
