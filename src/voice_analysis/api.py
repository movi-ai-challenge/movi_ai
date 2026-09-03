from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Optional

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)

from .contract_schemas import (
    ContractErrorBody,
    ContractErrorResponse,
    VoiceAnalyzeContractResponse,
)

from .contract_mapper import (
    map_entity_confidences,
    detect_missing_slots,
    map_entities,
    map_intent,
)

from .stt_batch_service import (
    SttBatchService,
    SttError,
)

from .stt_stream_service import STTStreamService

from .stream_session import AudioStream, StreamSession
from openai import (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
)
from .api_schemas import (
    VoiceAnalyzeRequest,
    VoiceFollowUpRequest,
    VoiceAnalyzeResponse,
    VoiceFollowUpResponse,
)


logger = logging.getLogger(__name__)

from .voice_service import (
    VoiceAnalysisService,
)


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(

    title=(
        "MOVI Voice Analysis API"
    ),

    description=(
        "MOVI STT 기반 음성 요구사항 분석 API"
    ),

    version="0.2.0",
)


# ============================================================
# Service
# ============================================================

voice_service: VoiceAnalysisService | None = None
voice_service_load_error: str | None = None


def load_voice_service() -> None:
    """분석기를 안전하게 초기화한다.

    OpenAI 환경설정이 없거나 Client 초기화가 실패해도 FastAPI 자체는
    기동하여 /health와 /ready에서 상태를 확인할 수 있게 한다.
    """

    global voice_service
    global voice_service_load_error

    voice_service = None
    voice_service_load_error = None

    try:
        voice_service = VoiceAnalysisService()
        logger.info("Voice analysis service loaded successfully")
    except Exception as error:
        voice_service_load_error = type(error).__name__
        logger.exception("Failed to load voice analysis service")


def voice_service_ready() -> bool:
    return (
        voice_service is not None
        and voice_service_load_error is None
    )


def require_voice_service() -> VoiceAnalysisService:
    if not voice_service_ready():
        raise HTTPException(
            status_code=503,
            detail={
                "code": "VOICE_ANALYZER_NOT_READY",
                "message": "음성 요구사항 분석기가 준비되지 않았습니다.",
            },
        )

    # voice_service_ready()에서 None이 아님을 확인했다.
    return voice_service


load_voice_service()


# ============================================================
# Root
# ============================================================

@app.get("/")
def root():

    return {

        "service": (
            "MOVI Voice Analysis API"
        ),

        "status": "running",

        "version": "0.2.0",
    }


# ============================================================
# Health
# ============================================================

@app.get("/health")
def health():
    """
    OPENAI_API_KEY 가 없으면 503 을 반환한다.

    클라이언트를 지연 생성하도록 바꾸면서 키가 없어도 서버가
    뜨게 됐다. 그대로 두면 배포는 성공했는데 모든 음성 요청이
    실패하는 상태가 된다. 배포 스크립트가 healthcheck 로
    롤백을 판단하므로 여기서 잡는다.
    """

    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unavailable",
                "service": "voice-analysis",
                "reason": "OPENAI_API_KEY 가 설정되지 않았습니다.",
            },
        )

    ready = voice_service_ready()

    return {

        "status": "ok" if ready else "degraded",

        "service": (
            "voice-analysis"
        ),

        "analyzer_loaded": ready,

        "error_code": (
            None
            if ready
            else "VOICE_ANALYZER_LOAD_FAILED"
        ),
    }


# ============================================================
# Readiness
# ============================================================

@app.get("/ready")
def ready():

    require_voice_service()

    return {
        "status": "ready",
        "service": "voice-analysis",
    }


# ============================================================
# 최초 Voice 분석
# ============================================================

@app.post(
    "/api/v1/voice/analyze",
    response_model=VoiceAnalyzeResponse,
)
def analyze_voice(
    request: VoiceAnalyzeRequest,
):

    try:

        service = require_voice_service()

        result = (
            service.analyze(
                request.transcript
            )
        )


        return VoiceAnalyzeResponse(

            status=(
                result.get(
                    "status",
                    "error",
                )
            ),

            intent=(
                result.get(
                    "intent"
                )
            ),

            transcript=(
                result.get(
                    "transcript"
                )
            ),

            entities=(
                result.get(
                    "entities",
                    {},
                )
            ),

            message=(
                result.get(
                    "message"
                )
            ),
        )


    except HTTPException:
        raise


    except (
        APITimeoutError,
        APIConnectionError,
        RateLimitError,
    ) as error:

        logger.warning(
            "Voice analyzer is temporarily unavailable: %s",
            type(error).__name__,
        )

        raise HTTPException(
            status_code=503,
            detail={
                "code": "VOICE_ANALYZER_UNAVAILABLE",
                "message": "음성 요구사항 분석 서비스를 일시적으로 사용할 수 없습니다.",
            },
        ) from error


    except ValueError as error:

        # 사용자 transcript나 계좌정보가 예외 문자열에 포함될 수 있으므로
        # 구체적인 오류값은 로그에 남기지 않는다.
        logger.warning(
            "Voice request rejected: error_type=%s",
            type(error).__name__,
        )

        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_VOICE_REQUEST",
                "message": "음성 분석 요청이 올바르지 않습니다.",
            },
        ) from error


    except Exception as error:

        logger.exception("Unexpected error during voice analysis")

        raise HTTPException(
            status_code=500,
            detail={
                "code": "VOICE_ANALYSIS_FAILED",
                "message": "음성 요구사항 분석 중 내부 오류가 발생했습니다.",
            },
        ) from error


# ============================================================
# Follow-up 분석
# ============================================================

@app.post(
    "/api/v1/voice/follow-up",
    response_model=VoiceFollowUpResponse,
)
def analyze_follow_up(
    request: VoiceFollowUpRequest,
):

    try:

        service = require_voice_service()

        return (
            service
            .analyze_follow_up(

                requested_field=(
                    request.requested_field
                ),

                text=(
                    request.transcript
                ),

                entities=(
                    request.entities
                ),
            )
        )


    except HTTPException:
        raise


    except (
        APITimeoutError,
        APIConnectionError,
        RateLimitError,
    ) as error:

        logger.warning(
            "Voice follow-up analyzer is temporarily unavailable: %s",
            type(error).__name__,
        )

        raise HTTPException(
            status_code=503,
            detail={
                "code": "VOICE_ANALYZER_UNAVAILABLE",
                "message": "음성 요구사항 분석 서비스를 일시적으로 사용할 수 없습니다.",
            },
        ) from error


    except ValueError as error:

        logger.warning(
            "Voice follow-up rejected: error_type=%s",
            type(error).__name__,
        )

        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_FOLLOW_UP_REQUEST",
                "message": "추가 정보 분석 요청이 올바르지 않습니다.",
            },
        ) from error


    except Exception as error:

        logger.exception("Unexpected error during voice follow-up analysis")

        raise HTTPException(
            status_code=500,
            detail={
                "code": "VOICE_FOLLOW_UP_FAILED",
                "message": "추가 정보 분석 중 내부 오류가 발생했습니다.",
            },
        ) from error

# ============================================================
# 내부 API — 백엔드 계약 경로
#
# 기준: movi_backend/docs/ai-api-contract.md v1.0  2절
#
# 위의 /api/v1/voice/* 는 텍스트를 받는 초기 인터페이스다.
# 백엔드는 음성 파일을 multipart 로 보내고 camelCase 응답을
# 기대하므로 계약 경로를 따로 둔다. 기존 경로는 유지한다.
# ============================================================

@app.post(
    "/internal/v1/voice/analyze",
    response_model=VoiceAnalyzeContractResponse,
    responses={
        400: {"model": ContractErrorResponse},
        422: {"model": ContractErrorResponse},
        502: {"model": ContractErrorResponse},
        500: {"model": ContractErrorResponse},
    },
)
def analyze_voice_internal(
    audio: UploadFile = File(
        description="WebM/Opus 또는 WAV. 최대 5MB",
    ),
    requestId: str = Form(...),
    voiceSessionId: int = Form(...),
    expectedIntent: Optional[str] = Form(default=None),
    expectedSlots: Optional[str] = Form(default=None),
):
    """
    이 함수는 일부러 ``def`` 다(``async def`` 가 아니다).

    STT 와 GPT 호출이 둘 다 동기이고 합쳐서 15~20초가 걸린다. ``async def`` 로 두면
    그동안 이벤트 루프가 통째로 멈춰 **다른 요청을 하나도 받지 못한다** — 실제로
    이것 때문에 같은 시간에 들어온 WebSocket 업그레이드가 타임아웃돼 음성 명령이
    실패했다(2026-09-03). ``def`` 로 두면 FastAPI 가 스레드풀에서 돌려 루프가 풀린다.
    """

    started = time.perf_counter()

    # --------------------------------------------------------
    # 1. STT
    # --------------------------------------------------------

    try:
        audio_bytes = audio.file.read()

        stt_result = _get_stt_service().transcribe(audio_bytes)

    except SttError as error:
        raise _contract_error(
            request_id=requestId,
            code=error.code,
            message=str(error),
        ) from error

    except Exception as error:
        raise _contract_error(
            request_id=requestId,
            code="STT_PROVIDER_ERROR",
            message=f"{type(error).__name__}: {error}",
        ) from error

    transcript = stt_result.transcript

    # --------------------------------------------------------
    # 2. 재질문 답변인지 판단
    #
    # expectedSlots 가 있으면 백엔드가 특정 슬롯을 물어본 상태다.
    # "오만 원" 같은 짧은 발화는 전체 분석을 돌리면 intent 를
    # 잃어버리므로 해당 슬롯만 추출한다.
    # --------------------------------------------------------

    follow_up_field = _resolve_follow_up_field(expectedSlots)

    try:
        if follow_up_field is not None:
            intent, confidence, raw_entities, raw_confidences = _analyze_follow_up(
                field_name=follow_up_field,
                transcript=transcript,
                expected_intent=expectedIntent,
            )
        else:
            analysis = voice_service.analyze_command(transcript)

            intent = map_intent(analysis.intent)
            confidence = float(analysis.intent_confidence)
            raw_entities = analysis.entities.model_dump()
            raw_confidences = analysis.entity_confidences.model_dump()

    except Exception as error:
        raise _contract_error(
            request_id=requestId,
            code="MODEL_INFERENCE_ERROR",
            message=f"{type(error).__name__}: {error}",
        ) from error

    # --------------------------------------------------------
    # 3. 계약 형식으로 변환
    # --------------------------------------------------------

    entities = map_entities(raw_entities)

    elapsed_ms = int(
        (time.perf_counter() - started) * 1000
    )

    return VoiceAnalyzeContractResponse(
        requestId=requestId,
        voiceSessionId=voiceSessionId,
        transcript=transcript,
        sttConfidence=stt_result.confidence,
        intent=intent,
        intentConfidence=confidence,
        entities=entities,
        entityConfidences=map_entity_confidences(
            raw_entities,
            raw_confidences,
            entities,
        ),
        detectedMissingEntities=detect_missing_slots(
            intent=intent,
            entities=entities,
        ),
        processingMs=elapsed_ms,
    )


# ============================================================
# 계약 경로 보조
# ============================================================

# 계약 슬롯명 -> 내부 follow-up field 명
_SLOT_TO_FIELD = {
    "AMOUNT": "amount",
    "RECIPIENT": "recipient_name",
    "BANK_NAME": "recipient_bank",
}

# 내부 follow-up field 명 -> 내부 엔티티 키
_FIELD_TO_ENTITY = {
    "amount": "amount",
    "recipient_name": "recipient_name",
    "recipient_bank": "recipient_bank",
    "recipient_account": "recipient_account",
}



def analyze_command_text(
    *,
    request_id: str,
    voice_session_id: int,
    transcript: str,
    stt_confidence: float,
    expected_intent: str | None = None,
    expected_slots: str | None = None,
) -> VoiceAnalyzeContractResponse:
    """
    인식된 문장에서 의도와 엔티티를 뽑아 계약 형식으로 만든다.

    배치 경로와 스트리밍 경로가 같은 함수를 쓴다. 두 경로가 각자 분석하면 같은
    말에 다른 판단이 나올 수 있고, 화면에 보인 문장과 실제로 실행되는 명령이
    어긋난다. 돈이 움직이는 흐름에서 그 어긋남은 그대로 사고가 된다.
    """
    started = time.perf_counter()

    follow_up_field = _resolve_follow_up_field(expected_slots)

    if follow_up_field is not None:
        intent, confidence, raw_entities, raw_confidences = _analyze_follow_up(
            field_name=follow_up_field,
            transcript=transcript,
            expected_intent=expected_intent,
        )
    else:
        analysis = voice_service.analyze_command(transcript)
        intent = map_intent(analysis.intent)
        confidence = float(analysis.intent_confidence)
        raw_entities = analysis.entities.model_dump()
        raw_confidences = analysis.entity_confidences.model_dump()

    entities = map_entities(raw_entities)

    return VoiceAnalyzeContractResponse(
        requestId=request_id,
        voiceSessionId=voice_session_id,
        transcript=transcript,
        sttConfidence=stt_confidence,
        intent=intent,
        intentConfidence=confidence,
        entities=entities,
        entityConfidences=map_entity_confidences(
            raw_entities,
            raw_confidences,
            entities,
        ),
        detectedMissingEntities=detect_missing_slots(
            intent=intent,
            entities=entities,
        ),
        processingMs=int((time.perf_counter() - started) * 1000),
    )


def _get_stt_service():
    """
    STT 클라이언트를 첫 호출 시점에 만든다.

    모듈 로드 시점에 만들면 GCP 자격증명이 없는 환경에서
    Voice API 전체가 기동하지 못한다. 계약 경로를 쓰지 않는
    배포에서도 서버는 떠야 한다.
    """

    global _stt_service

    if _stt_service is None:
        _stt_service = SttBatchService()

    return _stt_service


_stt_service = None


def _resolve_follow_up_field(
    expected_slots: Optional[str],
) -> Optional[str]:
    """
    expectedSlots(JSON 문자열)에서 첫 번째 슬롯을 꺼내
    내부 field 명으로 바꾼다.

    파싱에 실패하면 재질문이 아닌 것으로 보고 전체 분석으로 넘긴다.
    여기서 예외를 내면 백엔드가 형식을 조금 바꿨을 때
    음성 명령 전체가 실패한다.
    """

    if not expected_slots:
        return None

    try:
        slots = json.loads(expected_slots)
    except (TypeError, ValueError):
        return None

    if isinstance(slots, str):
        slots = [slots]

    if not isinstance(slots, list) or not slots:
        return None

    return _SLOT_TO_FIELD.get(str(slots[0]).upper())


def _analyze_follow_up(
    *,
    field_name: str,
    transcript: str,
    expected_intent: Optional[str],
):
    """
    재질문 답변에서 슬롯 하나만 추출한다.

    intent 는 백엔드가 알려준 expectedIntent 를 유지한다.
    "오만 원" 만으로 intent 를 다시 분류하면 UNKNOWN 이 되고,
    백엔드 세션이 진행 중이던 이체를 잃는다.
    """

    parsed = voice_service.parse_follow_up(
        field_name=field_name,
        text=transcript,
    )

    intent = "UNKNOWN"

    if expected_intent:
        candidate = expected_intent.strip().upper()

        if candidate in {
            "BALANCE", "TRANSFER", "HISTORY",
            "CONFIRM", "CANCEL", "UNKNOWN",
        }:
            intent = candidate

    entities: dict = {}
    confidences: dict = {}

    if parsed.success:
        entity_key = _FIELD_TO_ENTITY.get(
            parsed.field_name,
            parsed.field_name,
        )
        entities[entity_key] = parsed.value
        confidence = 0.9

        # 재질문 답변은 슬롯 하나만 뽑는 파서를 탄다. 성공 여부가 곧 확신도라
        # intent 확신도와 같은 값을 쓴다 - 여기서 null 로 두면 백엔드가
        # 방금 되물어 받은 답을 또 믿지 못해 같은 질문을 반복한다.
        confidences[entity_key] = confidence
    else:
        confidence = 0.3

    return intent, confidence, entities, confidences


def _contract_error(
    *,
    request_id: str,
    code: str,
    message: str,
) -> HTTPException:
    """
    ai-api-contract.md 2.7 의 코드 -> HTTP status 매핑.
    """

    status_by_code = {
        "UNSUPPORTED_AUDIO_FORMAT": 400,
        "AUDIO_TOO_LONG": 400,
        "EMPTY_TRANSCRIPT": 422,
        "STT_PROVIDER_ERROR": 502,
        "VOICE_ANALYSIS_TIMEOUT": 504,
        "MODEL_INFERENCE_ERROR": 500,
    }

    retryable_codes = {
        "EMPTY_TRANSCRIPT",
        "STT_PROVIDER_ERROR",
        "VOICE_ANALYSIS_TIMEOUT",
        "MODEL_INFERENCE_ERROR",
    }

    return HTTPException(
        status_code=status_by_code.get(code, 500),
        detail=ContractErrorResponse(
            requestId=request_id,
            error=ContractErrorBody(
                code=code,
                message=message,
                retryable=code in retryable_codes,
            ),
        ).model_dump(),
    )


# ============================================================
# 스트리밍 음성 인식 (WebSocket)
# ============================================================

_stt_stream_service: STTStreamService | None = None


def _get_stt_stream_service() -> STTStreamService:
    """
    Google 클라이언트 생성은 비싸다. 세션마다 만들지 않고 재사용한다.
    """
    global _stt_stream_service
    if _stt_stream_service is None:
        _stt_stream_service = STTStreamService()
    return _stt_stream_service


@app.websocket("/internal/v1/voice/stream")
async def stream_voice_internal(
    websocket: WebSocket,
    voiceSessionId: int = 0,
    expectedIntent: Optional[str] = None,
    expectedSlots: Optional[str] = None,
):
    """
    말하는 도중에 오디오 조각을 받아 인식 결과를 실시간으로 돌려준다.

    배치 경로(POST /internal/v1/voice/analyze)는 그대로 둔다. WebSocket 을 쓸 수
    없는 환경과, 녹음 파일을 통째로 올리는 기존 흐름이 계속 필요하다.

    주고받는 것
    ----------
    받는다  : 오디오 조각(binary). PCM16 / 16kHz / mono.
              텍스트 "EOS" 를 받으면 더 보낼 오디오가 없다는 뜻이다.
    보낸다  : {"type": "interim"|"final", "text", "activated", "command",
               "fullText", "confidence", "stability"}
              확정 발화에 호출어가 있으면 뒤이어
              {"type": "analysis", ...VoiceAnalyzeContractResponse} 를 보낸다.
              오류는 {"type": "error", "code", "message", "retryable"}

    호출어를 만나기 전까지 activated 는 false 이고 command 는 비어 있다.

    분석까지 여기서 마치는 이유는, 화면에 보인 문장과 실제로 실행되는 명령이
    같아야 하기 때문이다. 백엔드가 나중에 오디오를 다시 인식하면 두 값이 달라질
    수 있고, 사용자는 자기가 본 것과 다른 이체를 마주하게 된다.
    """
    await websocket.accept()

    audio_stream = AudioStream()
    session = StreamSession()
    received_final = False
    client_disconnected = False

    async def pump_audio() -> None:
        """
        WebSocket 수신을 큐로 옮긴다. 인식과 수신을 한 루프에 두면 Google 응답을
        기다리는 동안 오디오를 못 받아 조각이 밀린다.
        """
        nonlocal client_disconnected
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    client_disconnected = True
                    break
                chunk = message.get("bytes")
                if chunk:
                    await audio_stream.push(chunk)
                    continue
                if message.get("text") == "EOS":
                    break
        except WebSocketDisconnect:
            client_disconnected = True
        finally:
            await audio_stream.close()

    pump = asyncio.create_task(pump_audio())

    try:
        async for result in _get_stt_stream_service().recognize(audio_stream):
            message = session.consume(result)
            await websocket.send_json(message)

            if message["type"] == "final":
                received_final = True

            if message["type"] != "final" or not message["activated"]:
                continue
            command = (message["command"] or "").strip()
            if not command:
                continue

            # 분석 실패로 연결을 끊지 않는다. 인식된 문장은 이미 화면에 떠 있고,
            # 사용자에게는 그 뒤가 조용해지는 것보다 오류를 듣는 편이 낫다.
            try:
                # 스레드로 넘긴다. GPT 호출은 동기라 여기서 그냥 부르면 이벤트 루프가
                # 15초 넘게 멈추고, 그 사이 다른 사용자의 연결이 전부 타임아웃된다.
                analysis = await asyncio.to_thread(
                    analyze_command_text,
                    request_id=f"voice-stream-{voiceSessionId}",
                    voice_session_id=voiceSessionId,
                    transcript=command,
                    stt_confidence=float(message.get("confidence") or 0.0),
                    expected_intent=expectedIntent,
                    expected_slots=expectedSlots,
                )
                await websocket.send_json({
                    "type": "analysis",
                    **analysis.model_dump(),
                })
            except Exception as error:
                await websocket.send_json({
                    "type": "error",
                    "code": "MODEL_INFERENCE_ERROR",
                    "message": f"{type(error).__name__}: {error}",
                    "retryable": True,
                })

        # EOS까지 정상 수신했지만 Google STT가 interim만 반환한 경우다. 중간
        # 인식으로 금융 명령을 실행하지 않고, 연결을 닫기 전에 재시도 이유를
        # 명시적으로 알려 프론트가 일반 연결 장애와 구분하게 한다.
        if not received_final and not client_disconnected:
            await websocket.send_json({
                "type": "error",
                "code": "NO_FINAL_RESULT",
                "message": "음성을 최종 문장으로 확정하지 못했습니다.",
                "retryable": True,
            })

    except WebSocketDisconnect:
        pass

    except Exception as error:
        # 인식이 실패해도 연결을 그냥 끊지 않는다. 화면을 보지 않는 사용자에게는
        # "왜 멈췄는지" 알 방법이 응답뿐이다.
        try:
            await websocket.send_json({
                "type": "error",
                "code": "STT_PROVIDER_ERROR",
                "message": f"{type(error).__name__}: {error}",
                "retryable": True,
            })
        except Exception:
            pass

    finally:
        await audio_stream.close()
        pump.cancel()
        try:
            await websocket.close()
        except Exception:
            pass
