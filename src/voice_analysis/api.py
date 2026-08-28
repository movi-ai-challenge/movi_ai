from __future__ import annotations

import json
import time
from typing import Optional

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from .contract_schemas import (
    ContractErrorBody,
    ContractErrorResponse,
    VoiceAnalyzeContractResponse,
)

from .contract_mapper import (
    detect_missing_slots,
    empty_confidences,
    map_entities,
    map_intent,
)

from .stt_batch_service import (
    SttBatchService,
    SttError,
)

from .api_schemas import (
    VoiceAnalyzeRequest,
    VoiceFollowUpRequest,
    VoiceAnalyzeResponse,
)

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

    version="0.1.0",
)


# ============================================================
# Service
# ============================================================

voice_service = (
    VoiceAnalysisService()
)


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
    }


# ============================================================
# Health
# ============================================================

@app.get("/health")
def health():

    return {

        "status": "ok",

        "service": (
            "voice-analysis"
        ),
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

        result = (
            voice_service.analyze(
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


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                f"{type(error).__name__}: "
                f"{str(error)}"
            ),
        ) from error


# ============================================================
# Follow-up 분석
# ============================================================

@app.post(
    "/api/v1/voice/follow-up",
)
def analyze_follow_up(
    request: VoiceFollowUpRequest,
):

    try:

        return (
            voice_service
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


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                f"{type(error).__name__}: "
                f"{str(error)}"
            ),
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
async def analyze_voice_internal(
    audio: UploadFile = File(
        description="WebM/Opus 또는 WAV. 최대 5MB",
    ),
    requestId: str = Form(...),
    voiceSessionId: int = Form(...),
    expectedIntent: Optional[str] = Form(default=None),
    expectedSlots: Optional[str] = Form(default=None),
):
    started = time.perf_counter()

    # --------------------------------------------------------
    # 1. STT
    # --------------------------------------------------------

    try:
        audio_bytes = await audio.read()

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
            intent, confidence, raw_entities = _analyze_follow_up(
                field_name=follow_up_field,
                transcript=transcript,
                expected_intent=expectedIntent,
            )
        else:
            analysis = voice_service.analyze_command(transcript)

            intent = map_intent(analysis.intent)
            confidence = float(analysis.intent_confidence)
            raw_entities = analysis.entities.model_dump()

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
        entityConfidences=empty_confidences(),
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

    if parsed.success:
        entity_key = _FIELD_TO_ENTITY.get(
            parsed.field_name,
            parsed.field_name,
        )
        entities[entity_key] = parsed.value
        confidence = 0.9
    else:
        confidence = 0.3

    return intent, confidence, entities


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
