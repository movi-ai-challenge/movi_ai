from __future__ import annotations

from fastapi import (
    FastAPI,
    HTTPException,
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