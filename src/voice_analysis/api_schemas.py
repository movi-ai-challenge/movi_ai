from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ============================================================
# 최초 Voice 분석 Request
# ============================================================

class VoiceAnalyzeRequest(BaseModel):

    transcript: str = Field(
        min_length=1,
        description="Google STT 최종 인식 결과",
    )


# ============================================================
# Follow-up Request
# ============================================================

class VoiceFollowUpRequest(BaseModel):

    transcript: str = Field(
        min_length=1,
        description="추가 정보에 대한 사용자 음성 인식 결과",
    )

    requested_field: str = Field(
        description=(
            "Backend에서 추가로 요청한 Entity 이름"
        ),
    )

    entities: dict[str, Any] = Field(
        default_factory=dict,
        description="기존에 추출된 Entity",
    )


# ============================================================
# Voice Analyze Response
# ============================================================

class VoiceAnalyzeResponse(BaseModel):

    status: str

    intent: Optional[str] = None

    transcript: Optional[str] = None

    entities: dict[str, Any] = Field(
        default_factory=dict
    )

    message: Optional[str] = None