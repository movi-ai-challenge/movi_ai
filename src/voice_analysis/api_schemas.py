from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


MAX_TRANSCRIPT_LENGTH = 2_000
FollowUpField = Literal[
    "recipient_name",
    "recipient_bank",
    "recipient_account",
    "amount",
]


class TranscriptRequest(BaseModel):
    """STT 최종 문장을 받는 요청의 공통 검증 규칙."""

    transcript: str = Field(
        min_length=1,
        max_length=MAX_TRANSCRIPT_LENGTH,
        description="Google STT 최종 인식 결과",
    )

    @field_validator("transcript")
    @classmethod
    def validate_transcript(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("transcript는 공백일 수 없습니다.")
        return stripped


# ============================================================
# 최초 Voice 분석 Request
# ============================================================

class VoiceAnalyzeRequest(TranscriptRequest):
    """최초 음성 명령 분석 요청."""


# ============================================================
# Follow-up Request
# ============================================================

class VoiceFollowUpRequest(TranscriptRequest):

    requested_field: FollowUpField = Field(
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


class VoiceFollowUpResponse(BaseModel):
    status: Literal["analyzed", "parse_failed"]

    transcript: str

    entities: dict[str, Any] = Field(
        default_factory=dict
    )
