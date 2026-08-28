from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ============================================================
# 백엔드 내부 API 계약 스키마
#
# 기준: movi_backend/docs/ai-api-contract.md v1.0
#
# 이 파일의 필드명은 camelCase 다. 계약이 그렇게 정하고 있고,
# 백엔드 VoiceAnalysisResponse 레코드가 그대로 역직렬화한다.
# 프로젝트 내부 스키마(schemas.py)는 snake_case 를 유지한다.
# ============================================================


VoiceIntent = Literal[
    "BALANCE",
    "TRANSFER",
    "HISTORY",
    "CONFIRM",
    "CANCEL",
    "UNKNOWN",
]


VoiceSlot = Literal[
    "AMOUNT",
    "RECIPIENT",
    "SOURCE_ACCOUNT_ALIAS",
    "BANK_NAME",
    "START_DATE",
    "END_DATE",
]


class VoiceEntities(BaseModel):
    """
    백엔드 VoiceEntities 레코드와 1:1 대응.

    계약 규칙: 값이 없어도 키를 생략하지 않고 null 을 넣는다.
    """

    amount: Optional[int] = None
    recipient: Optional[str] = None
    sourceAccountAlias: Optional[str] = None
    bankName: Optional[str] = None

    # ISO-8601 date (yyyy-MM-dd). 백엔드가 LocalDate 로 파싱한다.
    startDate: Optional[str] = None
    endDate: Optional[str] = None


class VoiceEntityConfidences(BaseModel):
    """
    엔티티별 신뢰도.

    백엔드 검증기는 null 을 허용하고, 값이 있으면 0~1 범위만 확인한다.
    현재 per-entity 신뢰도는 보정된 근거가 없어 null 로 둔다.
    지어낸 숫자를 넣으면 백엔드 신뢰도 정책이 잘못된 근거로 동작한다.
    """

    amount: Optional[float] = None
    recipient: Optional[float] = None
    sourceAccountAlias: Optional[float] = None
    bankName: Optional[float] = None
    startDate: Optional[float] = None
    endDate: Optional[float] = None


class VoiceAnalyzeContractResponse(BaseModel):
    """
    POST /internal/v1/voice/analyze 정상 응답.
    """

    requestId: str
    voiceSessionId: int

    transcript: str

    sttConfidence: float = Field(ge=0.0, le=1.0)

    intent: VoiceIntent

    intentConfidence: float = Field(ge=0.0, le=1.0)

    entities: VoiceEntities
    entityConfidences: VoiceEntityConfidences

    detectedMissingEntities: list[VoiceSlot] = Field(
        default_factory=list
    )

    processingMs: int = Field(ge=0)


# ============================================================
# 오류 외피
# ============================================================

class ContractErrorBody(BaseModel):
    code: str
    message: str
    retryable: bool


class ContractErrorResponse(BaseModel):
    """
    계약 1절의 내부 오류 외피.

    message 는 사용자에게 직접 전달되지 않는다.
    백엔드가 ErrorCode.voiceMessage 로 변환한다.
    """

    requestId: Optional[str] = None
    error: ContractErrorBody
