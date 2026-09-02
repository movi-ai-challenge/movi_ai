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
    다만 **null 은 "모름"이 아니라 "신뢰할 수 없음"으로 읽힌다** — 백엔드는
    신뢰도가 0.80 미만인 엔티티를 버리고 사용자에게 되묻는다. 전부 null 로
    내려보내면 사용자가 아무리 정확히 말해도 재질문만 반복된다.

    값은 intent_confidence 와 같은 성격이다. **모델 자기보고이며 보정된
    수치가 아니다.** 그래도 내려보내는 편이 낫다고 판단한 이유는, 백엔드가
    되묻기 여부를 판단할 다른 근거가 없기 때문이다. 대신 확신이 낮으면 낮은
    값을 그대로 실어 보내 백엔드가 되묻게 한다 — 값을 높여 되묻기를 건너뛰게
    하는 것이 실제 위험이다.

    뽑지 못한 엔티티에는 넣지 않는다(null).
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
