from typing import Literal, Optional
from pydantic import BaseModel, Field


# ============================================================
# Intent
# ============================================================

IntentType = Literal[
    "read_screen",
    "transfer_money",
    "check_history",
    "check_balance",
    "check_savings",
    "confirm",
    "deny",
    "cancel",
    "unknown",
]


# ============================================================
# Entity
# ============================================================

class RequirementEntities(BaseModel):
    """
    사용자 음성 명령에서 추출할 금융 Entity.
    """

    # 송금 관련
    recipient_name: Optional[str] = None
    recipient_bank: Optional[str] = None
    recipient_account: Optional[str] = None
    amount: Optional[int] = Field(
        default=None,
        ge=0,
    )

    # 사용자의 출금 계좌
    source_bank: Optional[str] = None
    source_account: Optional[str] = None

    # 거래내역 조회 관련
    date_from: Optional[str] = None
    date_to: Optional[str] = None

    # 특정 조회 계좌
    bank: Optional[str] = None
    account: Optional[str] = None


class RequirementEntityConfidences(BaseModel):
    """
    엔티티별 확신도. 필드 이름은 RequirementEntities 와 1:1 로 맞춘다.

    intent_confidence 와 마찬가지로 **모델 자기보고 값이며 보정(calibration)된
    수치가 아니다.** 그럼에도 내려보내는 이유는, 백엔드가 엔티티마다 신뢰도를 보고
    낮으면 되묻는 정책을 쓰기 때문이다. 값을 비워 두면 백엔드는 "신뢰할 수 없음"으로
    읽어 모든 엔티티를 버리고, 사용자가 아무리 정확히 말해도 재질문만 반복된다.

    값을 지어내지 않는 것이 중요하다. 뽑지 못한 엔티티에는 넣지 않는다(null).
    확신이 낮으면 낮은 값을 그대로 넣어 백엔드가 되묻게 하는 것이 정상 동작이다.
    """

    recipient_name: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    recipient_bank: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    recipient_account: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    amount: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    source_bank: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    source_account: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    date_from: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    date_to: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    bank: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    account: Optional[float] = Field(default=None, ge=0.0, le=1.0)


# ============================================================
# Requirement Result
# ============================================================

class RequirementAnalysis(BaseModel):
    """
    GPT 요구사항 분석 최종 결과.
    """

    intent: IntentType

    # 0.0 ~ 1.0. ai-api-contract.md 의 intentConfidence 로 나간다.
    # 모델 자기보고 값이며 보정(calibration)된 수치가 아니다.
    intent_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )

    entities: RequirementEntities = Field(
        default_factory=RequirementEntities
    )

    # 엔티티별 확신도. entities 와 같은 필드 이름을 쓴다.
    entity_confidences: RequirementEntityConfidences = Field(
        default_factory=RequirementEntityConfidences
    )

    missing_fields: list[str] = Field(
        default_factory=list
    )

    original_text: str

    # 분석 결과에 대한 짧은 설명.
    # 디버깅용이며 실제 금융 실행 판단에는 사용하지 않는다.
    normalized_command: Optional[str] = None

class FollowUpEntityResult(BaseModel):
    """
    누락 필드에 대한 사용자 추가 답변 분석 결과.
    """

    field_name: str

    # 문자열/숫자 모두 받을 수 있도록 구성
    value: Optional[str | int] = None

    # 사용자의 답변에서 해당 값을
    # 정상적으로 추출했는지 여부
    success: bool

    original_text: str