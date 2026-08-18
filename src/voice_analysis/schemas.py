from typing import Literal, Optional
from pydantic import BaseModel, Field


# ============================================================
# Intent
# ============================================================

IntentType = Literal[
    "read_screen",
    "transfer_money",
    "check_history",
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


# ============================================================
# Requirement Result
# ============================================================

class RequirementAnalysis(BaseModel):
    """
    GPT 요구사항 분석 최종 결과.
    """

    intent: IntentType

    entities: RequirementEntities = Field(
        default_factory=RequirementEntities
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