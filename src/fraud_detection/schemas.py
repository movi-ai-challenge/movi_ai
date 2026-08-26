from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ============================================================
# Transaction
# ============================================================

class TransactionData(BaseModel):
    """
    FDS에서 사용하는 단일 거래 데이터.

    Spring Backend에서 전달하는 API DTO와
    Python 내부 Feature Engineering 사이의
    공통 입력 구조로 사용한다.
    """

    # 거래 식별자
    transaction_id: Optional[str] = None

    # 출금 계좌
    sender_account: str

    # 입금 계좌
    receiver_account: str

    # 출금 금융기관
    sender_bank: str

    # 입금 금융기관
    receiver_bank: str

    # 거래 유형
    #
    # 예:
    # transfer
    # withdrawal
    # deposit
    transaction_type: Optional[str] = None

    # 거래 금액
    amount: float = Field(
        gt=0,
        description="거래 금액",
    )

    # 거래 발생 시간
    #
    # ISO-8601 형식으로 Backend에서 전달
    #
    # Example:
    # 2026-08-26T00:30:00
    transaction_datetime: datetime

    # 거래 매체
    #
    # 예:
    # MOBILE
    # WEB
    # ATM
    medium: Optional[str] = None


# ============================================================
# Fraud Detection Request
# ============================================================

class FraudDetectionRequest(BaseModel):
    """
    Spring → Python FDS 요청.

    current_transaction:
        현재 탐지할 거래

    history:
        현재 거래보다 이전에 발생한
        동일 출금 계좌의 과거 거래 목록
    """

    current_transaction: TransactionData

    history: list[TransactionData] = Field(
        default_factory=list
    )


# ============================================================
# Fraud Detection Response
# ============================================================

class FraudDetectionResponse(BaseModel):
    """
    Python FDS → Spring 응답.

    현재 단계에서는 Isolation Forest 결과 중심.

    Rule Engine과 Final Risk Score가 구현되면
    이후 필드를 확장한다.
    """

    transaction_id: Optional[str] = None

    # Isolation Forest anomaly score
    #
    # 클수록 비정상 거래에 가깝도록 Python에서 변환
    anomaly_score: float

    # 현재 서비스 판단 Threshold
    threshold: float

    # threshold 기준 이상 거래 여부
    is_anomaly: bool

    # 사용한 모델
    model: str = "isolation_forest"

    # --------------------------------------------------------
    # 이후 Rule Engine 구현 시 사용
    # --------------------------------------------------------

    rule_score: Optional[float] = None

    final_risk_score: Optional[float] = None

    risk_level: Optional[str] = None

    triggered_rules: list[str] = Field(
        default_factory=list
    )