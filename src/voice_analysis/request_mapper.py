from __future__ import annotations

from typing import Any
from uuid import uuid4


# ============================================================
# Request ID
# ============================================================

def create_request_id() -> str:
    """
    하나의 음성 요청을 식별하기 위한 Request ID 생성.

    최초 요청에서 한 번 생성하고
    Follow-up 동안 동일한 ID를 유지한다.
    """

    return f"req-{uuid4()}"


# ============================================================
# Transfer Payload
# ============================================================

def map_transfer_payload(
    entities: dict[str, Any],
) -> dict[str, Any]:
    """
    계좌 이체 Intent의 Entity를 Backend Payload로 변환한다.

    없는 값은 None(null) 상태 그대로 전달한다.
    """

    return {
        "recipient_name": entities.get(
            "recipient_name"
        ),

        "recipient_bank": entities.get(
            "recipient_bank"
        ),

        "recipient_account": entities.get(
            "recipient_account"
        ),

        "amount": entities.get(
            "amount"
        ),

        "source_bank": entities.get(
            "source_bank"
        ),

        "source_account": entities.get(
            "source_account"
        ),
    }


# ============================================================
# Savings Payload
# ============================================================

def map_savings_payload(
    entities: dict[str, Any],
) -> dict[str, Any]:
    """
    현재 가입 중인 적금 조회.

    실제 적금 정보는 Backend DB에서 조회하므로
    현재는 별도 Payload가 필요하지 않다.
    """

    return {}


# ============================================================
# History Payload
# ============================================================

def map_history_payload(
    entities: dict[str, Any],
) -> dict[str, Any]:
    """
    거래내역 조회용 Payload.
    """

    return {
        "date_from": entities.get(
            "date_from"
        ),

        "date_to": entities.get(
            "date_to"
        ),

        "limit": entities.get(
            "limit",
            10,
        ),
    }


# ============================================================
# Backend Request
# ============================================================

def build_backend_request(
    *,
    request_id: str,
    transcript: str,
    intent: str,
    entities: dict[str, Any],
) -> dict[str, Any]:
    """
    Voice AI 분석 결과를 Backend API Request 형식으로 변환한다.

    Python에서는:
        - Intent 분석
        - Entity 추출
        - JSON Mapping

    까지만 담당한다.

    빈 값 / 필수값 검증은 Backend에서 담당한다.
    """

    if intent == "transfer_money":

        payload = map_transfer_payload(
            entities
        )

    elif intent == "check_savings":

        payload = map_savings_payload(
            entities
        )

    elif intent == "check_history":

        payload = map_history_payload(
            entities
        )

    else:

        # 현재 지원하지 않는 Intent가 들어왔을 경우
        # 분석 결과를 그대로 전달할 수 있도록 유지
        payload = entities.copy()


    return {
        "request_id": request_id,
        "intent": intent,
        "transcript": transcript,
        "payload": payload,
    }