from __future__ import annotations

from typing import Any
from uuid import uuid4


def create_request_id() -> str:
    """
    하나의 사용자 요청을 식별하기 위한 ID 생성.

    Multi-turn 동안 동일한 request_id를 유지한다.
    """

    return f"req-{uuid4()}"


def map_transfer_payload(
    entities: dict[str, Any],
) -> dict[str, Any]:

    return {
        "recipient_name": entities.get("recipient_name"),
        "recipient_bank": entities.get("recipient_bank"),
        "recipient_account": entities.get("recipient_account"),
        "amount": entities.get("amount"),
        "source_bank": entities.get("source_bank"),
        "source_account": entities.get("source_account"),
    }


def map_savings_payload(
    entities: dict[str, Any],
) -> dict[str, Any]:

    return {}


def map_history_payload(
    entities: dict[str, Any],
) -> dict[str, Any]:

    return {
        "date_from": entities.get("date_from"),
        "date_to": entities.get("date_to"),
        "limit": entities.get("limit", 10),
    }


def build_backend_request(
    *,
    request_id: str,
    transcript: str,
    intent: str,
    entities: dict[str, Any],
) -> dict[str, Any]:
    """
    Voice AI 분석 결과를 Spring Backend Request로 변환한다.

    빈 값 / 필수값 판단은 Backend 책임이므로
    null 값을 제거하지 않고 그대로 전달한다.
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

        payload = entities.copy()


    return {
        "request_id": request_id,
        "intent": intent,
        "transcript": transcript,
        "payload": payload,
    }