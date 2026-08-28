from __future__ import annotations

import re
from datetime import date
from typing import Any, Optional

from .contract_schemas import (
    VoiceEntities,
    VoiceEntityConfidences,
    VoiceSlot,
)


# ============================================================
# Intent 매핑
#
# 내부 intent(snake_case) -> 계약 intent(대문자)
#
# 계약 MVP intent 는 6개뿐이다(ai-api-contract.md 2.2).
# 내부에만 있는 intent 는 UNKNOWN 으로 내린다. 억지로 가까운 값에
# 붙이면 백엔드가 사용자가 요청하지 않은 금융 동작을 실행할 수 있다.
# ============================================================

INTENT_MAP: dict[str, str] = {
    "transfer_money": "TRANSFER",
    "check_history": "HISTORY",
    "check_balance": "BALANCE",
    "confirm": "CONFIRM",
    "cancel": "CANCEL",

    # "아니야" 는 진행 중이던 요청을 물리는 발화다.
    # 계약에 DENY 가 없고 취소가 가장 가까운 의미다.
    "deny": "CANCEL",

    # MVP 계약 비대상
    "check_savings": "UNKNOWN",
    "read_screen": "UNKNOWN",
    "unknown": "UNKNOWN",
}


# ============================================================
# 필수 슬롯
#
# integration-spec.md 6.3
#   amount, recipient 는 필수
#   sourceAccountAlias 는 선택 (기본 계좌 사용)
# ============================================================

REQUIRED_SLOTS: dict[str, list[tuple[str, str]]] = {
    "TRANSFER": [
        ("amount", "AMOUNT"),
        ("recipient", "RECIPIENT"),
    ],
}


_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def map_intent(internal_intent: str) -> str:
    return INTENT_MAP.get(internal_intent, "UNKNOWN")


def _clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    return text


def _clean_amount(value: Any) -> Optional[int]:
    if value is None:
        return None

    try:
        amount = int(value)
    except (TypeError, ValueError):
        return None

    if amount < 0:
        return None

    return amount


def _clean_date(value: Any) -> Optional[str]:
    """
    백엔드가 LocalDate 로 파싱하므로 yyyy-MM-dd 만 통과시킨다.

    모델이 "어제", "2026/08/01" 같은 값을 낼 수 있는데
    그대로 보내면 백엔드 역직렬화가 깨진다.
    """

    text = _clean_text(value)

    if text is None:
        return None

    if not _ISO_DATE.match(text):
        return None

    try:
        date.fromisoformat(text)
    except ValueError:
        return None

    return text


def map_entities(
    internal_entities: dict[str, Any],
) -> VoiceEntities:
    """
    내부 엔티티 -> 계약 엔티티.

    내부 스키마는 송금/조회를 위해 은행·계좌를 각각 나눠 갖지만
    계약은 bankName 하나뿐이다. 송금이면 수취인 은행이,
    조회면 대상 계좌 은행이 그 자리에 온다.
    """

    entities = internal_entities or {}

    bank_name = (
        _clean_text(entities.get("recipient_bank"))
        or _clean_text(entities.get("bank"))
    )

    source_alias = (
        _clean_text(entities.get("source_account"))
        or _clean_text(entities.get("source_bank"))
    )

    return VoiceEntities(
        amount=_clean_amount(entities.get("amount")),
        recipient=_clean_text(entities.get("recipient_name")),
        sourceAccountAlias=source_alias,
        bankName=bank_name,
        startDate=_clean_date(entities.get("date_from")),
        endDate=_clean_date(entities.get("date_to")),
    )


def detect_missing_slots(
    *,
    intent: str,
    entities: VoiceEntities,
) -> list[VoiceSlot]:
    """
    현재 발화만 기준으로 비어 있는 필수 슬롯을 알린다.

    AI 는 이전 턴의 슬롯을 알지 못한다. 백엔드가 세션에 저장된 값과
    병합한 뒤 최종 판단하므로, 이 값은 참고용이다.
    """

    required = REQUIRED_SLOTS.get(intent, [])

    missing: list[VoiceSlot] = []

    for field_name, slot_name in required:

        if getattr(entities, field_name, None) is None:
            missing.append(slot_name)

    return missing


def empty_confidences() -> VoiceEntityConfidences:
    return VoiceEntityConfidences()
