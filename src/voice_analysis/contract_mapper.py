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


# ============================================================
# 한 계약 필드에 내부 필드가 여러 개 대응하는 경우
#
# 내부 스키마는 송금/조회를 위해 은행·계좌를 각각 나눠 갖지만
# 계약은 bankName 하나뿐이다. 송금이면 수취인 은행이,
# 조회면 대상 계좌 은행이 그 자리에 온다.
#
# 신뢰도도 "이긴 필드"의 것을 따라가야 하므로, 어느 필드가 선택됐는지를
# 값과 함께 돌려주는 헬퍼를 둔다. 값과 신뢰도가 서로 다른 필드에서 오면
# 백엔드가 엉뚱한 근거로 되묻기를 건너뛴다.
# ============================================================

_BANK_NAME_KEYS = ("recipient_bank", "bank")
_SOURCE_ALIAS_KEYS = ("source_account", "source_bank")


def _pick_text(
    entities: dict[str, Any],
    keys: tuple[str, ...],
) -> tuple[Optional[str], Optional[str]]:
    """앞의 키부터 보고 처음 비어 있지 않은 값을 (값, 키) 로 돌려준다."""

    for key in keys:
        text = _clean_text(entities.get(key))

        if text is not None:
            return text, key

    return None, None


def map_entities(
    internal_entities: dict[str, Any],
) -> VoiceEntities:
    """
    내부 엔티티 -> 계약 엔티티.
    """

    entities = internal_entities or {}

    bank_name, _ = _pick_text(entities, _BANK_NAME_KEYS)
    source_alias, _ = _pick_text(entities, _SOURCE_ALIAS_KEYS)

    return VoiceEntities(
        amount=_clean_amount(entities.get("amount")),
        recipient=_clean_text(entities.get("recipient_name")),
        sourceAccountAlias=source_alias,
        bankName=bank_name,
        startDate=_clean_date(entities.get("date_from")),
        endDate=_clean_date(entities.get("date_to")),
    )


def _clean_confidence(value: Any) -> Optional[float]:
    """0.0~1.0 을 벗어나거나 숫자가 아니면 버린다."""

    if value is None:
        return None

    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None

    if confidence < 0.0 or confidence > 1.0:
        return None

    return confidence


def map_entity_confidences(
    internal_entities: dict[str, Any],
    internal_confidences: dict[str, Any],
    mapped_entities: VoiceEntities,
) -> VoiceEntityConfidences:
    """
    내부 엔티티 신뢰도 -> 계약 신뢰도.

    **값이 살아남은 엔티티에만 신뢰도를 붙인다.** 값이 정리 과정에서
    떨어졌는데(형식이 틀린 날짜, 음수 금액 등) 신뢰도만 남으면 백엔드는
    없는 정보를 있다고 읽는다. 반대로 값만 있고 신뢰도가 없으면 백엔드가
    사용자에게 되묻는데, 그쪽이 안전한 방향이라 그대로 둔다.
    """

    entities = internal_entities or {}
    confidences = internal_confidences or {}

    def confidence_if_present(field_value: Any, key: Optional[str]):
        if field_value is None or key is None:
            return None

        return _clean_confidence(confidences.get(key))

    _, bank_key = _pick_text(entities, _BANK_NAME_KEYS)
    _, alias_key = _pick_text(entities, _SOURCE_ALIAS_KEYS)

    return VoiceEntityConfidences(
        amount=confidence_if_present(mapped_entities.amount, "amount"),
        recipient=confidence_if_present(
            mapped_entities.recipient, "recipient_name"
        ),
        sourceAccountAlias=confidence_if_present(
            mapped_entities.sourceAccountAlias, alias_key
        ),
        bankName=confidence_if_present(mapped_entities.bankName, bank_key),
        startDate=confidence_if_present(mapped_entities.startDate, "date_from"),
        endDate=confidence_if_present(mapped_entities.endDate, "date_to"),
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
    """
    신뢰도를 하나도 알 수 없을 때 쓴다.

    이 값을 내려보내면 백엔드는 모든 엔티티를 신뢰할 수 없는 것으로 보고
    되묻는다. 분석 경로가 신뢰도를 낼 수 있다면
    map_entity_confidences() 를 쓴다.
    """

    return VoiceEntityConfidences()
