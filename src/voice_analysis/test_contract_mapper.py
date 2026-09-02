from .contract_mapper import (
    detect_missing_slots,
    map_entities,
    map_entity_confidences,
    map_intent,
)


# ============================================================
# Intent 매핑
# ============================================================

def test_intent_maps_to_contract_vocabulary():

    assert map_intent("transfer_money") == "TRANSFER"
    assert map_intent("check_history") == "HISTORY"
    assert map_intent("check_balance") == "BALANCE"
    assert map_intent("confirm") == "CONFIRM"
    assert map_intent("cancel") == "CANCEL"

    # "아니야" 는 진행 중이던 요청을 물리는 발화
    assert map_intent("deny") == "CANCEL"


def test_unsupported_intent_falls_back_to_unknown():
    """
    계약 MVP 밖의 intent 를 가까운 값에 붙이면
    백엔드가 요청하지 않은 금융 동작을 실행할 수 있다.
    """

    assert map_intent("check_savings") == "UNKNOWN"
    assert map_intent("read_screen") == "UNKNOWN"
    assert map_intent("unknown") == "UNKNOWN"
    assert map_intent("존재하지_않는_intent") == "UNKNOWN"


# ============================================================
# Entity 매핑
# ============================================================

def test_transfer_entities_map_to_contract_keys():

    entities = map_entities({
        "recipient_name": "김민수",
        "recipient_bank": "국민은행",
        "amount": 50000,
        "source_account": "주거래통장",
    })

    assert entities.recipient == "김민수"
    assert entities.amount == 50000
    assert entities.bankName == "국민은행"
    assert entities.sourceAccountAlias == "주거래통장"


def test_history_bank_falls_into_bank_name():
    """
    내부 스키마는 은행을 recipient_bank / bank 로 나눠 갖지만
    계약에는 bankName 하나뿐이다.
    """

    entities = map_entities({"bank": "신한은행"})

    assert entities.bankName == "신한은행"


def test_missing_values_become_null_not_omitted():
    """
    계약 규칙: 키를 생략하지 않고 값에 null 을 쓴다.
    """

    entities = map_entities({})

    dumped = entities.model_dump()

    assert set(dumped) == {
        "amount",
        "recipient",
        "sourceAccountAlias",
        "bankName",
        "startDate",
        "endDate",
    }
    assert all(value is None for value in dumped.values())


def test_blank_strings_are_treated_as_missing():

    entities = map_entities({
        "recipient_name": "   ",
        "recipient_bank": "",
    })

    assert entities.recipient is None
    assert entities.bankName is None


def test_non_iso_dates_are_dropped():
    """
    백엔드가 LocalDate 로 파싱하므로 yyyy-MM-dd 가 아니면
    보내지 않는다. 그대로 보내면 역직렬화가 깨진다.
    """

    entities = map_entities({
        "date_from": "어제",
        "date_to": "2026/08/01",
    })

    assert entities.startDate is None
    assert entities.endDate is None


def test_iso_dates_pass_through():

    entities = map_entities({
        "date_from": "2026-08-01",
        "date_to": "2026-08-28",
    })

    assert entities.startDate == "2026-08-01"
    assert entities.endDate == "2026-08-28"


def test_impossible_date_is_dropped():

    entities = map_entities({"date_from": "2026-02-31"})

    assert entities.startDate is None


def test_negative_amount_is_dropped():

    assert map_entities({"amount": -1000}).amount is None


def test_amount_accepts_numeric_string():

    assert map_entities({"amount": "50000"}).amount == 50000


# ============================================================
# 누락 슬롯
# ============================================================

def test_transfer_reports_missing_required_slots():

    entities = map_entities({"recipient_name": "김민수"})

    missing = detect_missing_slots(
        intent="TRANSFER",
        entities=entities,
    )

    assert missing == ["AMOUNT"]


def test_transfer_with_all_required_slots_reports_nothing():

    entities = map_entities({
        "recipient_name": "김민수",
        "amount": 50000,
    })

    missing = detect_missing_slots(
        intent="TRANSFER",
        entities=entities,
    )

    assert missing == []


def test_source_account_alias_is_not_required():
    """
    integration-spec 6.3: sourceAccountAlias 는 선택.
    없으면 기본 계좌를 쓴다.
    """

    entities = map_entities({
        "recipient_name": "김민수",
        "amount": 50000,
    })

    assert entities.sourceAccountAlias is None
    assert detect_missing_slots(
        intent="TRANSFER",
        entities=entities,
    ) == []


def test_non_transfer_intents_have_no_required_slots():

    empty = map_entities({})

    for intent in ("BALANCE", "HISTORY", "CONFIRM", "CANCEL", "UNKNOWN"):
        assert detect_missing_slots(
            intent=intent,
            entities=empty,
        ) == []


# ============================================================
# 엔티티 신뢰도 매핑
#
# 백엔드는 신뢰도가 없거나 0.80 미만인 엔티티를 버리고 되묻는다.
# 전부 null 로 내려보내면 사용자가 정확히 말해도 재질문만 반복된다.
# ============================================================

def _confidences(internal_entities, internal_confidences):
    entities = map_entities(internal_entities)
    return entities, map_entity_confidences(
        internal_entities,
        internal_confidences,
        entities,
    )


def test_뽑은_엔티티의_신뢰도를_그대로_싣는다():

    entities, confidences = _confidences(
        {"amount": 10000, "recipient_name": "엄마"},
        {"amount": 0.98, "recipient_name": 0.95},
    )

    assert entities.amount == 10000
    assert entities.recipient == "엄마"
    assert confidences.amount == 0.98
    assert confidences.recipient == 0.95


def test_낮은_신뢰도를_올리지_않는다():
    """백엔드가 되묻게 하는 것이 정상 동작이다."""

    _, confidences = _confidences(
        {"amount": 20000, "recipient_name": "어머님"},
        {"amount": 0.42, "recipient_name": 0.51},
    )

    assert confidences.amount == 0.42
    assert confidences.recipient == 0.51


def test_뽑지_못한_엔티티에는_신뢰도를_넣지_않는다():

    entities, confidences = _confidences(
        {"recipient_name": "엄마"},
        {"recipient_name": 0.95},
    )

    assert entities.amount is None
    assert confidences.amount is None
    assert confidences.recipient == 0.95


def test_값이_정리_과정에서_떨어지면_신뢰도도_함께_버린다():
    """
    형식이 틀린 날짜는 map_entities 가 버린다. 신뢰도만 남으면 백엔드는
    없는 정보를 있다고 읽는다.
    """

    entities, confidences = _confidences(
        {"date_from": "2026/08/01", "amount": -100},
        {"date_from": 0.9, "amount": 0.9},
    )

    assert entities.startDate is None
    assert entities.amount is None
    assert confidences.startDate is None
    assert confidences.amount is None


def test_범위를_벗어난_신뢰도는_버린다():

    _, confidences = _confidences(
        {"amount": 10000, "recipient_name": "엄마"},
        {"amount": 1.7, "recipient_name": "높음"},
    )

    assert confidences.amount is None
    assert confidences.recipient is None


def test_신뢰도가_없으면_null_로_둔다():
    """값만 있고 신뢰도가 없으면 백엔드가 되묻는다 - 안전한 방향이다."""

    entities, confidences = _confidences(
        {"amount": 10000, "recipient_name": "엄마"},
        {},
    )

    assert entities.amount == 10000
    assert confidences.amount is None


def test_bankName_은_선택된_필드의_신뢰도를_따라간다():
    """
    recipient_bank 와 bank 가 모두 있으면 recipient_bank 가 이긴다.
    신뢰도가 진 쪽을 따라가면 백엔드가 엉뚱한 근거로 판단한다.
    """

    entities, confidences = _confidences(
        {"recipient_bank": "국민은행", "bank": "신한은행"},
        {"recipient_bank": 0.93, "bank": 0.40},
    )

    assert entities.bankName == "국민은행"
    assert confidences.bankName == 0.93


def test_bankName_이_대체_필드에서_오면_그_신뢰도를_쓴다():

    entities, confidences = _confidences(
        {"bank": "신한은행"},
        {"bank": 0.88},
    )

    assert entities.bankName == "신한은행"
    assert confidences.bankName == 0.88


def test_sourceAccountAlias_도_선택된_필드를_따라간다():

    entities, confidences = _confidences(
        {"source_bank": "국민은행"},
        {"source_account": 0.99, "source_bank": 0.61},
    )

    assert entities.sourceAccountAlias == "국민은행"
    # source_account 는 비어 있어 선택되지 않았다. 그 신뢰도를 쓰면 안 된다.
    assert confidences.sourceAccountAlias == 0.61


def test_신뢰도_입력이_없어도_깨지지_않는다():

    entities, confidences = _confidences({"amount": 10000}, None)

    assert entities.amount == 10000
    assert confidences.amount is None
