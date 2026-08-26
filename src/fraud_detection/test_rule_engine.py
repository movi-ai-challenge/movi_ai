from __future__ import annotations

import json

from .rule_engine import (
    FraudRuleEngine,
)


def test_normal_transaction():

    engine = (
        FraudRuleEngine()
    )


    features = {

        "amount_ratio": 1.2,

        "amount_zscore": 0.5,

        "is_night": 0,

        "new_recipient": 0,

        "unusual_medium": 0,

        "same_bank": 1,

        "same_day_transaction_count": 1,

        "same_time_bucket_count": 0,
    }


    result = (
        engine.evaluate(
            features
        )
    )


    print()
    print("=" * 70)
    print("NORMAL TRANSACTION")
    print("=" * 70)

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )


    assert (
        result[
            "rule_score"
        ]
        ==
        0
    )


    assert (
        len(
            result[
                "triggered_rules"
            ]
        )
        ==
        0
    )


    print()
    print(
        "[PASS] Normal Transaction"
    )


def test_high_risk_transaction():

    engine = (
        FraudRuleEngine()
    )


    features = {

        "amount_ratio": 136.36,

        "amount_zscore": 394.44,

        "is_night": 1,

        "new_recipient": 1,

        "unusual_medium": 1,

        "same_bank": 0,

        "same_day_transaction_count": 0,

        "same_time_bucket_count": 0,
    }


    result = (
        engine.evaluate(
            features
        )
    )


    print()
    print("=" * 70)
    print("HIGH RISK TRANSACTION")
    print("=" * 70)

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )


    print()
    print(
        "Rule Score:",
        result[
            "rule_score"
        ]
    )


    print(
        "Triggered Rules:",
        result[
            "triggered_rules"
        ]
    )


    assert (
        "HIGH_AMOUNT_RATIO"
        in
        result[
            "triggered_rules"
        ]
    )


    assert (
        "EXTREME_AMOUNT_ZSCORE"
        in
        result[
            "triggered_rules"
        ]
    )


    assert (
        "NIGHT_TRANSACTION"
        in
        result[
            "triggered_rules"
        ]
    )


    assert (
        "NEW_RECIPIENT"
        in
        result[
            "triggered_rules"
        ]
    )


    assert (
        "UNUSUAL_MEDIUM"
        in
        result[
            "triggered_rules"
        ]
    )


    assert (
        result[
            "rule_score"
        ]
        >
        0
    )


    print()
    print(
        "[PASS] High Risk Transaction"
    )


if __name__ == "__main__":

    test_normal_transaction()

    test_high_risk_transaction()