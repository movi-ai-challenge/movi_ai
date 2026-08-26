from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math


# ============================================================
# Rule Result
# ============================================================

@dataclass
class RuleResult:

    name: str

    triggered: bool

    score: float

    reason: str


# ============================================================
# Rule Config
# ============================================================

RULE_CONFIG = {

    # 평소 거래 대비 5배 이상
    "HIGH_AMOUNT_RATIO": {
        "threshold": 5.0,
        "score": 25.0,
    },

    # 과거 평균 대비 5 표준편차 이상
    "EXTREME_AMOUNT_ZSCORE": {
        "threshold": 5.0,
        "score": 20.0,
    },

    "NIGHT_TRANSACTION": {
        "score": 15.0,
    },

    "NEW_RECIPIENT": {
        "score": 15.0,
    },

    "UNUSUAL_MEDIUM": {
        "score": 15.0,
    },

    "CROSS_BANK": {
        "score": 5.0,
    },

    # 당일 이미 3건 이상 존재
    "REPEATED_SAME_DAY": {
        "threshold": 3,
        "score": 10.0,
    },

    # 동일 시간 Bucket에서 이미 2건 이상
    "REPEATED_TIME_BUCKET": {
        "threshold": 2,
        "score": 10.0,
    },
}


# ============================================================
# Utility
# ============================================================

def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    if value is None:

        return default


    try:

        number = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return default


    if math.isnan(
        number
    ):

        return default


    if math.isinf(
        number
    ):

        return default


    return number


def safe_int(
    value: Any,
    default: int = 0,
) -> int:

    try:

        return int(
            safe_float(
                value,
                default,
            )
        )

    except (
        TypeError,
        ValueError,
    ):

        return default


# ============================================================
# Rule Engine
# ============================================================

class FraudRuleEngine:
    """
    MOVI Rule-based Fraud Detection Engine.

    Input
    -----
    Feature Engineering 결과 중
    현재 거래 1건.

    Output
    ------
    rule_score
    triggered_rules
    rule_details
    """

    def evaluate(
        self,
        features: dict[str, Any],
    ) -> dict[str, Any]:

        results: list[RuleResult] = []


        # ====================================================
        # 1. HIGH_AMOUNT_RATIO
        # ====================================================

        amount_ratio = safe_float(
            features.get(
                "amount_ratio"
            )
        )

        threshold = (
            RULE_CONFIG[
                "HIGH_AMOUNT_RATIO"
            ][
                "threshold"
            ]
        )

        triggered = (
            amount_ratio
            >=
            threshold
        )

        results.append(

            RuleResult(

                name=(
                    "HIGH_AMOUNT_RATIO"
                ),

                triggered=(
                    triggered
                ),

                score=(
                    RULE_CONFIG[
                        "HIGH_AMOUNT_RATIO"
                    ][
                        "score"
                    ]
                    if triggered
                    else 0.0
                ),

                reason=(
                    f"평소 거래 금액 대비 "
                    f"{amount_ratio:.2f}배"
                ),
            )
        )


        # ====================================================
        # 2. EXTREME_AMOUNT_ZSCORE
        # ====================================================

        amount_zscore = safe_float(
            features.get(
                "amount_zscore"
            )
        )

        threshold = (
            RULE_CONFIG[
                "EXTREME_AMOUNT_ZSCORE"
            ][
                "threshold"
            ]
        )

        triggered = (
            abs(
                amount_zscore
            )
            >=
            threshold
        )

        results.append(

            RuleResult(

                name=(
                    "EXTREME_AMOUNT_ZSCORE"
                ),

                triggered=(
                    triggered
                ),

                score=(
                    RULE_CONFIG[
                        "EXTREME_AMOUNT_ZSCORE"
                    ][
                        "score"
                    ]
                    if triggered
                    else 0.0
                ),

                reason=(
                    f"거래 금액 Z-Score "
                    f"{amount_zscore:.2f}"
                ),
            )
        )


        # ====================================================
        # 3. NIGHT_TRANSACTION
        # ====================================================

        is_night = safe_int(
            features.get(
                "is_night"
            )
        )

        triggered = (
            is_night
            == 1
        )

        results.append(

            RuleResult(

                name=(
                    "NIGHT_TRANSACTION"
                ),

                triggered=(
                    triggered
                ),

                score=(
                    RULE_CONFIG[
                        "NIGHT_TRANSACTION"
                    ][
                        "score"
                    ]
                    if triggered
                    else 0.0
                ),

                reason=(
                    "심야 시간대 거래"
                ),
            )
        )


        # ====================================================
        # 4. NEW_RECIPIENT
        # ====================================================

        new_recipient = safe_int(
            features.get(
                "new_recipient"
            )
        )

        triggered = (
            new_recipient
            == 1
        )

        results.append(

            RuleResult(

                name=(
                    "NEW_RECIPIENT"
                ),

                triggered=(
                    triggered
                ),

                score=(
                    RULE_CONFIG[
                        "NEW_RECIPIENT"
                    ][
                        "score"
                    ]
                    if triggered
                    else 0.0
                ),

                reason=(
                    "이전 거래 이력이 없는 신규 수취인"
                ),
            )
        )


        # ====================================================
        # 5. UNUSUAL_MEDIUM
        # ====================================================

        unusual_medium = safe_int(
            features.get(
                "unusual_medium"
            )
        )

        triggered = (
            unusual_medium
            == 1
        )

        results.append(

            RuleResult(

                name=(
                    "UNUSUAL_MEDIUM"
                ),

                triggered=(
                    triggered
                ),

                score=(
                    RULE_CONFIG[
                        "UNUSUAL_MEDIUM"
                    ][
                        "score"
                    ]
                    if triggered
                    else 0.0
                ),

                reason=(
                    "과거에 사용하지 않던 거래 매체"
                ),
            )
        )


        # ====================================================
        # 6. CROSS_BANK
        # ====================================================

        same_bank = safe_int(
            features.get(
                "same_bank"
            ),
            default=1,
        )

        triggered = (
            same_bank
            == 0
        )

        results.append(

            RuleResult(

                name=(
                    "CROSS_BANK"
                ),

                triggered=(
                    triggered
                ),

                score=(
                    RULE_CONFIG[
                        "CROSS_BANK"
                    ][
                        "score"
                    ]
                    if triggered
                    else 0.0
                ),

                reason=(
                    "출금은행과 입금은행이 다른 거래"
                ),
            )
        )


        # ====================================================
        # 7. REPEATED_SAME_DAY
        # ====================================================

        same_day_count = safe_int(
            features.get(
                "same_day_transaction_count"
            )
        )

        threshold = (
            RULE_CONFIG[
                "REPEATED_SAME_DAY"
            ][
                "threshold"
            ]
        )

        triggered = (
            same_day_count
            >=
            threshold
        )

        results.append(

            RuleResult(

                name=(
                    "REPEATED_SAME_DAY"
                ),

                triggered=(
                    triggered
                ),

                score=(
                    RULE_CONFIG[
                        "REPEATED_SAME_DAY"
                    ][
                        "score"
                    ]
                    if triggered
                    else 0.0
                ),

                reason=(
                    f"당일 이전 거래 "
                    f"{same_day_count}건"
                ),
            )
        )


        # ====================================================
        # 8. REPEATED_TIME_BUCKET
        # ====================================================

        time_bucket_count = safe_int(
            features.get(
                "same_time_bucket_count"
            )
        )

        threshold = (
            RULE_CONFIG[
                "REPEATED_TIME_BUCKET"
            ][
                "threshold"
            ]
        )

        triggered = (
            time_bucket_count
            >=
            threshold
        )

        results.append(

            RuleResult(

                name=(
                    "REPEATED_TIME_BUCKET"
                ),

                triggered=(
                    triggered
                ),

                score=(
                    RULE_CONFIG[
                        "REPEATED_TIME_BUCKET"
                    ][
                        "score"
                    ]
                    if triggered
                    else 0.0
                ),

                reason=(
                    "동일 시간대 이전 거래 "
                    f"{time_bucket_count}건"
                ),
            )
        )


        # ====================================================
        # Final Rule Score
        # ====================================================

        rule_score = sum(
            result.score
            for result in results
        )


        # 최대 100으로 제한
        rule_score = min(
            rule_score,
            100.0,
        )


        triggered_rules = [

            result.name

            for result in results

            if result.triggered

        ]


        rule_details = [

            {
                "name": (
                    result.name
                ),

                "triggered": (
                    result.triggered
                ),

                "score": (
                    result.score
                ),

                "reason": (
                    result.reason
                ),
            }

            for result in results
        ]


        return {

            "rule_score": (
                rule_score
            ),

            "triggered_rules": (
                triggered_rules
            ),

            "rule_details": (
                rule_details
            ),
        }