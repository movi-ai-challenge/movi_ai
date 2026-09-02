"""MOVI Rule Engine 단위 및 경계값 테스트.

프로젝트 루트 또는 tests 디렉터리에서 실행할 수 있다.

    python test_rule_engine.py

또는

    python -m unittest -v test_rule_engine.py
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


def _load_rule_engine():
    """프로젝트 배치 방식에 맞춰 Rule Engine 모듈을 불러온다."""

    try:
        from src.fraud_detection.rule_engine import FraudRuleEngine, RULE_CONFIG

        return FraudRuleEngine, RULE_CONFIG
    except ModuleNotFoundError:
        pass

    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent if current_dir.name == "tests" else current_dir

    for module_dir in (project_root / "src", project_root / "upload"):
        if module_dir.exists():
            sys.path.insert(0, str(module_dir))

    try:
        from fraud_detection.rule_engine import FraudRuleEngine, RULE_CONFIG
    except ModuleNotFoundError:
        from fraud_detection.rule_engine import FraudRuleEngine, RULE_CONFIG

    return FraudRuleEngine, RULE_CONFIG


FraudRuleEngine, RULE_CONFIG = _load_rule_engine()


def normal_features(**overrides):
    """어떤 Rule도 발생하지 않는 기본 Feature를 만든다."""

    features = {
        "amount_ratio": 1.0,
        "amount_zscore": 0.0,
        "is_night": 0,
        "new_recipient": 0,
        "unusual_medium": 0,
        "same_bank": 1,
        "same_day_transaction_count": 0,
        "same_time_bucket_count": 0,
    }
    features.update(overrides)
    return features


class FraudRuleEngineTest(unittest.TestCase):
    def setUp(self):
        self.engine = FraudRuleEngine()

    def evaluate(self, **overrides):
        return self.engine.evaluate(normal_features(**overrides))

    def assert_rule_result(self, result, expected_name, expected_score):
        self.assertEqual(result["triggered_rules"], [expected_name])
        self.assertEqual(result["rule_score"], expected_score)
        self.assertEqual(len(result["rule_details"]), len(RULE_CONFIG))

    def test_normal_transaction_triggers_no_rules(self):
        result = self.evaluate()

        self.assertEqual(result["rule_score"], 0.0)
        self.assertEqual(result["triggered_rules"], [])
        self.assertTrue(
            all(not detail["triggered"] for detail in result["rule_details"])
        )

    def test_high_amount_ratio_boundary(self):
        threshold = RULE_CONFIG["HIGH_AMOUNT_RATIO"]["threshold"]

        below = self.evaluate(amount_ratio=threshold - 0.001)
        at_threshold = self.evaluate(amount_ratio=threshold)

        self.assertNotIn("HIGH_AMOUNT_RATIO", below["triggered_rules"])
        self.assert_rule_result(
            at_threshold,
            "HIGH_AMOUNT_RATIO",
            RULE_CONFIG["HIGH_AMOUNT_RATIO"]["score"],
        )

    def test_extreme_amount_zscore_positive_and_negative_boundary(self):
        threshold = RULE_CONFIG["EXTREME_AMOUNT_ZSCORE"]["threshold"]

        below = self.evaluate(amount_zscore=threshold - 0.001)
        positive = self.evaluate(amount_zscore=threshold)
        negative = self.evaluate(amount_zscore=-threshold)

        self.assertNotIn("EXTREME_AMOUNT_ZSCORE", below["triggered_rules"])
        self.assert_rule_result(
            positive,
            "EXTREME_AMOUNT_ZSCORE",
            RULE_CONFIG["EXTREME_AMOUNT_ZSCORE"]["score"],
        )
        self.assert_rule_result(
            negative,
            "EXTREME_AMOUNT_ZSCORE",
            RULE_CONFIG["EXTREME_AMOUNT_ZSCORE"]["score"],
        )

    def test_night_transaction_only(self):
        self.assert_rule_result(
            self.evaluate(is_night=1),
            "NIGHT_TRANSACTION",
            RULE_CONFIG["NIGHT_TRANSACTION"]["score"],
        )

    def test_new_recipient_only(self):
        self.assert_rule_result(
            self.evaluate(new_recipient=1),
            "NEW_RECIPIENT",
            RULE_CONFIG["NEW_RECIPIENT"]["score"],
        )

    def test_unusual_medium_only(self):
        self.assert_rule_result(
            self.evaluate(unusual_medium=1),
            "UNUSUAL_MEDIUM",
            RULE_CONFIG["UNUSUAL_MEDIUM"]["score"],
        )

    def test_cross_bank_only(self):
        self.assert_rule_result(
            self.evaluate(same_bank=0),
            "CROSS_BANK",
            RULE_CONFIG["CROSS_BANK"]["score"],
        )

    def test_repeated_same_day_boundary(self):
        threshold = RULE_CONFIG["REPEATED_SAME_DAY"]["threshold"]

        below = self.evaluate(same_day_transaction_count=threshold - 1)
        at_threshold = self.evaluate(same_day_transaction_count=threshold)

        self.assertNotIn("REPEATED_SAME_DAY", below["triggered_rules"])
        self.assert_rule_result(
            at_threshold,
            "REPEATED_SAME_DAY",
            RULE_CONFIG["REPEATED_SAME_DAY"]["score"],
        )

    def test_repeated_time_bucket_boundary(self):
        threshold = RULE_CONFIG["REPEATED_TIME_BUCKET"]["threshold"]

        below = self.evaluate(same_time_bucket_count=threshold - 1)
        at_threshold = self.evaluate(same_time_bucket_count=threshold)

        self.assertNotIn("REPEATED_TIME_BUCKET", below["triggered_rules"])
        self.assert_rule_result(
            at_threshold,
            "REPEATED_TIME_BUCKET",
            RULE_CONFIG["REPEATED_TIME_BUCKET"]["score"],
        )

    def test_missing_features_use_safe_defaults(self):
        result = self.engine.evaluate({})

        self.assertEqual(result["rule_score"], 0.0)
        self.assertEqual(result["triggered_rules"], [])

    def test_invalid_numeric_features_use_safe_defaults(self):
        result = self.engine.evaluate(
            {
                "amount_ratio": "not-a-number",
                "amount_zscore": math.nan,
                "is_night": math.inf,
                "new_recipient": None,
                "unusual_medium": "unknown",
                "same_bank": "unknown",
                "same_day_transaction_count": None,
                "same_time_bucket_count": -math.inf,
            }
        )

        self.assertEqual(result["rule_score"], 0.0)
        self.assertEqual(result["triggered_rules"], [])

    def test_all_rules_trigger_and_score_is_capped_at_100(self):
        result = self.engine.evaluate(
            normal_features(
                amount_ratio=RULE_CONFIG["HIGH_AMOUNT_RATIO"]["threshold"],
                amount_zscore=RULE_CONFIG["EXTREME_AMOUNT_ZSCORE"]["threshold"],
                is_night=1,
                new_recipient=1,
                unusual_medium=1,
                same_bank=0,
                same_day_transaction_count=RULE_CONFIG["REPEATED_SAME_DAY"][
                    "threshold"
                ],
                same_time_bucket_count=RULE_CONFIG["REPEATED_TIME_BUCKET"][
                    "threshold"
                ],
            )
        )

        self.assertEqual(result["rule_score"], 100.0)
        self.assertEqual(set(result["triggered_rules"]), set(RULE_CONFIG))


if __name__ == "__main__":
    unittest.main(verbosity=2)
