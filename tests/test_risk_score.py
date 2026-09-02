"""MOVI Risk Score 정규화, 가중합 및 등급 경계 테스트.

프로젝트 루트 또는 tests 디렉터리에서 실행할 수 있다.

    python test_risk_score.py

또는

    python -m unittest -v test_risk_score.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


def _load_risk_score_module():
    """프로젝트 배치 방식에 맞춰 Risk Score 모듈을 불러온다."""

    try:
        from src.fraud_detection import risk_score

        return fraud_detection.risk_score
    except ModuleNotFoundError:
        pass

    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent if current_dir.name == "tests" else current_dir

    for module_dir in (project_root / "src", project_root / "upload"):
        if module_dir.exists():
            sys.path.insert(0, str(module_dir))

    try:
        from fraud_detection import risk_score
    except ModuleNotFoundError:
        import fraud_detection.risk_score as risk_score

    return risk_score


risk_score = _load_risk_score_module()


class RiskScoreTest(unittest.TestCase):
    def test_clamp_limits_values_to_zero_and_one_hundred(self):
        self.assertEqual(risk_score.clamp(-1.0), 0.0)
        self.assertEqual(risk_score.clamp(50.0), 50.0)
        self.assertEqual(risk_score.clamp(101.0), 100.0)

    def test_model_score_points_map_to_exact_risk_scores(self):
        for anomaly_score, expected_risk in risk_score.MODEL_SCORE_POINTS:
            with self.subTest(anomaly_score=anomaly_score):
                actual = risk_score.normalize_anomaly_score(anomaly_score)
                self.assertAlmostEqual(actual, expected_risk, places=7)

    def test_model_score_below_and_above_range_is_clamped(self):
        first_score = risk_score.MODEL_SCORE_POINTS[0][0]
        last_score = risk_score.MODEL_SCORE_POINTS[-1][0]

        self.assertEqual(
            risk_score.normalize_anomaly_score(first_score - 1.0),
            0.0,
        )
        self.assertEqual(
            risk_score.normalize_anomaly_score(last_score + 1.0),
            100.0,
        )

    def test_model_score_is_linearly_interpolated(self):
        x1, y1 = risk_score.MODEL_SCORE_POINTS[1]
        x2, y2 = risk_score.MODEL_SCORE_POINTS[2]
        midpoint = (x1 + x2) / 2
        expected = (y1 + y2) / 2

        actual = risk_score.normalize_anomaly_score(midpoint)

        self.assertAlmostEqual(actual, expected, places=7)

    def test_normalized_model_score_is_monotonic(self):
        anomaly_scores = [point[0] for point in risk_score.MODEL_SCORE_POINTS]
        anomaly_scores.extend(
            (left + right) / 2
            for left, right in zip(anomaly_scores, anomaly_scores[1:])
        )
        anomaly_scores.sort()

        normalized = [
            risk_score.normalize_anomaly_score(score) for score in anomaly_scores
        ]

        self.assertEqual(normalized, sorted(normalized))

    def test_risk_level_boundaries(self):
        self.assertEqual(
            risk_score.classify_risk_level(risk_score.MEDIUM_THRESHOLD - 0.01),
            "LOW",
        )
        self.assertEqual(
            risk_score.classify_risk_level(risk_score.MEDIUM_THRESHOLD),
            "MEDIUM",
        )
        self.assertEqual(
            risk_score.classify_risk_level(risk_score.HIGH_THRESHOLD - 0.01),
            "MEDIUM",
        )
        self.assertEqual(
            risk_score.classify_risk_level(risk_score.HIGH_THRESHOLD),
            "HIGH",
        )

    def test_weighted_final_score(self):
        anomaly_score = risk_score.MODEL_SCORE_POINTS[2][0]
        model_risk = risk_score.MODEL_SCORE_POINTS[2][1]
        rule_risk = 50.0
        expected = (
            model_risk * risk_score.MODEL_WEIGHT
            + rule_risk * risk_score.RULE_WEIGHT
        )

        result = risk_score.calculate_risk_score(
            anomaly_score=anomaly_score,
            rule_score=rule_risk,
        )

        self.assertEqual(result.model_risk_score, round(model_risk, 2))
        self.assertEqual(result.rule_score, rule_risk)
        self.assertEqual(result.final_risk_score, round(expected, 2))
        self.assertEqual(result.risk_level, "MEDIUM")

    def test_rule_score_is_clamped_before_weighting(self):
        anomaly_score = risk_score.MODEL_SCORE_POINTS[0][0]

        below = risk_score.calculate_risk_score(
            anomaly_score=anomaly_score,
            rule_score=-50.0,
        )
        above = risk_score.calculate_risk_score(
            anomaly_score=anomaly_score,
            rule_score=150.0,
        )

        self.assertEqual(below.rule_score, 0.0)
        self.assertEqual(below.final_risk_score, 0.0)
        self.assertEqual(above.rule_score, 100.0)
        self.assertEqual(
            above.final_risk_score,
            round(100.0 * risk_score.RULE_WEIGHT, 2),
        )

    def test_calculated_score_is_always_in_valid_range(self):
        for anomaly_score in (-10.0, 0.4, 10.0):
            for rule_score_value in (-100.0, 0.0, 100.0, 200.0):
                with self.subTest(
                    anomaly_score=anomaly_score,
                    rule_score=rule_score_value,
                ):
                    result = risk_score.calculate_risk_score(
                        anomaly_score=anomaly_score,
                        rule_score=rule_score_value,
                    )
                    self.assertGreaterEqual(result.final_risk_score, 0.0)
                    self.assertLessEqual(result.final_risk_score, 100.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
