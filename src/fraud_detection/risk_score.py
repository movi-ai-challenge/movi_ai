from __future__ import annotations

from dataclasses import dataclass


try:
    from .config import (
        MODEL_SCORE_POINTS,
        MODEL_WEIGHT,
        RISK_LEVEL_THRESHOLDS,
        RULE_WEIGHT,
    )
except ImportError:
    from fraud_detection.config import (
        MODEL_SCORE_POINTS,
        MODEL_WEIGHT,
        RISK_LEVEL_THRESHOLDS,
        RULE_WEIGHT,
    )


# 기존 외부 참조와 테스트의 호환성을 유지한다.
MEDIUM_THRESHOLD = RISK_LEVEL_THRESHOLDS["MEDIUM"]
HIGH_THRESHOLD = RISK_LEVEL_THRESHOLDS["HIGH"]


# ============================================================
# Result
# ============================================================

@dataclass
class RiskScoreResult:

    model_risk_score: float

    rule_score: float

    final_risk_score: float

    risk_level: str


# ============================================================
# Clamp
# ============================================================

def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 100.0,
) -> float:

    return max(
        minimum,
        min(
            value,
            maximum,
        ),
    )


# ============================================================
# Linear Interpolation
# ============================================================

def linear_interpolate(
    *,
    x: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    """
    두 기준점 사이의 값을 선형 보간한다.

    Example
    -------
    x1 = 0.419
    y1 = 40

    x2 = 0.446
    y2 = 70

    x = 0.443

    → 약 66점
    """

    if x2 == x1:

        return y1


    ratio = (
        (x - x1)
        /
        (x2 - x1)
    )


    return (
        y1
        +
        ratio
        *
        (y2 - y1)
    )


# ============================================================
# Isolation Forest Score → 0~100
# ============================================================

def normalize_anomaly_score(
    anomaly_score: float,
) -> float:
    """
    Isolation Forest anomaly_score를
    Validation 분포 기준 0~100 Risk Score로 변환한다.

    anomaly_score 자체를 단순히 ×100 하지 않는다.

    학습/Validation에서 확인한 실제 Score 분포를
    기준으로 Piecewise Linear Mapping을 적용한다.
    """

    score = float(
        anomaly_score
    )


    # ========================================================
    # 정상 평균 이하
    # ========================================================

    first_x, first_y = (
        MODEL_SCORE_POINTS[0]
    )


    if score <= first_x:

        return first_y


    # ========================================================
    # 기준점 사이 선형 보간
    # ========================================================

    for index in range(
        len(MODEL_SCORE_POINTS) - 1
    ):

        x1, y1 = (
            MODEL_SCORE_POINTS[
                index
            ]
        )

        x2, y2 = (
            MODEL_SCORE_POINTS[
                index + 1
            ]
        )


        if (
            x1
            <=
            score
            <=
            x2
        ):

            result = (
                linear_interpolate(

                    x=score,

                    x1=x1,
                    y1=y1,

                    x2=x2,
                    y2=y2,
                )
            )


            return clamp(
                result
            )


    # ========================================================
    # 가장 극단적인 기준보다 높은 경우
    # ========================================================

    return 100.0


# ============================================================
# Risk Level
# ============================================================

def classify_risk_level(
    final_risk_score: float,
) -> str:

    score = float(
        final_risk_score
    )


    if score >= HIGH_THRESHOLD:

        return "HIGH"


    if score >= MEDIUM_THRESHOLD:

        return "MEDIUM"


    return "LOW"


# ============================================================
# Final Risk Score
# ============================================================

def calculate_risk_score(
    *,
    anomaly_score: float,
    rule_score: float,
) -> RiskScoreResult:
    """
    Isolation Forest + Rule Engine을 통합하여
    최종 Risk Score를 계산한다.

    현재 정책:

        Model Risk 40%
        Rule Risk  60%

    추후 Validation 결과를 이용하여
    가중치를 재조정한다.
    """

    # ========================================================
    # 1. Model Score Normalize
    # ========================================================

    model_risk_score = (
        normalize_anomaly_score(
            anomaly_score
        )
    )


    # ========================================================
    # 2. Rule Score
    # ========================================================

    normalized_rule_score = (
        clamp(
            float(
                rule_score
            )
        )
    )


    # ========================================================
    # 3. Weighted Risk Score
    # ========================================================

    final_risk_score = (

        model_risk_score
        *
        MODEL_WEIGHT

        +

        normalized_rule_score
        *
        RULE_WEIGHT
    )


    final_risk_score = (
        clamp(
            final_risk_score
        )
    )


    # ========================================================
    # 4. Risk Level
    # ========================================================

    risk_level = (
        classify_risk_level(
            final_risk_score
        )
    )


    # ========================================================
    # 5. Result
    # ========================================================

    return RiskScoreResult(

        model_risk_score=round(
            model_risk_score,
            2,
        ),

        rule_score=round(
            normalized_rule_score,
            2,
        ),

        final_risk_score=round(
            final_risk_score,
            2,
        ),

        risk_level=(
            risk_level
        ),
    )
