from __future__ import annotations

from .risk_score import (
    normalize_anomaly_score,
    calculate_risk_score,
)


# ============================================================
# Model Score Normalize Test
# ============================================================

def test_model_normalization():

    print()
    print("=" * 70)
    print("MODEL SCORE NORMALIZATION")
    print("=" * 70)


    test_scores = [

        0.373701,

        0.400000,

        0.419761,

        0.443008,

        0.446117,

        0.472755,

        0.528859,
    ]


    for score in test_scores:

        risk = (
            normalize_anomaly_score(
                score
            )
        )


        print(
            f"anomaly_score={score:.6f}"
            f" → "
            f"model_risk={risk:.2f}"
        )


# ============================================================
# Normal Transaction
# ============================================================

def test_normal():

    result = (
        calculate_risk_score(

            anomaly_score=(
                0.390000
            ),

            rule_score=(
                0
            ),
        )
    )


    print()
    print("=" * 70)
    print("NORMAL TRANSACTION")
    print("=" * 70)

    print(
        result
    )


    assert (
        result.risk_level
        ==
        "LOW"
    )


    print()
    print(
        "[PASS] Normal Risk"
    )


# ============================================================
# Medium Transaction
# ============================================================

def test_medium():

    result = (
        calculate_risk_score(

            anomaly_score=(
                0.430000
            ),

            rule_score=(
                40
            ),
        )
    )


    print()
    print("=" * 70)
    print("MEDIUM TRANSACTION")
    print("=" * 70)

    print(
        result
    )


    assert (
        result.risk_level
        in {
            "MEDIUM",
            "HIGH",
        }
    )


    print()
    print(
        "[PASS] Medium Risk"
    )


# ============================================================
# High Risk Transaction
#
# 실제 API 테스트에서 나온 값 사용
# ============================================================

def test_high():

    result = (
        calculate_risk_score(

            anomaly_score=(
                0.443008
            ),

            rule_score=(
                95
            ),
        )
    )


    print()
    print("=" * 70)
    print("HIGH RISK TRANSACTION")
    print("=" * 70)

    print(
        "Model Risk Score:",
        result.model_risk_score,
    )

    print(
        "Rule Score:",
        result.rule_score,
    )

    print(
        "Final Risk Score:",
        result.final_risk_score,
    )

    print(
        "Risk Level:",
        result.risk_level,
    )


    assert (
        result.risk_level
        ==
        "HIGH"
    )


    assert (
        result.final_risk_score
        >=
        70
    )


    print()
    print(
        "[PASS] High Risk"
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    test_model_normalization()

    test_normal()

    test_medium()

    test_high()