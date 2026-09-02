"""MOVI FDS 5단계 위험 시나리오 통합 검증.

실행 전 FDS API를 먼저 실행한다.

    uvicorn src.fraud_detection.api:app --reload --port 8000

그다음 프로젝트 루트에서 이 파일을 실행한다.

    python test_risk_scenarios.py

다른 주소를 사용하는 경우 환경변수로 변경할 수 있다.

    FDS_API_URL=http://127.0.0.1:8000/api/v1/fraud/detect \
        python test_risk_scenarios.py
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_URL = os.getenv(
    "FDS_API_URL",
    "http://127.0.0.1:8000/api/v1/fraud/detect",
)
TIMEOUT_SECONDS = 15
RISK_LEVEL_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

REQUIRED_RESPONSE_FIELDS = {
    "transaction_id",
    "anomaly_score",
    "threshold",
    "is_anomaly",
    "model",
    "rule_score",
    "final_risk_score",
    "risk_level",
    "triggered_rules",
}


@dataclass(frozen=True)
class Scenario:
    """한 단계의 거래 입력과 기대 결과."""

    name: str
    description: str
    payload: dict[str, Any]
    expected_rules: frozenset[str]


def transaction(
    *,
    transaction_id: str,
    receiver_account: str,
    amount: int,
    transaction_datetime: str,
    medium: str = "MOBILE",
    sender_bank: str = "KB",
    receiver_bank: str = "KB",
) -> dict[str, Any]:
    """Spring FDS DTO 형식의 거래 한 건을 생성한다."""

    return {
        "transaction_id": transaction_id,
        "sender_account": "100001",
        "receiver_account": receiver_account,
        "sender_bank": sender_bank,
        "receiver_bank": receiver_bank,
        "transaction_type": "transfer",
        "amount": amount,
        "transaction_datetime": transaction_datetime,
        "medium": medium,
    }


def build_base_history() -> list[dict[str, Any]]:
    """정상 패턴을 만들기 위한 동일 계좌의 과거 거래 이력."""

    amounts = [90_000, 100_000, 110_000, 95_000, 105_000]
    datetimes = [
        "2026-08-20T12:10:00",
        "2026-08-21T12:20:00",
        "2026-08-22T12:30:00",
        "2026-08-23T12:40:00",
        "2026-08-24T12:50:00",
    ]

    return [
        transaction(
            transaction_id=f"history-{index}",
            receiver_account="200001",
            amount=amount,
            transaction_datetime=transaction_datetime,
        )
        for index, (amount, transaction_datetime) in enumerate(
            zip(amounts, datetimes, strict=True),
            start=1,
        )
    ]


def build_repeated_history() -> list[dict[str, Any]]:
    """Case 5에서 당일·동일 시간대 반복 Rule을 발생시킨다."""

    repeated = [
        transaction(
            transaction_id="repeat-1",
            receiver_account="200001",
            amount=92_000,
            transaction_datetime="2026-08-26T00:05:00",
        ),
        transaction(
            transaction_id="repeat-2",
            receiver_account="200001",
            amount=96_000,
            transaction_datetime="2026-08-26T00:10:00",
        ),
        transaction(
            transaction_id="repeat-3",
            receiver_account="200001",
            amount=98_000,
            transaction_datetime="2026-08-26T00:20:00",
        ),
    ]
    return build_base_history() + repeated


def request_payload(
    current_transaction: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "current_transaction": current_transaction,
        "history": history if history is not None else build_base_history(),
    }


def build_scenarios() -> list[Scenario]:
    """위험 조건을 한 단계씩 추가한 5개 시나리오를 만든다."""

    normal_receiver = "200001"
    new_receiver = "999999"

    return [
        Scenario(
            name="CASE_1_NORMAL",
            description="평소와 동일한 금액·수취인·시간·매체",
            payload=request_payload(
                transaction(
                    transaction_id="case-1",
                    receiver_account=normal_receiver,
                    amount=100_000,
                    transaction_datetime="2026-08-26T12:30:00",
                )
            ),
            expected_rules=frozenset(),
        ),
        Scenario(
            name="CASE_2_HIGH_AMOUNT",
            description="평소보다 매우 큰 금액",
            payload=request_payload(
                transaction(
                    transaction_id="case-2",
                    receiver_account=normal_receiver,
                    amount=10_000_000,
                    transaction_datetime="2026-08-26T12:30:00",
                )
            ),
            expected_rules=frozenset(
                {"HIGH_AMOUNT_RATIO", "EXTREME_AMOUNT_ZSCORE"}
            ),
        ),
        Scenario(
            name="CASE_3_NEW_RECIPIENT",
            description="고액 + 신규 수취인",
            payload=request_payload(
                transaction(
                    transaction_id="case-3",
                    receiver_account=new_receiver,
                    amount=10_000_000,
                    transaction_datetime="2026-08-26T12:30:00",
                )
            ),
            expected_rules=frozenset(
                {
                    "HIGH_AMOUNT_RATIO",
                    "EXTREME_AMOUNT_ZSCORE",
                    "NEW_RECIPIENT",
                }
            ),
        ),
        Scenario(
            name="CASE_4_NIGHT_NEW_MEDIUM",
            description="고액 + 신규 수취인 + 심야 + 신규 매체",
            payload=request_payload(
                transaction(
                    transaction_id="case-4",
                    receiver_account=new_receiver,
                    amount=10_000_000,
                    transaction_datetime="2026-08-26T00:30:00",
                    medium="WEB",
                )
            ),
            expected_rules=frozenset(
                {
                    "HIGH_AMOUNT_RATIO",
                    "EXTREME_AMOUNT_ZSCORE",
                    "NEW_RECIPIENT",
                    "NIGHT_TRANSACTION",
                    "UNUSUAL_MEDIUM",
                }
            ),
        ),
        Scenario(
            name="CASE_5_REPEATED",
            description="Case 4 위험 조건 + 당일·동일 시간대 반복 거래",
            payload=request_payload(
                transaction(
                    transaction_id="case-5",
                    receiver_account=new_receiver,
                    amount=10_000_000,
                    transaction_datetime="2026-08-26T00:30:00",
                    medium="WEB",
                ),
                history=build_repeated_history(),
            ),
            expected_rules=frozenset(
                {
                    "HIGH_AMOUNT_RATIO",
                    "EXTREME_AMOUNT_ZSCORE",
                    "NEW_RECIPIENT",
                    "NIGHT_TRANSACTION",
                    "UNUSUAL_MEDIUM",
                    "REPEATED_SAME_DAY",
                    "REPEATED_TIME_BUCKET",
                }
            ),
        ),
    ]


def call_fds_api(payload: dict[str, Any]) -> dict[str, Any]:
    """FDS API를 호출하고 JSON 응답을 반환한다."""

    request = Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            body = response.read().decode("utf-8")
    except HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"FDS API가 HTTP {error.code}을 반환했습니다: {error_body}"
        ) from error
    except URLError as error:
        raise RuntimeError(
            "FDS API에 연결할 수 없습니다. "
            "uvicorn 서버가 8000번 포트에서 실행 중인지 확인하세요."
        ) from error

    try:
        result = json.loads(body)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"FDS 응답이 JSON이 아닙니다: {body}") from error

    if not isinstance(result, dict):
        raise RuntimeError("FDS 응답의 최상위 값은 JSON 객체여야 합니다.")

    return result


def validate_response(scenario: Scenario, result: dict[str, Any]) -> list[str]:
    """응답 Schema와 개별 시나리오 기대 Rule을 검사한다."""

    failures: list[str] = []
    missing_fields = REQUIRED_RESPONSE_FIELDS - result.keys()
    if missing_fields:
        failures.append(f"응답 필드 누락: {sorted(missing_fields)}")
        return failures

    if result["transaction_id"] != scenario.payload["current_transaction"]["transaction_id"]:
        failures.append("요청과 응답의 transaction_id가 다름")

    for field in ("anomaly_score", "threshold", "rule_score", "final_risk_score"):
        if not isinstance(result[field], (int, float)):
            failures.append(f"{field}가 숫자가 아님: {result[field]!r}")

    if not 0 <= float(result["rule_score"]) <= 100:
        failures.append(f"rule_score 범위 오류: {result['rule_score']}")

    if not 0 <= float(result["final_risk_score"]) <= 100:
        failures.append(f"final_risk_score 범위 오류: {result['final_risk_score']}")

    if result["risk_level"] not in RISK_LEVEL_ORDER:
        failures.append(f"알 수 없는 risk_level: {result['risk_level']!r}")

    actual_rules = set(result["triggered_rules"])
    missing_rules = scenario.expected_rules - actual_rules
    if missing_rules:
        failures.append(f"예상 Rule 미발생: {sorted(missing_rules)}")

    if scenario.name == "CASE_1_NORMAL" and actual_rules:
        failures.append(f"정상 거래에서 Rule 발생: {sorted(actual_rules)}")

    return failures


def validate_progression(results: list[dict[str, Any]]) -> list[str]:
    """위험 조건 증가에 따라 점수와 등급이 역행하지 않는지 검사한다."""

    failures: list[str] = []

    for previous_index, current_index in zip(
        range(len(results) - 1),
        range(1, len(results)),
        strict=True,
    ):
        previous = results[previous_index]
        current = results[current_index]
        previous_case = previous_index + 1
        current_case = current_index + 1

        if float(current["rule_score"]) <= float(previous["rule_score"]):
            failures.append(
                f"Case {previous_case} → {current_case}: "
                f"Rule Score가 증가하지 않음 "
                f"({previous['rule_score']} → {current['rule_score']})"
            )

        # 모델 점수는 History 변화에 따라 조금 달라질 수 있으므로 최종 점수는
        # 역행 여부를 검사한다. Rule Score는 위에서 단계별 증가를 엄격히 검사한다.
        if float(current["final_risk_score"]) < float(previous["final_risk_score"]):
            failures.append(
                f"Case {previous_case} → {current_case}: "
                f"Final Risk Score가 감소함 "
                f"({previous['final_risk_score']} → {current['final_risk_score']})"
            )

        previous_level = RISK_LEVEL_ORDER[previous["risk_level"]]
        current_level = RISK_LEVEL_ORDER[current["risk_level"]]
        if current_level < previous_level:
            failures.append(
                f"Case {previous_case} → {current_case}: "
                f"Risk Level이 역행함 "
                f"({previous['risk_level']} → {current['risk_level']})"
            )

    return failures


def print_result(index: int, scenario: Scenario, result: dict[str, Any]) -> None:
    print(f"\n[{index}] {scenario.name}")
    print(f"설명            : {scenario.description}")
    print(f"Anomaly Score   : {result['anomaly_score']}")
    print(f"Threshold       : {result['threshold']}")
    print(f"Model Anomaly   : {result['is_anomaly']}")
    print(f"Rule Score      : {result['rule_score']}")
    print(f"Final Risk Score: {result['final_risk_score']}")
    print(f"Risk Level      : {result['risk_level']}")
    print(f"Triggered Rules : {', '.join(result['triggered_rules']) or '-'}")


def main() -> int:
    scenarios = build_scenarios()
    results: list[dict[str, Any]] = []
    failures: list[str] = []

    print("=" * 72)
    print("MOVI FDS 5단계 Risk Scenario 검증")
    print(f"API: {API_URL}")
    print("=" * 72)

    for index, scenario in enumerate(scenarios, start=1):
        try:
            result = call_fds_api(scenario.payload)
        except RuntimeError as error:
            print(f"\n[실패] {scenario.name}: {error}", file=sys.stderr)
            return 1

        results.append(result)
        print_result(index, scenario, result)

        scenario_failures = validate_response(scenario, result)
        failures.extend(
            f"{scenario.name}: {failure}" for failure in scenario_failures
        )

    failures.extend(validate_progression(results))

    print("\n" + "=" * 72)
    if failures:
        print("검증 실패")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("검증 성공")
    print("- 5개 API 응답 Schema 정상")
    print("- 단계별 예상 Rule 정상 발생")
    print("- Rule Score 단계별 증가")
    print("- Final Risk Score 및 Risk Level 역행 없음")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
