"""실제 OpenAI 연결로 MOVI 핵심 음성 명령을 검증하는 수동 스모크 테스트.

일반 단위 테스트와 달리 외부 API를 호출하므로 unittest discover 대상에서
제외한다. 프로젝트 루트에서 다음처럼 직접 실행한다.

    python tests/live_voice_smoke.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        """python-dotenv가 없을 때 셸 환경변수만 사용한다."""

        return False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class SmokeCase:
    name: str
    transcript: str
    expected_intent: str
    expected_entities: dict[str, Any]
    expected_null_entities: tuple[str, ...]


CASES = (
    SmokeCase(
        name="화면 읽기",
        transcript="모비야 화면 읽어줘",
        expected_intent="read_screen",
        expected_entities={},
        expected_null_entities=(
            "recipient_name",
            "recipient_bank",
            "recipient_account",
            "amount",
            "source_bank",
            "source_account",
            "date_from",
            "date_to",
            "bank",
            "account",
        ),
    ),
    SmokeCase(
        name="계좌 이체",
        transcript=(
            "모비야 김민수에게 국민은행 1234567890 계좌로 "
            "오만원 보내줘"
        ),
        expected_intent="transfer_money",
        expected_entities={
            "recipient_name": "김민수",
            "recipient_bank": "국민은행",
            "recipient_account": "1234567890",
            "amount": 50_000,
        },
        expected_null_entities=(
            "source_bank",
            "source_account",
            "date_from",
            "date_to",
            "bank",
            "account",
        ),
    ),
    SmokeCase(
        name="거래내역 조회",
        transcript="모비야 거래내역 알려줘",
        expected_intent="check_history",
        expected_entities={},
        expected_null_entities=(
            "recipient_name",
            "recipient_bank",
            "recipient_account",
            "amount",
            "source_bank",
            "source_account",
            "date_from",
            "date_to",
            "bank",
            "account",
        ),
    ),
    SmokeCase(
        name="적금 조회",
        transcript="모비야 가입한 적금 알려줘",
        expected_intent="check_savings",
        expected_entities={},
        expected_null_entities=(
            "recipient_name",
            "recipient_bank",
            "recipient_account",
            "amount",
            "source_bank",
            "source_account",
            "date_from",
            "date_to",
            "bank",
            "account",
        ),
    ),
)


def digits_only(value: Any) -> str:
    return "".join(character for character in str(value) if character.isdigit())


def entity_matches(field: str, actual: Any, expected: Any) -> bool:
    if field in {"recipient_account", "source_account", "account"}:
        return digits_only(actual) == digits_only(expected)
    return actual == expected


def validate(case: SmokeCase, result: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    if result.get("status") != "analyzed":
        errors.append(
            f"status: expected='analyzed', actual={result.get('status')!r}"
        )

    if result.get("intent") != case.expected_intent:
        errors.append(
            "intent: "
            f"expected={case.expected_intent!r}, actual={result.get('intent')!r}"
        )

    entities = result.get("entities")
    if not isinstance(entities, dict):
        errors.append("entities가 객체가 아닙니다.")
        return errors

    for field, expected in case.expected_entities.items():
        actual = entities.get(field)
        if not entity_matches(field, actual, expected):
            errors.append(
                f"entities.{field}: expected={expected!r}, actual={actual!r}"
            )

    for field in case.expected_null_entities:
        actual = entities.get(field)
        if actual is not None:
            errors.append(
                f"entities.{field}: 말하지 않은 값은 null이어야 하지만 "
                f"actual={actual!r}"
            )

    return errors


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        print("[설정 오류] OPENAI_API_KEY가 설정되지 않았습니다.")
        print(".env 또는 현재 셸 환경변수에 키를 설정한 뒤 다시 실행하세요.")
        return 2

    try:
        from src.voice_analysis.config import OPENAI_MODEL
        from src.voice_analysis.voice_service import VoiceAnalysisService

        service = VoiceAnalysisService()
    except Exception as error:
        print(f"[초기화 실패] {type(error).__name__}: {error}")
        return 2

    print("=" * 72)
    print("MOVI Voice 실제 GPT 스모크 테스트")
    print(f"Model: {OPENAI_MODEL}")
    print("=" * 72)

    failed = 0

    for index, case in enumerate(CASES, start=1):
        print(f"\n[{index}] {case.name}")
        print(f"입력   : {case.transcript}")

        try:
            result = service.analyze(case.transcript)
        except Exception as error:
            failed += 1
            print(f"결과   : FAIL ({type(error).__name__}: {error})")
            continue

        errors = validate(case, result)
        if errors:
            failed += 1
            print("결과   : FAIL")
            for error in errors:
                print(f"  - {error}")
            continue

        print(f"Intent : {result['intent']}")
        print("결과   : PASS")

    print("\n" + "=" * 72)
    if failed:
        print(f"검증 실패: {failed}/{len(CASES)}개 시나리오 실패")
        return 1

    print(f"검증 성공: {len(CASES)}개 실제 GPT 시나리오 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
