"""외부 API 호출 없이 VoiceAnalysisService의 분기 로직을 검증한다."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from _optional_dependency_stubs import install_optional_dependency_stubs


install_optional_dependency_stubs()


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.voice_analysis.schemas import (  # noqa: E402
    FollowUpEntityResult,
    RequirementAnalysis,
    RequirementEntities,
)
from src.voice_analysis.voice_service import VoiceAnalysisService  # noqa: E402


class FakeWakeWordDetector:
    def __init__(self, *, activated: bool, command: str):
        self.activated = activated
        self.command = command

    def detect(self, text: str):
        return SimpleNamespace(
            activated=self.activated,
            command=self.command,
            original_text=text,
        )


class FakeRequirementAnalyzer:
    def __init__(self, result: RequirementAnalysis):
        self.result = result

    def analyze(self, command: str) -> RequirementAnalysis:
        return self.result


class FakeFollowUpParser:
    def __init__(self, result: FollowUpEntityResult):
        self.result = result

    def parse(self, *, field_name: str, user_text: str) -> FollowUpEntityResult:
        return self.result


def build_service(
    *,
    activated: bool = True,
    command: str = "김민수에게 오만원 보내줘",
    intent: str = "transfer_money",
) -> VoiceAnalysisService:
    service = VoiceAnalysisService.__new__(VoiceAnalysisService)
    service.wake_word_detector = FakeWakeWordDetector(
        activated=activated,
        command=command,
    )
    service.requirement_analyzer = FakeRequirementAnalyzer(
        RequirementAnalysis(
            intent=intent,
            entities=RequirementEntities(
                recipient_name="김민수" if intent == "transfer_money" else None,
                amount=50_000 if intent == "transfer_money" else None,
            ),
            missing_fields=[],
            original_text=command,
        )
    )
    service.follow_up_parser = FakeFollowUpParser(
        FollowUpEntityResult(
            field_name="recipient_bank",
            value="국민은행",
            success=True,
            original_text="국민은행이야",
        )
    )
    return service


class VoiceAnalysisServiceTest(unittest.TestCase):
    def test_empty_input_returns_error(self) -> None:
        result = build_service().analyze("   ")

        self.assertEqual(result["status"], "error")
        self.assertIsNone(result.get("intent"))

    def test_missing_wake_word_is_ignored(self) -> None:
        result = build_service(activated=False, command="").analyze(
            "김민수에게 오만원 보내줘"
        )

        self.assertEqual(result["status"], "ignored")

    def test_wake_word_without_command_awaits_command(self) -> None:
        result = build_service(command="").analyze("모비야")

        self.assertEqual(result["status"], "awaiting_command")

    def test_transfer_entities_preserve_null_values(self) -> None:
        result = build_service().analyze("모비야 김민수에게 오만원 보내줘")

        self.assertEqual(result["status"], "analyzed")
        self.assertEqual(result["intent"], "transfer_money")
        self.assertEqual(result["entities"]["recipient_name"], "김민수")
        self.assertEqual(result["entities"]["amount"], 50_000)
        self.assertIsNone(result["entities"]["recipient_bank"])
        self.assertIsNone(result["entities"]["recipient_account"])

    def test_unknown_intent_is_unsupported(self) -> None:
        result = build_service(intent="unknown").analyze("모비야 오늘 날씨 알려줘")

        self.assertEqual(result["status"], "unsupported")
        self.assertEqual(result["intent"], "unknown")

    def test_follow_up_success_updates_requested_entity_only(self) -> None:
        service = build_service()
        original_entities = {
            "recipient_name": "김민수",
            "recipient_bank": None,
            "amount": 50_000,
        }

        result = service.analyze_follow_up(
            requested_field="recipient_bank",
            text="국민은행이야",
            entities=original_entities,
        )

        self.assertEqual(result["status"], "analyzed")
        self.assertEqual(result["entities"]["recipient_bank"], "국민은행")
        self.assertIsNone(original_entities["recipient_bank"])

    def test_follow_up_failure_keeps_entities_unchanged(self) -> None:
        service = build_service()
        service.follow_up_parser = FakeFollowUpParser(
            FollowUpEntityResult(
                field_name="recipient_bank",
                value=None,
                success=False,
                original_text="잘 모르겠어",
            )
        )
        entities = {"recipient_bank": None}

        result = service.analyze_follow_up(
            requested_field="recipient_bank",
            text="잘 모르겠어",
            entities=entities,
        )

        self.assertEqual(result["status"], "parse_failed")
        self.assertEqual(result["entities"], entities)


if __name__ == "__main__":
    unittest.main(verbosity=2)
