"""Voice FastAPI의 입력 검증, 준비 상태, 안전한 오류 응답을 검증한다."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from _optional_dependency_stubs import install_optional_dependency_stubs


install_optional_dependency_stubs()


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from fastapi.testclient import TestClient
except ModuleNotFoundError:
    TestClient = None

if TestClient is not None:
    from src.voice_analysis import api as voice_api  # noqa: E402
    from src.voice_analysis.api_schemas import MAX_TRANSCRIPT_LENGTH  # noqa: E402
else:
    voice_api = None
    MAX_TRANSCRIPT_LENGTH = 2_000


def ready_service() -> Mock:
    service = Mock()
    service.analyze.return_value = {
        "status": "analyzed",
        "intent": "transfer_money",
        "transcript": "김민수에게 오만원 보내줘",
        "entities": {
            "recipient_name": "김민수",
            "recipient_bank": None,
            "recipient_account": None,
            "amount": 50_000,
        },
        "message": None,
    }
    service.analyze_follow_up.return_value = {
        "status": "analyzed",
        "transcript": "국민은행이야",
        "entities": {
            "recipient_name": "김민수",
            "recipient_bank": "국민은행",
            "amount": 50_000,
        },
    }
    return service


@unittest.skipIf(TestClient is None, "fastapi가 설치되지 않은 검증 환경")
class VoiceApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(
            voice_api.app,
            raise_server_exceptions=False,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()

    def assert_no_internal_details(self, response) -> None:
        for fragment in (
            "Traceback",
            "/Users/",
            "/workspace/",
            "site-packages",
            "secret internal failure",
        ):
            self.assertNotIn(fragment, response.text)

    def test_health_and_ready_when_analyzer_is_loaded(self) -> None:
        with (
            patch.object(voice_api, "voice_service", ready_service()),
            patch.object(voice_api, "voice_service_load_error", None),
        ):
            health = self.client.get("/health")
            ready = self.client.get("/ready")

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")
        self.assertTrue(health.json()["analyzer_loaded"])
        self.assertEqual(ready.status_code, 200)
        self.assertEqual(ready.json()["status"], "ready")

    def test_degraded_and_503_when_analyzer_is_not_loaded(self) -> None:
        with (
            patch.object(voice_api, "voice_service", None),
            patch.object(voice_api, "voice_service_load_error", "simulated"),
        ):
            health = self.client.get("/health")
            ready = self.client.get("/ready")
            analyze = self.client.post(
                "/api/v1/voice/analyze",
                json={"transcript": "모비야 화면 읽어줘"},
            )

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "degraded")
        self.assertFalse(health.json()["analyzer_loaded"])
        for response in (ready, analyze):
            self.assertEqual(response.status_code, 503)
            self.assertEqual(
                response.json()["detail"]["code"],
                "VOICE_ANALYZER_NOT_READY",
            )
            self.assert_no_internal_details(response)

    def test_analyze_returns_intent_entities_and_nulls(self) -> None:
        service = ready_service()
        with (
            patch.object(voice_api, "voice_service", service),
            patch.object(voice_api, "voice_service_load_error", None),
        ):
            response = self.client.post(
                "/api/v1/voice/analyze",
                json={"transcript": "  모비야 김민수에게 오만원 보내줘  "},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["intent"], "transfer_money")
        self.assertEqual(body["entities"]["amount"], 50_000)
        self.assertIsNone(body["entities"]["recipient_bank"])
        service.analyze.assert_called_once_with("모비야 김민수에게 오만원 보내줘")

    def test_empty_or_whitespace_transcript_is_rejected(self) -> None:
        for transcript in ("", "   "):
            with self.subTest(transcript=repr(transcript)):
                response = self.client.post(
                    "/api/v1/voice/analyze",
                    json={"transcript": transcript},
                )
                self.assertEqual(response.status_code, 422)
                self.assert_no_internal_details(response)

    def test_too_long_transcript_is_rejected(self) -> None:
        response = self.client.post(
            "/api/v1/voice/analyze",
            json={"transcript": "가" * (MAX_TRANSCRIPT_LENGTH + 1)},
        )

        self.assertEqual(response.status_code, 422)

    def test_follow_up_returns_updated_entities(self) -> None:
        service = ready_service()
        with (
            patch.object(voice_api, "voice_service", service),
            patch.object(voice_api, "voice_service_load_error", None),
        ):
            response = self.client.post(
                "/api/v1/voice/follow-up",
                json={
                    "transcript": "국민은행이야",
                    "requested_field": "recipient_bank",
                    "entities": {
                        "recipient_name": "김민수",
                        "recipient_bank": None,
                        "amount": 50_000,
                    },
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "analyzed")
        self.assertEqual(response.json()["entities"]["recipient_bank"], "국민은행")

    def test_unsupported_follow_up_field_is_rejected(self) -> None:
        response = self.client.post(
            "/api/v1/voice/follow-up",
            json={
                "transcript": "국민은행이야",
                "requested_field": "source_bank",
                "entities": {},
            },
        )

        self.assertEqual(response.status_code, 422)

    def test_value_error_returns_safe_400(self) -> None:
        service = ready_service()
        service.analyze.side_effect = ValueError("private input detail")
        with (
            patch.object(voice_api, "voice_service", service),
            patch.object(voice_api, "voice_service_load_error", None),
        ):
            response = self.client.post(
                "/api/v1/voice/analyze",
                json={"transcript": "모비야 화면 읽어줘"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json()["detail"]["code"],
            "INVALID_VOICE_REQUEST",
        )
        self.assertNotIn("private input detail", response.text)

    def test_unexpected_error_returns_safe_500(self) -> None:
        service = ready_service()
        service.analyze.side_effect = RuntimeError("secret internal failure")
        with (
            patch.object(voice_api, "voice_service", service),
            patch.object(voice_api, "voice_service_load_error", None),
            patch.object(voice_api.logger, "exception"),
        ):
            response = self.client.post(
                "/api/v1/voice/analyze",
                json={"transcript": "모비야 화면 읽어줘"},
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json()["detail"]["code"],
            "VOICE_ANALYSIS_FAILED",
        )
        self.assert_no_internal_details(response)


if __name__ == "__main__":
    unittest.main(verbosity=2)
