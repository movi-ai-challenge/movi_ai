"""MOVI FDS API 입력 검증·장애 처리 테스트.

프로젝트 루트에서 실행한다.

    python tests/test_fraud_api_errors.py

이 테스트는 별도의 uvicorn 서버를 실행하지 않아도 된다. FastAPI 앱을
프로세스 내부에서 호출하며, 모델 미준비와 내부 오류는 mock으로 재현한다.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fraud_detection import api as fraud_api  # noqa: E402


def transaction(
    *,
    transaction_id: str = "api-test-current",
    amount: int | float = 100_000,
    transaction_datetime: str = "2026-08-26T12:30:00",
) -> dict:
    return {
        "transaction_id": transaction_id,
        "sender_account": "100001",
        "receiver_account": "200001",
        "sender_bank": "KB",
        "receiver_bank": "KB",
        "transaction_type": "transfer",
        "amount": amount,
        "transaction_datetime": transaction_datetime,
        "medium": "MOBILE",
    }


def valid_payload() -> dict:
    return {
        "current_transaction": transaction(),
        "history": [
            transaction(
                transaction_id="api-test-history",
                amount=90_000,
                transaction_datetime="2026-08-25T12:30:00",
            )
        ],
    }


class FraudApiErrorTest(unittest.TestCase):
    """API 경계 입력과 장애 응답을 검증한다."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(
            fraud_api.app,
            raise_server_exceptions=False,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()

    def assert_no_internal_details(self, response) -> None:
        """클라이언트 응답에 내부 경로나 Traceback이 없는지 확인한다."""

        body = response.text
        forbidden_fragments = (
            "Traceback",
            "/Users/",
            "/workspace/",
            "site-packages",
            "FileNotFoundError",
            "KeyError",
            "RuntimeError",
        )
        for fragment in forbidden_fragments:
            self.assertNotIn(fragment, body)

    def assert_validation_error(self, response) -> None:
        """Schema 또는 업무 입력 오류가 안전한 4xx인지 확인한다."""

        self.assertIn(response.status_code, {400, 422}, response.text)
        self.assert_no_internal_details(response)

        body = response.json()
        self.assertIn("detail", body)
        if response.status_code == 400:
            self.assertIsInstance(body["detail"], dict)
            self.assertEqual(
                body["detail"].get("code"),
                "INVALID_TRANSACTION_REQUEST",
            )
        else:
            self.assertIsInstance(body["detail"], list)

    def test_health_response_is_available_and_safe(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["service"], "fraud-detection")
        self.assertIn(body["status"], {"ok", "degraded"})
        self.assertIsInstance(body["model_loaded"], bool)
        self.assert_no_internal_details(response)

    def test_ready_matches_current_model_state(self) -> None:
        health = self.client.get("/health").json()
        response = self.client.get("/ready")

        if health["model_loaded"]:
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "ready")
        else:
            self.assertEqual(response.status_code, 503)
            self.assertEqual(
                response.json()["detail"]["code"],
                "MODEL_NOT_READY",
            )
        self.assert_no_internal_details(response)

    def test_malformed_json_is_rejected(self) -> None:
        response = self.client.post(
            "/api/v1/fraud/detect",
            content=b'{"current_transaction":',
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 422)
        self.assert_no_internal_details(response)

    def test_empty_request_is_rejected(self) -> None:
        response = self.client.post("/api/v1/fraud/detect", json={})
        self.assertEqual(response.status_code, 422)
        self.assert_no_internal_details(response)

    def test_missing_amount_is_rejected(self) -> None:
        payload = valid_payload()
        del payload["current_transaction"]["amount"]

        response = self.client.post("/api/v1/fraud/detect", json=payload)
        self.assertEqual(response.status_code, 422)
        self.assert_no_internal_details(response)

    def test_zero_amount_is_rejected(self) -> None:
        payload = valid_payload()
        payload["current_transaction"]["amount"] = 0

        response = self.client.post("/api/v1/fraud/detect", json=payload)
        self.assert_validation_error(response)

    def test_negative_amount_is_rejected(self) -> None:
        payload = valid_payload()
        payload["current_transaction"]["amount"] = -1

        response = self.client.post("/api/v1/fraud/detect", json=payload)
        self.assert_validation_error(response)

    def test_invalid_datetime_is_rejected(self) -> None:
        payload = valid_payload()
        payload["current_transaction"]["transaction_datetime"] = "not-a-date"

        response = self.client.post("/api/v1/fraud/detect", json=payload)
        self.assert_validation_error(response)

    def test_history_must_be_a_list(self) -> None:
        payload = valid_payload()
        payload["history"] = {"unexpected": "object"}

        response = self.client.post("/api/v1/fraud/detect", json=payload)
        self.assertEqual(response.status_code, 422)
        self.assert_no_internal_details(response)

    def test_duplicate_transaction_id_is_rejected(self) -> None:
        payload = valid_payload()
        payload["history"][0]["transaction_id"] = (
            payload["current_transaction"]["transaction_id"]
        )

        response = self.client.post("/api/v1/fraud/detect", json=payload)
        self.assert_validation_error(response)

    def test_model_not_ready_returns_503(self) -> None:
        service = Mock()
        service.ready.return_value = False

        with patch.object(fraud_api, "fraud_service", service):
            ready_response = self.client.get("/ready")
            detect_response = self.client.post(
                "/api/v1/fraud/detect",
                json=valid_payload(),
            )

        for response in (ready_response, detect_response):
            self.assertEqual(response.status_code, 503)
            self.assertEqual(
                response.json()["detail"]["code"],
                "MODEL_NOT_READY",
            )
            self.assert_no_internal_details(response)

    def test_unexpected_internal_error_is_hidden(self) -> None:
        service = Mock()
        service.ready.return_value = True
        service.detect.side_effect = RuntimeError("secret internal failure")

        with (
            patch.object(fraud_api, "fraud_service", service),
            patch.object(fraud_api.logger, "exception"),
        ):
            response = self.client.post(
                "/api/v1/fraud/detect",
                json=valid_payload(),
            )

        self.assertEqual(response.status_code, 500)
        body = response.json()
        self.assertEqual(body["detail"]["code"], "INTERNAL_FDS_ERROR")
        self.assertEqual(
            body["detail"]["message"],
            "이상거래 분석 중 내부 오류가 발생했습니다.",
        )
        self.assertNotIn("secret internal failure", response.text)
        self.assert_no_internal_details(response)


if __name__ == "__main__":
    unittest.main(verbosity=2)
