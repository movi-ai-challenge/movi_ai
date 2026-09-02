"""공식 FraudDetectionService의 통합 단위 테스트."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fraud_detection.fraud_service import (  # noqa: E402
    FraudDetectionService,
    FraudDetectionServiceError,
    InvalidTransactionRequest,
)
from src.fraud_detection.schemas import (  # noqa: E402
    FraudDetectionRequest,
    TransactionData,
)


class FakePreprocessor:
    def transform(self, dataframe):
        return np.zeros((len(dataframe), 1), dtype=np.float64)


class FakeIsolationForest:
    def score_samples(self, transformed):
        return np.full(len(transformed), -0.48, dtype=np.float64)


def fake_feature_engineer(dataframe):
    result = dataframe.copy()
    result["amount_ratio"] = 10.0
    result["amount_zscore"] = 6.0
    result["is_night"] = 1
    result["new_recipient"] = 1
    result["unusual_medium"] = 1
    result["same_bank"] = 0
    result["historical_transaction_count"] = 5
    result["same_day_transaction_count"] = 0
    result["same_time_bucket_count"] = 0
    return result


def transaction(
    transaction_id: str,
    *,
    amount: float = 100_000,
    datetime_text: str = "2026-08-26T12:30:00",
) -> TransactionData:
    return TransactionData(
        transaction_id=transaction_id,
        sender_account="100001",
        receiver_account="200001",
        sender_bank="KB",
        receiver_bank="KB",
        transaction_type="transfer",
        amount=amount,
        transaction_datetime=datetime_text,
        medium="MOBILE",
    )


def request(*, duplicate_id: bool = False) -> FraudDetectionRequest:
    current_id = "service-current"
    return FraudDetectionRequest(
        current_transaction=transaction(
            current_id,
            amount=10_000_000,
        ),
        history=[
            transaction(
                current_id if duplicate_id else "service-history",
                datetime_text="2026-08-25T12:30:00",
            )
        ],
    )


def ready_service() -> FraudDetectionService:
    service = FraudDetectionService(
        feature_engineer=fake_feature_engineer,
        autoload=False,
    )
    service.bundle = {
        "model": FakeIsolationForest(),
        "preprocessor": FakePreprocessor(),
    }
    service.model = service.bundle["model"]
    service.preprocessor = service.bundle["preprocessor"]
    service.threshold = 0.446117
    service.model_load_error = None
    return service


class FraudDetectionServiceTest(unittest.TestCase):
    def test_autoload_disabled_service_is_not_ready(self) -> None:
        service = FraudDetectionService(autoload=False)

        self.assertFalse(service.ready())

    def test_not_ready_service_rejects_detection(self) -> None:
        service = FraudDetectionService(autoload=False)

        with self.assertRaises(FraudDetectionServiceError):
            service.detect(request())

    def test_duplicate_transaction_id_is_rejected_in_service(self) -> None:
        service = ready_service()

        with self.assertRaises(InvalidTransactionRequest):
            service.detect(request(duplicate_id=True))

    def test_model_rule_and_risk_are_returned_from_one_service(self) -> None:
        result = ready_service().detect(request())

        self.assertEqual(result["transaction_id"], "service-current")
        self.assertAlmostEqual(result["anomaly_score"], 0.48)
        self.assertEqual(result["threshold"], 0.446117)
        self.assertTrue(result["is_anomaly"])
        self.assertGreater(result["rule_score"], 0)
        self.assertGreater(result["final_risk_score"], 0)
        self.assertEqual(result["risk_level"], "HIGH")
        self.assertIn("HIGH_AMOUNT_RATIO", result["triggered_rules"])
        self.assertIn("EXTREME_AMOUNT_ZSCORE", result["triggered_rules"])

    def test_internal_feature_error_is_wrapped(self) -> None:
        service = ready_service()

        def broken_feature_engineer(dataframe):
            raise RuntimeError("secret feature failure")

        service.feature_engineer = broken_feature_engineer

        with self.assertRaises(FraudDetectionServiceError) as context:
            service.detect(request())

        self.assertNotIn("secret feature failure", str(context.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
