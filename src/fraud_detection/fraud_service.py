"""MOVI FDS의 공식 추론 서비스.

FastAPI와 분리된 하나의 서비스에서 다음 흐름을 수행한다.

    Spring DTO → AIHub Mapping → Feature Engineering
    → Isolation Forest → Rule Engine → Final Risk Score
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
from scipy import sparse

from .config import ELECTRONIC_CONFIG
from .feature_engineering import engineer_electronic_features
from .risk_score import calculate_risk_score
from .rule_engine import FraudRuleEngine
from .schemas import FraudDetectionRequest
from .transaction_mapper import (
    find_current_transaction_index,
    request_to_dataframe,
)


logger = logging.getLogger(__name__)
FeatureEngineer = Callable[[Any], Any]


class InvalidTransactionRequest(ValueError):
    """클라이언트가 수정해야 하는 거래 요청 오류."""


class FraudDetectionServiceError(RuntimeError):
    """외부에 상세 내용을 노출하지 않아야 하는 FDS 내부 오류."""


class FraudDetectionService:
    """모델·규칙·위험점수를 결합한 단일 FDS 서비스."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        threshold_path: str | Path | None = None,
        threshold: float | None = None,
        feature_engineer: FeatureEngineer = engineer_electronic_features,
        rule_engine: FraudRuleEngine | None = None,
        *,
        autoload: bool = True,
    ) -> None:
        default_model_path = ELECTRONIC_CONFIG["model_path"]
        default_threshold_path = ELECTRONIC_CONFIG["threshold_path"]

        self.model_path = Path(
            model_path
            or os.getenv("FDS_MODEL_PATH", str(default_model_path))
        )
        self.threshold_path = Path(
            threshold_path
            or os.getenv("FDS_THRESHOLD_PATH", str(default_threshold_path))
        )
        self.threshold_override = threshold
        self.feature_engineer = feature_engineer
        self.rule_engine = rule_engine or FraudRuleEngine()

        self.bundle: dict[str, Any] | None = None
        self.model: Any = None
        self.preprocessor: Any = None
        self.threshold: float | None = None
        self.dataset_type = "electronic"
        self.model_load_error: str | None = None

        if autoload:
            self.load_resources()

    def load_resources(self) -> None:
        """모델, 전처리기, Threshold를 로드한다.

        실패를 상태로 보관하여 API 서버의 /health는 계속 응답할 수 있게 한다.
        """

        self.bundle = None
        self.model = None
        self.preprocessor = None
        self.threshold = None
        self.model_load_error = None

        try:
            if not self.model_path.exists():
                raise FileNotFoundError("학습된 모델 파일이 없습니다.")

            loaded_bundle = joblib.load(self.model_path)
            if not isinstance(loaded_bundle, dict):
                raise ValueError("모델 Bundle 형식이 올바르지 않습니다.")

            loaded_model = loaded_bundle.get("model")
            loaded_preprocessor = loaded_bundle.get("preprocessor")
            if loaded_model is None or loaded_preprocessor is None:
                raise ValueError("모델 Bundle에 필수 구성요소가 없습니다.")

            loaded_threshold = self.threshold_override
            if loaded_threshold is None:
                if not self.threshold_path.exists():
                    raise FileNotFoundError("Threshold 파일이 없습니다.")
                with self.threshold_path.open("r", encoding="utf-8") as file:
                    threshold_config = json.load(file)
                loaded_threshold = float(threshold_config["threshold"])

            loaded_threshold = float(loaded_threshold)
            if not np.isfinite(loaded_threshold):
                raise ValueError("Threshold는 유한한 숫자여야 합니다.")

            self.bundle = loaded_bundle
            self.model = loaded_model
            self.preprocessor = loaded_preprocessor
            self.threshold = loaded_threshold
            self.dataset_type = loaded_bundle.get("dataset_type", "electronic")
            logger.info("FDS model resources loaded successfully")

        except Exception as error:
            self.model_load_error = type(error).__name__
            logger.exception("Failed to load FDS model resources")

    def ready(self) -> bool:
        """전체 추론에 필요한 리소스가 준비됐는지 반환한다."""

        return (
            self.model is not None
            and self.preprocessor is not None
            and self.threshold is not None
            and self.model_load_error is None
        )

    @staticmethod
    def validate_unique_transaction_ids(
        request: FraudDetectionRequest,
    ) -> None:
        """현재 거래와 History 전체에서 거래 ID 중복을 차단한다."""

        transactions = [
            request.current_transaction,
            *(request.history or []),
        ]
        transaction_ids = [
            str(transaction.transaction_id).strip()
            for transaction in transactions
        ]

        if len(transaction_ids) != len(set(transaction_ids)):
            raise InvalidTransactionRequest(
                "transaction_id는 current_transaction과 history에서 "
                "중복될 수 없습니다."
            )

    def detect(
        self,
        request: FraudDetectionRequest,
    ) -> dict[str, Any]:
        """현재 거래 한 건의 모델·규칙·최종 위험 결과를 반환한다."""

        if not self.ready():
            raise FraudDetectionServiceError(
                "이상거래 탐지 모델이 준비되지 않았습니다."
            )

        self.validate_unique_transaction_ids(request)

        try:
            dataframe = request_to_dataframe(request)
            if dataframe.empty:
                raise InvalidTransactionRequest("거래 데이터가 없습니다.")

            transaction_id = request.current_transaction.transaction_id
            current_index = find_current_transaction_index(
                dataframe=dataframe,
                transaction_id=transaction_id,
            )

            logger.info(
                "FDS request mapped: total_rows=%d current_index=%d",
                len(dataframe),
                current_index,
            )

            feature_input = dataframe.drop(
                columns=["_transaction_id", "_transaction_datetime"],
                errors="ignore",
            )
            engineered = self.feature_engineer(feature_input)
            if engineered.empty:
                raise FraudDetectionServiceError(
                    "Feature Engineering 결과가 비어 있습니다."
                )
            if current_index >= len(engineered):
                raise FraudDetectionServiceError(
                    "현재 거래 Feature 위치가 결과 범위를 벗어났습니다."
                )

            current_features = engineered.iloc[[current_index]]
            feature_values = current_features.iloc[0].to_dict()

            rule_result = self.rule_engine.evaluate(feature_values)
            rule_score = float(rule_result["rule_score"])
            triggered_rules = list(rule_result["triggered_rules"])

            transformed = self.preprocessor.transform(current_features)
            transformed = transformed.astype(np.float32)
            if sparse.issparse(transformed):
                transformed = transformed.tocsr().astype(np.float32)

            raw_score = float(self.model.score_samples(transformed)[0])
            anomaly_score = -raw_score
            is_anomaly = anomaly_score >= self.threshold

            risk_result = calculate_risk_score(
                anomaly_score=anomaly_score,
                rule_score=rule_score,
            )

            logger.info(
                "FDS inference completed: anomaly=%s rule_risk=%.2f "
                "final_risk=%.2f risk_level=%s",
                is_anomaly,
                rule_score,
                risk_result.final_risk_score,
                risk_result.risk_level,
            )

            return {
                "transaction_id": transaction_id,
                "anomaly_score": round(anomaly_score, 6),
                "threshold": round(self.threshold, 6),
                "is_anomaly": bool(is_anomaly),
                "model": "isolation_forest",
                "rule_score": round(rule_score, 2),
                "final_risk_score": risk_result.final_risk_score,
                "risk_level": risk_result.risk_level,
                "triggered_rules": triggered_rules,
            }

        except InvalidTransactionRequest:
            raise
        except FraudDetectionServiceError:
            raise
        except Exception as error:
            raise FraudDetectionServiceError(
                "이상거래 추론 과정에서 내부 오류가 발생했습니다."
            ) from error
