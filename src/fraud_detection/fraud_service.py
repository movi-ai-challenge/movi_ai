from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib

from .schemas import FraudDetectionRequest

from .transaction_mapper import (
    request_to_dataframe,
    find_current_transaction_index,
)

from .feature_engineering import (
    transform_features,
)


# ============================================================
# Config
# ============================================================

DEFAULT_MODEL_PATH = (
    "models/electronic/isolation_forest.joblib"
)

DEFAULT_THRESHOLD = 0.44611697


# ============================================================
# Exception
# ============================================================

class FraudDetectionServiceError(
    RuntimeError
):
    """
    FDS Service에서 발생하는 오류.
    """

    pass


# ============================================================
# Fraud Detection Service
# ============================================================

class FraudDetectionService:
    """
    MOVI 이상거래 탐지 Service.

    역할
    ----
    1. 저장된 Model Bundle Load
    2. Spring 거래 Request 변환
    3. Historical Feature Engineering
    4. 기존 Preprocessor Transform
    5. Isolation Forest 추론
    6. anomaly_score 반환

    주의
    ----
    추론에서는 Preprocessor를 절대 다시 fit하지 않는다.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        threshold: float | None = None,
    ):

        # ====================================================
        # Model Path
        # ====================================================

        if model_path is None:

            model_path = os.getenv(
                "FDS_MODEL_PATH",
                DEFAULT_MODEL_PATH,
            )

        self.model_path = Path(
            model_path
        )


        # ====================================================
        # Threshold
        # ====================================================

        if threshold is None:

            threshold = float(
                os.getenv(
                    "FDS_THRESHOLD",
                    str(DEFAULT_THRESHOLD),
                )
            )

        self.threshold = threshold


        # ====================================================
        # Model Bundle Load
        # ====================================================

        self.bundle = (
            self._load_model_bundle()
        )


        self.model = (
            self.bundle["model"]
        )

        self.preprocessor = (
            self.bundle["preprocessor"]
        )


        self.dataset_type = (
            self.bundle.get(
                "dataset_type",
                "electronic",
            )
        )


    # ============================================================
    # Model Bundle Load
    # ============================================================

    def _load_model_bundle(
        self,
    ) -> dict[str, Any]:

        if not self.model_path.exists():

            raise FraudDetectionServiceError(
                "Isolation Forest 모델 파일을 "
                f"찾을 수 없습니다: "
                f"{self.model_path}"
            )


        try:

            bundle = joblib.load(
                self.model_path
            )

        except Exception as error:

            raise FraudDetectionServiceError(
                "Isolation Forest Model Bundle "
                f"로드 실패: {error}"
            ) from error


        if not isinstance(
            bundle,
            dict,
        ):

            raise FraudDetectionServiceError(
                "Model Bundle이 dict 형식이 아닙니다."
            )


        required_keys = [
            "model",
            "preprocessor",
        ]


        missing_keys = [

            key

            for key in required_keys

            if key not in bundle

        ]


        if missing_keys:

            raise FraudDetectionServiceError(
                "Model Bundle 필수 값이 없습니다: "
                f"{missing_keys}"
            )


        return bundle


    # ============================================================
    # Fraud Detection
    # ============================================================

    def detect(
        self,
        request: FraudDetectionRequest,
    ) -> dict[str, Any]:
        """
        거래 한 건에 대해 이상거래 탐지를 수행한다.

        Spring에서:
            current_transaction
            history

        를 전달하면 현재 거래의 이상도를 반환한다.
        """

        # ====================================================
        # 1. API Request → AIHub DataFrame
        # ====================================================

        try:

            raw_df = request_to_dataframe(
                request
            )

        except Exception as error:

            raise FraudDetectionServiceError(
                f"거래 데이터 변환 실패: {error}"
            ) from error


        if raw_df.empty:

            raise FraudDetectionServiceError(
                "거래 데이터가 없습니다."
            )


        # ====================================================
        # 2. 현재 거래 Index
        # ====================================================

        current_transaction_id = (
            request
            .current_transaction
            .transaction_id
        )


        try:

            current_index = (
                find_current_transaction_index(

                    dataframe=raw_df,

                    transaction_id=(
                        current_transaction_id
                    ),
                )
            )

        except Exception as error:

            raise FraudDetectionServiceError(
                "현재 거래를 찾지 못했습니다: "
                f"{error}"
            ) from error


        # ====================================================
        # 3. Feature Engineering + Transform
        #
        # 기존 학습 Preprocessor 사용
        #
        # 절대 fit_transform() 하지 않음
        # ====================================================

        try:

            X = transform_features(

                df=raw_df,

                dataset_type=(
                    self.dataset_type
                ),

                preprocessor=(
                    self.preprocessor
                ),
            )

        except Exception as error:

            raise FraudDetectionServiceError(
                "Feature Transform 실패: "
                f"{error}"
            ) from error


        # ====================================================
        # 4. 현재 거래 한 행 선택
        #
        # transform_features는 Row 순서를 유지하므로
        # Raw DataFrame에서 찾은 Index를 그대로 사용한다.
        # ====================================================

        try:

            current_X = X[
                current_index:
                current_index + 1
            ]

        except Exception as error:

            raise FraudDetectionServiceError(
                "현재 거래 Feature 추출 실패: "
                f"{error}"
            ) from error


        # ====================================================
        # 5. Isolation Forest
        # ====================================================

        try:

            raw_score = float(

                self.model.score_samples(
                    current_X
                )[0]

            )

        except Exception as error:

            raise FraudDetectionServiceError(
                "Isolation Forest 추론 실패: "
                f"{error}"
            ) from error


        # ====================================================
        # 6. Score 방향 변환
        #
        # sklearn:
        #   더 작을수록 이상
        #
        # MOVI:
        #   더 클수록 위험
        # ====================================================

        anomaly_score = (
            -raw_score
        )


        # ====================================================
        # 7. Threshold
        # ====================================================

        is_anomaly = (
            anomaly_score
            >=
            self.threshold
        )


        # ====================================================
        # 8. 결과
        # ====================================================

        return {

            "transaction_id": (
                current_transaction_id
            ),

            "raw_score": (
                raw_score
            ),

            "anomaly_score": (
                anomaly_score
            ),

            "threshold": (
                self.threshold
            ),

            "is_anomaly": (
                is_anomaly
            ),

            "model": (
                "isolation_forest"
            ),

            "history_count": (
                len(
                    request.history
                )
            ),
        }