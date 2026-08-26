from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import joblib
import pandas as pd

from .schemas import FraudDetectionRequest
from .inference_feature_builder import (
    build_inference_features,
)


# ============================================================
# Exception
# ============================================================

class FraudInferenceError(RuntimeError):
    """
    FDS 추론 과정에서 발생하는 오류.
    """

    pass


# ============================================================
# Model Bundle Load
# ============================================================

def load_model_bundle(
    model_path: str | Path,
) -> dict[str, Any]:
    """
    train_iforest.py에서 저장한
    Isolation Forest Model Bundle을 로드한다.

    예상 Bundle 구조
    ----------------
    {
        "model": IsolationForest,
        "preprocessor": ColumnTransformer,
        ...
    }
    """

    model_path = Path(
        model_path
    )


    if not model_path.exists():

        raise FraudInferenceError(
            f"모델 파일을 찾을 수 없습니다: "
            f"{model_path}"
        )


    try:

        bundle = joblib.load(
            model_path
        )

    except Exception as error:

        raise FraudInferenceError(
            f"모델 Bundle 로드 실패: {error}"
        ) from error


    if not isinstance(
        bundle,
        dict,
    ):

        raise FraudInferenceError(
            "Model Bundle 형식이 dict가 아닙니다."
        )


    if "model" not in bundle:

        raise FraudInferenceError(
            "Model Bundle에 'model'이 없습니다."
        )


    if "preprocessor" not in bundle:

        raise FraudInferenceError(
            "Model Bundle에 'preprocessor'가 없습니다."
        )


    return bundle


# ============================================================
# Threshold
# ============================================================

def get_threshold(
    bundle: dict[str, Any],
    default: float = 0.446117,
) -> float:
    """
    Model Bundle 내부에 threshold가 있다면 사용하고,
    없다면 현재 Validation에서 얻은 threshold를 사용한다.

    현재 Baseline Best F1 Threshold:
        약 0.446117
    """

    threshold = bundle.get(
        "threshold"
    )


    if threshold is None:

        threshold = bundle.get(
            "best_threshold"
        )


    if threshold is None:

        threshold = default


    return float(
        threshold
    )


# ============================================================
# Feature Column 정리
# ============================================================

def prepare_model_features(
    feature_df: pd.DataFrame,
    bundle: dict[str, Any],
) -> pd.DataFrame:
    """
    Feature Engineering 결과에서
    Model Preprocessor에 전달할 컬럼을 정리한다.

    Bundle에 feature_names가 저장되어 있다면
    학습 당시 컬럼 순서를 그대로 맞춘다.
    """

    if feature_df.empty:

        raise FraudInferenceError(
            "모델에 전달할 Feature가 없습니다."
        )


    feature_names = bundle.get(
        "feature_names"
    )


    if not feature_names:

        return feature_df.copy()


    missing_columns = [
        column
        for column in feature_names
        if column not in feature_df.columns
    ]


    if missing_columns:

        raise FraudInferenceError(
            "학습 시 사용한 Feature가 "
            "Inference Data에 없습니다: "
            f"{missing_columns}"
        )


    return feature_df[
        feature_names
    ].copy()


# ============================================================
# Isolation Forest Prediction
# ============================================================

def predict_isolation_forest(
    *,
    feature_df: pd.DataFrame,
    bundle: dict[str, Any],
) -> dict[str, Any]:
    """
    Feature 1행을 Isolation Forest에 입력하고
    anomaly_score를 계산한다.

    sklearn IsolationForest:
        score_samples 값이 작을수록 이상치에 가까움

    MOVI API:
        anomaly_score가 클수록 위험하도록

        anomaly_score = -score_samples

    로 변환한다.
    """

    model = bundle[
        "model"
    ]

    preprocessor = bundle[
        "preprocessor"
    ]


    model_features = (
        prepare_model_features(
            feature_df,
            bundle,
        )
    )


    # ========================================================
    # Preprocessing
    # ========================================================

    try:

        transformed = (
            preprocessor.transform(
                model_features
            )
        )

    except Exception as error:

        raise FraudInferenceError(
            f"Preprocessor Transform 실패: "
            f"{error}"
        ) from error


    # ========================================================
    # Isolation Forest Score
    # ========================================================

    try:

        raw_score = float(
            model.score_samples(
                transformed
            )[0]
        )

    except Exception as error:

        raise FraudInferenceError(
            f"Isolation Forest 추론 실패: "
            f"{error}"
        ) from error


    anomaly_score = (
        -raw_score
    )


    threshold = get_threshold(
        bundle
    )


    is_anomaly = (
        anomaly_score
        >= threshold
    )


    return {

        "raw_score": raw_score,

        "anomaly_score": (
            anomaly_score
        ),

        "threshold": threshold,

        "is_anomaly": (
            is_anomaly
        ),

        "model": (
            "isolation_forest"
        ),
    }


# ============================================================
# Full Inference
# ============================================================

def detect_fraud(
    *,
    request: FraudDetectionRequest,
    bundle: dict[str, Any],
    feature_engineer: Callable[
        [pd.DataFrame],
        pd.DataFrame,
    ],
) -> dict[str, Any]:
    """
    하나의 거래에 대한 전체 Isolation Forest 추론.

    순서
    ----
    Request
        ↓
    API → AIHub Mapping
        ↓
    Historical Feature Engineering
        ↓
    현재 거래 Feature 추출
        ↓
    Preprocessor
        ↓
    Isolation Forest
        ↓
    anomaly_score
    """

    # ========================================================
    # 1. Feature 생성
    # ========================================================

    current_feature_df = (
        build_inference_features(

            request=request,

            feature_engineer=(
                feature_engineer
            ),
        )
    )


    # ========================================================
    # 2. Isolation Forest
    # ========================================================

    prediction = (
        predict_isolation_forest(

            feature_df=(
                current_feature_df
            ),

            bundle=bundle,
        )
    )


    # ========================================================
    # 3. Transaction 정보 추가
    # ========================================================

    return {

        "transaction_id": (
            request
            .current_transaction
            .transaction_id
        ),

        **prediction,
    }