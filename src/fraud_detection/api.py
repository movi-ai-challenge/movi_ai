from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from scipy import sparse


try:
    from .config import (
        ELECTRONIC_CONFIG,
    )

    from .feature_engineering import (
        engineer_electronic_features,
    )

    from .schemas import (
        FraudDetectionRequest,
        FraudDetectionResponse,
    )

except ImportError:

    from config import (
        ELECTRONIC_CONFIG,
    )

    from feature_engineering import (
        engineer_electronic_features,
    )

    from schemas import (
        FraudDetectionRequest,
        FraudDetectionResponse,
    )


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="MOVI Fraud Detection API",
    version="0.1.0",
)


# ============================================================
# Model Load
# ============================================================

MODEL_PATH = (
    ELECTRONIC_CONFIG[
        "model_path"
    ]
)

THRESHOLD_PATH = (
    ELECTRONIC_CONFIG[
        "threshold_path"
    ]
)


if not MODEL_PATH.exists():

    raise RuntimeError(
        f"학습된 모델이 없습니다: "
        f"{MODEL_PATH}"
    )


bundle = joblib.load(
    MODEL_PATH
)

model = bundle[
    "model"
]

preprocessor = bundle[
    "preprocessor"
]


# ============================================================
# Threshold Load
# ============================================================

if not THRESHOLD_PATH.exists():

    raise RuntimeError(
        f"Threshold 파일이 없습니다: "
        f"{THRESHOLD_PATH}"
    )


with open(
    THRESHOLD_PATH,
    "r",
    encoding="utf-8",
) as file:

    threshold_config = (
        json.load(file)
    )


threshold = float(
    threshold_config[
        "threshold"
    ]
)


# ============================================================
# Request → AIHub 형식 변환
# ============================================================

def convert_transaction(
    transaction,
) -> dict:

    return {

        "출금계좌일련번호":
            transaction.sender_account,

        "입금계좌일련번호":
            transaction.receiver_account,

        "출금금융회사일련번호":
            transaction.sender_bank,

        "입금금융회사일련번호":
            transaction.receiver_bank,

        "자금구분":
            transaction.transaction_type,

        "거래금액":
            transaction.amount,

        "거래시간대":
            transaction.transaction_hour,

        "매체구분":
            transaction.medium,

        "거래일자":
            transaction.transaction_date,
    }


# ============================================================
# Health Check
# ============================================================

@app.get(
    "/health"
)
def health_check():

    return {
        "status": "ok",
        "service": "fraud-detection",
        "model_loaded": True,
    }


# ============================================================
# Fraud Detection
# ============================================================

@app.post(
    "/api/v1/fraud/detect",
    response_model=FraudDetectionResponse,
)
def detect_fraud(
    request: FraudDetectionRequest,
):

    try:

        # ====================================================
        # 1. History 변환
        # ====================================================

        rows = []

        for transaction in request.history:

            rows.append(
                convert_transaction(
                    transaction
                )
            )


        # ====================================================
        # 2. 현재 거래 마지막에 추가
        # ====================================================

        rows.append(
            convert_transaction(
                request.transaction
            )
        )


        df = pd.DataFrame(
            rows
        )


        # ====================================================
        # 3. Feature Engineering
        # ====================================================

        engineered_df = (
            engineer_electronic_features(
                df
            )
        )


        # ====================================================
        # 4. 현재 거래 Feature만 사용
        # ====================================================

        current_features = (
            engineered_df
            .iloc[
                [-1]
            ]
        )


        # ====================================================
        # 5. Train Preprocessor
        # ====================================================

        X = (
            preprocessor
            .transform(
                current_features
            )
        )


        X = X.astype(
            np.float32
        )


        if sparse.issparse(X):

            X = (
                X
                .tocsr()
                .astype(
                    np.float32
                )
            )


        # ====================================================
        # 6. Isolation Forest
        # ====================================================

        raw_score = float(
            model.score_samples(
                X
            )[0]
        )


        anomaly_score = (
            -raw_score
        )


        # ====================================================
        # 7. Threshold
        # ====================================================

        is_fraud = (
            anomaly_score
            >= threshold
        )


        # ====================================================
        # 8. 임시 Risk Level
        #
        # 이후 Risk Score 구현 시 교체
        # ====================================================

        if is_fraud:

            risk_level = "HIGH"

        else:

            risk_level = "LOW"


        # ====================================================
        # Response
        # ====================================================

        return FraudDetectionResponse(

            transaction_id=(
                request.transaction_id
            ),

            anomaly_score=round(
                anomaly_score,
                6,
            ),

            threshold=round(
                threshold,
                6,
            ),

            is_fraud=bool(
                is_fraud
            ),

            risk_level=(
                risk_level
            ),

            model=(
                "isolation_forest"
            ),
        )


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )