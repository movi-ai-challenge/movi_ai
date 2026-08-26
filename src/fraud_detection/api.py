from __future__ import annotations

import json

import joblib
import numpy as np

from fastapi import (
    FastAPI,
    HTTPException,
)

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

    from .transaction_mapper import (
        request_to_dataframe,
        find_current_transaction_index,
    )

    from .rule_engine import (
        FraudRuleEngine,
    )

    from .risk_score import (
        calculate_risk_score,
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

    from transaction_mapper import (
        request_to_dataframe,
        find_current_transaction_index,
    )

    from rule_engine import (
        FraudRuleEngine,
    )

    from risk_score import (
        calculate_risk_score,
    )


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="MOVI Fraud Detection API",
    description=(
        "Isolation Forest + Rule Engine 기반 "
        "MOVI 이상거래 탐지 API"
    ),
    version="0.4.0",
)


# ============================================================
# Model / Threshold Config
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


# ============================================================
# Model Load
#
# 서버 시작 시 한 번만 Load.
# 요청마다 다시 로드하지 않는다.
# ============================================================

if not MODEL_PATH.exists():

    raise RuntimeError(
        "학습된 모델이 없습니다: "
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
        "Threshold 파일이 없습니다: "
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
# Rule Engine
# ============================================================

rule_engine = (
    FraudRuleEngine()
)


# ============================================================
# Root
# ============================================================

@app.get("/")
def root():

    return {

        "service": (
            "MOVI Fraud Detection API"
        ),

        "status": "running",

        "version": "0.4.0",

        "components": [
            "isolation_forest",
            "rule_engine",
            "risk_score",
        ],
    }


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
def health_check():

    return {

        "status": "ok",

        "service": (
            "fraud-detection"
        ),

        "model_loaded": (
            model is not None
        ),

        "model": (
            "isolation_forest"
        ),

        "rule_engine_loaded": (
            rule_engine is not None
        ),

        "risk_score_enabled": True,

        "threshold": round(
            threshold,
            6,
        ),
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
        # 1. API Request
        #       ↓
        # AIHub 스타일 DataFrame
        #
        # history + current transaction
        # ====================================================

        df = (
            request_to_dataframe(
                request
            )
        )


        if df.empty:

            raise ValueError(
                "거래 데이터가 없습니다."
            )


        # ====================================================
        # 2. 현재 거래 ID
        # ====================================================

        current_transaction_id = (
            request
            .current_transaction
            .transaction_id
        )


        # ====================================================
        # 3. 현재 거래 Index 확인
        # ====================================================

        current_index = (
            find_current_transaction_index(

                dataframe=df,

                transaction_id=(
                    current_transaction_id
                ),
            )
        )


        # ====================================================
        # Debug - Raw Transaction
        # ====================================================

        print()
        print("=" * 70)
        print("[RAW TRANSACTION DF]")
        print("=" * 70)

        print(
            df.to_string(
                index=False
            )
        )

        print()

        print(
            "[CURRENT INDEX]",
            current_index,
        )


        # ====================================================
        # 4. Feature Engineering 입력 생성
        #
        # API 내부 추적용 Column 제거
        # ====================================================

        feature_input_df = (
            df.drop(
                columns=[
                    "_transaction_id",
                    "_transaction_datetime",
                ],
                errors="ignore",
            )
        )


        # ====================================================
        # 5. Feature Engineering
        #
        # history + current 전체를 함께 넣는다.
        #
        # Historical Feature가 현재 거래의
        # 이전 거래 기준으로 계산되어야 하기 때문.
        # ====================================================

        engineered_df = (
            engineer_electronic_features(
                feature_input_df
            )
        )


        if engineered_df.empty:

            raise ValueError(
                "Feature Engineering 결과가 "
                "비어 있습니다."
            )


        # ====================================================
        # Debug - Engineered Features
        # ====================================================

        print()
        print("=" * 70)
        print("[ENGINEERED FEATURES]")
        print("=" * 70)

        print(
            engineered_df.to_string(
                index=False
            )
        )


        # ====================================================
        # 6. 현재 거래 Feature
        # ====================================================

        if (
            current_index
            >=
            len(
                engineered_df
            )
        ):

            raise ValueError(

                "현재 거래 Index가 "
                "Feature Engineering 결과 범위를 "
                "벗어났습니다. "

                f"current_index={current_index}, "

                f"rows={len(engineered_df)}"
            )


        current_features = (
            engineered_df.iloc[
                [
                    current_index
                ]
            ]
        )


        # ====================================================
        # 7. Rule Engine
        #
        # Preprocessor 적용 전 원본 Feature 사용
        # ====================================================

        current_feature_dict = (
            current_features
            .iloc[0]
            .to_dict()
        )


        rule_result = (
            rule_engine.evaluate(
                current_feature_dict
            )
        )


        rule_score = float(
            rule_result[
                "rule_score"
            ]
        )


        triggered_rules = (
            rule_result[
                "triggered_rules"
            ]
        )


        # ====================================================
        # Debug - Rule Engine
        # ====================================================

        print()
        print("=" * 70)
        print("[RULE ENGINE]")
        print("=" * 70)

        print(
            "rule_score:",
            rule_score,
        )

        print(
            "triggered_rules:",
            triggered_rules,
        )


        for detail in (
            rule_result[
                "rule_details"
            ]
        ):

            if detail[
                "triggered"
            ]:

                print(
                    "-",
                    detail[
                        "name"
                    ],
                    ":",
                    detail[
                        "reason"
                    ],
                    f"(+{detail['score']})",
                )


        # ====================================================
        # 8. 저장된 Preprocessor
        #
        # 추론에서는 반드시 transform만 사용.
        #
        # fit_transform X
        # transform O
        # ====================================================

        X = (
            preprocessor.transform(
                current_features
            )
        )


        # ====================================================
        # 9. float32 통일
        # ====================================================

        X = X.astype(
            np.float32
        )


        if sparse.issparse(
            X
        ):

            X = (
                X
                .tocsr()
                .astype(
                    np.float32
                )
            )


        # ====================================================
        # 10. Isolation Forest
        # ====================================================

        raw_score = float(

            model.score_samples(
                X
            )[0]

        )


        # ====================================================
        # 11. Anomaly Score
        #
        # sklearn score_samples:
        # 작은 값 → 이상
        #
        # MOVI:
        # 큰 값 → 위험
        #
        # 따라서 부호 반전
        # ====================================================

        anomaly_score = (
            -raw_score
        )


        # ====================================================
        # 12. Isolation Forest 판단
        # ====================================================

        is_anomaly = (
            anomaly_score
            >=
            threshold
        )


        # ====================================================
        # 13. Final Risk Score
        #
        # anomaly_score
        #     ↓
        # model_risk_score 0~100
        #
        # +
        #
        # rule_score 0~100
        #
        #     ↓
        #
        # final_risk_score 0~100
        # ====================================================

        risk_result = (
            calculate_risk_score(

                anomaly_score=(
                    anomaly_score
                ),

                rule_score=(
                    rule_score
                ),
            )
        )


        model_risk_score = (
            risk_result
            .model_risk_score
        )


        final_risk_score = (
            risk_result
            .final_risk_score
        )


        risk_level = (
            risk_result
            .risk_level
        )


        # ====================================================
        # Debug - Isolation Forest
        # ====================================================

        print()
        print("=" * 70)
        print("[ISOLATION FOREST]")
        print("=" * 70)

        print(
            "transaction_id:",
            current_transaction_id,
        )

        print(
            "raw_score:",
            raw_score,
        )

        print(
            "anomaly_score:",
            anomaly_score,
        )

        print(
            "threshold:",
            threshold,
        )

        print(
            "is_anomaly:",
            is_anomaly,
        )


        # ====================================================
        # Debug - Final Risk
        # ====================================================

        print()
        print("=" * 70)
        print("[FINAL RISK]")
        print("=" * 70)

        print(
            "model_risk_score:",
            model_risk_score,
        )

        print(
            "rule_score:",
            rule_score,
        )

        print(
            "final_risk_score:",
            final_risk_score,
        )

        print(
            "risk_level:",
            risk_level,
        )


        # ====================================================
        # 14. API Response
        # ====================================================

        return FraudDetectionResponse(

            transaction_id=(
                current_transaction_id
            ),

            anomaly_score=round(
                anomaly_score,
                6,
            ),

            threshold=round(
                threshold,
                6,
            ),

            is_anomaly=bool(
                is_anomaly
            ),

            model=(
                "isolation_forest"
            ),

            rule_score=round(
                rule_score,
                2,
            ),

            final_risk_score=(
                final_risk_score
            ),

            risk_level=(
                risk_level
            ),

            triggered_rules=(
                triggered_rules
            ),
        )


    # ========================================================
    # Bad Request
    # ========================================================

    except ValueError as error:

        print()
        print(
            "[FDS BAD REQUEST]",
            str(error),
        )


        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


    # ========================================================
    # Internal Error
    # ========================================================

    except Exception as error:

        print()
        print(
            "[FDS INTERNAL ERROR]",
            type(error).__name__,
            str(error),
        )


        raise HTTPException(

            status_code=500,

            detail=(
                f"{type(error).__name__}: "
                f"{str(error)}"
            ),

        ) from error