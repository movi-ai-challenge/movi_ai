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


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="MOVI Fraud Detection API",
    description="MOVI 이상거래 탐지 API",
    version="0.2.0",
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
# 서버 실행 시 모델을 한 번만 로드한다.
# 요청마다 joblib.load()를 수행하지 않는다.
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
# Root
# ============================================================

@app.get("/")
def root():

    return {

        "service": (
            "MOVI Fraud Detection API"
        ),

        "status": "running",
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
        # 1. Request → AIHub 형식 DataFrame
        #
        # history
        #   +
        # current_transaction
        #
        # 전체를 DataFrame으로 변환한다.
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
        # 3. 현재 거래 위치 확인
        #
        # Feature Engineering 전에
        # 현재 거래 위치를 확보한다.
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
        # 4. Feature Engineering용 DataFrame 생성
        #
        # 아래 컬럼은 API 내부 추적용이므로
        # AIHub Feature Engineering에서는 제거한다.
        #
        # _transaction_id
        # _transaction_datetime
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
        # Debug - Feature Input
        # ====================================================

        print()
        print("=" * 70)
        print("[FEATURE INPUT]")
        print("=" * 70)

        print(
            feature_input_df.to_string(
                index=False
            )
        )


        # ====================================================
        # 5. Feature Engineering
        #
        # 중요:
        #
        # 현재 거래만 넣는 것이 아니라
        #
        # history + current
        #
        # 전체를 함께 넣어야 한다.
        #
        # 그래야:
        #
        # amount_ratio
        # amount_zscore
        # new_recipient
        # unusual_medium
        # historical_transaction_count
        # same_day_transaction_count
        # same_time_bucket_count
        #
        # 등이 정상 계산된다.
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

        print()

        print(
            "[ENGINEERED COLUMNS]"
        )

        print(
            engineered_df
            .columns
            .tolist()
        )


        # ====================================================
        # 6. 현재 거래 Feature만 추출
        #
        # transaction_mapper에서 시간순 정렬 후
        # current_index를 찾았고,
        #
        # engineer_electronic_features가 Row 개수와
        # Row 순서를 유지한다는 전제.
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
        # 7. 저장된 Preprocessor 적용
        #
        # 추론이므로 절대 다시 fit하지 않는다.
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
        # 8. float32 통일
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
        # 9. Isolation Forest
        # ====================================================

        raw_score = float(

            model.score_samples(
                X
            )[0]

        )


        # ====================================================
        # 10. Anomaly Score
        #
        # sklearn Isolation Forest
        #
        # score_samples:
        #     작을수록 이상
        #
        # MOVI:
        #     anomaly_score가 클수록 위험
        #
        # 따라서 부호를 반전한다.
        # ====================================================

        anomaly_score = (
            -raw_score
        )


        # ====================================================
        # 11. Threshold
        # ====================================================

        is_anomaly = (
            anomaly_score
            >=
            threshold
        )


        # ====================================================
        # 12. 임시 Risk Level
        #
        # 현재는 Isolation Forest 결과만 사용.
        #
        # 다음 단계:
        #
        # Isolation Forest
        #       +
        # Rule Engine
        #       ↓
        # Final Risk Score
        #
        # 구현 후 교체한다.
        # ====================================================

        if is_anomaly:

            risk_level = (
                "HIGH"
            )

        else:

            risk_level = (
                "LOW"
            )


        # ====================================================
        # Debug - Score
        # ====================================================

        print()
        print("=" * 70)
        print("[FDS RESULT]")
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
        # 13. Response
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

            risk_level=(
                risk_level
            ),

            model=(
                "isolation_forest"
            ),
        )


    # ========================================================
    # 잘못된 거래 입력 / Mapping 오류
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
    # 기타 내부 오류
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