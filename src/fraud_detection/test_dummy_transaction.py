from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from scipy import sparse


# ============================================================
# Import
# ============================================================

try:
    from .config import (
        ELECTRONIC_CONFIG,
    )

    from .feature_engineering import (
        engineer_electronic_features,
    )

except ImportError:

    from config import (
        ELECTRONIC_CONFIG,
    )

    from feature_engineering import (
        engineer_electronic_features,
    )


# ============================================================
# 1. Dummy 거래 생성
# ============================================================

def create_dummy_transactions() -> pd.DataFrame:
    """
    한 사용자의 평소 거래 History를 만든 뒤

    마지막 2개 거래를:

    1. 정상에 가까운 거래
    2. 의도적으로 만든 이상거래

    로 구성한다.

    Historical Feature가 있으므로
    현재 거래 1건만 넣는 것이 아니라
    과거 거래 History와 함께 넣어야 한다.
    """

    transactions = [

        # ====================================================
        # 과거 정상 거래 History
        # ====================================================

        {
            "출금계좌일련번호": 10001,
            "입금계좌일련번호": 20001,

            "출금금융회사일련번호": 110,
            "입금금융회사일련번호": 110,

            "자금구분": 1,

            "거래금액": 100_000,

            "거래시간대": 9,

            "매체구분": 2,

            "거래일자": 20260818,
        },

        {
            "출금계좌일련번호": 10001,
            "입금계좌일련번호": 20001,

            "출금금융회사일련번호": 110,
            "입금금융회사일련번호": 110,

            "자금구분": 1,

            "거래금액": 120_000,

            "거래시간대": 12,

            "매체구분": 2,

            "거래일자": 20260819,
        },

        {
            "출금계좌일련번호": 10001,
            "입금계좌일련번호": 20001,

            "출금금융회사일련번호": 110,
            "입금금융회사일련번호": 110,

            "자금구분": 1,

            "거래금액": 80_000,

            "거래시간대": 15,

            "매체구분": 2,

            "거래일자": 20260820,
        },

        {
            "출금계좌일련번호": 10001,
            "입금계좌일련번호": 20001,

            "출금금융회사일련번호": 110,
            "입금금융회사일련번호": 110,

            "자금구분": 1,

            "거래금액": 110_000,

            "거래시간대": 12,

            "매체구분": 2,

            "거래일자": 20260821,
        },

        {
            "출금계좌일련번호": 10001,
            "입금계좌일련번호": 20001,

            "출금금융회사일련번호": 110,
            "입금금융회사일련번호": 110,

            "자금구분": 1,

            "거래금액": 90_000,

            "거래시간대": 15,

            "매체구분": 2,

            "거래일자": 20260822,
        },

        {
            "출금계좌일련번호": 10001,
            "입금계좌일련번호": 20001,

            "출금금융회사일련번호": 110,
            "입금금융회사일련번호": 110,

            "자금구분": 1,

            "거래금액": 100_000,

            "거래시간대": 12,

            "매체구분": 2,

            "거래일자": 20260823,
        },


        # ====================================================
        # 정상 테스트 거래
        #
        # 평소 거래와 거의 동일
        # ====================================================

        {
            "출금계좌일련번호": 10001,

            # 기존 수취인
            "입금계좌일련번호": 20001,

            "출금금융회사일련번호": 110,
            "입금금융회사일련번호": 110,

            "자금구분": 1,

            # 평소 수준
            "거래금액": 105_000,

            # 일반 시간
            "거래시간대": 12,

            # 기존 매체
            "매체구분": 2,

            "거래일자": 20260824,
        },


        # ====================================================
        # 이상 테스트 거래
        #
        # 평소 약 10만원
        #
        # → 1천만원
        # → 심야
        # → 신규 수취인
        # → 타행
        # → 신규 매체
        # ====================================================

        {
            "출금계좌일련번호": 10001,

            # 신규 수취인
            "입금계좌일련번호": 99999,

            "출금금융회사일련번호": 110,

            # 타행
            "입금금융회사일련번호": 999,

            "자금구분": 1,

            # 비정상적으로 큰 금액
            "거래금액": 10_000_000,

            # 심야
            "거래시간대": 3,

            # 평소 사용하지 않던 매체
            "매체구분": 7,

            "거래일자": 20260825,
        },
    ]

    return pd.DataFrame(
        transactions
    )


# ============================================================
# 2. Feature 출력
# ============================================================

def print_test_features(
    engineered_df: pd.DataFrame,
) -> None:

    columns = [
        "log_amount",
        "amount_ratio",
        "amount_zscore",
        "is_night",
        "is_weekend",
        "same_bank",
        "new_recipient",
        "unusual_medium",
        "historical_transaction_count",
        "same_day_transaction_count",
        "same_time_bucket_count",
    ]

    print()
    print("=" * 80)
    print("DUMMY FEATURE COMPARISON")
    print("=" * 80)

    print()
    print("[NORMAL TRANSACTION]")

    print(
        engineered_df[
            columns
        ].iloc[-2]
    )

    print()
    print("[ANOMALOUS TRANSACTION]")

    print(
        engineered_df[
            columns
        ].iloc[-1]
    )

    print("=" * 80)


# ============================================================
# 3. Score 테스트
# ============================================================

def main():

    # ========================================================
    # Model Bundle Load
    # ========================================================

    model_path = (
        ELECTRONIC_CONFIG[
            "model_path"
        ]
    )


    if not model_path.exists():

        raise FileNotFoundError(
            "학습된 Model Bundle이 없습니다.\n"
            f"path: {model_path}\n\n"
            "먼저 train_iforest.py를 실행하세요."
        )


    print()
    print(
        "Model Loading:",
        model_path,
    )


    bundle = joblib.load(
        model_path
    )


    model = bundle[
        "model"
    ]

    preprocessor = bundle[
        "preprocessor"
    ]


    # ========================================================
    # Dummy 데이터
    # ========================================================

    dummy_df = (
        create_dummy_transactions()
    )


    # ========================================================
    # Historical Feature Engineering
    # ========================================================

    engineered_df = (
        engineer_electronic_features(
            dummy_df
        )
    )


    print_test_features(
        engineered_df
    )


    # ========================================================
    # Preprocessor
    #
    # 반드시 Train에서 Fit한 객체를 사용한다.
    # 여기에서 fit_transform() 하면 안 된다.
    # ========================================================

    X = preprocessor.transform(
        engineered_df
    )


    X = X.astype(
        np.float32
    )


    # score/predict에서는 CSR 사용 권장
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


    # ========================================================
    # Score
    # ========================================================

    scores = model.score_samples(
        X
    )


    decision_scores = (
        model.decision_function(
            X
        )
    )


    predictions = model.predict(
        X
    )


    normal_index = len(
        dummy_df
    ) - 2

    anomaly_index = len(
        dummy_df
    ) - 1


    normal_score = float(
        scores[
            normal_index
        ]
    )

    anomaly_score = float(
        scores[
            anomaly_index
        ]
    )


    normal_decision = float(
        decision_scores[
            normal_index
        ]
    )

    anomaly_decision = float(
        decision_scores[
            anomaly_index
        ]
    )


    normal_prediction = int(
        predictions[
            normal_index
        ]
    )

    anomaly_prediction = int(
        predictions[
            anomaly_index
        ]
    )


    # ========================================================
    # 결과 출력
    # ========================================================

    print()
    print("=" * 80)
    print("ISOLATION FOREST RESULT")
    print("=" * 80)


    print()
    print("[NORMAL TRANSACTION]")

    print(
        f"score_samples      : "
        f"{normal_score:.6f}"
    )

    print(
        f"decision_function  : "
        f"{normal_decision:.6f}"
    )

    print(
        "prediction         :",
        (
            "NORMAL"
            if normal_prediction == 1
            else "ANOMALY"
        ),
    )


    print()
    print("[ANOMALOUS TRANSACTION]")

    print(
        f"score_samples      : "
        f"{anomaly_score:.6f}"
    )

    print(
        f"decision_function  : "
        f"{anomaly_decision:.6f}"
    )

    print(
        "prediction         :",
        (
            "NORMAL"
            if anomaly_prediction == 1
            else "ANOMALY"
        ),
    )


    print()
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)


    score_difference = (
        normal_score
        - anomaly_score
    )


    print(
        f"Normal Score : "
        f"{normal_score:.6f}"
    )

    print(
        f"Anomaly Score: "
        f"{anomaly_score:.6f}"
    )

    print(
        f"Difference   : "
        f"{score_difference:.6f}"
    )


    print()

    # Isolation Forest:
    # score_samples 값이 낮을수록
    # 더 이상치에 가깝다.

    if anomaly_score < normal_score:

        print(
            "[PASS]"
        )

        print(
            "의도적으로 만든 이상거래가 "
            "정상거래보다 낮은 Score를 가집니다."
        )

    else:

        print(
            "[CHECK]"
        )

        print(
            "이상거래 Score가 정상거래보다 "
            "낮게 나오지 않았습니다."
        )

        print(
            "Feature 또는 모델 학습 결과를 "
            "추가로 확인해야 합니다."
        )


    print("=" * 80)


if __name__ == "__main__":
    main()