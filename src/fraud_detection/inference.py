from __future__ import annotations

"""학습 결과 확인과 오프라인 추론을 위한 기존 모듈.

FastAPI 운영 요청에는 사용하지 않는다. 온라인 공식 경로는
``api.py → FraudDetectionService``이다.
"""

import json
from typing import Any

import numpy as np
import pandas as pd

from scipy import sparse


# ============================================================
# Import
# ============================================================

try:
    from .config import (
        ELECTRONIC_CONFIG,
        CARD_CONFIG,
    )

    from .feature_engineering import (
        get_feature_config,
        transform_features,
    )

    from .train_iforest import (
        load_model_bundle,
    )

except ImportError:

    from fraud_detection.config import (
        ELECTRONIC_CONFIG,
        CARD_CONFIG,
    )

    from feature_engineering import (
        get_feature_config,
        transform_features,
    )

    from train_iforest import (
        load_model_bundle,
    )


# ============================================================
# 1. Dataset Config
# ============================================================

DATASET_CONFIGS = {
    "electronic": ELECTRONIC_CONFIG,
    "card": CARD_CONFIG,
}


def get_dataset_config(
    dataset_type: str,
) -> dict:
    """
    dataset_type에 해당하는 설정 반환.
    """

    if dataset_type not in DATASET_CONFIGS:

        raise ValueError(
            f"지원하지 않는 dataset_type입니다: "
            f"{dataset_type}\n"
            f"가능한 값: {list(DATASET_CONFIGS.keys())}"
        )

    return DATASET_CONFIGS[
        dataset_type
    ]


# ============================================================
# 2. Fraud Detector
# ============================================================

class FraudDetector:
    """
    Isolation Forest 기반 이상거래 탐지기.

    서버 실행 시 객체를 한 번 생성하고,
    이후 여러 거래에 대해 반복해서 predict()를 호출한다.

    Example
    -------
    detector = FraudDetector("electronic")

    result = detector.predict({
        "출금금융회사일련번호": 155,
        "입금금융회사일련번호": 155,
        "자금구분": 1,
        "거래금액": 100000,
        "거래시간대": 15,
        "매체구분": 1,
        "거래일자": 20260812
    })
    """

    def __init__(
        self,
        dataset_type: str,
    ):

        self.dataset_type = (
            dataset_type
        )

        self.dataset_config = (
            get_dataset_config(
                dataset_type
            )
        )

        self.feature_config = (
            get_feature_config(
                dataset_type
            )
        )


        # ====================================================
        # Model Bundle Load
        # ====================================================

        bundle = load_model_bundle(
            dataset_type
        )

        self.bundle = bundle

        self.model = bundle[
            "model"
        ]

        self.preprocessor = bundle[
            "preprocessor"
        ]


        # ====================================================
        # 모델 Dataset 확인
        # ====================================================

        model_dataset_type = bundle[
            "dataset_type"
        ]

        if (
            model_dataset_type
            != dataset_type
        ):

            raise ValueError(
                "로드된 모델의 dataset_type과 "
                "요청 dataset_type이 다릅니다.\n"
                f"Model: {model_dataset_type}\n"
                f"Request: {dataset_type}"
            )


        # ====================================================
        # Threshold Load
        # ====================================================

        threshold_data = (
            self._load_threshold()
        )

        self.threshold_data = (
            threshold_data
        )

        self.threshold = float(
            threshold_data[
                "threshold"
            ]
        )


        # ====================================================
        # Risk Score 계산용 Scale
        # ====================================================

        self.risk_scale = (
            self._calculate_risk_scale()
        )


        print()
        print("=" * 70)
        print("FRAUD DETECTOR LOADED")
        print("=" * 70)

        print(
            f"Dataset   : "
            f"{self.dataset_type}"
        )

        print(
            f"Threshold : "
            f"{self.threshold:.8f}"
        )

        print(
            f"Features  : "
            f"{self.bundle['n_features']}"
        )

        print(
            f"Risk Scale: "
            f"{self.risk_scale:.8f}"
        )

        print("=" * 70)


    # ========================================================
    # 3. Threshold Load
    # ========================================================

    def _load_threshold(
        self,
    ) -> dict:
        """
        evaluate.py에서 생성한 threshold.json을 읽는다.
        """

        threshold_path = (
            self.dataset_config[
                "threshold_path"
            ]
        )

        if not threshold_path.exists():

            raise FileNotFoundError(
                "Threshold 파일이 존재하지 않습니다.\n"
                "evaluate.py를 먼저 실행하세요.\n"
                f"path: {threshold_path}"
            )


        with open(
            threshold_path,
            "r",
            encoding="utf-8",
        ) as file:

            threshold_data = (
                json.load(
                    file
                )
            )


        # ====================================================
        # 기본 검증
        # ====================================================

        if (
            threshold_data.get(
                "dataset_type"
            )
            != self.dataset_type
        ):

            raise ValueError(
                "Threshold Dataset과 "
                "Model Dataset이 다릅니다."
            )


        if (
            "threshold"
            not in threshold_data
        ):

            raise ValueError(
                "threshold.json에 "
                "threshold 값이 없습니다."
            )


        return threshold_data


    # ========================================================
    # 4. Risk Score Scale 계산
    # ========================================================

    def _calculate_risk_scale(
        self,
    ) -> float:
        """
        정상 Train Score 분포의 표준편차를 이용하여
        Risk Score 계산 Scale을 결정한다.

        train_iforest.py에서 저장한 score std 사용.

        Risk Score는 확률이 아니라
        사용자/Backend가 이해하기 쉬운
        0~100 위험 지표이다.
        """

        score_summary = (
            self.bundle.get(
                "training_score_summary",
                {},
            )
        )

        score_std = float(
            score_summary.get(
                "std",
                0.05,
            )
        )


        # 지나치게 작은 값 방지
        return max(
            score_std,
            1e-4,
        )


    # ========================================================
    # 5. 입력 데이터 검증
    # ========================================================

    def validate_transaction(
        self,
        transaction: dict[str, Any],
    ) -> None:
        """
        Feature Engineering에 필요한 필수 입력값이
        존재하는지 검사한다.
        """

        required_columns = (
            self.feature_config[
                "required_columns"
            ]
        )

        missing_columns = [

            column

            for column
            in required_columns

            if column
            not in transaction
        ]


        if missing_columns:

            raise ValueError(
                "거래 데이터에 필요한 필드가 없습니다.\n"
                f"누락 필드: {missing_columns}"
            )


    # ========================================================
    # 6. DataFrame 생성
    # ========================================================

    def transaction_to_dataframe(
        self,
        transaction: dict[str, Any],
    ) -> pd.DataFrame:
        """
        단일 JSON/Dictionary 거래를
        pandas DataFrame 1행으로 변환한다.
        """

        self.validate_transaction(
            transaction
        )

        df = pd.DataFrame(
            [transaction]
        )

        return df


    # ========================================================
    # 7. Anomaly Score 계산
    # ========================================================

    def calculate_anomaly_scores(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:
        """
        DataFrame에 대한 Anomaly Score 계산.

        sklearn Isolation Forest:
            낮을수록 이상

        MOVI:
            높을수록 위험

        따라서:

            anomaly_score =
                -model.score_samples(X)
        """

        # ====================================================
        # Train에서 학습한 Preprocessor만 사용
        # ====================================================

        X = transform_features(
            df,
            self.dataset_type,
            self.preprocessor,
        )


        # ====================================================
        # Sparse Matrix 처리
        # ====================================================

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

        else:

            X = np.asarray(
                X,
                dtype=np.float32,
            )


        # ====================================================
        # Isolation Forest
        # ====================================================

        raw_scores = (
            self.model
            .score_samples(
                X
            )
        )


        # ====================================================
        # 높을수록 위험하도록 부호 반전
        # ====================================================

        anomaly_scores = (
            -raw_scores
        )


        return anomaly_scores


    # ========================================================
    # 8. Anomaly Score → Risk Score
    # ========================================================

    def anomaly_to_risk_score(
        self,
        anomaly_score: float,
    ) -> float:
        """
        Isolation Forest의 Anomaly Score를
        0 ~ 100 Risk Score로 변환한다.

        Threshold를 Risk Score 50으로 설정한다.

        anomaly_score < threshold
            → Risk Score < 50

        anomaly_score >= threshold
            → Risk Score >= 50


        주의
        ----
        Risk Score는 Fraud 확률이 아니다.

        모델 점수를 서비스에서 이해하기 쉬운
        위험 지표로 변환한 값이다.
        """

        # ====================================================
        # Threshold 기준 거리
        # ====================================================

        z = (
            anomaly_score
            - self.threshold
        ) / self.risk_scale


        # exp overflow 방지
        z = np.clip(
            z,
            -20,
            20,
        )


        # ====================================================
        # Sigmoid
        #
        # z = 0
        # → risk = 50
        # ====================================================

        risk_score = (
            100
            /
            (
                1
                + np.exp(
                    -z
                )
            )
        )


        return round(
            float(
                risk_score
            ),
            2,
        )


    # ========================================================
    # 9. Risk Level
    # ========================================================

    @staticmethod
    def get_risk_level(
        risk_score: float,
    ) -> str:
        """
        Frontend / Backend에서 표시하기 쉬운
        Risk Level 반환.

        이것은 서비스 표시용 등급이며
        Fraud 판정 자체는 Threshold로 수행한다.
        """

        if risk_score < 30:

            return "LOW"

        elif risk_score < 50:

            return "MEDIUM"

        elif risk_score < 70:

            return "HIGH"

        else:

            return "CRITICAL"


    # ========================================================
    # 10. 단일 거래 Prediction
    # ========================================================

    def predict(
        self,
        transaction: dict[str, Any],
    ) -> dict:
        """
        단일 거래 이상거래 탐지.

        Returns
        -------
        {
            "dataset_type": "electronic",
            "is_fraud": true,
            "risk_score": 83.5,
            "risk_level": "CRITICAL",
            "anomaly_score": 0.61,
            "threshold": 0.52,
            "decision_margin": 0.09
        }
        """

        df = (
            self.transaction_to_dataframe(
                transaction
            )
        )


        anomaly_score = float(
            self.calculate_anomaly_scores(
                df
            )[0]
        )


        # ====================================================
        # Fraud 판정
        # ====================================================

        is_fraud = bool(
            anomaly_score
            >= self.threshold
        )


        # ====================================================
        # Risk Score
        # ====================================================

        risk_score = (
            self.anomaly_to_risk_score(
                anomaly_score
            )
        )


        # ====================================================
        # Threshold와의 거리
        # ====================================================

        decision_margin = (
            anomaly_score
            - self.threshold
        )


        result = {

            "dataset_type": (
                self.dataset_type
            ),

            "is_fraud": (
                is_fraud
            ),

            "risk_score": (
                risk_score
            ),

            "risk_level": (
                self.get_risk_level(
                    risk_score
                )
            ),

            "anomaly_score": round(
                anomaly_score,
                8,
            ),

            "threshold": round(
                self.threshold,
                8,
            ),

            "decision_margin": round(
                float(
                    decision_margin
                ),
                8,
            ),
        }


        return result


    # ========================================================
    # 11. Batch Prediction
    # ========================================================

    def predict_batch(
        self,
        transactions: list[
            dict[str, Any]
        ],
    ) -> list[dict]:
        """
        여러 거래를 한 번에 처리한다.

        API 요청을 여러 건 묶어서 처리할 때 사용 가능.
        """

        if not transactions:

            return []


        # ====================================================
        # 모든 거래 필드 검증
        # ====================================================

        for index, transaction in enumerate(
            transactions
        ):

            try:

                self.validate_transaction(
                    transaction
                )

            except ValueError as error:

                raise ValueError(
                    f"transactions[{index}] 오류:\n"
                    f"{error}"
                ) from error


        # ====================================================
        # DataFrame
        # ====================================================

        df = pd.DataFrame(
            transactions
        )


        # ====================================================
        # 한 번에 Score 계산
        # ====================================================

        anomaly_scores = (
            self.calculate_anomaly_scores(
                df
            )
        )


        results = []


        for index, anomaly_score in enumerate(
            anomaly_scores
        ):

            anomaly_score = float(
                anomaly_score
            )


            is_fraud = bool(
                anomaly_score
                >= self.threshold
            )


            risk_score = (
                self.anomaly_to_risk_score(
                    anomaly_score
                )
            )


            decision_margin = (
                anomaly_score
                - self.threshold
            )


            result = {

                "index": index,

                "dataset_type": (
                    self.dataset_type
                ),

                "is_fraud": (
                    is_fraud
                ),

                "risk_score": (
                    risk_score
                ),

                "risk_level": (
                    self.get_risk_level(
                        risk_score
                    )
                ),

                "anomaly_score": round(
                    anomaly_score,
                    8,
                ),

                "threshold": round(
                    self.threshold,
                    8,
                ),

                "decision_margin": round(
                    float(
                        decision_margin
                    ),
                    8,
                ),
            }


            results.append(
                result
            )


        return results


# ============================================================
# 12. 간단한 함수형 Interface
# ============================================================

def detect_fraud(
    transaction: dict[str, Any],
    dataset_type: str = "electronic",
) -> dict:
    """
    간단하게 호출하기 위한 Wrapper.

    주의:
    실제 서버에서는 요청마다 이 함수를 사용하는 것보다

        detector = FraudDetector(...)

    를 서버 시작 시 한 번 생성한 뒤

        detector.predict(...)

    를 반복 호출하는 것이 더 효율적이다.
    """

    detector = FraudDetector(
        dataset_type
    )

    return detector.predict(
        transaction
    )


# ============================================================
# 13. 실행 테스트
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # 실제 Validation 데이터 1건을 가져와
    # inference가 정상 동작하는지만 확인한다.
    #
    # 이 코드는 테스트용이며,
    # 실제 서비스에서는 JSON 입력을 받게 된다.
    # ========================================================

    try:

        from .data_loader import (
            load_dataset
        )

    except ImportError:

        from data_loader import (
            load_dataset
        )


    print()
    print("Inference 테스트 시작")
    print()


    # ========================================================
    # Detector는 한 번만 Load
    # ========================================================

    detector = FraudDetector(
        dataset_type="electronic"
    )


    # ========================================================
    # Validation 1건 Load
    # ========================================================

    test_df = load_dataset(
        dataset_type="electronic",
        split="validation",
        max_files=1,
        nrows_per_file=1,
    )


    # ========================================================
    # 모델 입력에 필요한 Feature만 추출
    # ========================================================

    required_columns = (
        detector
        .feature_config[
            "required_columns"
        ]
    )


    test_transaction = (
        test_df[
            required_columns
        ]
        .iloc[0]
        .to_dict()
    )


    print()
    print("=" * 70)
    print("INPUT TRANSACTION")
    print("=" * 70)

    print(
        json.dumps(
            test_transaction,
            ensure_ascii=False,
            indent=4,
            default=str,
        )
    )


    # ========================================================
    # Prediction
    # ========================================================

    result = detector.predict(
        test_transaction
    )


    print()
    print("=" * 70)
    print("FRAUD DETECTION RESULT")
    print("=" * 70)

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=4,
        )
    )

    print("=" * 70)
