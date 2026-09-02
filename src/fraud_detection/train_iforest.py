from __future__ import annotations

from datetime import datetime
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn

from scipy import sparse
from sklearn.ensemble import IsolationForest


# ============================================================
# Import
# ============================================================

try:
    from .config import (
        ELECTRONIC_CONFIG,
        CARD_CONFIG,
        ISOLATION_FOREST_PARAMS,
        RANDOM_STATE,
        TRAIN_SAMPLE_SIZE,
        create_output_directories,
    )

    from .data_loader import (
        load_dataset,
    )

    from .feature_engineering import (
        get_feature_config,
        engineer_features,
        build_preprocessor,
        get_transformed_feature_names,
    )

except ImportError:

    from fraud_detection.config import (
        ELECTRONIC_CONFIG,
        CARD_CONFIG,
        ISOLATION_FOREST_PARAMS,
        RANDOM_STATE,
        TRAIN_SAMPLE_SIZE,
        create_output_directories,
    )

    from data_loader import (
        load_dataset,
    )

    from feature_engineering import (
        get_feature_config,
        engineer_features,
        build_preprocessor,
        get_transformed_feature_names,
    )


# ============================================================
# 1. Dataset Config
# ============================================================

DATASET_CONFIGS = {
    "electronic": ELECTRONIC_CONFIG,
    "card": CARD_CONFIG,
}


# ============================================================
# 2. Dataset Config 조회
# ============================================================

def get_dataset_config(
    dataset_type: str,
) -> dict:
    """
    dataset_type에 해당하는 config 반환.
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
# 3. 학습에 필요한 컬럼 목록 생성
# ============================================================

def get_training_columns(
    dataset_type: str,
) -> list[str]:
    """
    학습에 필요한 원본 CSV 컬럼 목록을 반환한다.

    Feature Engineering에 필요한 컬럼
    +
    이상거래여부(Target)

    Target은 Isolation Forest 입력에는 사용하지 않고,
    Historical Feature 계산 후 정상거래만 추출할 때 사용한다.
    """

    dataset_config = get_dataset_config(
        dataset_type
    )

    feature_config = get_feature_config(
        dataset_type
    )

    target_column = dataset_config[
        "target"
    ]

    required_columns = feature_config[
        "required_columns"
    ]

    columns = list(
        dict.fromkeys(
            required_columns
            + [target_column]
        )
    )

    return columns


# ============================================================
# 4. Historical Feature 기반 학습 데이터 준비
# ============================================================

def prepare_historical_training_sample(
    dataset_type: str,
    *,
    sample_size: int = TRAIN_SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
    max_files: int | None = None,
    nrows_per_file: int | None = None,
) -> tuple[pd.DataFrame, int, int]:
    """
    Historical Feature를 사용하는 Isolation Forest 학습 데이터를 준비한다.

    중요
    ----------------------------------------------------------

    잘못된 순서:

        정상거래 추출
        ↓
        Random Sampling
        ↓
        Historical Feature Engineering

    위 방식은 거래 이력이 사라지므로 amount_ratio,
    new_recipient 등의 값이 왜곡될 수 있다.


    올바른 순서:

        전체 거래 Load
        ↓
        시간순 정렬
        ↓
        전체 거래 기준 Historical Feature Engineering
        ↓
        정상거래만 추출
        ↓
        Random Sampling


    Returns
    -------
    normal_features
        Feature Engineering이 끝난 정상거래 Sample

    total_normal_seen
        전체 로드 데이터 중 정상거래 수

    total_rows_seen
        전체 로드 거래 수
    """

    dataset_config = get_dataset_config(
        dataset_type
    )

    target_column = dataset_config[
        "target"
    ]

    usecols = get_training_columns(
        dataset_type
    )


    print()
    print("=" * 70)
    print("HISTORICAL TRAINING DATA PREPARATION")
    print("=" * 70)

    print(
        f"Dataset       : {dataset_type}"
    )

    print(
        f"Target Sample : {sample_size:,}"
    )

    if max_files is not None:
        print(
            f"Max Files     : {max_files}"
        )

    if nrows_per_file is not None:
        print(
            f"Rows / File   : {nrows_per_file:,}"
        )

    print("=" * 70)


    # ========================================================
    # STEP 1
    # 전체 거래 Load
    #
    # load_dataset() 내부에서
    # 거래일자 + 거래시간대 기준으로
    # 과거 → 미래 정렬이 이루어진다.
    # ========================================================

    print()
    print("[1/4] 전체 거래 Load")

    raw_df = load_dataset(
        dataset_type=dataset_type,
        split="train",
        usecols=usecols,
        max_files=max_files,
        nrows_per_file=nrows_per_file,
    )

    total_rows_seen = len(
        raw_df
    )

    print()
    print(
        f"전체 거래 수 : {total_rows_seen:,}"
    )


    # ========================================================
    # STEP 2
    # 전체 거래 기준 Feature Engineering
    #
    # 정상/이상 여부와 관계없이 실제 거래 History를
    # 그대로 사용해야 Historical Feature가 정확하다.
    # ========================================================

    print()
    print(
        "[2/4] Historical Feature Engineering"
    )

    engineered_df = engineer_features(
        raw_df,
        dataset_type,
    )


    print()
    print(
        f"Engineered Rows    : "
        f"{len(engineered_df):,}"
    )

    print(
        f"Engineered Columns : "
        f"{len(engineered_df.columns):,}"
    )


    # ========================================================
    # STEP 3
    # 정상거래만 추출
    #
    # Isolation Forest는 정상 패턴만 학습한다.
    #
    # 단, Feature 계산은 이미 전체 거래를 기준으로
    # 완료한 이후이다.
    # ========================================================

    print()
    print(
        "[3/4] 정상거래 추출"
    )

    target = pd.to_numeric(
        raw_df[target_column],
        errors="coerce",
    )


    if target.isna().any():

        raise ValueError(
            "Target 컬럼에 NaN 또는 "
            "숫자로 변환할 수 없는 값이 존재합니다."
        )


    normal_mask = (
        target == 0
    )


    normal_features = (
        engineered_df
        .loc[normal_mask]
        .copy()
    )


    total_normal_seen = len(
        normal_features
    )


    anomaly_count = int(
        (target == 1).sum()
    )


    print(
        f"정상거래 : "
        f"{total_normal_seen:,}"
    )

    print(
        f"이상거래 : "
        f"{anomaly_count:,}"
    )


    # ========================================================
    # STEP 4
    # 정상거래 Random Sampling
    #
    # Feature Engineering 이후 Sample하므로
    # 거래 History는 이미 보존되어 있다.
    # ========================================================

    print()
    print(
        "[4/4] 정상거래 Random Sampling"
    )


    if (
        sample_size is not None
        and len(normal_features) > sample_size
    ):

        normal_features = (
            normal_features
            .sample(
                n=sample_size,
                random_state=random_state,
            )
            .reset_index(
                drop=True
            )
        )

    else:

        normal_features = (
            normal_features
            .reset_index(
                drop=True
            )
        )


    print()
    print("=" * 70)
    print("TRAINING DATA 준비 완료")
    print("=" * 70)

    print(
        f"전체 거래       : "
        f"{total_rows_seen:,}"
    )

    print(
        f"전체 정상거래   : "
        f"{total_normal_seen:,}"
    )

    print(
        f"최종 학습 Sample: "
        f"{len(normal_features):,}"
    )

    print("=" * 70)


    return (
        normal_features,
        total_normal_seen,
        total_rows_seen,
    )


# ============================================================
# 5. Isolation Forest 생성
# ============================================================

def build_isolation_forest(
) -> IsolationForest:
    """
    config.py에 정의된 Parameter를 사용하여
    Isolation Forest를 생성한다.
    """

    model = IsolationForest(
        **ISOLATION_FOREST_PARAMS
    )

    return model


# ============================================================
# 6. 학습 Score Summary
# ============================================================

def calculate_training_score_summary(
    model: IsolationForest,
    X,
) -> dict:
    """
    정상 Train Sample에 대한 score_samples 분포를 계산한다.

    Isolation Forest:

        score가 낮을수록 이상치에 가깝다.

    Validation 단계에서 실제 이상거래 Score와
    비교하여 Threshold를 결정할 때 참고한다.
    """

    scores = model.score_samples(
        X
    )

    summary = {

        "min": float(
            np.min(scores)
        ),

        "max": float(
            np.max(scores)
        ),

        "mean": float(
            np.mean(scores)
        ),

        "std": float(
            np.std(scores)
        ),

        "q01": float(
            np.quantile(
                scores,
                0.01,
            )
        ),

        "q05": float(
            np.quantile(
                scores,
                0.05,
            )
        ),

        "q50": float(
            np.quantile(
                scores,
                0.50,
            )
        ),

        "q95": float(
            np.quantile(
                scores,
                0.95,
            )
        ),

        "q99": float(
            np.quantile(
                scores,
                0.99,
            )
        ),
    }

    return summary


# ============================================================
# 7. 학습 예측 결과 확인
# ============================================================

def calculate_training_prediction_summary(
    model: IsolationForest,
    X,
) -> dict:
    """
    IsolationForest 기본 predict() 결과를 확인한다.

    predict:
        1  -> Normal
        -1 -> Anomaly

    주의:
        이것은 최종 Threshold가 아니다.

        최종 Threshold는 evaluate.py에서
        실제 Validation Label을 기준으로 결정한다.
    """

    predictions = model.predict(
        X
    )


    normal_count = int(
        np.sum(
            predictions == 1
        )
    )


    anomaly_count = int(
        np.sum(
            predictions == -1
        )
    )


    total = len(
        predictions
    )


    return {

        "normal_count":
            normal_count,

        "anomaly_count":
            anomaly_count,

        "normal_ratio":
            normal_count / total
            if total > 0
            else 0.0,

        "anomaly_ratio":
            anomaly_count / total
            if total > 0
            else 0.0,
    }


# ============================================================
# 8. Model Bundle 생성
# ============================================================

def create_model_bundle(
    *,
    dataset_type: str,
    model: IsolationForest,
    preprocessor,
    feature_names: list[str],
    training_sample_size: int,
    total_normal_seen: int,
    total_rows_seen: int,
    training_score_summary: dict,
    training_prediction_summary: dict,
    training_seconds: float,
) -> dict:
    """
    Model + Preprocessor + Metadata를 하나의 객체로 묶는다.

    evaluate.py / inference.py에서는
    해당 Joblib Bundle 하나만 Load하면 된다.
    """

    bundle = {

        # ====================================================
        # 핵심 객체
        # ====================================================

        "model": model,

        "preprocessor": preprocessor,


        # ====================================================
        # Dataset
        # ====================================================

        "dataset_type":
            dataset_type,

        "target_column":
            get_dataset_config(
                dataset_type
            )["target"],


        # ====================================================
        # Feature
        # ====================================================

        "feature_names":
            feature_names,

        "n_features":
            len(
                feature_names
            ),


        # ====================================================
        # Training
        # ====================================================

        "training_sample_size":
            training_sample_size,

        "total_rows_seen":
            total_rows_seen,

        "total_normal_seen":
            total_normal_seen,

        "isolation_forest_params":
            model.get_params(),

        "training_score_summary":
            training_score_summary,

        "training_prediction_summary":
            training_prediction_summary,

        "training_seconds":
            training_seconds,

        "trained_at": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),


        # ====================================================
        # 환경
        # ====================================================

        "environment": {

            "python":
                platform.python_version(),

            "scikit_learn":
                sklearn.__version__,

            "pandas":
                pd.__version__,

            "numpy":
                np.__version__,
        },
    }

    return bundle


# ============================================================
# 9. 모델 저장
# ============================================================

def save_model_bundle(
    bundle: dict,
    model_path: Path,
) -> None:
    """
    Model Bundle을 Joblib 파일로 저장한다.
    """

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    joblib.dump(
        bundle,
        model_path,
        compress=3,
    )


    print()
    print("=" * 70)
    print("MODEL SAVED")
    print("=" * 70)

    print(
        f"Path : {model_path}"
    )

    print(
        f"Size : "
        f"{model_path.stat().st_size / (1024 ** 2):.2f} MB"
    )

    print("=" * 70)


# ============================================================
# 10. 학습 결과 출력
# ============================================================

def print_training_summary(
    bundle: dict,
) -> None:
    """
    학습 결과 Summary 출력.
    """

    print()
    print("=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)


    print(
        f"Dataset        : "
        f"{bundle['dataset_type']}"
    )

    print(
        f"Rows Seen      : "
        f"{bundle['total_rows_seen']:,}"
    )

    print(
        f"Normal Seen    : "
        f"{bundle['total_normal_seen']:,}"
    )

    print(
        f"Train Samples  : "
        f"{bundle['training_sample_size']:,}"
    )

    print(
        f"Features       : "
        f"{bundle['n_features']:,}"
    )

    print(
        f"Training Time  : "
        f"{bundle['training_seconds']:.2f} sec"
    )


    # ========================================================
    # Isolation Forest 설정
    # ========================================================

    print()
    print("[Isolation Forest]")

    params = bundle[
        "isolation_forest_params"
    ]

    print(
        f"n_estimators   : "
        f"{params['n_estimators']}"
    )

    print(
        f"max_samples    : "
        f"{params['max_samples']}"
    )

    print(
        f"contamination  : "
        f"{params['contamination']}"
    )

    print(
        f"n_jobs         : "
        f"{params['n_jobs']}"
    )


    # ========================================================
    # Score
    # ========================================================

    print()
    print("[Normal Training Score]")

    score_summary = bundle[
        "training_score_summary"
    ]

    for key, value in (
        score_summary.items()
    ):

        print(
            f"{key:<5} : "
            f"{value:.6f}"
        )


    # ========================================================
    # Prediction
    # ========================================================

    print()
    print("[Training Prediction]")

    prediction_summary = bundle[
        "training_prediction_summary"
    ]

    print(
        "Normal : "
        f"{prediction_summary['normal_count']:,} "
        f"({prediction_summary['normal_ratio'] * 100:.2f}%)"
    )

    print(
        "Anomaly: "
        f"{prediction_summary['anomaly_count']:,} "
        f"({prediction_summary['anomaly_ratio'] * 100:.2f}%)"
    )


    # ========================================================
    # Environment
    # ========================================================

    print()
    print("[Environment]")

    for key, value in (
        bundle[
            "environment"
        ].items()
    ):

        print(
            f"{key:<15}: {value}"
        )


    print("=" * 70)


# ============================================================
# 11. 전체 학습 Pipeline
# ============================================================

def train_isolation_forest(
    dataset_type: str,
    *,
    sample_size: int = TRAIN_SAMPLE_SIZE,
    max_files: int | None = None,
    nrows_per_file: int | None = None,
) -> dict:
    """
    Isolation Forest 전체 학습 Pipeline.

    Flow
    ----------------------------------------------------------

    Train CSV
        ↓
    시간순 Load
        ↓
    전체 거래 Historical Feature Engineering
        ↓
    정상거래 추출
        ↓
    정상거래 Random Sampling
        ↓
    Preprocessor Fit
        ↓
    Isolation Forest Fit
        ↓
    Train Score 확인
        ↓
    Model Bundle 저장
    """

    # ========================================================
    # 저장 폴더 생성
    # ========================================================

    create_output_directories()


    dataset_config = (
        get_dataset_config(
            dataset_type
        )
    )


    print()
    print("#" * 70)

    print(
        f"Isolation Forest Training : "
        f"{dataset_type}"
    )

    print("#" * 70)


    # ========================================================
    # STEP 1
    # Historical Feature 포함 학습 데이터 준비
    # ========================================================

    print()
    print(
        "[STEP 1/5] "
        "Training Data Preparation"
    )


    (
        normal_features,
        total_normal_seen,
        total_rows_seen,
    ) = prepare_historical_training_sample(

        dataset_type=dataset_type,

        sample_size=sample_size,

        random_state=RANDOM_STATE,

        max_files=max_files,

        nrows_per_file=nrows_per_file,
    )


    # ========================================================
    # STEP 2
    # Preprocessor Fit
    #
    # Feature Engineering은 이미 완료되었으므로
    # 여기서는 다시 engineer_features()를 호출하지 않는다.
    # ========================================================

    print()
    print(
        "[STEP 2/5] "
        "Preprocessor Fit"
    )


    preprocessor = build_preprocessor(
        dataset_type
    )


    X_train = (
        preprocessor
        .fit_transform(
            normal_features
        )
    )


    X_train = X_train.astype(
        np.float32
    )


    # ========================================================
    # Sparse Matrix 처리
    #
    # IsolationForest.fit()에서는
    # CSC Sparse Matrix 사용
    # ========================================================

    if sparse.issparse(
        X_train
    ):

        X_train = (
            X_train
            .tocsc()
            .astype(
                np.float32
            )
        )

    else:

        X_train = np.asarray(
            X_train,
            dtype=np.float32,
        )


    feature_names = (
        get_transformed_feature_names(
            preprocessor
        )
    )


    print()
    print(
        f"Train Matrix Shape : "
        f"{X_train.shape}"
    )

    print(
        f"Feature Count      : "
        f"{len(feature_names):,}"
    )

    print(
        f"Matrix Type        : "
        f"{type(X_train)}"
    )

    print(
        f"Matrix Dtype       : "
        f"{X_train.dtype}"
    )


    # ========================================================
    # STEP 3
    # Isolation Forest 생성
    # ========================================================

    print()
    print(
        "[STEP 3/5] "
        "Isolation Forest 생성"
    )


    model = build_isolation_forest()


    print(
        model
    )


    # ========================================================
    # STEP 4
    # Model 학습
    # ========================================================

    print()
    print(
        "[STEP 4/5] "
        "Isolation Forest 학습"
    )

    print(
        "학습 시작..."
    )


    start_time = (
        time.perf_counter()
    )


    model.fit(
        X_train
    )


    training_seconds = (
        time.perf_counter()
        - start_time
    )


    print(
        f"학습 완료: "
        f"{training_seconds:.2f}초"
    )


    # ========================================================
    # Train Score
    # ========================================================

    training_score_summary = (
        calculate_training_score_summary(
            model,
            X_train,
        )
    )


    # ========================================================
    # 기본 Prediction
    # ========================================================

    training_prediction_summary = (
        calculate_training_prediction_summary(
            model,
            X_train,
        )
    )


    # ========================================================
    # STEP 5
    # Model Bundle 저장
    # ========================================================

    print()
    print(
        "[STEP 5/5] "
        "Model 저장"
    )


    bundle = create_model_bundle(

        dataset_type=dataset_type,

        model=model,

        preprocessor=preprocessor,

        feature_names=feature_names,

        training_sample_size=len(
            normal_features
        ),

        total_normal_seen=(
            total_normal_seen
        ),

        total_rows_seen=(
            total_rows_seen
        ),

        training_score_summary=(
            training_score_summary
        ),

        training_prediction_summary=(
            training_prediction_summary
        ),

        training_seconds=(
            training_seconds
        ),
    )


    save_model_bundle(
        bundle=bundle,
        model_path=dataset_config[
            "model_path"
        ],
    )


    print_training_summary(
        bundle
    )


    return bundle


# ============================================================
# 12. 모델 Load 테스트
# ============================================================

def load_model_bundle(
    dataset_type: str,
) -> dict:
    """
    저장된 Model Bundle Load.

    evaluate.py / inference.py에서도
    동일한 방식으로 사용한다.
    """

    dataset_config = (
        get_dataset_config(
            dataset_type
        )
    )

    model_path = dataset_config[
        "model_path"
    ]


    if not model_path.exists():

        raise FileNotFoundError(
            "학습된 모델 파일이 없습니다.\n"
            f"path: {model_path}"
        )


    bundle = joblib.load(
        model_path
    )


    required_keys = [
        "model",
        "preprocessor",
        "dataset_type",
        "feature_names",
    ]


    missing_keys = [
        key
        for key in required_keys
        if key not in bundle
    ]


    if missing_keys:

        raise ValueError(
            "Model Bundle 구조가 올바르지 않습니다.\n"
            f"누락 Key: {missing_keys}"
        )


    return bundle


# ============================================================
# 13. 실행
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # 현재는 전자금융공동망부터 학습
    #
    # 처음에는 전체 수백만 건을 사용하지 않는다.
    #
    # 1개 CSV에서 최대 100,000행으로
    # 전체 Pipeline을 먼저 검증한다.
    #
    # 정상적으로 학습된 후:
    #
    # max_files=None
    # nrows_per_file=None
    #
    # 로 변경하면 전체 Train 데이터 사용 가능.
    # ========================================================

    bundle = train_isolation_forest(

        dataset_type="electronic",

        # 정상거래 Sample 수
        sample_size=TRAIN_SAMPLE_SIZE,

        # 테스트 단계
        max_files=None,

        # 파일당 최대 10만 행
        nrows_per_file=None,
    )


    # ========================================================
    # 저장 파일 Load 검증
    # ========================================================

    loaded_bundle = (
        load_model_bundle(
            "electronic"
        )
    )


    print()
    print("=" * 70)
    print("MODEL LOAD TEST")
    print("=" * 70)

    print(
        "Dataset:",
        loaded_bundle[
            "dataset_type"
        ],
    )

    print(
        "Features:",
        loaded_bundle[
            "n_features"
        ],
    )

    print(
        "Load 성공"
    )

    print("=" * 70)