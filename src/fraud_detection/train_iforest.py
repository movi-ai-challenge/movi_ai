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
        iter_dataset_chunks,
    )

    from .feature_engineering import (
        get_feature_config,
        fit_transform_features,
        get_transformed_feature_names,
    )

except ImportError:

    from config import (
        ELECTRONIC_CONFIG,
        CARD_CONFIG,
        ISOLATION_FOREST_PARAMS,
        RANDOM_STATE,
        TRAIN_SAMPLE_SIZE,
        create_output_directories,
    )

    from data_loader import (
        iter_dataset_chunks,
    )

    from feature_engineering import (
        get_feature_config,
        fit_transform_features,
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
    CSV에서 실제 학습에 필요한 컬럼만 읽기 위한 함수.

    전체 CSV 컬럼을 모두 읽지 않고

        Feature Engineering에 필요한 컬럼
        +
        이상거래여부

    만 읽는다.

    메모리 사용량 감소 목적.
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
# 4. 정상거래 추출
# ============================================================

def extract_normal_transactions(
    chunk: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    """
    이상거래여부 == 0 인 정상거래만 반환한다.

    Isolation Forest Baseline은
    정상 패턴을 기준으로 학습한다.
    """

    target = pd.to_numeric(
        chunk[target_column],
        errors="coerce",
    )

    normal_mask = (
        target == 0
    )

    normal_df = (
        chunk.loc[normal_mask]
        .copy()
    )

    return normal_df


# ============================================================
# 5. 정상거래 Uniform Sampling
# ============================================================

def collect_normal_sample(
    dataset_type: str,
    sample_size: int = TRAIN_SAMPLE_SIZE,
    chunksize: int = 100_000,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, int]:
    """
    Train 데이터 전체를 Chunk 단위로 순회하면서
    정상거래 중 sample_size개를 균등하게 Sampling한다.

    단순히 앞의 30만 건만 사용하는 것이 아니라
    모든 Train 정상거래에 랜덤 우선순위를 부여하고
    그중 가장 작은 sample_size개를 유지한다.

    이를 통해 특정 월 / 특정 파일 앞부분에
    학습 데이터가 편향되는 것을 줄인다.

    Returns
    -------
    sample_df
        최종 정상거래 Sample

    total_normal_seen
        전체 Train에서 확인한 정상거래 개수
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

    rng = np.random.default_rng(
        random_state
    )

    # 최종적으로 유지할 Sample
    reservoir = None

    total_rows_seen = 0
    total_normal_seen = 0
    chunk_count = 0


    print()
    print("=" * 70)
    print("NORMAL TRANSACTION SAMPLING")
    print("=" * 70)

    print(
        f"Dataset      : {dataset_type}"
    )

    print(
        f"Target Sample: {sample_size:,}"
    )

    print(
        f"Chunk Size   : {chunksize:,}"
    )

    print("=" * 70)


    # ========================================================
    # 전체 Train 데이터 순회
    # ========================================================

    for chunk in iter_dataset_chunks(
        dataset_type=dataset_type,
        split="train",
        chunksize=chunksize,
        usecols=usecols,
    ):

        chunk_count += 1

        total_rows_seen += len(
            chunk
        )


        # ====================================================
        # 정상거래만 추출
        # ====================================================

        normal_chunk = (
            extract_normal_transactions(
                chunk,
                target_column,
            )
        )

        normal_count = len(
            normal_chunk
        )

        total_normal_seen += (
            normal_count
        )


        # 정상 데이터가 없는 Chunk는 Skip
        if normal_count == 0:

            continue


        # ====================================================
        # 각 행에 Random Priority 부여
        # ====================================================

        normal_chunk[
            "__sample_priority__"
        ] = rng.random(
            normal_count
        )


        # ====================================================
        # 기존 Sample + 새로운 Chunk 결합
        # ====================================================

        if reservoir is None:

            candidates = (
                normal_chunk
            )

        else:

            candidates = pd.concat(
                [
                    reservoir,
                    normal_chunk,
                ],
                ignore_index=True,
                copy=False,
            )


        # ====================================================
        # Priority가 가장 작은 K개만 유지
        #
        # 각 행에 독립적인 난수를 부여한 뒤
        # 전역에서 작은 K개를 유지하므로
        # 전체 Train 데이터에서 Random Sample하는 것과
        # 같은 효과를 얻는다.
        # ====================================================

        if len(candidates) > sample_size:

            reservoir = (
                candidates
                .nsmallest(
                    sample_size,
                    "__sample_priority__",
                )
                .reset_index(
                    drop=True
                )
            )

        else:

            reservoir = (
                candidates
                .reset_index(
                    drop=True
                )
            )


        # ====================================================
        # 진행상황 출력
        # ====================================================

        print(
            f"[Chunk {chunk_count:>3}] "
            f"전체={total_rows_seen:>10,} | "
            f"정상={total_normal_seen:>10,} | "
            f"보관={len(reservoir):>8,}"
        )


    # ========================================================
    # 데이터가 아예 없는 경우
    # ========================================================

    if reservoir is None:

        raise RuntimeError(
            "정상거래 데이터를 찾을 수 없습니다."
        )


    # ========================================================
    # Priority 임시 컬럼 제거
    # ========================================================

    reservoir.drop(
        columns=[
            "__sample_priority__"
        ],
        inplace=True,
    )


    # ========================================================
    # Sample Size 검증
    # ========================================================

    if len(reservoir) < sample_size:

        print()
        print(
            "[WARNING] 전체 정상거래 수가 "
            "요청 Sample Size보다 작습니다."
        )

        print(
            f"요청 : {sample_size:,}"
        )

        print(
            f"실제 : {len(reservoir):,}"
        )


    print()
    print("=" * 70)

    print(
        "Sampling 완료"
    )

    print(
        f"전체 Train Row   : "
        f"{total_rows_seen:,}"
    )

    print(
        f"전체 정상거래    : "
        f"{total_normal_seen:,}"
    )

    print(
        f"최종 학습 Sample : "
        f"{len(reservoir):,}"
    )

    print("=" * 70)


    return (
        reservoir,
        total_normal_seen,
    )


# ============================================================
# 6. Isolation Forest 생성
# ============================================================

def build_isolation_forest(
) -> IsolationForest:
    """
    config.py에 정의한 Parameter로
    Isolation Forest 생성.
    """

    model = IsolationForest(
        **ISOLATION_FOREST_PARAMS
    )

    return model


# ============================================================
# 7. 학습 Score Summary
# ============================================================

def calculate_training_score_summary(
    model: IsolationForest,
    X,
) -> dict:
    """
    정상 Train Sample에 대한 score_samples 분포를 계산한다.

    Isolation Forest:
        score가 낮을수록 이상치.

    향후 evaluate.py에서 Validation Score와
    비교하기 위한 참고용 Metadata.
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
    training_score_summary: dict,
    training_seconds: float,
) -> dict:
    """
    Model + Preprocessor + Metadata를 하나의 객체로 묶는다.

    evaluate.py / inference.py에서 이 파일 하나만 Load하면 된다.
    """

    bundle = {

        # ====================================================
        # 핵심 객체
        # ====================================================

        "model": model,

        "preprocessor": preprocessor,


        # ====================================================
        # Dataset 정보
        # ====================================================

        "dataset_type": (
            dataset_type
        ),

        "target_column": (
            get_dataset_config(
                dataset_type
            )["target"]
        ),


        # ====================================================
        # Feature 정보
        # ====================================================

        "feature_names": (
            feature_names
        ),

        "n_features": len(
            feature_names
        ),


        # ====================================================
        # Training 정보
        # ====================================================

        "training_sample_size": (
            training_sample_size
        ),

        "total_normal_seen": (
            total_normal_seen
        ),

        "isolation_forest_params": (
            model.get_params()
        ),

        "training_score_summary": (
            training_score_summary
        ),

        "training_seconds": (
            training_seconds
        ),

        "trained_at": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),


        # ====================================================
        # 환경 정보
        #
        # 저장한 모델을 나중에 재현하거나
        # 버전 문제를 추적하기 위해 저장
        # ====================================================

        "environment": {

            "python": (
                platform.python_version()
            ),

            "scikit_learn": (
                sklearn.__version__
            ),

            "pandas": (
                pd.__version__
            ),

            "numpy": (
                np.__version__
            ),
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
    학습 결과 Bundle을 Joblib 파일로 저장.
    """

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        bundle,
        model_path,

        # 적당한 압축
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
    학습 결과를 콘솔에 출력.
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
        f"Train Samples  : "
        f"{bundle['training_sample_size']:,}"
    )

    print(
        f"Normal Seen    : "
        f"{bundle['total_normal_seen']:,}"
    )

    print(
        f"Features       : "
        f"{bundle['n_features']:,}"
    )

    print(
        f"Training Time  : "
        f"{bundle['training_seconds']:.2f} sec"
    )

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
    chunksize: int = 100_000,
) -> dict:
    """
    Isolation Forest 전체 학습 Pipeline.

    Flow
    ----------------------------------------------------------

    Train CSV
        ↓
    Chunk Loading
        ↓
    정상거래 추출
        ↓
    Uniform Sampling
        ↓
    Feature Engineering
        ↓
    Preprocessor Fit
        ↓
    Isolation Forest Fit
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
    # 정상거래 Sampling
    # ========================================================

    print()
    print(
        "[STEP 1/5] "
        "정상거래 Sampling"
    )

    normal_df, total_normal_seen = (
        collect_normal_sample(
            dataset_type=dataset_type,
            sample_size=sample_size,
            chunksize=chunksize,
        )
    )


    # ========================================================
    # STEP 2
    # Feature Engineering + Preprocessor Fit
    # ========================================================

    print()
    print(
        "[STEP 2/5] "
        "Feature Engineering"
    )


    X_train, preprocessor = (
        fit_transform_features(
            normal_df,
            dataset_type,
        )
    )


    # ========================================================
    # Sparse Matrix 처리
    #
    # IsolationForest.fit()에서는
    # CSC Sparse Matrix를 효율적으로 사용할 수 있다.
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

        X_train = (
            np.asarray(
                X_train,
                dtype=np.float32,
            )
        )


    feature_names = (
        get_transformed_feature_names(
            preprocessor
        )
    )


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


    model = (
        build_isolation_forest()
    )


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
    # 정상 Train Score 확인
    # ========================================================

    score_summary = (
        calculate_training_score_summary(
            model,
            X_train,
        )
    )


    # ========================================================
    # STEP 5
    # 저장
    # ========================================================

    print()
    print(
        "[STEP 5/5] "
        "Model 저장"
    )


    bundle = (
        create_model_bundle(
            dataset_type=dataset_type,
            model=model,
            preprocessor=preprocessor,
            feature_names=feature_names,
            training_sample_size=len(
                normal_df
            ),
            total_normal_seen=total_normal_seen,
            training_score_summary=score_summary,
            training_seconds=training_seconds,
        )
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
    저장한 Model Bundle Load.

    evaluate.py / inference.py에서도
    동일한 방식으로 사용 예정.
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
    # ========================================================

    bundle = train_isolation_forest(
        dataset_type="electronic",

        # Baseline
        sample_size=TRAIN_SAMPLE_SIZE,

        # CSV는 10만 행씩 읽기
        chunksize=100_000,
    )


    # ========================================================
    # 저장 파일이 정상 Load 되는지 마지막 검증
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