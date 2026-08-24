from __future__ import annotations

from datetime import datetime
import json

import numpy as np
import pandas as pd

from scipy import sparse

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
)


# ============================================================
# Import
# ============================================================

try:
    from .config import (
        ELECTRONIC_CONFIG,
        CARD_CONFIG,
        create_output_directories,
    )

    from .data_loader import (
        load_dataset,
    )

    from .feature_engineering import (
        get_feature_config,
        engineer_features,
    )

    from .train_iforest import (
        load_model_bundle,
    )

except ImportError:

    from config import (
        ELECTRONIC_CONFIG,
        CARD_CONFIG,
        create_output_directories,
    )

    from data_loader import (
        load_dataset,
    )

    from feature_engineering import (
        get_feature_config,
        engineer_features,
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
    Dataset 설정 반환.
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
# 2. Validation에서 필요한 컬럼
# ============================================================

def get_evaluation_columns(
    dataset_type: str,
) -> list[str]:
    """
    Validation CSV에서 평가에 필요한 컬럼만 읽는다.

    Feature Engineering용 원본 컬럼
    +
    이상거래여부
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
# 3. Target 검증
# ============================================================

def extract_target(
    df: pd.DataFrame,
    target_column: str,
) -> np.ndarray:
    """
    이상거래여부를 0 / 1 numpy 배열로 변환한다.
    """

    if target_column not in df.columns:

        raise ValueError(
            f"Validation Target 컬럼이 없습니다: "
            f"{target_column}"
        )

    target = pd.to_numeric(
        df[target_column],
        errors="coerce",
    )

    if target.isna().any():

        raise ValueError(
            "Validation Target에 NaN 또는 "
            "숫자가 아닌 값이 존재합니다."
        )

    target = target.astype(
        np.int8
    )

    unique_values = set(
        target.unique()
    )

    if not unique_values.issubset(
        {0, 1}
    ):

        raise ValueError(
            "이상거래여부는 0 또는 1이어야 합니다.\n"
            f"현재 값: {unique_values}"
        )

    return target.to_numpy()


# ============================================================
# 4. Validation Score 계산
# ============================================================

def collect_validation_scores(
    dataset_type: str,
    *,
    max_files: int | None = None,
    nrows_per_file: int | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
]:
    """
    Validation 데이터를 전체 시간순으로 불러온 뒤
    Historical Feature를 한 번에 생성한다.

    중요
    ----------------------------------------------------------

    Historical Feature:

        amount_ratio
        amount_zscore
        new_recipient
        unusual_medium
        historical_transaction_count
        same_day_transaction_count
        same_time_bucket_count

    는 거래 History가 필요하다.

    따라서 Chunk마다 독립적으로 Feature Engineering을 하면
    Chunk 경계에서 History가 끊어지므로 정확한 평가가 아니다.


    Flow
    ----------------------------------------------------------

    Validation 전체 Load
        ↓
    시간순 정렬
        ↓
    Historical Feature Engineering
        ↓
    Train Preprocessor Transform
        ↓
    Isolation Forest Score


    Returns
    -------
    y_true
        실제 이상거래 여부

    anomaly_scores
        높을수록 이상거래에 가까운 Score

    engineered_df
        분석용 Feature DataFrame
    """

    dataset_config = (
        get_dataset_config(
            dataset_type
        )
    )

    target_column = dataset_config[
        "target"
    ]


    # ========================================================
    # Model Bundle
    # ========================================================

    bundle = load_model_bundle(
        dataset_type
    )

    model = bundle[
        "model"
    ]

    preprocessor = bundle[
        "preprocessor"
    ]


    if (
        bundle["dataset_type"]
        != dataset_type
    ):

        raise ValueError(
            "Model의 dataset_type과 "
            "평가할 dataset_type이 다릅니다."
        )


    usecols = get_evaluation_columns(
        dataset_type
    )


    print()
    print("=" * 70)
    print("VALIDATION SCORING")
    print("=" * 70)

    print(
        f"Dataset : {dataset_type}"
    )

    if max_files is not None:
        print(
            f"Max Files : {max_files}"
        )

    if nrows_per_file is not None:
        print(
            f"Rows / File : {nrows_per_file:,}"
        )

    print("=" * 70)


    # ========================================================
    # STEP 1
    # Validation 전체 Load
    #
    # load_dataset 내부에서 시간순 정렬 수행
    # ========================================================

    print()
    print(
        "[1/4] Validation 전체 Load"
    )


    validation_df = load_dataset(

        dataset_type=dataset_type,

        split="validation",

        usecols=usecols,

        max_files=max_files,

        nrows_per_file=nrows_per_file,
    )


    print()
    print(
        f"Validation Rows : "
        f"{len(validation_df):,}"
    )


    # ========================================================
    # STEP 2
    # Target 추출
    # ========================================================

    print()
    print(
        "[2/4] Target 추출"
    )


    y_true = extract_target(
        validation_df,
        target_column,
    )


    print(
        f"Normal : "
        f"{np.sum(y_true == 0):,}"
    )

    print(
        f"Fraud  : "
        f"{np.sum(y_true == 1):,}"
    )


    # ========================================================
    # STEP 3
    # Historical Feature Engineering
    # ========================================================

    print()
    print(
        "[3/4] Historical Feature Engineering"
    )


    engineered_df = engineer_features(
        validation_df,
        dataset_type,
    )


    print(
        f"Engineered Rows    : "
        f"{len(engineered_df):,}"
    )

    print(
        f"Engineered Columns : "
        f"{len(engineered_df.columns):,}"
    )


    # ========================================================
    # STEP 4
    # Train Preprocessor Transform
    # ========================================================

    print()
    print(
        "[4/4] Preprocessor + Isolation Forest"
    )


    X_validation = (
        preprocessor
        .transform(
            engineered_df
        )
    )


    X_validation = (
        X_validation
        .astype(
            np.float32
        )
    )


    if sparse.issparse(
        X_validation
    ):

        X_validation = (
            X_validation
            .tocsr()
            .astype(
                np.float32
            )
        )

    else:

        X_validation = np.asarray(
            X_validation,
            dtype=np.float32,
        )


    # ========================================================
    # Isolation Forest Score
    #
    # score_samples:
    # 낮을수록 이상
    #
    # MOVI anomaly_score:
    # 높을수록 위험
    # ========================================================

    raw_scores = (
        model
        .score_samples(
            X_validation
        )
    )


    anomaly_scores = (
        -raw_scores
    )


    print()
    print("=" * 70)
    print("Validation Scoring 완료")
    print("=" * 70)

    print(
        f"Total Rows : "
        f"{len(y_true):,}"
    )

    print(
        f"Normal     : "
        f"{np.sum(y_true == 0):,}"
    )

    print(
        f"Fraud      : "
        f"{np.sum(y_true == 1):,}"
    )

    print("=" * 70)


    return (
        y_true,
        anomaly_scores,
        engineered_df,
    )


# ============================================================
# 5. Score 분포 분석
# ============================================================

def calculate_score_distribution(
    y_true: np.ndarray,
    anomaly_scores: np.ndarray,
) -> dict:
    """
    정상 / 이상거래 각각의 Score 분포를 계산한다.

    Dummy Test에서 Score 차이가 작아 보였기 때문에
    실제 Validation 분포에서 Score의 상대적 위치를
    확인하기 위한 함수.
    """

    normal_scores = (
        anomaly_scores[
            y_true == 0
        ]
    )

    fraud_scores = (
        anomaly_scores[
            y_true == 1
        ]
    )


    def summarize(
        scores: np.ndarray,
    ) -> dict:

        if len(scores) == 0:

            return {
                "count": 0,
            }

        return {

            "count": int(
                len(scores)
            ),

            "min": float(
                np.min(scores)
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

            "q25": float(
                np.quantile(
                    scores,
                    0.25,
                )
            ),

            "median": float(
                np.quantile(
                    scores,
                    0.50,
                )
            ),

            "q75": float(
                np.quantile(
                    scores,
                    0.75,
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

            "max": float(
                np.max(scores)
            ),

            "mean": float(
                np.mean(scores)
            ),

            "std": float(
                np.std(scores)
            ),
        }


    return {

        "normal": summarize(
            normal_scores
        ),

        "fraud": summarize(
            fraud_scores
        ),
    }


# ============================================================
# 6. Threshold 탐색
# ============================================================

def find_best_f1_threshold(
    y_true: np.ndarray,
    anomaly_scores: np.ndarray,
) -> dict:
    """
    Precision-Recall Curve 기반으로
    F1 Score가 가장 높은 Threshold 선택.

    anomaly_score >= threshold
        → Fraud
    """

    unique_classes = np.unique(
        y_true
    )

    if len(unique_classes) < 2:

        raise ValueError(
            "Validation 데이터에 정상/이상 "
            "두 클래스가 모두 필요합니다."
        )


    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            anomaly_scores,
        )
    )


    if len(thresholds) == 0:

        raise RuntimeError(
            "Threshold 후보를 생성할 수 없습니다."
        )


    precision_for_thresholds = (
        precision[:-1]
    )

    recall_for_thresholds = (
        recall[:-1]
    )


    denominator = (
        precision_for_thresholds
        + recall_for_thresholds
    )


    f1_scores = np.divide(

        2
        * precision_for_thresholds
        * recall_for_thresholds,

        denominator,

        out=np.zeros_like(
            denominator
        ),

        where=(
            denominator != 0
        ),
    )


    best_index = int(
        np.argmax(
            f1_scores
        )
    )


    return {

        "threshold": float(
            thresholds[
                best_index
            ]
        ),

        "precision": float(
            precision_for_thresholds[
                best_index
            ]
        ),

        "recall": float(
            recall_for_thresholds[
                best_index
            ]
        ),

        "f1": float(
            f1_scores[
                best_index
            ]
        ),
    }


# ============================================================
# 7. Threshold 후보 비교
# ============================================================

def compare_threshold_candidates(
    y_true: np.ndarray,
    anomaly_scores: np.ndarray,
    best_threshold: float,
) -> list[dict]:
    """
    Best F1 Threshold 주변과 Score Quantile 기반 후보를 비교한다.

    FDS에서는 반드시 F1 하나만 보고 Threshold를
    결정할 필요는 없다.

    Recall 우선 정책 등으로 변경할 수 있도록
    여러 후보를 출력한다.
    """

    normal_scores = (
        anomaly_scores[
            y_true == 0
        ]
    )


    candidates = [

        best_threshold,

        float(
            np.quantile(
                anomaly_scores,
                0.90,
            )
        ),

        float(
            np.quantile(
                anomaly_scores,
                0.95,
            )
        ),

        float(
            np.quantile(
                anomaly_scores,
                0.99,
            )
        ),

        float(
            np.quantile(
                normal_scores,
                0.95,
            )
        ),

        float(
            np.quantile(
                normal_scores,
                0.99,
            )
        ),
    ]


    # 중복 제거
    candidates = sorted(
        set(
            candidates
        )
    )


    results = []


    for threshold in candidates:

        y_pred = (
            anomaly_scores
            >= threshold
        ).astype(
            np.int8
        )


        precision = (
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        )

        recall = (
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        )

        f1 = (
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        )


        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=[
                0,
                1,
            ],
        )


        tn, fp, fn, tp = (
            matrix.ravel()
        )


        results.append({

            "threshold": float(
                threshold
            ),

            "precision": float(
                precision
            ),

            "recall": float(
                recall
            ),

            "f1": float(
                f1
            ),

            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        })


    return results


# ============================================================
# 8. Threshold 기반 Prediction
# ============================================================

def predict_with_threshold(
    anomaly_scores: np.ndarray,
    threshold: float,
) -> np.ndarray:

    return (
        anomaly_scores
        >= threshold
    ).astype(
        np.int8
    )


# ============================================================
# 9. 최종 평가 Metric
# ============================================================

def calculate_metrics(
    y_true: np.ndarray,
    anomaly_scores: np.ndarray,
    threshold: float,
) -> dict:

    y_pred = predict_with_threshold(
        anomaly_scores,
        threshold,
    )


    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1,
        ],
    )


    tn, fp, fn, tp = (
        matrix.ravel()
    )


    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )


    # ========================================================
    # Ranking Metrics
    # ========================================================

    average_precision = (
        average_precision_score(
            y_true,
            anomaly_scores,
        )
    )


    roc_auc = roc_auc_score(
        y_true,
        anomaly_scores,
    )


    pr_precision, pr_recall, _ = (
        precision_recall_curve(
            y_true,
            anomaly_scores,
        )
    )


    pr_auc = auc(
        pr_recall[::-1],
        pr_precision[::-1],
    )


    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )


    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0.0
    )


    fraud_rate = float(
        np.mean(
            y_true == 1
        )
    )


    score_distribution = (
        calculate_score_distribution(
            y_true,
            anomaly_scores,
        )
    )


    return {

        "threshold": float(
            threshold
        ),

        "accuracy": float(
            accuracy
        ),

        "precision": float(
            precision
        ),

        "recall": float(
            recall
        ),

        "f1": float(
            f1
        ),

        "average_precision": float(
            average_precision
        ),

        "pr_auc": float(
            pr_auc
        ),

        "roc_auc": float(
            roc_auc
        ),

        "false_positive_rate": float(
            false_positive_rate
        ),

        "false_negative_rate": float(
            false_negative_rate
        ),

        "fraud_rate": float(
            fraud_rate
        ),

        "confusion_matrix": {

            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },

        "validation": {

            "total": int(
                len(y_true)
            ),

            "normal": int(
                np.sum(
                    y_true == 0
                )
            ),

            "fraud": int(
                np.sum(
                    y_true == 1
                )
            ),
        },

        "score_summary": {

            "min": float(
                np.min(
                    anomaly_scores
                )
            ),

            "max": float(
                np.max(
                    anomaly_scores
                )
            ),

            "mean": float(
                np.mean(
                    anomaly_scores
                )
            ),

            "std": float(
                np.std(
                    anomaly_scores
                )
            ),

            "normal_mean": float(
                np.mean(
                    anomaly_scores[
                        y_true == 0
                    ]
                )
            ),

            "fraud_mean": float(
                np.mean(
                    anomaly_scores[
                        y_true == 1
                    ]
                )
            ),
        },

        "score_distribution":
            score_distribution,
    }


# ============================================================
# 10. JSON 저장 Helper
# ============================================================

def save_json(
    data: dict,
    file_path,
) -> None:

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    with open(
        file_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=4,
        )


# ============================================================
# 11. Threshold 저장
# ============================================================

def save_threshold(
    dataset_type: str,
    threshold_result: dict,
    validation_size: int,
) -> dict:

    config = get_dataset_config(
        dataset_type
    )


    threshold_data = {

        "dataset_type":
            dataset_type,

        "score_definition":
            "anomaly_score = -model.score_samples(X)",

        "rule":
            "fraud if anomaly_score >= threshold",

        "selection_method":
            "max_f1_on_validation",

        "threshold":
            threshold_result[
                "threshold"
            ],

        "precision_at_threshold":
            threshold_result[
                "precision"
            ],

        "recall_at_threshold":
            threshold_result[
                "recall"
            ],

        "f1_at_threshold":
            threshold_result[
                "f1"
            ],

        "validation_size": int(
            validation_size
        ),

        "created_at": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),
    }


    save_json(
        threshold_data,
        config[
            "threshold_path"
        ],
    )


    return threshold_data


# ============================================================
# 12. Metrics 저장
# ============================================================

def save_metrics(
    dataset_type: str,
    metrics: dict,
    threshold_candidates: list[dict],
) -> None:

    config = get_dataset_config(
        dataset_type
    )


    report = {

        "dataset_type":
            dataset_type,

        "evaluated_at": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),

        "threshold_candidates":
            threshold_candidates,

        **metrics,
    }


    save_json(
        report,
        config[
            "report_path"
        ],
    )


# ============================================================
# 13. Score 분포 출력
# ============================================================

def print_score_distribution(
    distribution: dict,
) -> None:

    print()
    print("[Score Distribution]")

    print()
    print(
        f"{'Statistic':<12}"
        f"{'Normal':>14}"
        f"{'Fraud':>14}"
    )

    print(
        "-" * 40
    )


    keys = [
        "mean",
        "std",
        "min",
        "q01",
        "q05",
        "q25",
        "median",
        "q75",
        "q95",
        "q99",
        "max",
    ]


    normal = distribution[
        "normal"
    ]

    fraud = distribution[
        "fraud"
    ]


    for key in keys:

        print(
            f"{key:<12}"
            f"{normal[key]:>14.6f}"
            f"{fraud[key]:>14.6f}"
        )


# ============================================================
# 14. Threshold 후보 출력
# ============================================================

def print_threshold_candidates(
    candidates: list[dict],
) -> None:

    print()
    print("[Threshold Candidates]")

    print(
        f"{'Threshold':>12} "
        f"{'Precision':>10} "
        f"{'Recall':>10} "
        f"{'F1':>10} "
        f"{'FP':>10} "
        f"{'FN':>10}"
    )

    print(
        "-" * 70
    )


    for result in candidates:

        print(
            f"{result['threshold']:>12.6f} "
            f"{result['precision']:>10.4f} "
            f"{result['recall']:>10.4f} "
            f"{result['f1']:>10.4f} "
            f"{result['fp']:>10,} "
            f"{result['fn']:>10,}"
        )


# ============================================================
# 15. 평가 결과 출력
# ============================================================

def print_evaluation_summary(
    dataset_type: str,
    metrics: dict,
) -> None:

    cm = metrics[
        "confusion_matrix"
    ]

    validation = metrics[
        "validation"
    ]

    scores = metrics[
        "score_summary"
    ]


    print()
    print("=" * 70)
    print("EVALUATION RESULT")
    print("=" * 70)


    print(
        f"Dataset          : "
        f"{dataset_type}"
    )

    print(
        f"Validation Rows  : "
        f"{validation['total']:,}"
    )

    print(
        f"Normal           : "
        f"{validation['normal']:,}"
    )

    print(
        f"Fraud            : "
        f"{validation['fraud']:,}"
    )


    print()
    print("[Threshold]")

    print(
        f"{metrics['threshold']:.8f}"
    )


    print()
    print("[Main Metrics]")

    print(
        f"Precision        : "
        f"{metrics['precision']:.6f}"
    )

    print(
        f"Recall           : "
        f"{metrics['recall']:.6f}"
    )

    print(
        f"F1               : "
        f"{metrics['f1']:.6f}"
    )

    print(
        f"Average Precision: "
        f"{metrics['average_precision']:.6f}"
    )

    print(
        f"PR-AUC           : "
        f"{metrics['pr_auc']:.6f}"
    )

    print(
        f"ROC-AUC          : "
        f"{metrics['roc_auc']:.6f}"
    )

    print(
        f"Accuracy         : "
        f"{metrics['accuracy']:.6f}"
    )


    print()
    print("[Error Rates]")

    print(
        f"False Positive   : "
        f"{metrics['false_positive_rate']:.6f}"
    )

    print(
        f"False Negative   : "
        f"{metrics['false_negative_rate']:.6f}"
    )


    print()
    print("[Confusion Matrix]")

    print()
    print(
        "                 "
        "Pred Normal   Pred Fraud"
    )

    print(
        f"Actual Normal    "
        f"{cm['tn']:>11,} "
        f"{cm['fp']:>12,}"
    )

    print(
        f"Actual Fraud     "
        f"{cm['fn']:>11,} "
        f"{cm['tp']:>12,}"
    )


    print()
    print("[Anomaly Score]")

    print(
        f"Normal Mean      : "
        f"{scores['normal_mean']:.6f}"
    )

    print(
        f"Fraud Mean       : "
        f"{scores['fraud_mean']:.6f}"
    )

    print(
        f"Score Min        : "
        f"{scores['min']:.6f}"
    )

    print(
        f"Score Max        : "
        f"{scores['max']:.6f}"
    )


    print_score_distribution(
        metrics[
            "score_distribution"
        ]
    )


    print("=" * 70)


# ============================================================
# 16. 전체 Evaluation Pipeline
# ============================================================

def evaluate_model(
    dataset_type: str,
    *,
    max_files: int | None = None,
    nrows_per_file: int | None = None,
) -> dict:
    """
    Evaluation Pipeline.

    Validation
        ↓
    시간순 정렬
        ↓
    Historical Feature Engineering
        ↓
    Train Preprocessor
        ↓
    Isolation Forest
        ↓
    anomaly_score
        ↓
    실제 Label 비교
        ↓
    Threshold 탐색
        ↓
    Metrics
    """

    create_output_directories()


    print()
    print("#" * 70)

    print(
        f"Isolation Forest Evaluation : "
        f"{dataset_type}"
    )

    print("#" * 70)


    # ========================================================
    # STEP 1
    # Validation Score
    # ========================================================

    print()
    print(
        "[STEP 1/5] "
        "Validation Score 계산"
    )


    (
        y_true,
        anomaly_scores,
        engineered_df,
    ) = collect_validation_scores(

        dataset_type,

        max_files=max_files,

        nrows_per_file=nrows_per_file,
    )


    # ========================================================
    # STEP 2
    # 실제 Score 분포
    # ========================================================

    print()
    print(
        "[STEP 2/5] "
        "정상 / 이상 Score 분포 분석"
    )


    score_distribution = (
        calculate_score_distribution(
            y_true,
            anomaly_scores,
        )
    )


    print_score_distribution(
        score_distribution
    )


    # ========================================================
    # STEP 3
    # Best Threshold
    # ========================================================

    print()
    print(
        "[STEP 3/5] "
        "Best F1 Threshold 탐색"
    )


    threshold_result = (
        find_best_f1_threshold(
            y_true,
            anomaly_scores,
        )
    )


    print()
    print(
        f"Threshold : "
        f"{threshold_result['threshold']:.8f}"
    )

    print(
        f"Precision : "
        f"{threshold_result['precision']:.6f}"
    )

    print(
        f"Recall    : "
        f"{threshold_result['recall']:.6f}"
    )

    print(
        f"F1        : "
        f"{threshold_result['f1']:.6f}"
    )


    # ========================================================
    # STEP 4
    # 후보 Threshold 비교
    # ========================================================

    print()
    print(
        "[STEP 4/5] "
        "Threshold 후보 비교"
    )


    threshold_candidates = (
        compare_threshold_candidates(

            y_true,

            anomaly_scores,

            threshold_result[
                "threshold"
            ],
        )
    )


    print_threshold_candidates(
        threshold_candidates
    )


    # ========================================================
    # STEP 5
    # Metric + 저장
    # ========================================================

    print()
    print(
        "[STEP 5/5] "
        "Metric 계산 / 결과 저장"
    )


    metrics = calculate_metrics(

        y_true,

        anomaly_scores,

        threshold_result[
            "threshold"
        ],
    )


    threshold_data = (
        save_threshold(

            dataset_type,

            threshold_result,

            len(
                y_true
            ),
        )
    )


    save_metrics(

        dataset_type,

        metrics,

        threshold_candidates,
    )


    config = get_dataset_config(
        dataset_type
    )


    print()
    print(
        "Threshold 저장:"
    )

    print(
        config[
            "threshold_path"
        ]
    )


    print()
    print(
        "Metrics 저장:"
    )

    print(
        config[
            "report_path"
        ]
    )


    print_evaluation_summary(

        dataset_type,

        metrics,
    )


    return {

        "threshold":
            threshold_data,

        "metrics":
            metrics,

        "threshold_candidates":
            threshold_candidates,
    }


# ============================================================
# 17. 실행
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # 처음에는 Validation 일부로 Pipeline 확인
    #
    # 정상적으로 동작하면:
    #
    # max_files=None
    # nrows_per_file=None
    #
    # 로 변경하여 전체 Validation 평가
    # ========================================================

    results = evaluate_model(

        dataset_type="electronic",

        # 초기 테스트
        max_files=None,

        nrows_per_file=None,
    )