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
        iter_dataset_chunks,
    )

    from .feature_engineering import (
        get_feature_config,
        transform_features,
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
        iter_dataset_chunks,
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

    Feature Engineering용 컬럼
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
    이상거래여부를 0/1 numpy 배열로 변환한다.
    """

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
    chunksize: int = 100_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validation 전체를 Chunk 단위로 읽으면서

    1. Train Preprocessor로 Feature 변환
    2. Isolation Forest Score 계산
    3. Label 수집

    Returns
    -------
    y_true
        실제 이상거래 여부

    anomaly_scores
        높을수록 이상거래에 가까운 Score
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
    # 학습 완료 Model Bundle Load
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


    # Model / 요청 Dataset 일치 여부
    if bundle["dataset_type"] != dataset_type:

        raise ValueError(
            "Model의 dataset_type과 "
            "평가할 dataset_type이 다릅니다."
        )


    usecols = get_evaluation_columns(
        dataset_type
    )


    all_targets = []
    all_scores = []

    total_rows = 0
    chunk_count = 0


    print()
    print("=" * 70)
    print("VALIDATION SCORING")
    print("=" * 70)

    print(
        f"Dataset    : {dataset_type}"
    )

    print(
        f"Chunk Size : {chunksize:,}"
    )

    print("=" * 70)


    # ========================================================
    # Validation Chunk 순회
    # ========================================================

    for chunk in iter_dataset_chunks(
        dataset_type=dataset_type,
        split="validation",
        chunksize=chunksize,
        usecols=usecols,
    ):

        chunk_count += 1

        # ====================================================
        # Target
        # ====================================================

        y_chunk = extract_target(
            chunk,
            target_column,
        )


        # ====================================================
        # Feature Transform
        #
        # 주의:
        # Validation에서는 fit_transform 금지
        # Train에서 Fit된 Preprocessor만 사용
        # ====================================================

        X_chunk = transform_features(
            chunk,
            dataset_type,
            preprocessor,
        )


        # ====================================================
        # score_samples에는 CSR 형태 사용
        # ====================================================

        if sparse.issparse(
            X_chunk
        ):

            X_chunk = (
                X_chunk
                .tocsr()
                .astype(
                    np.float32
                )
            )

        else:

            X_chunk = np.asarray(
                X_chunk,
                dtype=np.float32,
            )


        # ====================================================
        # Isolation Forest Score
        #
        # sklearn:
        # 낮을수록 이상
        #
        # MOVI:
        # 높을수록 위험
        #
        # 따라서 부호 반전
        # ====================================================

        raw_scores = model.score_samples(
            X_chunk
        )

        anomaly_scores = (
            -raw_scores
        )


        all_targets.append(
            y_chunk
        )

        all_scores.append(
            anomaly_scores
        )


        total_rows += len(
            chunk
        )


        print(
            f"[Chunk {chunk_count:>3}] "
            f"누적 평가 거래="
            f"{total_rows:>10,}"
        )


    # ========================================================
    # 하나의 배열로 합치기
    # ========================================================

    if not all_targets:

        raise RuntimeError(
            "Validation 데이터가 없습니다."
        )


    y_true = np.concatenate(
        all_targets
    )

    anomaly_scores = np.concatenate(
        all_scores
    )


    print()
    print("=" * 70)
    print("Validation Scoring 완료")

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
    )


# ============================================================
# 5. Threshold 탐색
# ============================================================

def find_best_f1_threshold(
    y_true: np.ndarray,
    anomaly_scores: np.ndarray,
) -> dict:
    """
    Precision-Recall Curve를 이용하여
    F1 Score가 가장 높은 Threshold를 찾는다.

    Prediction:

        anomaly_score >= threshold
            → 이상거래 1

        anomaly_score < threshold
            → 정상거래 0
    """

    # Validation에 두 Class 모두 있어야 평가 가능
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


    # precision, recall은 thresholds보다
    # 원소가 하나 더 많음
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

        where=denominator != 0,
    )


    best_index = int(
        np.argmax(
            f1_scores
        )
    )


    best_threshold = float(
        thresholds[
            best_index
        ]
    )


    result = {

        "threshold": (
            best_threshold
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


    return result


# ============================================================
# 6. Threshold 기반 Prediction
# ============================================================

def predict_with_threshold(
    anomaly_scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    위험점수가 Threshold 이상이면 Fraud로 판정.
    """

    return (
        anomaly_scores
        >= threshold
    ).astype(
        np.int8
    )


# ============================================================
# 7. 최종 평가 Metric 계산
# ============================================================

def calculate_metrics(
    y_true: np.ndarray,
    anomaly_scores: np.ndarray,
    threshold: float,
) -> dict:
    """
    선택된 Threshold 기준 최종 평가 Metric 계산.
    """

    y_pred = predict_with_threshold(
        anomaly_scores,
        threshold,
    )


    # ========================================================
    # Confusion Matrix
    # ========================================================

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )


    tn, fp, fn, tp = (
        matrix.ravel()
    )


    # ========================================================
    # 기본 Metric
    # ========================================================

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
    # Threshold와 무관한 Ranking Metric
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


    # ========================================================
    # PR Curve AUC
    # ========================================================

    pr_precision, pr_recall, _ = (
        precision_recall_curve(
            y_true,
            anomaly_scores,
        )
    )

    # recall이 큰 값 → 작은 값 순으로 반환되므로
    # AUC 계산을 위해 역순으로 정렬
    pr_auc = auc(
        pr_recall[::-1],
        pr_precision[::-1],
    )


    # ========================================================
    # 추가 FDS 지표
    # ========================================================

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


    metrics = {

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

        "fraud_rate": (
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
    }


    return metrics


# ============================================================
# 8. JSON 저장 Helper
# ============================================================

def save_json(
    data: dict,
    file_path,
) -> None:
    """
    Dictionary를 JSON 파일로 저장한다.
    """

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
# 9. Threshold 저장
# ============================================================

def save_threshold(
    dataset_type: str,
    threshold_result: dict,
    validation_size: int,
) -> dict:
    """
    inference.py에서 사용할
    Threshold 설정 파일 저장.
    """

    config = get_dataset_config(
        dataset_type
    )


    threshold_data = {

        "dataset_type": (
            dataset_type
        ),

        # 점수 정의를 명시적으로 저장
        "score_definition": (
            "anomaly_score = "
            "-model.score_samples(X)"
        ),

        "rule": (
            "fraud if "
            "anomaly_score >= threshold"
        ),

        "selection_method": (
            "max_f1_on_validation"
        ),

        "threshold": (
            threshold_result[
                "threshold"
            ]
        ),

        "precision_at_threshold": (
            threshold_result[
                "precision"
            ]
        ),

        "recall_at_threshold": (
            threshold_result[
                "recall"
            ]
        ),

        "f1_at_threshold": (
            threshold_result[
                "f1"
            ]
        ),

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
# 10. Metrics 저장
# ============================================================

def save_metrics(
    dataset_type: str,
    metrics: dict,
) -> None:
    """
    평가 결과를 reports 폴더에 저장.
    """

    config = get_dataset_config(
        dataset_type
    )


    report = {

        "dataset_type": (
            dataset_type
        ),

        "evaluated_at": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),

        **metrics,
    }


    save_json(
        report,
        config[
            "report_path"
        ],
    )


# ============================================================
# 11. 평가 결과 출력
# ============================================================

def print_evaluation_summary(
    dataset_type: str,
    metrics: dict,
) -> None:
    """
    주요 평가 결과 Console 출력.
    """

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
    print("                 Pred Normal   Pred Fraud")

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


    print("=" * 70)


# ============================================================
# 12. 전체 Evaluation Pipeline
# ============================================================

def evaluate_model(
    dataset_type: str,
    *,
    chunksize: int = 100_000,
) -> dict:
    """
    전체 평가 Pipeline.

    Validation
        ↓
    Train Preprocessor
        ↓
    Isolation Forest
        ↓
    Anomaly Score
        ↓
    Best F1 Threshold
        ↓
    Precision / Recall / F1
        ↓
    threshold.json
        +
    metrics.json
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
    # Validation Score 계산
    # ========================================================

    print()
    print(
        "[STEP 1/4] "
        "Validation Score 계산"
    )


    y_true, anomaly_scores = (
        collect_validation_scores(
            dataset_type,
            chunksize=chunksize,
        )
    )


    # ========================================================
    # STEP 2
    # Threshold 탐색
    # ========================================================

    print()
    print(
        "[STEP 2/4] "
        "최적 Threshold 탐색"
    )


    threshold_result = (
        find_best_f1_threshold(
            y_true,
            anomaly_scores,
        )
    )


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
    # STEP 3
    # 최종 평가
    # ========================================================

    print()
    print(
        "[STEP 3/4] "
        "평가 Metric 계산"
    )


    metrics = calculate_metrics(
        y_true,
        anomaly_scores,
        threshold_result[
            "threshold"
        ],
    )


    # ========================================================
    # STEP 4
    # 결과 저장
    # ========================================================

    print()
    print(
        "[STEP 4/4] "
        "Threshold / Metrics 저장"
    )


    threshold_data = save_threshold(
        dataset_type,
        threshold_result,
        len(y_true),
    )


    save_metrics(
        dataset_type,
        metrics,
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

        "threshold": (
            threshold_data
        ),

        "metrics": (
            metrics
        ),
    }


# ============================================================
# 13. 실행
# ============================================================

if __name__ == "__main__":

    # 현재는 전자금융공동망 평가
    results = evaluate_model(

        dataset_type="electronic",

        chunksize=100_000,
    )