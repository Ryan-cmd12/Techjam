from __future__ import annotations

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_metrics(
    labels,
    probabilities,
    threshold: float = 0.5,
) -> dict[str, float]:

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    predictions = (
        probabilities
        >= threshold
    ).astype(
        np.int64
    )

    accuracy = (
        accuracy_score(
            labels,
            predictions,
        )
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            labels,
            predictions,
        )
    )

    precision = (
        precision_score(
            labels,
            predictions,
            zero_division=0,
        )
    )

    recall = (
        recall_score(
            labels,
            predictions,
            zero_division=0,
        )
    )

    f1 = (
        f1_score(
            labels,
            predictions,
            zero_division=0,
        )
    )

    unique_labels = (
        np.unique(
            labels
        )
    )

    if len(unique_labels) == 2:

        auroc = (
            roc_auc_score(
                labels,
                probabilities,
            )
        )

        average_precision = (
            average_precision_score(
                labels,
                probabilities,
            )
        )

    else:

        auroc = float(
            "nan"
        )

        average_precision = float(
            "nan"
        )

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[
            0,
            1,
        ],
    )

    tn, fp, fn, tp = (
        matrix.ravel()
    )

    if (
        fp + tn
    ) > 0:

        false_positive_rate = (
            fp
            / (
                fp + tn
            )
        )

    else:

        false_positive_rate = (
            float(
                "nan"
            )
        )

    if (
        fn + tp
    ) > 0:

        false_negative_rate = (
            fn
            / (
                fn + tp
            )
        )

    else:

        false_negative_rate = (
            float(
                "nan"
            )
        )

    return {
        "accuracy":
            float(
                accuracy
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy
            ),

        "precision":
            float(
                precision
            ),

        "recall":
            float(
                recall
            ),

        "f1":
            float(
                f1
            ),

        "auroc":
            float(
                auroc
            ),

        "average_precision":
            float(
                average_precision
            ),

        "false_positive_rate":
            float(
                false_positive_rate
            ),

        "false_negative_rate":
            float(
                false_negative_rate
            ),

        "true_negatives":
            int(
                tn
            ),

        "false_positives":
            int(
                fp
            ),

        "false_negatives":
            int(
                fn
            ),

        "true_positives":
            int(
                tp
            ),
    }


def print_metrics(
    metrics: dict,
    title: str,
) -> None:

    print(
        "\n=============================="
    )

    print(
        title.upper()
    )

    print(
        "=============================="
    )

    print(
        f"Accuracy:           "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Balanced Accuracy:  "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Precision:          "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall:             "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1:                 "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"AUROC:              "
        f"{metrics['auroc']:.4f}"
    )

    print(
        f"Average Precision:  "
        f"{metrics['average_precision']:.4f}"
    )

    print(
        f"False Positive Rate:"
        f" {metrics['false_positive_rate']:.4f}"
    )

    print(
        f"False Negative Rate:"
        f" {metrics['false_negative_rate']:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        f"TN: "
        f"{metrics['true_negatives']}   "
        f"FP: "
        f"{metrics['false_positives']}"
    )

    print(
        f"FN: "
        f"{metrics['false_negatives']}   "
        f"TP: "
        f"{metrics['true_positives']}"
    )