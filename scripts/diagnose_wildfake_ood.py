from __future__ import annotations

import argparse

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
)


SCORE_COLUMNS = {
    "final":
        "pred",

    "base_fusion":
        "base_pred",

    "semantic":
        "semantic_pred",

    "forensic":
        "forensic_pred",
}


def safe_auroc(
    labels,
    scores,
):

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    if len(
        np.unique(
            labels
        )
    ) < 2:

        return float(
            "nan"
        )

    return float(
        roc_auc_score(
            labels,
            scores,
        )
    )


def safe_ap(
    labels,
    scores,
):

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    if len(
        np.unique(
            labels
        )
    ) < 2:

        return float(
            "nan"
        )

    return float(
        average_precision_score(
            labels,
            scores,
        )
    )


def evaluate_score(
    dataframe,
    score_column,
):

    labels = (
        dataframe[
            "label"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    scores = (
        dataframe[
            score_column
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    predictions = (
        scores >= 0.5
    ).astype(
        np.int64
    )

    real_scores = scores[
        labels == 0
    ]

    fake_scores = scores[
        labels == 1
    ]

    auroc = safe_auroc(
        labels,
        scores,
    )

    return {
        "auroc":
            auroc,

        # Diagnostic only.
        #
        # If AUROC is 0.35 then the inverse ordering
        # would be 0.65. We NEVER use this to flip
        # predictions; it only measures inversion.
        "inverse_auroc":
            (
                1.0 - auroc
                if np.isfinite(
                    auroc
                )
                else float(
                    "nan"
                )
            ),

        "average_precision":
            safe_ap(
                labels,
                scores,
            ),

        "accuracy":
            float(
                accuracy_score(
                    labels,
                    predictions,
                )
            ),

        "mean_real_score":
            float(
                real_scores.mean()
            ),

        "mean_fake_score":
            float(
                fake_scores.mean()
            ),

        "median_real_score":
            float(
                np.median(
                    real_scores
                )
            ),

        "median_fake_score":
            float(
                np.median(
                    fake_scores
                )
            ),

        "score_gap_fake_minus_real":
            float(
                fake_scores.mean()
                - real_scores.mean()
            ),

        "fake_recall":
            float(
                (
                    fake_scores
                    >= 0.5
                ).mean()
            ),

        "real_specificity":
            float(
                (
                    real_scores
                    < 0.5
                ).mean()
            ),
    }


def analyze_condition(
    dataframe,
):

    rows = []

    for (
        display_name,
        score_column,
    ) in SCORE_COLUMNS.items():

        if (
            score_column
            not in dataframe.columns
        ):

            continue

        metrics = evaluate_score(
            dataframe=
                dataframe,

            score_column=
                score_column,
        )

        rows.append(
            {
                "branch":
                    display_name,

                **metrics,
            }
        )

    return pd.DataFrame(
        rows
    )


def build_condition_comparison(
    dataframe,
):

    rows = []

    for (
        condition_key,
        condition,
    ) in dataframe.groupby(
        "condition_key"
    ):

        condition_name = (
            condition[
                "condition_name"
            ].iloc[
                0
            ]
        )

        branch_metrics = (
            analyze_condition(
                condition
            )
        )

        for _, row in (
            branch_metrics.iterrows()
        ):

            rows.append(
                {
                    "condition_key":
                        condition_key,

                    "condition_name":
                        condition_name,

                    **row.to_dict(),
                }
            )

    return pd.DataFrame(
        rows
    )


def build_gate_summary(
    dataframe,
):

    rows = []

    for (
        condition_key,
        condition,
    ) in dataframe.groupby(
        "condition_key"
    ):

        real = condition[
            condition[
                "label"
            ] == 0
        ]

        fake = condition[
            condition[
                "label"
            ] == 1
        ]

        rows.append(
            {
                "condition_key":
                    condition_key,

                "condition_name":
                    condition[
                        "condition_name"
                    ].iloc[
                        0
                    ],

                "semantic_weight_all":
                    float(
                        condition[
                            "semantic_weight"
                        ].mean()
                    ),

                "forensic_weight_all":
                    float(
                        condition[
                            "forensic_weight"
                        ].mean()
                    ),

                "semantic_weight_real":
                    float(
                        real[
                            "semantic_weight"
                        ].mean()
                    ),

                "forensic_weight_real":
                    float(
                        real[
                            "forensic_weight"
                        ].mean()
                    ),

                "semantic_weight_fake":
                    float(
                        fake[
                            "semantic_weight"
                        ].mean()
                    ),

                "forensic_weight_fake":
                    float(
                        fake[
                            "forensic_weight"
                        ].mean()
                    ),

                "predicted_severity_real":
                    float(
                        real[
                            "predicted_severity"
                        ].mean()
                    ),

                "predicted_severity_fake":
                    float(
                        fake[
                            "predicted_severity"
                        ].mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_gate_effect(
    dataframe,
):

    rows = []

    for (
        condition_key,
        condition,
    ) in dataframe.groupby(
        "condition_key"
    ):

        labels = (
            condition[
                "label"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        base = (
            condition[
                "base_pred"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        final = (
            condition[
                "pred"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        base_prediction = (
            base >= 0.5
        ).astype(
            np.int64
        )

        final_prediction = (
            final >= 0.5
        ).astype(
            np.int64
        )

        base_correct = (
            base_prediction
            == labels
        )

        final_correct = (
            final_prediction
            == labels
        )

        helped = (
            final_correct
            & ~base_correct
        )

        hurt = (
            base_correct
            & ~final_correct
        )

        rows.append(
            {
                "condition_key":
                    condition_key,

                "condition_name":
                    condition[
                        "condition_name"
                    ].iloc[
                        0
                    ],

                "samples":
                    len(
                        condition
                    ),

                "gate_helped":
                    int(
                        helped.sum()
                    ),

                "gate_hurt":
                    int(
                        hurt.sum()
                    ),

                "net_gate_fixes":
                    int(
                        helped.sum()
                        - hurt.sum()
                    ),

                "mean_absolute_change":
                    float(
                        np.mean(
                            np.abs(
                                final
                                - base
                            )
                        )
                    ),

                "base_auroc":
                    safe_auroc(
                        labels,
                        base,
                    ),

                "final_auroc":
                    safe_auroc(
                        labels,
                        final,
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def print_branch_table(
    dataframe,
):

    for condition_name in (
        dataframe[
            "condition_name"
        ]
        .unique()
    ):

        condition = dataframe[
            dataframe[
                "condition_name"
            ]
            == condition_name
        ]

        print(
            "\n========================================"
        )

        print(
            condition_name
        )

        print(
            "========================================"
        )

        print(
            "\n"
            f"{'Branch':16s}"
            f"{'AUROC':>10s}"
            f"{'Inv.AUC':>10s}"
            f"{'Real μ':>10s}"
            f"{'Fake μ':>10s}"
            f"{'Gap':>10s}"
        )

        print(
            "-" * 66
        )

        for _, row in (
            condition.iterrows()
        ):

            print(
                f"{row['branch']:16s}"
                f"{row['auroc']:10.4f}"
                f"{row['inverse_auroc']:10.4f}"
                f"{row['mean_real_score']:10.4f}"
                f"{row['mean_fake_score']:10.4f}"
                f"{row['score_gap_fake_minus_real']:10.4f}"
            )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        default=(
            "outputs/evaluation/"
            "wildfake_zero_shot/"
            "predictions.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/evaluation/"
            "wildfake_zero_shot/"
            "diagnostics"
        ),
    )

    args = parser.parse_args()

    dataframe = pd.read_csv(
        args.predictions
    )

    required = {
        "label",
        "condition_key",
        "condition_name",
        "pred",
        "base_pred",
        "semantic_pred",
        "forensic_pred",
        "semantic_weight",
        "forensic_weight",
        "predicted_severity",
    }

    missing = (
        required
        - set(
            dataframe.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Prediction CSV is missing:\n"
            + "\n".join(
                sorted(
                    missing
                )
            )
        )

    output_directory = Path(
        args.output_dir
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    branch_metrics = (
        build_condition_comparison(
            dataframe
        )
    )

    gate_summary = (
        build_gate_summary(
            dataframe
        )
    )

    gate_effect = (
        build_gate_effect(
            dataframe
        )
    )

    branch_metrics.to_csv(
        output_directory
        / "branch_metrics.csv",

        index=False,
    )

    gate_summary.to_csv(
        output_directory
        / "gate_summary.csv",

        index=False,
    )

    gate_effect.to_csv(
        output_directory
        / "gate_effect.csv",

        index=False,
    )

    print(
        "\n========================================"
    )

    print(
        "WILDFAKE DDIM OOD DIAGNOSTIC"
    )

    print(
        "========================================"
    )

    print_branch_table(
        branch_metrics
    )

    print(
        "\n========================================"
    )

    print(
        "GATE EFFECT"
    )

    print(
        "========================================"
    )

    print(
        gate_effect.to_string(
            index=False
        )
    )

    print(
        "\n========================================"
    )

    print(
        "GATE WEIGHTS"
    )

    print(
        "========================================"
    )

    print(
        gate_summary.to_string(
            index=False
        )
    )

    print(
        "\nSaved:"
    )

    print(
        output_directory.resolve()
    )


if __name__ == "__main__":

    main()