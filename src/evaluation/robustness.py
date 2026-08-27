from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import (
    DataLoader,
    Dataset,
)

from tqdm import tqdm

from src.evaluation.corruption_dataset import (
    CorruptedEvaluationDataset,
)

from src.training.metrics import (
    compute_binary_metrics,
)


@dataclass(
    frozen=True
)
class CorruptionSpec:

    key: str

    name: str

    corruption_type: str

    severity: (
        float
        | int
        | None
    )


def build_corruption_specs(
    config_entries: list[dict],
) -> list[
    CorruptionSpec
]:

    specs = []

    for entry in config_entries:

        required = {
            "key",
            "name",
            "type",
        }

        missing = (
            required
            - set(
                entry.keys()
            )
        )

        if missing:

            raise ValueError(
                "Robustness condition is "
                "missing required fields: "
                f"{sorted(missing)}"
            )

        specs.append(
            CorruptionSpec(
                key=str(
                    entry[
                        "key"
                    ]
                ),

                name=str(
                    entry[
                        "name"
                    ]
                ),

                corruption_type=str(
                    entry[
                        "type"
                    ]
                ),

                severity=(
                    entry.get(
                        "severity"
                    )
                ),
            )
        )

    return specs


@torch.no_grad()
def evaluate_corruption_condition(
    model,
    dataloader: DataLoader,
    device: torch.device,
    spec: CorruptionSpec,
    threshold: float = 0.5,
    use_amp: bool = True,
) -> tuple[
    float,
    dict,
    pd.DataFrame,
]:

    model.eval()

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    amp_enabled = (
        use_amp
        and device.type
        == "cuda"
    )

    running_loss = (
        0.0
    )

    sample_count = (
        0
    )

    all_labels = []

    all_probabilities = []

    all_paths = []

    progress_bar = tqdm(
        dataloader,
        desc=spec.name,
    )

    for batch in progress_bar:

        pixel_values = (
            batch[
                "pixel_values"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        labels = (
            batch[
                "labels"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        with torch.autocast(
            device_type=
                device.type,

            dtype=(
                torch.float16
                if device.type
                == "cuda"

                else torch.bfloat16
            ),

            enabled=
                amp_enabled,
        ):

            logits = model(
                pixel_values
            )

            loss = criterion(
                logits,
                labels,
            )

        probabilities = (
            torch.sigmoid(
                logits
            )
        )

        batch_size = (
            labels.shape[
                0
            ]
        )

        running_loss += (
            loss.item()
            * batch_size
        )

        sample_count += (
            batch_size
        )

        all_labels.extend(
            labels
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

        all_probabilities.extend(
            probabilities
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

        if (
            "image_path"
            in batch
        ):

            all_paths.extend(
                batch[
                    "image_path"
                ]
            )

    average_loss = (
        running_loss
        / max(
            sample_count,
            1,
        )
    )

    metrics = (
        compute_binary_metrics(
            labels=
                all_labels,

            probabilities=
                all_probabilities,

            threshold=
                threshold,
        )
    )

    predictions = (
        np.asarray(
            all_probabilities,
            dtype=np.float64,
        )
        >= threshold
    ).astype(
        np.int64
    )

    prediction_dataframe = (
        pd.DataFrame(
            {
                "image_path":
                    all_paths,

                "label":
                    np.asarray(
                        all_labels,
                        dtype=np.int64,
                    ),

                "pred":
                    np.asarray(
                        all_probabilities,
                        dtype=np.float64,
                    ),

                "prediction":
                    predictions,

                "condition_key":
                    spec.key,

                "condition_name":
                    spec.name,

                "corruption_type":
                    spec.corruption_type,

                "severity":
                    (
                        spec.severity
                    ),
            }
        )
    )

    return (
        average_loss,
        metrics,
        prediction_dataframe,
    )


def calculate_retention(
    current: float,
    clean: float,
) -> float:

    if (
        clean is None
        or clean == 0
        or np.isnan(
            clean
        )
    ):

        return float(
            "nan"
        )

    return (
        current
        / clean
    )


def add_robustness_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    clean_rows = (
        dataframe[
            dataframe[
                "corruption_type"
            ]
            == "clean"
        ]
    )

    if len(
        clean_rows
    ) != 1:

        raise ValueError(
            "Robustness benchmark "
            "must contain exactly "
            "one clean condition."
        )

    clean_row = (
        clean_rows.iloc[
            0
        ]
    )

    clean_auroc = float(
        clean_row[
            "auroc"
        ]
    )

    clean_accuracy = float(
        clean_row[
            "accuracy"
        ]
    )

    clean_f1 = float(
        clean_row[
            "f1"
        ]
    )

    result = (
        dataframe.copy()
    )

    result[
        "auroc_drop"
    ] = (
        clean_auroc
        - result[
            "auroc"
        ]
    )

    result[
        "accuracy_drop"
    ] = (
        clean_accuracy
        - result[
            "accuracy"
        ]
    )

    result[
        "f1_drop"
    ] = (
        clean_f1
        - result[
            "f1"
        ]
    )

    result[
        "auroc_retention"
    ] = (
        result[
            "auroc"
        ]
        / clean_auroc
    )

    result[
        "accuracy_retention"
    ] = (
        result[
            "accuracy"
        ]
        / clean_accuracy
    )

    if clean_f1 != 0:

        result[
            "f1_retention"
        ] = (
            result[
                "f1"
            ]
            / clean_f1
        )

    else:

        result[
            "f1_retention"
        ] = (
            float(
                "nan"
            )
        )

    result[
        "auroc_retention_pct"
    ] = (
        result[
            "auroc_retention"
        ]
        * 100.0
    )

    result[
        "accuracy_retention_pct"
    ] = (
        result[
            "accuracy_retention"
        ]
        * 100.0
    )

    result[
        "f1_retention_pct"
    ] = (
        result[
            "f1_retention"
        ]
        * 100.0
    )

    return result


def build_summary(
    result_dataframe: pd.DataFrame,
) -> dict:

    clean_row = (
        result_dataframe[
            result_dataframe[
                "corruption_type"
            ]
            == "clean"
        ]
        .iloc[
            0
        ]
    )

    corrupted = (
        result_dataframe[
            result_dataframe[
                "corruption_type"
            ]
            != "clean"
        ]
    )

    if len(
        corrupted
    ) == 0:

        raise ValueError(
            "No corrupted conditions "
            "were evaluated."
        )

    worst_auroc_row = (
        corrupted.loc[
            corrupted[
                "auroc"
            ].idxmin()
        ]
    )

    worst_accuracy_row = (
        corrupted.loc[
            corrupted[
                "accuracy"
            ].idxmin()
        ]
    )

    summary = {
        "clean_auroc":
            float(
                clean_row[
                    "auroc"
                ]
            ),

        "clean_accuracy":
            float(
                clean_row[
                    "accuracy"
                ]
            ),

        "clean_f1":
            float(
                clean_row[
                    "f1"
                ]
            ),

        "mean_corrupted_auroc":
            float(
                corrupted[
                    "auroc"
                ].mean()
            ),

        "mean_corrupted_accuracy":
            float(
                corrupted[
                    "accuracy"
                ].mean()
            ),

        "mean_corrupted_f1":
            float(
                corrupted[
                    "f1"
                ].mean()
            ),

        "mean_auroc_retention":
            float(
                corrupted[
                    "auroc_retention"
                ].mean()
            ),

        "mean_auroc_retention_pct":
            float(
                corrupted[
                    "auroc_retention_pct"
                ].mean()
            ),

        "worst_auroc":
            float(
                worst_auroc_row[
                    "auroc"
                ]
            ),

        "worst_auroc_condition":
            str(
                worst_auroc_row[
                    "condition_name"
                ]
            ),

        "worst_accuracy":
            float(
                worst_accuracy_row[
                    "accuracy"
                ]
            ),

        "worst_accuracy_condition":
            str(
                worst_accuracy_row[
                    "condition_name"
                ]
            ),

        "robustness_gap_auroc":
            float(
                clean_row[
                    "auroc"
                ]
                - corrupted[
                    "auroc"
                ].mean()
            ),
    }

    return summary


def save_robustness_plot(
    result_dataframe: pd.DataFrame,
    output_path: str | Path,
) -> None:

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels = (
        result_dataframe[
            "condition_name"
        ]
        .tolist()
    )

    values = (
        result_dataframe[
            "auroc"
        ]
        .tolist()
    )

    clean_auroc = float(
        result_dataframe[
            result_dataframe[
                "corruption_type"
            ]
            == "clean"
        ][
            "auroc"
        ]
        .iloc[
            0
        ]
    )

    figure = plt.figure(
        figsize=(
            16,
            7,
        )
    )

    axis = (
        figure.add_subplot(
            1,
            1,
            1,
        )
    )

    positions = (
        np.arange(
            len(
                labels
            )
        )
    )

    axis.bar(
        positions,
        values,
    )

    axis.axhline(
        y=clean_auroc,
        linestyle="--",
        label=(
            f"Clean AUROC "
            f"{clean_auroc:.3f}"
        ),
    )

    axis.set_title(
        "Baseline Robustness — AUROC"
    )

    axis.set_ylabel(
        "AUROC"
    )

    axis.set_xlabel(
        "Evaluation Condition"
    )

    axis.set_ylim(
        0.0,
        1.05,
    )

    axis.set_xticks(
        positions
    )

    axis.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def print_condition_result(
    spec: CorruptionSpec,
    loss: float,
    metrics: dict,
) -> None:

    print(
        "\n----------------------------------------"
    )

    print(
        spec.name
    )

    print(
        "----------------------------------------"
    )

    print(
        f"Loss:               "
        f"{loss:.6f}"
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
        f"FPR:                "
        f"{metrics['false_positive_rate']:.4f}"
    )

    print(
        f"FNR:                "
        f"{metrics['false_negative_rate']:.4f}"
    )


def print_robustness_summary(
    summary: dict,
) -> None:

    print(
        "\n"
        "========================================"
    )

    print(
        "ROBUSTNESS SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        f"\nClean AUROC: "
        f"{summary['clean_auroc']:.4f}"
    )

    print(
        f"Mean corrupted AUROC: "
        f"{summary['mean_corrupted_auroc']:.4f}"
    )

    print(
        f"AUROC robustness gap: "
        f"{summary['robustness_gap_auroc']:.4f}"
    )

    print(
        f"\nMean AUROC retention: "
        f"{summary['mean_auroc_retention_pct']:.2f}%"
    )

    print(
        "\nWorst AUROC condition:"
    )

    print(
        f"{summary['worst_auroc_condition']} "
        f"→ "
        f"{summary['worst_auroc']:.4f}"
    )

    print(
        "\nWorst accuracy condition:"
    )

    print(
        f"{summary['worst_accuracy_condition']} "
        f"→ "
        f"{summary['worst_accuracy']:.4f}"
    )


def run_robustness_benchmark(
    model,
    base_dataset,
    collator,
    specs,
    device,
    batch_size,
    num_workers,
    threshold,
    seed,
    output_directory,
    use_amp=True,
    output_prefix="baseline",
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict,
]:

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    condition_records = []

    all_prediction_frames = []

    pin_memory = (
        device.type
        == "cuda"
    )

    for condition_index, spec in enumerate(
        specs,
        start=1,
    ):

        print(
            "\n"
            "========================================"
        )

        print(
            f"CONDITION "
            f"{condition_index}/"
            f"{len(specs)}"
        )

        print(
            "========================================"
        )

        print(
            f"\n{spec.name}"
        )

        corrupted_dataset = (
            CorruptedEvaluationDataset(
                base_dataset=
                    base_dataset,

                corruption_type=
                    spec.corruption_type,

                severity=
                    spec.severity,

                seed=
                    seed,
            )
        )

        dataloader = (
            DataLoader(
                corrupted_dataset,

                batch_size=
                    batch_size,

                shuffle=
                    False,

                num_workers=
                    num_workers,

                pin_memory=
                    pin_memory,

                persistent_workers=(
                    num_workers > 0
                ),

                collate_fn=
                    collator,
            )
        )

        (
            loss,
            metrics,
            prediction_dataframe,
        ) = (
            evaluate_corruption_condition(
                model=
                    model,

                dataloader=
                    dataloader,

                device=
                    device,

                spec=
                    spec,

                threshold=
                    threshold,

                use_amp=
                    use_amp,
            )
        )

        print_condition_result(
            spec=spec,
            loss=loss,
            metrics=metrics,
        )

        record = {
            "condition_key":
                spec.key,

            "condition_name":
                spec.name,

            "corruption_type":
                spec.corruption_type,

            "severity":
                spec.severity,

            "loss":
                float(
                    loss
                ),

            **metrics,
        }

        condition_records.append(
            record
        )

        all_prediction_frames.append(
            prediction_dataframe
        )

    result_dataframe = (
        pd.DataFrame(
            condition_records
        )
    )

    result_dataframe = (
        add_robustness_columns(
            result_dataframe
        )
    )

    prediction_dataframe = (
        pd.concat(
            all_prediction_frames,
            ignore_index=True,
        )
    )

    summary = (
        build_summary(
            result_dataframe
        )
    )

    metrics_path = (
        output_directory
        / f"{output_prefix}_robustness.csv"
    )

    predictions_path = (
        output_directory
        / (
            f"{output_prefix}_"
            f"robustness_predictions.csv"
        )
    )

    summary_path = (
        output_directory
        / (
            f"{output_prefix}_"
            f"robustness_summary.json"
        )
    )

    plot_path = (
        output_directory
        / (
            f"{output_prefix}_"
            f"robustness_auroc.png"
        )
    )

    result_dataframe.to_csv(
        metrics_path,
        index=False,
    )

    prediction_dataframe.to_csv(
        predictions_path,
        index=False,
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    save_robustness_plot(
        result_dataframe=
            result_dataframe,

        output_path=
            plot_path,
    )

    print_robustness_summary(
        summary
    )

    print(
        "\nSaved robustness table:"
    )

    print(
        metrics_path
    )

    print(
        "\nSaved per-image predictions:"
    )

    print(
        predictions_path
    )

    print(
        "\nSaved summary:"
    )

    print(
        summary_path
    )

    print(
        "\nSaved figure:"
    )

    print(
        plot_path
    )

    return (
        result_dataframe,
        prediction_dataframe,
        summary,
    )