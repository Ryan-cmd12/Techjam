from __future__ import annotations

import json

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import (
    DataLoader,
)

from tqdm import tqdm

from src.evaluation.laundering_dataset import (
    LaunderingEvaluationDataset,
    LaunderingSpec,
)

from src.training.corruption_targets import (
    ID_TO_CORRUPTION,
)

from src.training.metrics import (
    compute_binary_metrics,
)


@torch.no_grad()
def evaluate_laundering_condition(
    model,
    dataloader,
    device,
    threshold: float,
    spec: LaunderingSpec,
):

    model.eval()

    labels_all = []
    probabilities = []

    semantic_probabilities = []
    forensic_probabilities = []

    semantic_weights = []
    forensic_weights = []

    predicted_severities = []
    predicted_corruption_ids = []

    image_paths = []

    for batch in tqdm(
        dataloader,
        desc=spec.name,
    ):

        pixel_values = (
            batch[
                "pixel_values"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        forensic_tiles = (
            batch[
                "forensic_tiles"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        tile_mask = (
            batch[
                "tile_mask"
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

        output = (
            model.forward_with_details(
                pixel_values=
                    pixel_values,
                forensic_tiles=
                    forensic_tiles,
                tile_mask=
                    tile_mask,
            )
        )

        final_probability = (
            torch.sigmoid(
                output[
                    "logits"
                ]
            )
        )

        semantic_probability = (
            torch.sigmoid(
                output[
                    "semantic_logits"
                ]
            )
        )

        forensic_probability = (
            torch.sigmoid(
                output[
                    "forensic_logits"
                ]
            )
        )

        corruption_ids = (
            output[
                "corruption_type_probabilities"
            ]
            .argmax(
                dim=-1
            )
        )

        labels_all.extend(
            labels
            .cpu()
            .numpy()
            .tolist()
        )

        probabilities.extend(
            final_probability
            .cpu()
            .numpy()
            .tolist()
        )

        semantic_probabilities.extend(
            semantic_probability
            .cpu()
            .numpy()
            .tolist()
        )

        forensic_probabilities.extend(
            forensic_probability
            .cpu()
            .numpy()
            .tolist()
        )

        semantic_weights.extend(
            output[
                "semantic_weight"
            ]
            .cpu()
            .numpy()
            .tolist()
        )

        forensic_weights.extend(
            output[
                "forensic_weight"
            ]
            .cpu()
            .numpy()
            .tolist()
        )

        predicted_severities.extend(
            output[
                "corruption_severity"
            ]
            .cpu()
            .numpy()
            .tolist()
        )

        predicted_corruption_ids.extend(
            corruption_ids
            .cpu()
            .numpy()
            .tolist()
        )

        image_paths.extend(
            batch[
                "image_path"
            ]
        )

    metrics = (
        compute_binary_metrics(
            labels=
                labels_all,
            probabilities=
                probabilities,
            threshold=
                threshold,
        )
    )

    predictions = (
        np.asarray(
            probabilities
        )
        >= threshold
    ).astype(
        np.int64
    )

    predicted_corruption_names = [
        ID_TO_CORRUPTION.get(
            int(index),
            "unknown",
        )
        for index
        in predicted_corruption_ids
    ]

    unique_types, type_counts = (
        np.unique(
            predicted_corruption_names,
            return_counts=True,
        )
    )

    dominant_corruption = (
        str(
            unique_types[
                np.argmax(
                    type_counts
                )
            ]
        )
    )

    dataframe = pd.DataFrame(
        {
            "image_path":
                image_paths,

            "label":
                np.asarray(
                    labels_all,
                    dtype=np.int64,
                ),

            "pred":
                np.asarray(
                    probabilities,
                    dtype=np.float64,
                ),

            "prediction":
                predictions,

            "semantic_pred":
                semantic_probabilities,

            "forensic_pred":
                forensic_probabilities,

            "semantic_weight":
                semantic_weights,

            "forensic_weight":
                forensic_weights,

            "predicted_corruption":
                predicted_corruption_names,

            "predicted_severity":
                predicted_severities,

            "pipeline_key":
                spec.key,

            "pipeline_name":
                spec.name,
        }
    )

    record = {
        "pipeline_key":
            spec.key,

        "pipeline_name":
            spec.name,

        "num_steps":
            len(
                spec.steps
            ),

        "pipeline":
            " → ".join(
                (
                    f"{step.corruption_type}"
                    f"({step.severity})"
                )
                for step
                in spec.steps
            )
            if spec.steps
            else "clean",

        **metrics,

        "semantic_weight":
            float(
                np.mean(
                    semantic_weights
                )
            ),

        "forensic_weight":
            float(
                np.mean(
                    forensic_weights
                )
            ),

        "predicted_severity":
            float(
                np.mean(
                    predicted_severities
                )
            ),

        "dominant_predicted_corruption":
            dominant_corruption,
    }

    return (
        record,
        dataframe,
    )


def add_laundering_retention(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    clean_rows = dataframe[
        dataframe[
            "pipeline_key"
        ]
        == "clean"
    ]

    if len(clean_rows) != 1:

        raise ValueError(
            "Laundering suite must "
            "contain exactly one clean "
            "reference condition."
        )

    clean = clean_rows.iloc[
        0
    ]

    clean_auroc = float(
        clean[
            "auroc"
        ]
    )

    clean_accuracy = float(
        clean[
            "accuracy"
        ]
    )

    clean_f1 = float(
        clean[
            "f1"
        ]
    )

    result = dataframe.copy()

    result[
        "auroc_drop"
    ] = (
        clean_auroc
        - result[
            "auroc"
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
        "auroc_retention_pct"
    ] = (
        result[
            "auroc_retention"
        ]
        * 100.0
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
        "accuracy_retention"
    ] = (
        result[
            "accuracy"
        ]
        / clean_accuracy
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
        "f1_retention"
    ] = (
        result[
            "f1"
        ]
        / clean_f1
        if clean_f1 != 0
        else np.nan
    )

    return result


def build_laundering_summary(
    dataframe: pd.DataFrame,
) -> dict:

    clean = dataframe[
        dataframe[
            "pipeline_key"
        ]
        == "clean"
    ].iloc[0]

    corrupted = dataframe[
        dataframe[
            "pipeline_key"
        ]
        != "clean"
    ]

    worst_auroc = corrupted.loc[
        corrupted[
            "auroc"
        ].idxmin()
    ]

    worst_accuracy = corrupted.loc[
        corrupted[
            "accuracy"
        ].idxmin()
    ]

    return {
        "clean_auroc":
            float(
                clean[
                    "auroc"
                ]
            ),

        "mean_laundered_auroc":
            float(
                corrupted[
                    "auroc"
                ].mean()
            ),

        "mean_laundering_retention":
            float(
                corrupted[
                    "auroc_retention"
                ].mean()
            ),

        "mean_laundering_retention_pct":
            float(
                corrupted[
                    "auroc_retention_pct"
                ].mean()
            ),

        "worst_pipeline":
            str(
                worst_auroc[
                    "pipeline_name"
                ]
            ),

        "worst_auroc":
            float(
                worst_auroc[
                    "auroc"
                ]
            ),

        "worst_accuracy_pipeline":
            str(
                worst_accuracy[
                    "pipeline_name"
                ]
            ),

        "worst_accuracy":
            float(
                worst_accuracy[
                    "accuracy"
                ]
            ),

        "clean_semantic_weight":
            float(
                clean[
                    "semantic_weight"
                ]
            ),

        "clean_forensic_weight":
            float(
                clean[
                    "forensic_weight"
                ]
            ),

        "mean_laundered_semantic_weight":
            float(
                corrupted[
                    "semantic_weight"
                ].mean()
            ),

        "mean_laundered_forensic_weight":
            float(
                corrupted[
                    "forensic_weight"
                ].mean()
            ),
    }


def print_laundering_summary(
    summary: dict,
):

    print(
        "\n========================================"
    )

    print(
        "LAUNDERING ROBUSTNESS SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        f"\nClean AUROC: "
        f"{summary['clean_auroc']:.4f}"
    )

    print(
        f"Mean laundered AUROC: "
        f"{summary['mean_laundered_auroc']:.4f}"
    )

    print(
        f"Mean AUROC retention: "
        f"{summary['mean_laundering_retention_pct']:.2f}%"
    )

    print(
        "\nWorst pipeline:"
    )

    print(
        f"{summary['worst_pipeline']} "
        f"→ "
        f"{summary['worst_auroc']:.4f}"
    )

    print(
        "\nGate behavior:"
    )

    print(
        f"Clean semantic:   "
        f"{summary['clean_semantic_weight']:.4f}"
    )

    print(
        f"Clean forensic:   "
        f"{summary['clean_forensic_weight']:.4f}"
    )

    print(
        f"Launder semantic: "
        f"{summary['mean_laundered_semantic_weight']:.4f}"
    )

    print(
        f"Launder forensic: "
        f"{summary['mean_laundered_forensic_weight']:.4f}"
    )


def run_laundering_benchmark(
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
):

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []
    prediction_frames = []

    for index, spec in enumerate(
        specs,
        start=1,
    ):

        print(
            "\n========================================"
        )

        print(
            f"PIPELINE {index}/{len(specs)}"
        )

        print(
            "========================================"
        )

        print(
            f"\n{spec.name}"
        )

        if spec.steps:

            for step_index, step in enumerate(
                spec.steps,
                start=1,
            ):

                print(
                    f"  {step_index}. "
                    f"{step.corruption_type} "
                    f"{step.severity}"
                )

        else:

            print(
                "  no transformations"
            )

        dataset = (
            LaunderingEvaluationDataset(
                base_dataset=
                    base_dataset,
                spec=
                    spec,
                seed=
                    seed,
            )
        )

        dataloader = (
            DataLoader(
                dataset,

                batch_size=
                    batch_size,

                shuffle=
                    False,

                num_workers=
                    num_workers,

                pin_memory=(
                    device.type
                    == "cuda"
                ),

                persistent_workers=(
                    num_workers > 0
                ),

                collate_fn=
                    collator,
            )
        )

        (
            record,
            predictions,
        ) = (
            evaluate_laundering_condition(
                model=
                    model,
                dataloader=
                    dataloader,
                device=
                    device,
                threshold=
                    threshold,
                spec=
                    spec,
            )
        )

        records.append(
            record
        )

        prediction_frames.append(
            predictions
        )

        print(
            "\nResults:"
        )

        print(
            f"AUROC:          "
            f"{record['auroc']:.4f}"
        )

        print(
            f"Accuracy:       "
            f"{record['accuracy']:.4f}"
        )

        print(
            f"F1:             "
            f"{record['f1']:.4f}"
        )

        print(
            f"Semantic weight:"
            f" {record['semantic_weight']:.4f}"
        )

        print(
            f"Forensic weight:"
            f" {record['forensic_weight']:.4f}"
        )

        print(
            f"Estimated sev.: "
            f"{record['predicted_severity']:.4f}"
        )

    results = pd.DataFrame(
        records
    )

    results = (
        add_laundering_retention(
            results
        )
    )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    summary = (
        build_laundering_summary(
            results
        )
    )

    results.to_csv(
        output_directory
        / "laundering_metrics.csv",
        index=False,
    )

    predictions.to_csv(
        output_directory
        / "laundering_predictions.csv",
        index=False,
    )

    with (
        output_directory
        / "laundering_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    print_laundering_summary(
        summary
    )

    return (
        results,
        predictions,
        summary,
    )