from __future__ import annotations

import json

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.corruption_dataset import (
    CorruptedEvaluationDataset,
)

from src.evaluation.robustness import (
    add_robustness_columns,
    build_summary,
)

from src.training.metrics import (
    compute_binary_metrics,
)


@torch.no_grad()
def evaluate_forensic_condition(
    model,
    dataloader,
    device,
    threshold,
    condition,
):

    model.eval()

    labels_all = []

    final_probs = []
    semantic_probs = []
    forensic_probs = []

    paths = []

    for batch in tqdm(
        dataloader,
        desc=condition.name,
    ):

        pixel_values = batch[
            "pixel_values"
        ].to(
            device,
            non_blocking=True,
        )

        forensic_images = batch[
            "forensic_images"
        ].to(
            device,
            non_blocking=True,
        )

        labels = batch[
            "labels"
        ].to(
            device,
            non_blocking=True,
        )

        output = model.forward_with_details(
            pixel_values=
                pixel_values,

            forensic_images=
                forensic_images,
        )

        final = torch.sigmoid(
            output["logits"]
        )

        semantic = torch.sigmoid(
            output[
                "semantic_logits"
            ]
        )

        forensic = torch.sigmoid(
            output[
                "forensic_logits"
            ]
        )

        labels_all.extend(
            labels.cpu().numpy().tolist()
        )

        final_probs.extend(
            final.cpu().numpy().tolist()
        )

        semantic_probs.extend(
            semantic.cpu().numpy().tolist()
        )

        forensic_probs.extend(
            forensic.cpu().numpy().tolist()
        )

        paths.extend(
            batch["image_path"]
        )

    final_metrics = compute_binary_metrics(
        labels_all,
        final_probs,
        threshold,
    )

    semantic_metrics = (
        compute_binary_metrics(
            labels_all,
            semantic_probs,
            threshold,
        )
    )

    forensic_metrics = (
        compute_binary_metrics(
            labels_all,
            forensic_probs,
            threshold,
        )
    )

    predictions = pd.DataFrame(
        {
            "image_path":
                paths,

            "label":
                labels_all,

            "pred":
                final_probs,

            "semantic_pred":
                semantic_probs,

            "forensic_pred":
                forensic_probs,

            "condition_key":
                condition.key,

            "condition_name":
                condition.name,

            "corruption_type":
                condition.corruption_type,

            "severity":
                condition.severity,
        }
    )

    return (
        final_metrics,
        semantic_metrics,
        forensic_metrics,
        predictions,
    )


def run_forensic_robustness(
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

    for spec in specs:

        dataset = (
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

        loader = DataLoader(
            dataset,

            batch_size=
                batch_size,

            shuffle=False,

            num_workers=
                num_workers,

            pin_memory=(
                device.type == "cuda"
            ),

            persistent_workers=(
                num_workers > 0
            ),

            collate_fn=
                collator,
        )

        (
            final_metrics,
            semantic_metrics,
            forensic_metrics,
            predictions,
        ) = evaluate_forensic_condition(
            model=
                model,

            dataloader=
                loader,

            device=
                device,

            threshold=
                threshold,

            condition=
                spec,
        )

        records.append(
            {
                "condition_key":
                    spec.key,

                "condition_name":
                    spec.name,

                "corruption_type":
                    spec.corruption_type,

                "severity":
                    spec.severity,

                **final_metrics,

                "semantic_auroc":
                    semantic_metrics[
                        "auroc"
                    ],

                "forensic_auroc":
                    forensic_metrics[
                        "auroc"
                    ],
            }
        )

        prediction_frames.append(
            predictions
        )

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
            f"Final AUROC:    "
            f"{final_metrics['auroc']:.4f}"
        )

        print(
            f"Semantic AUROC: "
            f"{semantic_metrics['auroc']:.4f}"
        )

        print(
            f"Forensic AUROC: "
            f"{forensic_metrics['auroc']:.4f}"
        )

        print(
            f"Accuracy:       "
            f"{final_metrics['accuracy']:.4f}"
        )

    results = pd.DataFrame(
        records
    )

    results = add_robustness_columns(
        results
    )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    summary = build_summary(
        results
    )

    results.to_csv(
        output_directory
        / "forensic_fusion_robustness.csv",
        index=False,
    )

    predictions.to_csv(
        output_directory
        / "forensic_fusion_predictions.csv",
        index=False,
    )

    with (
        output_directory
        / "forensic_fusion_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    print(
        "\n========================================"
    )

    print(
        "FORENSIC FUSION ROBUSTNESS SUMMARY"
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
        f"Mean retention: "
        f"{summary['mean_auroc_retention_pct']:.2f}%"
    )

    print(
        f"Worst condition: "
        f"{summary['worst_auroc_condition']}"
    )

    print(
        f"Worst AUROC: "
        f"{summary['worst_auroc']:.4f}"
    )

    return results