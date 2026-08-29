from __future__ import annotations

import json

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm

from src.training.forensic_checkpoint import (
    save_forensic_checkpoint,
)

from src.training.metrics import (
    compute_binary_metrics,
    print_metrics,
)


def cosine_consistency(
    clean_features,
    corrupted_features,
):
    return (
        1.0
        - F.cosine_similarity(
            clean_features,
            corrupted_features,
            dim=-1,
        )
    ).mean()


def probability_consistency(
    clean_logits,
    corrupted_logits,
):
    return F.mse_loss(
        torch.sigmoid(
            clean_logits
        ),
        torch.sigmoid(
            corrupted_logits
        ),
    )


def move_batch(
    batch,
    device,
):
    return (
        batch[
            "clean_pixel_values"
        ].to(
            device,
            non_blocking=True,
        ),

        batch[
            "corrupted_pixel_values"
        ].to(
            device,
            non_blocking=True,
        ),

        batch[
            "clean_forensic_images"
        ].to(
            device,
            non_blocking=True,
        ),

        batch[
            "corrupted_forensic_images"
        ].to(
            device,
            non_blocking=True,
        ),

        batch[
            "labels"
        ].to(
            device,
            non_blocking=True,
        ),
    )


def train_forensic_epoch(
    model,
    dataloader,
    optimizer,
    device,
    auxiliary_branch_weight,
    feature_consistency_weight,
    prediction_consistency_weight,
    gradient_clip_norm,
    use_amp,
):

    model.train()

    criterion = nn.BCEWithLogitsLoss()

    amp_enabled = (
        use_amp
        and device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    totals = {
        "total_loss": 0.0,
        "classification_loss": 0.0,
        "auxiliary_loss": 0.0,
        "feature_consistency_loss": 0.0,
        "prediction_consistency_loss": 0.0,
    }

    sample_count = 0

    progress = tqdm(
        dataloader,
        desc="Forensic Fusion Training",
    )

    for batch in progress:

        (
            clean_pixels,
            corrupt_pixels,
            clean_forensic,
            corrupt_forensic,
            labels,
        ) = move_batch(
            batch,
            device,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type=device.type,
            dtype=(
                torch.float16
                if device.type == "cuda"
                else torch.bfloat16
            ),
            enabled=amp_enabled,
        ):

            output = model.forward_pair(
                clean_pixel_values=
                    clean_pixels,

                corrupted_pixel_values=
                    corrupt_pixels,

                clean_forensic_images=
                    clean_forensic,

                corrupted_forensic_images=
                    corrupt_forensic,
            )

            clean_final_loss = criterion(
                output["clean_logits"],
                labels,
            )

            corrupt_final_loss = criterion(
                output["corrupted_logits"],
                labels,
            )

            classification_loss = (
                clean_final_loss
                + corrupt_final_loss
            ) / 2.0

            auxiliary_losses = [
                criterion(
                    output[
                        "clean_semantic_logits"
                    ],
                    labels,
                ),

                criterion(
                    output[
                        "corrupted_semantic_logits"
                    ],
                    labels,
                ),

                criterion(
                    output[
                        "clean_forensic_logits"
                    ],
                    labels,
                ),

                criterion(
                    output[
                        "corrupted_forensic_logits"
                    ],
                    labels,
                ),
            ]

            auxiliary_loss = (
                sum(auxiliary_losses)
                / len(auxiliary_losses)
            )

            feature_loss = (
                cosine_consistency(
                    output[
                        "clean_fused_features"
                    ],
                    output[
                        "corrupted_fused_features"
                    ],
                )
            )

            prediction_loss = (
                probability_consistency(
                    output[
                        "clean_logits"
                    ],
                    output[
                        "corrupted_logits"
                    ],
                )
            )

            total_loss = (
                classification_loss

                + auxiliary_branch_weight
                * auxiliary_loss

                + feature_consistency_weight
                * feature_loss

                + prediction_consistency_weight
                * prediction_loss
            )

        scaler.scale(
            total_loss
        ).backward()

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            (
                p
                for p in model.parameters()
                if p.requires_grad
            ),
            max_norm=
                gradient_clip_norm,
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        batch_size = labels.shape[0]

        sample_count += batch_size

        values = {
            "total_loss":
                total_loss.item(),

            "classification_loss":
                classification_loss.item(),

            "auxiliary_loss":
                auxiliary_loss.item(),

            "feature_consistency_loss":
                feature_loss.item(),

            "prediction_consistency_loss":
                prediction_loss.item(),
        }

        for key, value in values.items():
            totals[key] += (
                value
                * batch_size
            )

        progress.set_postfix(
            loss=(
                f"{totals['total_loss'] / sample_count:.4f}"
            )
        )

    return {
        key:
            value
            / sample_count

        for key, value in totals.items()
    }


@torch.no_grad()
def validate_forensic_model(
    model,
    dataloader,
    device,
    threshold,
):

    model.eval()

    labels_all = []

    clean_final = []
    corrupt_final = []

    clean_semantic = []
    corrupt_semantic = []

    clean_forensic = []
    corrupt_forensic = []

    shifts = []

    for batch in tqdm(
        dataloader,
        desc="Forensic Validation",
    ):

        (
            clean_pixels,
            corrupt_pixels,
            clean_forensic_images,
            corrupt_forensic_images,
            labels,
        ) = move_batch(
            batch,
            device,
        )

        output = model.forward_pair(
            clean_pixel_values=
                clean_pixels,

            corrupted_pixel_values=
                corrupt_pixels,

            clean_forensic_images=
                clean_forensic_images,

            corrupted_forensic_images=
                corrupt_forensic_images,
        )

        clean_probs = torch.sigmoid(
            output[
                "clean_logits"
            ]
        )

        corrupt_probs = torch.sigmoid(
            output[
                "corrupted_logits"
            ]
        )

        labels_all.extend(
            labels.cpu().numpy().tolist()
        )

        clean_final.extend(
            clean_probs.cpu().numpy().tolist()
        )

        corrupt_final.extend(
            corrupt_probs.cpu().numpy().tolist()
        )

        clean_semantic.extend(
            torch.sigmoid(
                output[
                    "clean_semantic_logits"
                ]
            )
            .cpu()
            .numpy()
            .tolist()
        )

        corrupt_semantic.extend(
            torch.sigmoid(
                output[
                    "corrupted_semantic_logits"
                ]
            )
            .cpu()
            .numpy()
            .tolist()
        )

        clean_forensic.extend(
            torch.sigmoid(
                output[
                    "clean_forensic_logits"
                ]
            )
            .cpu()
            .numpy()
            .tolist()
        )

        corrupt_forensic.extend(
            torch.sigmoid(
                output[
                    "corrupted_forensic_logits"
                ]
            )
            .cpu()
            .numpy()
            .tolist()
        )

        shifts.extend(
            torch.abs(
                clean_probs
                - corrupt_probs
            )
            .cpu()
            .numpy()
            .tolist()
        )

    clean_metrics = compute_binary_metrics(
        labels_all,
        clean_final,
        threshold,
    )

    corrupt_metrics = compute_binary_metrics(
        labels_all,
        corrupt_final,
        threshold,
    )

    semantic_metrics = (
        compute_binary_metrics(
            labels_all,
            corrupt_semantic,
            threshold,
        )
    )

    forensic_metrics = (
        compute_binary_metrics(
            labels_all,
            corrupt_forensic,
            threshold,
        )
    )

    robust_score = (
        clean_metrics["auroc"]
        + corrupt_metrics["auroc"]
    ) / 2.0

    return {
        "clean_metrics":
            clean_metrics,

        "corrupted_metrics":
            corrupt_metrics,

        "semantic_corrupted_metrics":
            semantic_metrics,

        "forensic_corrupted_metrics":
            forensic_metrics,

        "mean_prediction_shift":
            float(
                np.mean(shifts)
            ),

        "robust_score":
            float(
                robust_score
            ),
    }


def train_forensic_model(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    device,
    epochs,
    checkpoint_directory,
    config,
    threshold,
    auxiliary_branch_weight,
    feature_consistency_weight,
    prediction_consistency_weight,
    gradient_clip_norm,
    use_amp,
):

    checkpoint_directory = Path(
        checkpoint_directory
    )

    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_score = float(
        "-inf"
    )

    history = []

    for epoch in range(
        1,
        epochs + 1,
    ):

        print(
            "\n========================================"
        )

        print(
            f"FORENSIC FUSION EPOCH {epoch}/{epochs}"
        )

        print(
            "========================================"
        )

        losses = train_forensic_epoch(
            model=
                model,

            dataloader=
                train_dataloader,

            optimizer=
                optimizer,

            device=
                device,

            auxiliary_branch_weight=
                auxiliary_branch_weight,

            feature_consistency_weight=
                feature_consistency_weight,

            prediction_consistency_weight=
                prediction_consistency_weight,

            gradient_clip_norm=
                gradient_clip_norm,

            use_amp=
                use_amp,
        )

        validation = validate_forensic_model(
            model,
            val_dataloader,
            device,
            threshold,
        )

        print(
            "\nTraining losses:"
        )

        for name, value in losses.items():
            print(
                f"{name:34s} "
                f"{value:.6f}"
            )

        print_metrics(
            validation[
                "clean_metrics"
            ],
            "Fusion Clean Validation",
        )

        print_metrics(
            validation[
                "corrupted_metrics"
            ],
            "Fusion Corrupted Validation",
        )

        print(
            "\nBranch diagnostics"
        )

        print(
            f"Semantic corrupted AUROC: "
            f"{validation['semantic_corrupted_metrics']['auroc']:.4f}"
        )

        print(
            f"Forensic corrupted AUROC: "
            f"{validation['forensic_corrupted_metrics']['auroc']:.4f}"
        )

        print(
            f"Final corrupted AUROC:    "
            f"{validation['corrupted_metrics']['auroc']:.4f}"
        )

        print(
            f"Prediction shift:         "
            f"{validation['mean_prediction_shift']:.4f}"
        )

        print(
            f"Robust score:             "
            f"{validation['robust_score']:.4f}"
        )

        record = {
            "epoch":
                epoch,

            **losses,

            "clean_auroc":
                validation[
                    "clean_metrics"
                ][
                    "auroc"
                ],

            "corrupted_auroc":
                validation[
                    "corrupted_metrics"
                ][
                    "auroc"
                ],

            "semantic_corrupted_auroc":
                validation[
                    "semantic_corrupted_metrics"
                ][
                    "auroc"
                ],

            "forensic_corrupted_auroc":
                validation[
                    "forensic_corrupted_metrics"
                ][
                    "auroc"
                ],

            "mean_prediction_shift":
                validation[
                    "mean_prediction_shift"
                ],

            "robust_score":
                validation[
                    "robust_score"
                ],
        }

        history.append(
            record
        )

        save_forensic_checkpoint(
            checkpoint_directory
            / "forensic_fusion_last.pt",

            model,
            optimizer,
            epoch,
            validation,
            config,
        )

        if (
            validation["robust_score"]
            > best_score
        ):
            best_score = (
                validation[
                    "robust_score"
                ]
            )

            save_forensic_checkpoint(
                checkpoint_directory
                / "forensic_fusion_best.pt",

                model,
                optimizer,
                epoch,
                validation,
                config,
            )

            print(
                "\nNEW BEST FORENSIC FUSION MODEL"
            )

    with (
        checkpoint_directory
        / "forensic_fusion_history.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            history,
            file,
            indent=4,
        )

    print(
        "\n========================================"
    )

    print(
        "FORENSIC FUSION TRAINING COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        f"\nBest robust score: "
        f"{best_score:.6f}"
    )