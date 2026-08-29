from __future__ import annotations

import json

from pathlib import Path

import numpy as np

import torch
import torch.nn.functional as F

from torch import nn

from tqdm import tqdm

from src.training.metrics import (
    compute_binary_metrics,
    print_metrics,
)

from src.training.native_tile_checkpoint import (
    save_native_tile_checkpoint,
)


# ============================================================
# LOSSES
# ============================================================


def cosine_consistency_loss(
    clean_features:
        torch.Tensor,

    corrupted_features:
        torch.Tensor,
) -> torch.Tensor:

    similarity = (
        F.cosine_similarity(
            clean_features,
            corrupted_features,
            dim=-1,
        )
    )

    return (
        1.0
        - similarity
    ).mean()


def probability_consistency_loss(
    clean_logits:
        torch.Tensor,

    corrupted_logits:
        torch.Tensor,
) -> torch.Tensor:

    clean_probabilities = (
        torch.sigmoid(
            clean_logits
        )
    )

    corrupted_probabilities = (
        torch.sigmoid(
            corrupted_logits
        )
    )

    return F.mse_loss(
        clean_probabilities,
        corrupted_probabilities,
    )


def attention_entropy_loss(
    attention:
        torch.Tensor,

    tile_mask:
        torch.Tensor,
) -> torch.Tensor:
    """
    Negative entropy.

    Minimizing this weakly encourages higher entropy,
    stopping attention from collapsing too early.

    This weight should stay very small.
    """

    valid_attention = (
        attention.clamp_min(
            1e-8
        )
    )

    entropy = -(
        valid_attention
        * torch.log(
            valid_attention
        )
        * tile_mask.float()
    ).sum(
        dim=1
    )

    valid_count = (
        tile_mask.sum(
            dim=1
        )
        .float()
        .clamp_min(
            1.0
        )
    )

    # Normalize roughly by maximum possible entropy.
    max_entropy = torch.log(
        valid_count
    ).clamp_min(
        1.0
    )

    normalized_entropy = (
        entropy
        / max_entropy
    )

    return -(
        normalized_entropy.mean()
    )


# ============================================================
# DEVICE
# ============================================================


def move_batch(
    batch: dict,
    device: torch.device,
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
            "clean_forensic_tiles"
        ].to(
            device,
            non_blocking=True,
        ),

        batch[
            "corrupted_forensic_tiles"
        ].to(
            device,
            non_blocking=True,
        ),

        batch[
            "tile_mask"
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


# ============================================================
# TRAIN
# ============================================================


def train_native_tile_epoch(
    model,
    dataloader,
    optimizer,
    device,
    auxiliary_branch_weight,
    feature_consistency_weight,
    prediction_consistency_weight,
    attention_entropy_weight,
    gradient_clip_norm,
    use_amp,
):

    model.train()

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    amp_enabled = (
        use_amp
        and device.type
        == "cuda"
    )

    scaler = (
        torch.amp.GradScaler(
            "cuda",
            enabled=
                amp_enabled,
        )
    )

    totals = {
        "total_loss":
            0.0,

        "classification_loss":
            0.0,

        "auxiliary_loss":
            0.0,

        "feature_consistency_loss":
            0.0,

        "prediction_consistency_loss":
            0.0,

        "attention_entropy_loss":
            0.0,
    }

    sample_count = 0

    progress = tqdm(
        dataloader,
        desc="Native Tile Training",
    )

    for batch in progress:

        (
            clean_pixels,
            corrupted_pixels,
            clean_tiles,
            corrupted_tiles,
            tile_mask,
            labels,
        ) = move_batch(
            batch,
            device,
        )

        optimizer.zero_grad(
            set_to_none=True
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

            output = model.forward_pair(

                clean_pixel_values=
                    clean_pixels,

                corrupted_pixel_values=
                    corrupted_pixels,

                clean_forensic_tiles=
                    clean_tiles,

                corrupted_forensic_tiles=
                    corrupted_tiles,

                tile_mask=
                    tile_mask,
            )

            # ==============================================
            # MAIN CLASSIFICATION
            # ==============================================

            clean_loss = criterion(
                output[
                    "clean_logits"
                ],
                labels,
            )

            corrupted_loss = criterion(
                output[
                    "corrupted_logits"
                ],
                labels,
            )

            classification_loss = (
                clean_loss
                + corrupted_loss
            ) / 2.0

            # ==============================================
            # AUXILIARY BRANCH SUPERVISION
            # ==============================================

            auxiliary_loss = (
                criterion(
                    output[
                        "clean_semantic_logits"
                    ],
                    labels,
                )

                + criterion(
                    output[
                        "corrupted_semantic_logits"
                    ],
                    labels,
                )

                + criterion(
                    output[
                        "clean_forensic_logits"
                    ],
                    labels,
                )

                + criterion(
                    output[
                        "corrupted_forensic_logits"
                    ],
                    labels,
                )
            ) / 4.0

            # ==============================================
            # ROBUSTNESS CONSISTENCY
            # ==============================================

            feature_loss = (
                cosine_consistency_loss(

                    output[
                        "clean_fused_features"
                    ],

                    output[
                        "corrupted_fused_features"
                    ],
                )
            )

            prediction_loss = (
                probability_consistency_loss(

                    output[
                        "clean_logits"
                    ],

                    output[
                        "corrupted_logits"
                    ],
                )
            )

            # ==============================================
            # ATTENTION
            # ==============================================

            clean_attention_loss = (
                attention_entropy_loss(

                    output[
                        "clean_attention"
                    ],

                    tile_mask,
                )
            )

            corrupted_attention_loss = (
                attention_entropy_loss(

                    output[
                        "corrupted_attention"
                    ],

                    tile_mask,
                )
            )

            entropy_loss = (
                clean_attention_loss
                + corrupted_attention_loss
            ) / 2.0

            # ==============================================
            # TOTAL
            # ==============================================

            total_loss = (
                classification_loss

                + auxiliary_branch_weight
                * auxiliary_loss

                + feature_consistency_weight
                * feature_loss

                + prediction_consistency_weight
                * prediction_loss

                + attention_entropy_weight
                * entropy_loss
            )

        scaler.scale(
            total_loss
        ).backward()

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            (
                parameter
                for parameter
                in model.parameters()
                if parameter.requires_grad
            ),

            max_norm=
                gradient_clip_norm,
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        batch_size = (
            labels.shape[
                0
            ]
        )

        sample_count += (
            batch_size
        )

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

            "attention_entropy_loss":
                entropy_loss.item(),
        }

        for key, value in (
            values.items()
        ):

            totals[
                key
            ] += (
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
            / max(
                sample_count,
                1,
            )

        for key, value
        in totals.items()
    }


# ============================================================
# VALIDATION
# ============================================================


@torch.no_grad()
def validate_native_tile_model(
    model,
    dataloader,
    device,
    threshold,
):

    model.eval()

    labels_all = []

    clean_probabilities = []

    corrupted_probabilities = []

    clean_semantic_probabilities = []

    corrupted_semantic_probabilities = []

    clean_forensic_probabilities = []

    corrupted_forensic_probabilities = []

    prediction_shifts = []

    attention_shifts = []

    for batch in tqdm(
        dataloader,
        desc="Native Tile Validation",
    ):

        (
            clean_pixels,
            corrupted_pixels,
            clean_tiles,
            corrupted_tiles,
            tile_mask,
            labels,
        ) = move_batch(
            batch,
            device,
        )

        output = (
            model.forward_pair(

                clean_pixel_values=
                    clean_pixels,

                corrupted_pixel_values=
                    corrupted_pixels,

                clean_forensic_tiles=
                    clean_tiles,

                corrupted_forensic_tiles=
                    corrupted_tiles,

                tile_mask=
                    tile_mask,
            )
        )

        clean_final = torch.sigmoid(
            output[
                "clean_logits"
            ]
        )

        corrupted_final = torch.sigmoid(
            output[
                "corrupted_logits"
            ]
        )

        labels_all.extend(
            labels
            .cpu()
            .numpy()
            .tolist()
        )

        clean_probabilities.extend(
            clean_final
            .cpu()
            .numpy()
            .tolist()
        )

        corrupted_probabilities.extend(
            corrupted_final
            .cpu()
            .numpy()
            .tolist()
        )

        clean_semantic_probabilities.extend(
            torch.sigmoid(
                output[
                    "clean_semantic_logits"
                ]
            )
            .cpu()
            .numpy()
            .tolist()
        )

        corrupted_semantic_probabilities.extend(
            torch.sigmoid(
                output[
                    "corrupted_semantic_logits"
                ]
            )
            .cpu()
            .numpy()
            .tolist()
        )

        clean_forensic_probabilities.extend(
            torch.sigmoid(
                output[
                    "clean_forensic_logits"
                ]
            )
            .cpu()
            .numpy()
            .tolist()
        )

        corrupted_forensic_probabilities.extend(
            torch.sigmoid(
                output[
                    "corrupted_forensic_logits"
                ]
            )
            .cpu()
            .numpy()
            .tolist()
        )

        prediction_shifts.extend(
            torch.abs(
                clean_final
                - corrupted_final
            )
            .cpu()
            .numpy()
            .tolist()
        )

        attention_difference = (
            torch.abs(
                output[
                    "clean_attention"
                ]
                - output[
                    "corrupted_attention"
                ]
            )

            * tile_mask.float()
        )

        attention_difference = (
            attention_difference.sum(
                dim=1
            )

            / tile_mask.sum(
                dim=1
            )
            .clamp_min(
                1
            )
        )

        attention_shifts.extend(
            attention_difference
            .cpu()
            .numpy()
            .tolist()
        )

    clean_metrics = (
        compute_binary_metrics(
            labels=
                labels_all,

            probabilities=
                clean_probabilities,

            threshold=
                threshold,
        )
    )

    corrupted_metrics = (
        compute_binary_metrics(
            labels=
                labels_all,

            probabilities=
                corrupted_probabilities,

            threshold=
                threshold,
        )
    )

    semantic_metrics = (
        compute_binary_metrics(
            labels=
                labels_all,

            probabilities=
                corrupted_semantic_probabilities,

            threshold=
                threshold,
        )
    )

    forensic_metrics = (
        compute_binary_metrics(
            labels=
                labels_all,

            probabilities=
                corrupted_forensic_probabilities,

            threshold=
                threshold,
        )
    )

    robust_score = (
        clean_metrics[
            "auroc"
        ]

        + corrupted_metrics[
            "auroc"
        ]
    ) / 2.0

    return {
        "clean_metrics":
            clean_metrics,

        "corrupted_metrics":
            corrupted_metrics,

        "semantic_corrupted_metrics":
            semantic_metrics,

        "forensic_corrupted_metrics":
            forensic_metrics,

        "mean_prediction_shift":
            float(
                np.mean(
                    prediction_shifts
                )
            ),

        "mean_attention_shift":
            float(
                np.mean(
                    attention_shifts
                )
            ),

        "robust_score":
            float(
                robust_score
            ),
    }


# ============================================================
# FULL TRAIN
# ============================================================


def train_native_tile_model(
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
    attention_entropy_weight,
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
            "\n"
            "========================================"
        )

        print(
            f"NATIVE TILE EPOCH "
            f"{epoch}/{epochs}"
        )

        print(
            "========================================"
        )

        losses = (
            train_native_tile_epoch(

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

                attention_entropy_weight=
                    attention_entropy_weight,

                gradient_clip_norm=
                    gradient_clip_norm,

                use_amp=
                    use_amp,
            )
        )

        validation = (
            validate_native_tile_model(

                model=
                    model,

                dataloader=
                    val_dataloader,

                device=
                    device,

                threshold=
                    threshold,
            )
        )

        print(
            "\nTraining losses:"
        )

        for (
            key,
            value,
        ) in losses.items():

            print(
                f"{key:34s} "
                f"{value:.6f}"
            )

        print_metrics(
            validation[
                "clean_metrics"
            ],

            "Native Tile Clean Validation",
        )

        print_metrics(
            validation[
                "corrupted_metrics"
            ],

            "Native Tile Corrupted Validation",
        )

        print(
            "\nBranch diagnostics:"
        )

        print(
            f"Semantic AUROC: "
            f"{validation['semantic_corrupted_metrics']['auroc']:.4f}"
        )

        print(
            f"Forensic AUROC: "
            f"{validation['forensic_corrupted_metrics']['auroc']:.4f}"
        )

        print(
            f"Final AUROC:    "
            f"{validation['corrupted_metrics']['auroc']:.4f}"
        )

        print(
            "\nStability:"
        )

        print(
            f"Prediction shift: "
            f"{validation['mean_prediction_shift']:.6f}"
        )

        print(
            f"Attention shift:  "
            f"{validation['mean_attention_shift']:.6f}"
        )

        print(
            f"\nRobust score: "
            f"{validation['robust_score']:.6f}"
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

            "prediction_shift":
                validation[
                    "mean_prediction_shift"
                ],

            "attention_shift":
                validation[
                    "mean_attention_shift"
                ],

            "robust_score":
                validation[
                    "robust_score"
                ],
        }

        history.append(
            record
        )

        save_native_tile_checkpoint(

            path=(
                checkpoint_directory
                / "native_tile_last.pt"
            ),

            model=
                model,

            optimizer=
                optimizer,

            epoch=
                epoch,

            metrics=
                validation,

            config=
                config,
        )

        if (
            validation[
                "robust_score"
            ]
            > best_score
        ):

            best_score = (
                validation[
                    "robust_score"
                ]
            )

            save_native_tile_checkpoint(

                path=(
                    checkpoint_directory
                    / "native_tile_best.pt"
                ),

                model=
                    model,

                optimizer=
                    optimizer,

                epoch=
                    epoch,

                metrics=
                    validation,

                config=
                    config,
            )

            print(
                "\nNEW BEST NATIVE TILE MODEL"
            )

    with (
        checkpoint_directory
        / "native_tile_history.json"
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
        "NATIVE TILE TRAINING COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        f"\nBest robust score: "
        f"{best_score:.6f}"
    )