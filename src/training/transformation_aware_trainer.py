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

from src.training.transformation_aware_checkpoint import (
    save_transformation_aware_checkpoint,
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


def build_gate_targets(
    semantic_logits,
    forensic_logits,
    labels,
    temperature,
):

    semantic_losses = (
        F.binary_cross_entropy_with_logits(

            semantic_logits,
            labels,

            reduction="none",
        )
    )

    forensic_losses = (
        F.binary_cross_entropy_with_logits(

            forensic_logits,
            labels,

            reduction="none",
        )
    )

    losses = torch.stack(
        [
            semantic_losses,
            forensic_losses,
        ],
        dim=-1,
    )

    reliability = torch.softmax(
        -losses
        / temperature,
        dim=-1,
    )

    return reliability.detach()


def gate_reliability_loss(
    predicted_weights,
    target_weights,
):

    return F.kl_div(

        torch.log(
            predicted_weights
            .clamp_min(
                1e-8
            )
        ),

        target_weights,

        reduction="batchmean",
    )


def feature_consistency_loss(
    clean,
    corrupted,
):

    return (
        1.0
        - F.cosine_similarity(
            clean,
            corrupted,
            dim=-1,
        )
    ).mean()


def probability_consistency_loss(
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


def train_epoch(
    model,
    dataloader,
    optimizer,
    device,
    auxiliary_branch_weight,
    feature_consistency_weight,
    prediction_consistency_weight,
    gate_reliability_weight,
    gate_target_temperature,
    gradient_clip_norm,
    use_amp,
):

    model.train()

    # Keep frozen corruption estimator deterministic.
    model.corruption_estimator.eval()

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    amp_enabled = (
        use_amp
        and device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=
            amp_enabled,
    )

    totals = {
        "total":
            0.0,

        "classification":
            0.0,

        "auxiliary":
            0.0,

        "feature_consistency":
            0.0,

        "prediction_consistency":
            0.0,

        "gate_reliability":
            0.0,
    }

    sample_count = 0

    progress = tqdm(
        dataloader,
        desc=(
            "Transformation-Aware Training"
        ),
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
                if device.type == "cuda"
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

            # ==========================================
            # FINAL CLASSIFICATION
            # ==========================================

            clean_classification = criterion(

                output[
                    "clean_logits"
                ],

                labels,
            )

            corrupted_classification = criterion(

                output[
                    "corrupted_logits"
                ],

                labels,
            )

            classification_loss = (
                clean_classification
                + corrupted_classification
            ) / 2.0

            # ==========================================
            # AUXILIARY BRANCHES
            # ==========================================

            auxiliary_loss = (

                criterion(
                    output[
                        "clean_semantic_logits"
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
                        "corrupted_semantic_logits"
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

            # ==========================================
            # CONSISTENCY
            # ==========================================

            feature_loss = (
                feature_consistency_loss(

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

            # ==========================================
            # RELIABILITY SUPERVISION
            # ==========================================

            clean_gate_target = (
                build_gate_targets(

                    semantic_logits=
                        output[
                            "clean_semantic_logits"
                        ],

                    forensic_logits=
                        output[
                            "clean_forensic_logits"
                        ],

                    labels=
                        labels,

                    temperature=
                        gate_target_temperature,
                )
            )

            corrupted_gate_target = (
                build_gate_targets(

                    semantic_logits=
                        output[
                            "corrupted_semantic_logits"
                        ],

                    forensic_logits=
                        output[
                            "corrupted_forensic_logits"
                        ],

                    labels=
                        labels,

                    temperature=
                        gate_target_temperature,
                )
            )

            clean_gate_loss = (
                gate_reliability_loss(

                    predicted_weights=
                        output[
                            "clean_gate_weights"
                        ],

                    target_weights=
                        clean_gate_target,
                )
            )

            corrupted_gate_loss = (
                gate_reliability_loss(

                    predicted_weights=
                        output[
                            "corrupted_gate_weights"
                        ],

                    target_weights=
                        corrupted_gate_target,
                )
            )

            reliability_loss = (
                clean_gate_loss
                + corrupted_gate_loss
            ) / 2.0

            # ==========================================
            # TOTAL
            # ==========================================

            total_loss = (

                classification_loss

                + auxiliary_branch_weight
                * auxiliary_loss

                + feature_consistency_weight
                * feature_loss

                + prediction_consistency_weight
                * prediction_loss

                + gate_reliability_weight
                * reliability_loss
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
            "total":
                total_loss.item(),

            "classification":
                classification_loss.item(),

            "auxiliary":
                auxiliary_loss.item(),

            "feature_consistency":
                feature_loss.item(),

            "prediction_consistency":
                prediction_loss.item(),

            "gate_reliability":
                reliability_loss.item(),
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
                f"{totals['total'] / sample_count:.4f}"
            ),

            scale=(
                f"{model.adaptive_scale.item():.3f}"
            ),
        )

    return {
        key:
            value
            / sample_count

        for key, value
        in totals.items()
    }


@torch.no_grad()
def validate_model(
    model,
    dataloader,
    device,
    threshold,
):

    model.eval()

    labels_all = []

    clean_predictions = []
    corrupted_predictions = []

    semantic_weights = []
    forensic_weights = []

    clean_semantic_weights = []
    clean_forensic_weights = []

    prediction_shift = []

    for batch in tqdm(
        dataloader,
        desc="Transformation-Aware Validation",
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

        clean_prob = torch.sigmoid(
            output[
                "clean_logits"
            ]
        )

        corrupted_prob = torch.sigmoid(
            output[
                "corrupted_logits"
            ]
        )

        labels_all.extend(
            labels.cpu().numpy().tolist()
        )

        clean_predictions.extend(
            clean_prob.cpu().numpy().tolist()
        )

        corrupted_predictions.extend(
            corrupted_prob.cpu().numpy().tolist()
        )

        semantic_weights.extend(
            output[
                "corrupted_semantic_weight"
            ]
            .cpu()
            .numpy()
            .tolist()
        )

        forensic_weights.extend(
            output[
                "corrupted_forensic_weight"
            ]
            .cpu()
            .numpy()
            .tolist()
        )

        clean_semantic_weights.extend(
            output[
                "clean_semantic_weight"
            ]
            .cpu()
            .numpy()
            .tolist()
        )

        clean_forensic_weights.extend(
            output[
                "clean_forensic_weight"
            ]
            .cpu()
            .numpy()
            .tolist()
        )

        prediction_shift.extend(
            torch.abs(
                clean_prob
                - corrupted_prob
            )
            .cpu()
            .numpy()
            .tolist()
        )

    clean_metrics = (
        compute_binary_metrics(

            labels=
                labels_all,

            probabilities=
                clean_predictions,

            threshold=
                threshold,
        )
    )

    corrupted_metrics = (
        compute_binary_metrics(

            labels=
                labels_all,

            probabilities=
                corrupted_predictions,

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

        "mean_clean_semantic_weight":
            float(
                np.mean(
                    clean_semantic_weights
                )
            ),

        "mean_clean_forensic_weight":
            float(
                np.mean(
                    clean_forensic_weights
                )
            ),

        "mean_corrupted_semantic_weight":
            float(
                np.mean(
                    semantic_weights
                )
            ),

        "mean_corrupted_forensic_weight":
            float(
                np.mean(
                    forensic_weights
                )
            ),

        "mean_prediction_shift":
            float(
                np.mean(
                    prediction_shift
                )
            ),

        "robust_score":
            float(
                robust_score
            ),
    }


def train_transformation_aware_model(
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
    gate_reliability_weight,
    gate_target_temperature,
    gradient_clip_norm,
    use_amp,
):

    checkpoint_directory = Path(
        checkpoint_directory
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
            f"TRANSFORMATION-AWARE "
            f"EPOCH {epoch}/{epochs}"
        )

        print(
            "========================================"
        )

        losses = train_epoch(

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

            gate_reliability_weight=
                gate_reliability_weight,

            gate_target_temperature=
                gate_target_temperature,

            gradient_clip_norm=
                gradient_clip_norm,

            use_amp=
                use_amp,
        )

        validation = validate_model(

            model=
                model,

            dataloader=
                val_dataloader,

            device=
                device,

            threshold=
                threshold,
        )

        print_metrics(
            validation[
                "clean_metrics"
            ],

            "Transformation-Aware Clean",
        )

        print_metrics(
            validation[
                "corrupted_metrics"
            ],

            "Transformation-Aware Corrupted",
        )

        print(
            "\nReliability gate:"
        )

        print(
            f"Clean semantic:     "
            f"{validation['mean_clean_semantic_weight']:.4f}"
        )

        print(
            f"Clean forensic:     "
            f"{validation['mean_clean_forensic_weight']:.4f}"
        )

        print(
            f"Corrupt semantic:   "
            f"{validation['mean_corrupted_semantic_weight']:.4f}"
        )

        print(
            f"Corrupt forensic:   "
            f"{validation['mean_corrupted_forensic_weight']:.4f}"
        )

        print(
            f"\nPrediction shift: "
            f"{validation['mean_prediction_shift']:.5f}"
        )

        print(
            f"Adaptive scale:   "
            f"{model.adaptive_scale.item():.5f}"
        )

        print(
            f"Robust score:     "
            f"{validation['robust_score']:.5f}"
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

            "clean_semantic_weight":
                validation[
                    "mean_clean_semantic_weight"
                ],

            "clean_forensic_weight":
                validation[
                    "mean_clean_forensic_weight"
                ],

            "corrupted_semantic_weight":
                validation[
                    "mean_corrupted_semantic_weight"
                ],

            "corrupted_forensic_weight":
                validation[
                    "mean_corrupted_forensic_weight"
                ],

            "adaptive_scale":
                float(
                    model.adaptive_scale.item()
                ),

            "robust_score":
                validation[
                    "robust_score"
                ],
        }

        history.append(
            record
        )

        save_transformation_aware_checkpoint(

            checkpoint_directory
            / "transformation_aware_last.pt",

            model,
            optimizer,
            epoch,
            validation,
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

            save_transformation_aware_checkpoint(

                checkpoint_directory
                / "transformation_aware_best.pt",

                model,
                optimizer,
                epoch,
                validation,
                config,
            )

            print(
                "\nNEW BEST "
                "TRANSFORMATION-AWARE MODEL"
            )

    with (
        checkpoint_directory
        / "transformation_aware_history.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            history,
            file,
            indent=4,
        )