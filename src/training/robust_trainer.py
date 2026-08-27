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

from src.training.robust_checkpoint import (
    save_robust_checkpoint,
)


def cosine_consistency_loss(
    clean_features:
        torch.Tensor,

    corrupted_features:
        torch.Tensor,
) -> torch.Tensor:

    similarities = (
        F.cosine_similarity(
            clean_features,
            corrupted_features,
            dim=-1,
        )
    )

    return (
        1.0
        - similarities
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

    return (
        F.mse_loss(
            clean_probabilities,
            corrupted_probabilities,
        )
    )


def move_paired_batch(
    batch: dict,
    device: torch.device,
):

    clean_pixels = (
        batch[
            "clean_pixel_values"
        ]
        .to(
            device,
            non_blocking=True,
        )
    )

    corrupted_pixels = (
        batch[
            "corrupted_pixel_values"
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

    return (
        clean_pixels,
        corrupted_pixels,
        labels,
    )


def train_robust_epoch(
    model,
    dataloader,
    optimizer,
    device: torch.device,
    feature_consistency_weight: float,
    prediction_consistency_weight: float,
    gradient_clip_norm: float,
    use_amp: bool,
) -> dict[str, float]:

    model.train()

    classification_criterion = (
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
        "total_loss": 0.0,
        "classification_loss": 0.0,
        "clean_loss": 0.0,
        "corrupted_loss": 0.0,
        "feature_consistency_loss": 0.0,
        "prediction_consistency_loss": 0.0,
    }

    sample_count = 0

    progress = tqdm(
        dataloader,
        desc="Robust Training",
    )

    for batch in progress:

        (
            clean_pixels,
            corrupted_pixels,
            labels,
        ) = (
            move_paired_batch(
                batch,
                device,
            )
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

            outputs = (
                model.forward_pair(
                    clean_pixel_values=
                        clean_pixels,

                    corrupted_pixel_values=
                        corrupted_pixels,
                )
            )

            clean_logits = (
                outputs[
                    "clean_logits"
                ]
            )

            corrupted_logits = (
                outputs[
                    "corrupted_logits"
                ]
            )

            clean_features = (
                outputs[
                    "clean_features"
                ]
            )

            corrupted_features = (
                outputs[
                    "corrupted_features"
                ]
            )

            clean_loss = (
                classification_criterion(
                    clean_logits,
                    labels,
                )
            )

            corrupted_loss = (
                classification_criterion(
                    corrupted_logits,
                    labels,
                )
            )

            classification_loss = (
                0.5
                * (
                    clean_loss
                    + corrupted_loss
                )
            )

            feature_loss = (
                cosine_consistency_loss(
                    clean_features=
                        clean_features,

                    corrupted_features=
                        corrupted_features,
                )
            )

            prediction_loss = (
                probability_consistency_loss(
                    clean_logits=
                        clean_logits,

                    corrupted_logits=
                        corrupted_logits,
                )
            )

            total_loss = (
                classification_loss

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

        if (
            gradient_clip_norm
            is not None
            and gradient_clip_norm
            > 0
        ):

            torch.nn.utils.clip_grad_norm_(
                model.get_trainable_parameters(),
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

        totals[
            "total_loss"
        ] += (
            total_loss.item()
            * batch_size
        )

        totals[
            "classification_loss"
        ] += (
            classification_loss.item()
            * batch_size
        )

        totals[
            "clean_loss"
        ] += (
            clean_loss.item()
            * batch_size
        )

        totals[
            "corrupted_loss"
        ] += (
            corrupted_loss.item()
            * batch_size
        )

        totals[
            "feature_consistency_loss"
        ] += (
            feature_loss.item()
            * batch_size
        )

        totals[
            "prediction_consistency_loss"
        ] += (
            prediction_loss.item()
            * batch_size
        )

        progress.set_postfix(
            loss=(
                f"{totals['total_loss'] / sample_count:.4f}"
            )
        )

    averages = {
        key:
            value
            / max(
                sample_count,
                1,
            )

        for key, value
        in totals.items()
    }

    return averages


@torch.no_grad()
def evaluate_paired_validation(
    model,
    dataloader,
    device: torch.device,
    threshold: float,
) -> dict:

    model.eval()

    clean_labels = []
    clean_probs = []

    corrupted_labels = []
    corrupted_probs = []

    clean_corrupt_differences = []

    progress = tqdm(
        dataloader,
        desc="Paired Validation",
    )

    for batch in progress:

        (
            clean_pixels,
            corrupted_pixels,
            labels,
        ) = (
            move_paired_batch(
                batch,
                device,
            )
        )

        outputs = (
            model.forward_pair(
                clean_pixel_values=
                    clean_pixels,

                corrupted_pixel_values=
                    corrupted_pixels,
            )
        )

        clean_probabilities = (
            torch.sigmoid(
                outputs[
                    "clean_logits"
                ]
            )
        )

        corrupted_probabilities = (
            torch.sigmoid(
                outputs[
                    "corrupted_logits"
                ]
            )
        )

        clean_labels.extend(
            labels
            .cpu()
            .numpy()
            .tolist()
        )

        corrupted_labels.extend(
            labels
            .cpu()
            .numpy()
            .tolist()
        )

        clean_probs.extend(
            clean_probabilities
            .cpu()
            .numpy()
            .tolist()
        )

        corrupted_probs.extend(
            corrupted_probabilities
            .cpu()
            .numpy()
            .tolist()
        )

        differences = (
            torch.abs(
                clean_probabilities
                - corrupted_probabilities
            )
        )

        clean_corrupt_differences.extend(
            differences
            .cpu()
            .numpy()
            .tolist()
        )

    clean_metrics = (
        compute_binary_metrics(
            labels=
                clean_labels,

            probabilities=
                clean_probs,

            threshold=
                threshold,
        )
    )

    corrupted_metrics = (
        compute_binary_metrics(
            labels=
                corrupted_labels,

            probabilities=
                corrupted_probs,

            threshold=
                threshold,
        )
    )

    mean_prediction_shift = (
        float(
            np.mean(
                clean_corrupt_differences
            )
        )
    )

    robust_score = (
        (
            clean_metrics[
                "auroc"
            ]
            + corrupted_metrics[
                "auroc"
            ]
        )
        / 2.0
    )

    return {
        "clean_metrics":
            clean_metrics,

        "corrupted_metrics":
            corrupted_metrics,

        "mean_prediction_shift":
            mean_prediction_shift,

        "robust_score":
            float(
                robust_score
            ),
    }


def train_robust_model(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    device: torch.device,
    epochs: int,
    checkpoint_directory:
        str | Path,
    config: dict,
    threshold: float,
    feature_consistency_weight: float,
    prediction_consistency_weight: float,
    gradient_clip_norm: float,
    use_amp: bool,
):

    checkpoint_directory = Path(
        checkpoint_directory
    )

    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    history = []

    best_robust_score = (
        float(
            "-inf"
        )
    )

    for epoch in range(
        1,
        epochs + 1,
    ):

        print(
            "\n"
            "========================================"
        )

        print(
            f"ROBUST EPOCH "
            f"{epoch}/{epochs}"
        )

        print(
            "========================================"
        )

        train_losses = (
            train_robust_epoch(
                model=
                    model,

                dataloader=
                    train_dataloader,

                optimizer=
                    optimizer,

                device=
                    device,

                feature_consistency_weight=
                    feature_consistency_weight,

                prediction_consistency_weight=
                    prediction_consistency_weight,

                gradient_clip_norm=
                    gradient_clip_norm,

                use_amp=
                    use_amp,
            )
        )

        validation = (
            evaluate_paired_validation(
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

        for key, value in (
            train_losses.items()
        ):

            print(
                f"{key:32s}"
                f"{value:.6f}"
            )

        print_metrics(
            validation[
                "clean_metrics"
            ],
            title="Robust Model Clean Validation",
        )

        print_metrics(
            validation[
                "corrupted_metrics"
            ],
            title="Robust Model Corrupted Validation",
        )

        print(
            "\nPrediction stability:"
        )

        print(
            f"Mean |Pclean - Pcorrupt|: "
            f"{validation['mean_prediction_shift']:.6f}"
        )

        print(
            f"Robust score: "
            f"{validation['robust_score']:.6f}"
        )

        epoch_record = {
            "epoch":
                epoch,

            **train_losses,

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

            "clean_accuracy":
                validation[
                    "clean_metrics"
                ][
                    "accuracy"
                ],

            "corrupted_accuracy":
                validation[
                    "corrupted_metrics"
                ][
                    "accuracy"
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
            epoch_record
        )

        last_path = (
            checkpoint_directory
            / "robust_last.pt"
        )

        save_robust_checkpoint(
            path=
                last_path,

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
            > best_robust_score
        ):

            best_robust_score = (
                validation[
                    "robust_score"
                ]
            )

            best_path = (
                checkpoint_directory
                / "robust_best.pt"
            )

            save_robust_checkpoint(
                path=
                    best_path,

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
                "\nNEW BEST ROBUST MODEL"
            )

            print(
                f"Robust score: "
                f"{best_robust_score:.6f}"
            )

    history_path = (
        checkpoint_directory
        / "robust_history.json"
    )

    with history_path.open(
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
        "ROBUST TRAINING COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        f"\nBest robust score: "
        f"{best_robust_score:.6f}"
    )