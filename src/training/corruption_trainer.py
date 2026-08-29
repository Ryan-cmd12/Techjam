from __future__ import annotations

import json

from pathlib import Path

import numpy as np

import torch
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

from tqdm import tqdm

from src.training.corruption_checkpoint import (
    save_corruption_estimator_checkpoint,
)

from src.training.corruption_targets import (
    CORRUPTION_TYPES,
    build_severity_targets,
    build_type_targets,
)


def freeze_detector(
    detector,
):

    detector.eval()

    for parameter in (
        detector.parameters()
    ):

        parameter.requires_grad = (
            False
        )


def extract_detector_features(
    detector,
    batch,
    device,
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

    with torch.no_grad():

        output = (
            detector.forward_with_details(

                pixel_values=
                    pixel_values,

                forensic_tiles=
                    forensic_tiles,

                tile_mask=
                    tile_mask,
            )
        )

    return (
        output[
            "semantic_features"
        ].detach(),

        output[
            "forensic_features"
        ].detach(),
    )


def train_corruption_epoch(
    detector,
    estimator,
    dataloader,
    optimizer,
    device,
    type_weight,
    severity_weight,
):

    detector.eval()

    estimator.train()

    running = {
        "total":
            0.0,

        "type":
            0.0,

        "severity":
            0.0,
    }

    samples = 0

    progress = tqdm(
        dataloader,
        desc="Corruption Estimator Training",
    )

    for batch in progress:

        (
            semantic_features,
            forensic_features,
        ) = extract_detector_features(

            detector=
                detector,

            batch=
                batch,

            device=
                device,
        )

        type_targets = (
            build_type_targets(

                corruption_types=
                    batch[
                        "corruption_type"
                    ],

                device=
                    device,
            )
        )

        severity_targets = (
            build_severity_targets(
                batch=
                    batch,

                device=
                    device,
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        output = estimator(

            semantic_features=
                semantic_features,

            forensic_features=
                forensic_features,
        )

        type_loss = (
            F.cross_entropy(

                output[
                    "type_logits"
                ],

                type_targets,
            )
        )

        severity_loss = (
            F.mse_loss(

                output[
                    "severity"
                ],

                severity_targets,
            )
        )

        total_loss = (
            type_weight
            * type_loss

            + severity_weight
            * severity_loss
        )

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            estimator.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        batch_size = (
            type_targets.shape[
                0
            ]
        )

        samples += batch_size

        running[
            "total"
        ] += (
            total_loss.item()
            * batch_size
        )

        running[
            "type"
        ] += (
            type_loss.item()
            * batch_size
        )

        running[
            "severity"
        ] += (
            severity_loss.item()
            * batch_size
        )

        progress.set_postfix(
            loss=(
                f"{running['total'] / samples:.4f}"
            )
        )

    return {
        "loss":
            running[
                "total"
            ]
            / samples,

        "type_loss":
            running[
                "type"
            ]
            / samples,

        "severity_loss":
            running[
                "severity"
            ]
            / samples,
    }


@torch.no_grad()
def validate_corruption_estimator(
    detector,
    estimator,
    dataloader,
    device,
):

    detector.eval()

    estimator.eval()

    true_types = []

    predicted_types = []

    true_severity = []

    predicted_severity = []

    for batch in tqdm(
        dataloader,
        desc="Corruption Estimator Validation",
    ):

        (
            semantic_features,
            forensic_features,
        ) = extract_detector_features(

            detector=
                detector,

            batch=
                batch,

            device=
                device,
        )

        type_targets = (
            build_type_targets(

                batch[
                    "corruption_type"
                ],

                device,
            )
        )

        severity_targets = (
            build_severity_targets(
                batch,
                device,
            )
        )

        output = estimator(
            semantic_features,
            forensic_features,
        )

        predictions = (
            output[
                "type_logits"
            ]
            .argmax(
                dim=-1
            )
        )

        true_types.extend(
            type_targets
            .cpu()
            .numpy()
            .tolist()
        )

        predicted_types.extend(
            predictions
            .cpu()
            .numpy()
            .tolist()
        )

        true_severity.extend(
            severity_targets
            .cpu()
            .numpy()
            .tolist()
        )

        predicted_severity.extend(
            output[
                "severity"
            ]
            .cpu()
            .numpy()
            .tolist()
        )

    type_accuracy = (
        accuracy_score(
            true_types,
            predicted_types,
        )
    )

    macro_f1 = (
        f1_score(
            true_types,
            predicted_types,
            average="macro",
            zero_division=0,
        )
    )

    severity_mae = float(
        np.mean(
            np.abs(
                np.asarray(
                    true_severity
                )
                - np.asarray(
                    predicted_severity
                )
            )
        )
    )

    per_class_accuracy = {}

    true_array = np.asarray(
        true_types
    )

    predicted_array = np.asarray(
        predicted_types
    )

    for class_id, class_name in enumerate(
        CORRUPTION_TYPES
    ):

        mask = (
            true_array
            == class_id
        )

        if mask.sum() == 0:

            accuracy = float(
                "nan"
            )

        else:

            accuracy = float(
                (
                    predicted_array[
                        mask
                    ]
                    == true_array[
                        mask
                    ]
                ).mean()
            )

        per_class_accuracy[
            class_name
        ] = accuracy

    return {
        "type_accuracy":
            float(
                type_accuracy
            ),

        "macro_f1":
            float(
                macro_f1
            ),

        "severity_mae":
            severity_mae,

        "per_class_accuracy":
            per_class_accuracy,
    }


def train_corruption_estimator(
    detector,
    estimator,
    train_dataloader,
    val_dataloader,
    optimizer,
    device,
    epochs,
    checkpoint_directory,
    config,
    type_weight,
    severity_weight,
):

    checkpoint_directory = Path(
        checkpoint_directory
    )

    best_score = float(
        "-inf"
    )

    history = []

    freeze_detector(
        detector
    )

    for epoch in range(
        1,
        epochs + 1,
    ):

        print(
            "\n========================================"
        )

        print(
            f"CORRUPTION ESTIMATOR "
            f"EPOCH {epoch}/{epochs}"
        )

        print(
            "========================================"
        )

        losses = (
            train_corruption_epoch(

                detector=
                    detector,

                estimator=
                    estimator,

                dataloader=
                    train_dataloader,

                optimizer=
                    optimizer,

                device=
                    device,

                type_weight=
                    type_weight,

                severity_weight=
                    severity_weight,
            )
        )

        metrics = (
            validate_corruption_estimator(

                detector=
                    detector,

                estimator=
                    estimator,

                dataloader=
                    val_dataloader,

                device=
                    device,
            )
        )

        print(
            "\nTraining:"
        )

        print(
            f"Total loss:    "
            f"{losses['loss']:.5f}"
        )

        print(
            f"Type loss:     "
            f"{losses['type_loss']:.5f}"
        )

        print(
            f"Severity loss: "
            f"{losses['severity_loss']:.5f}"
        )

        print(
            "\nValidation:"
        )

        print(
            f"Type accuracy: "
            f"{metrics['type_accuracy']:.4f}"
        )

        print(
            f"Macro F1:      "
            f"{metrics['macro_f1']:.4f}"
        )

        print(
            f"Severity MAE:  "
            f"{metrics['severity_mae']:.4f}"
        )

        print(
            "\nPer-corruption accuracy:"
        )

        for (
            name,
            accuracy,
        ) in metrics[
            "per_class_accuracy"
        ].items():

            print(
                f"{name:15s} "
                f"{accuracy:.4f}"
            )

        record = {
            "epoch":
                epoch,

            **losses,

            "type_accuracy":
                metrics[
                    "type_accuracy"
                ],

            "macro_f1":
                metrics[
                    "macro_f1"
                ],

            "severity_mae":
                metrics[
                    "severity_mae"
                ],
        }

        history.append(
            record
        )

        score = (
            metrics[
                "macro_f1"
            ]
            - 0.25
            * metrics[
                "severity_mae"
            ]
        )

        save_corruption_estimator_checkpoint(

            path=(
                checkpoint_directory
                / "corruption_estimator_last.pt"
            ),

            estimator=
                estimator,

            optimizer=
                optimizer,

            epoch=
                epoch,

            metrics=
                metrics,

            config=
                config,
        )

        if score > best_score:

            best_score = score

            save_corruption_estimator_checkpoint(

                path=(
                    checkpoint_directory
                    / "corruption_estimator_best.pt"
                ),

                estimator=
                    estimator,

                optimizer=
                    optimizer,

                epoch=
                    epoch,

                metrics=
                    metrics,

                config=
                    config,
            )

            print(
                "\nNEW BEST CORRUPTION ESTIMATOR"
            )

    with (
        checkpoint_directory
        / "corruption_estimator_history.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            history,
            file,
            indent=4,
        )