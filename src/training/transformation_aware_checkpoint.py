from __future__ import annotations

from pathlib import Path

import torch

from src.training.corruption_checkpoint import (
    load_corruption_estimator_checkpoint,
)

from src.training.native_tile_checkpoint import (
    load_native_tile_checkpoint,
)


def warm_start_transformation_aware_model(
    model,
    native_tile_checkpoint,
    corruption_checkpoint,
    device,
    freeze_corruption_estimator=True,
):

    print(
        "\n========================================"
    )

    print(
        "TRANSFORMATION-AWARE WARM START"
    )

    print(
        "========================================"
    )

    load_native_tile_checkpoint(

        path=
            native_tile_checkpoint,

        model=
            model,

        device=
            device,
    )

    print(
        "\nLoaded native-tile detector:"
    )

    print(
        native_tile_checkpoint
    )

    load_corruption_estimator_checkpoint(

        path=
            corruption_checkpoint,

        estimator=
            model.corruption_estimator,

        device=
            device,
    )

    print(
        "\nLoaded corruption estimator:"
    )

    print(
        corruption_checkpoint
    )

    if freeze_corruption_estimator:

        for parameter in (
            model.corruption_estimator
            .parameters()
        ):

            parameter.requires_grad = (
                False
            )

        model.corruption_estimator.eval()

        print(
            "\nCorruption estimator frozen."
        )


def save_transformation_aware_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    metrics,
    config,
):

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch":
                epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "metrics":
                metrics,

            "config":
                config,
        },

        path,
    )


def load_transformation_aware_checkpoint(
    path,
    model,
    device="cpu",
):

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    return checkpoint