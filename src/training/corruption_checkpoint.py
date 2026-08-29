from __future__ import annotations

from pathlib import Path

import torch


def save_corruption_estimator_checkpoint(
    path: str | Path,
    estimator,
    optimizer,
    epoch: int,
    metrics: dict,
    config: dict,
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

            "estimator_state_dict":
                estimator.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "metrics":
                metrics,

            "config":
                config,
        },

        path,
    )


def load_corruption_estimator_checkpoint(
    path: str | Path,
    estimator,
    device="cpu",
):

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    estimator.load_state_dict(
        checkpoint[
            "estimator_state_dict"
        ]
    )

    return checkpoint