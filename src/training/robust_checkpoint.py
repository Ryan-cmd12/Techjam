from __future__ import annotations

from pathlib import Path

import torch


def save_robust_checkpoint(
    path: str | Path,
    model,
    optimizer,
    epoch: int,
    metrics: dict,
    config: dict,
) -> None:

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch":
            epoch,

        "clip_model_name":
            model.clip_model_name,

        "adapter_state_dict":
            model.adapter.state_dict(),

        "classifier_state_dict":
            model.classifier.state_dict(),

        "optimizer_state_dict":
            (
                optimizer.state_dict()
                if optimizer is not None
                else None
            ),

        "metrics":
            metrics,

        "config":
            config,
    }

    torch.save(
        checkpoint,
        path,
    )


def load_robust_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    device="cpu",
) -> dict:

    path = Path(
        path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Robust checkpoint "
            f"does not exist: "
            f"{path}"
        )

    checkpoint = (
        torch.load(
            path,
            map_location=
                device,
            weights_only=False,
        )
    )

    model.adapter.load_state_dict(
        checkpoint[
            "adapter_state_dict"
        ]
    )

    model.classifier.load_state_dict(
        checkpoint[
            "classifier_state_dict"
        ]
    )

    if (
        optimizer is not None
        and checkpoint.get(
            "optimizer_state_dict"
        )
        is not None
    ):

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

    return checkpoint