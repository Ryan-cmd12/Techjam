from __future__ import annotations

from pathlib import Path

import torch


def warm_start_from_robust_checkpoint(
    path: str | Path,
    model,
    device="cpu",
) -> dict:

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model.semantic_adapter.load_state_dict(
        checkpoint[
            "adapter_state_dict"
        ]
    )

    model.semantic_head.load_state_dict(
        checkpoint[
            "classifier_state_dict"
        ]
    )

    print(
        "\nLoaded semantic warm start:"
    )

    print(
        path
    )

    print(
        f"Robust checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    return checkpoint


def save_forensic_checkpoint(
    path: str | Path,
    model,
    optimizer,
    epoch: int,
    metrics: dict,
    config: dict,
) -> None:

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch":
            epoch,

        "clip_model_name":
            model.clip_model_name,

        "semantic_adapter_state_dict":
            model.semantic_adapter.state_dict(),

        "semantic_head_state_dict":
            model.semantic_head.state_dict(),

        "forensic_encoder_state_dict":
            model.forensic_encoder.state_dict(),

        "forensic_head_state_dict":
            model.forensic_head.state_dict(),

        "semantic_projection_state_dict":
            model.semantic_projection.state_dict(),

        "fusion_head_state_dict":
            model.fusion_head.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "metrics":
            metrics,

        "config":
            config,
    }

    torch.save(
        checkpoint,
        path,
    )


def load_forensic_checkpoint(
    path: str | Path,
    model,
    device="cpu",
) -> dict:

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model.semantic_adapter.load_state_dict(
        checkpoint[
            "semantic_adapter_state_dict"
        ]
    )

    model.semantic_head.load_state_dict(
        checkpoint[
            "semantic_head_state_dict"
        ]
    )

    model.forensic_encoder.load_state_dict(
        checkpoint[
            "forensic_encoder_state_dict"
        ]
    )

    model.forensic_head.load_state_dict(
        checkpoint[
            "forensic_head_state_dict"
        ]
    )

    model.semantic_projection.load_state_dict(
        checkpoint[
            "semantic_projection_state_dict"
        ]
    )

    model.fusion_head.load_state_dict(
        checkpoint[
            "fusion_head_state_dict"
        ]
    )

    return checkpoint