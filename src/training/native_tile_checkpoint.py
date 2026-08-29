from __future__ import annotations

from pathlib import Path

import torch


def warm_start_native_tile_model(
    path: str | Path,
    model,
    device="cpu",
) -> dict:

    path = Path(
        path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Warm-start checkpoint "
            f"does not exist: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    print(
        "\n========================================"
    )

    print(
        "NATIVE TILE WARM START"
    )

    print(
        "========================================"
    )

    # --------------------------------------------------
    # Semantic branch
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Old forensic encoder → tile forensic encoder
    #
    # Old ForensicEncoder state contains:
    #
    # encoder.*
    # projection.*
    #
    # which matches our tile encoder.
    # --------------------------------------------------

    old_forensic_state = (
        checkpoint[
            "forensic_encoder_state_dict"
        ]
    )

    load_result = (
        model.tile_encoder
        .load_state_dict(
            old_forensic_state,
            strict=False,
        )
    )

    print(
        "\nTile encoder warm start:"
    )

    print(
        f"Missing keys: "
        f"{load_result.missing_keys}"
    )

    print(
        f"Unexpected keys: "
        f"{load_result.unexpected_keys}"
    )

    # --------------------------------------------------
    # Existing heads/fusion
    # --------------------------------------------------

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

    print(
        "\nLoaded from:"
    )

    print(
        path
    )

    print(
        f"\nCheckpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        "\nNew component:"
    )

    print(
        "Tile attention module "
        "(randomly initialized)"
    )

    return checkpoint


def save_native_tile_checkpoint(
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

        "semantic_adapter_state_dict":
            model.semantic_adapter
            .state_dict(),

        "semantic_head_state_dict":
            model.semantic_head
            .state_dict(),

        "tile_encoder_state_dict":
            model.tile_encoder
            .state_dict(),

        "tile_attention_state_dict":
            model.tile_attention
            .state_dict(),

        "forensic_head_state_dict":
            model.forensic_head
            .state_dict(),

        "semantic_projection_state_dict":
            model.semantic_projection
            .state_dict(),

        "fusion_head_state_dict":
            model.fusion_head
            .state_dict(),

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


def load_native_tile_checkpoint(
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

    model.tile_encoder.load_state_dict(
        checkpoint[
            "tile_encoder_state_dict"
        ]
    )

    model.tile_attention.load_state_dict(
        checkpoint[
            "tile_attention_state_dict"
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