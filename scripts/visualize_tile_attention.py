from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pandas as pd
import torch

from torch.utils.data import (
    DataLoader,
)

from src.data.dataset import (
    AIGCImageDataset,
)

from src.data.image_utils import (
    load_rgb_image,
)

from src.data.native_tile_collate import (
    NativeTileCLIPBatchCollator,
)

from src.models.native_tile_fusion_detector import (
    NativeTileFusionDetector,
)

from src.training.native_tile_checkpoint import (
    load_native_tile_checkpoint,
)

from src.utils.config import (
    load_config,
)

from src.utils.device import (
    get_device,
)


def main():

    config = load_config(
        "configs/base.yaml"
    )

    device = get_device()

    manifest_path = (
        "data/manifests/"
        "sid_test.csv"
    )

    dataset = (
        AIGCImageDataset(
            manifest_path=
                manifest_path,

            return_metadata=
                True,
        )
    )

    tile_cfg = (
        config[
            "native_tiles"
        ]
    )

    collator = (
        NativeTileCLIPBatchCollator(

            model_name=
                config[
                    "model"
                ][
                    "clip_model"
                ],

            tile_size=
                int(
                    tile_cfg[
                        "tile_size"
                    ]
                ),

            max_tiles=
                int(
                    tile_cfg[
                        "max_tiles"
                    ]
                ),

            feature_map_size=
                int(
                    tile_cfg[
                        "feature_map_size"
                    ]
                ),

            sampling_mode="grid",

            seed=42,
        )
    )

    dataloader = (
        DataLoader(

            dataset,

            batch_size=1,

            shuffle=False,

            num_workers=0,

            collate_fn=
                collator,
        )
    )

    batch = next(
        iter(
            dataloader
        )
    )

    model_cfg = (
        config[
            "model"
        ]
    )

    robust_cfg = (
        config[
            "robust_training"
        ]
    )

    model = (
        NativeTileFusionDetector(

            clip_model_name=
                model_cfg[
                    "clip_model"
                ],

            semantic_hidden_dim=
                int(
                    model_cfg[
                        "classifier"
                    ][
                        "hidden_dim"
                    ]
                ),

            semantic_dropout=
                float(
                    model_cfg[
                        "classifier"
                    ][
                        "dropout"
                    ]
                ),

            adapter_bottleneck_dim=
                int(
                    robust_cfg[
                        "adapter"
                    ][
                        "bottleneck_dim"
                    ]
                ),

            adapter_dropout=
                float(
                    robust_cfg[
                        "adapter"
                    ][
                        "dropout"
                    ]
                ),

            forensic_embedding_dim=
                int(
                    tile_cfg[
                        "encoder"
                    ][
                        "embedding_dim"
                    ]
                ),

            forensic_base_channels=
                int(
                    tile_cfg[
                        "encoder"
                    ][
                        "base_channels"
                    ]
                ),

            forensic_dropout=
                float(
                    tile_cfg[
                        "encoder"
                    ][
                        "dropout"
                    ]
                ),

            attention_hidden_dim=
                int(
                    tile_cfg[
                        "attention"
                    ][
                        "hidden_dim"
                    ]
                ),

            attention_dropout=
                float(
                    tile_cfg[
                        "attention"
                    ][
                        "dropout"
                    ]
                ),

            fusion_projection_dim=
                int(
                    tile_cfg[
                        "fusion"
                    ][
                        "projection_dim"
                    ]
                ),

            fusion_hidden_dim=
                int(
                    tile_cfg[
                        "fusion"
                    ][
                        "hidden_dim"
                    ]
                ),

            fusion_dropout=
                float(
                    tile_cfg[
                        "fusion"
                    ][
                        "dropout"
                    ]
                ),
        )
        .to(
            device
        )
    )

    load_native_tile_checkpoint(

        path=(
            "checkpoints/"
            "native_tile_best.pt"
        ),

        model=
            model,

        device=
            device,
    )

    model.eval()

    with torch.no_grad():

        output = (
            model.forward_with_details(

                pixel_values=
                    batch[
                        "pixel_values"
                    ]
                    .to(
                        device
                    ),

                forensic_tiles=
                    batch[
                        "forensic_tiles"
                    ]
                    .to(
                        device
                    ),

                tile_mask=
                    batch[
                        "tile_mask"
                    ]
                    .to(
                        device
                    ),
            )
        )

    probability = (
        torch.sigmoid(
            output[
                "logits"
            ]
        )[0].item()
    )

    semantic_probability = (
        torch.sigmoid(
            output[
                "semantic_logits"
            ]
        )[0].item()
    )

    forensic_probability = (
        torch.sigmoid(
            output[
                "forensic_logits"
            ]
        )[0].item()
    )

    attention = (
        output[
            "attention"
        ][0]
        .cpu()
        .numpy()
    )

    mask = (
        batch[
            "tile_mask"
        ][0]
        .numpy()
    )

    boxes = (
        batch[
            "tile_boxes"
        ][0]
        .numpy()
    )

    image_path = Path(
        batch[
            "image_path"
        ][0]
    )

    image = load_rgb_image(
        image_path
    )

    width, height = (
        image.size
    )

    figure = plt.figure(
        figsize=(
            12,
            8,
        )
    )

    axis = figure.add_subplot(
        1,
        1,
        1,
    )

    axis.imshow(
        image
    )

    valid_weights = (
        attention[
            mask
        ]
    )

    maximum_weight = max(
        valid_weights.max(),
        1e-8,
    )

    tile_number = 0

    for (
        weight,
        box,
        valid,
    ) in zip(
        attention,
        boxes,
        mask,
    ):

        if not valid:
            continue

        tile_number += 1

        left = (
            box[0]
            * width
        )

        top = (
            box[1]
            * height
        )

        right = (
            box[2]
            * width
        )

        bottom = (
            box[3]
            * height
        )

        rectangle = (
            patches.Rectangle(

                (
                    left,
                    top,
                ),

                right - left,

                bottom - top,

                linewidth=(
                    1.5
                    + 5.0
                    * (
                        weight
                        / maximum_weight
                    )
                ),

                fill=False,
            )
        )

        axis.add_patch(
            rectangle
        )

        axis.text(
            left + 4,
            top + 16,

            (
                f"T{tile_number}: "
                f"{weight:.2f}"
            ),

            fontsize=9,

            bbox={
                "facecolor":
                    "white",

                "alpha":
                    0.75,

                "pad":
                    2,
            },
        )

    axis.set_title(
        (
            f"AI probability: "
            f"{probability:.3f}\n"

            f"Semantic: "
            f"{semantic_probability:.3f} | "

            f"Forensic: "
            f"{forensic_probability:.3f}"
        )
    )

    axis.axis(
        "off"
    )

    figure.tight_layout()

    output_path = Path(
        "outputs/figures/"
        "tile_attention.png"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        "\nSaved:"
    )

    print(
        output_path.resolve()
    )

    print(
        "\nAI probability:"
    )

    print(
        probability
    )

    print(
        "\nSemantic probability:"
    )

    print(
        semantic_probability
    )

    print(
        "\nForensic probability:"
    )

    print(
        forensic_probability
    )

    print(
        "\nAttention:"
    )

    print(
        attention[
            mask
        ]
    )


if __name__ == "__main__":
    main()