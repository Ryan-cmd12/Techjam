from __future__ import annotations

import argparse

from src.data.dataset import (
    AIGCImageDataset,
)

from src.data.native_tile_collate import (
    NativeTileCLIPBatchCollator,
)

from src.evaluation.native_tile_robustness import (
    run_native_tile_robustness,
)

from src.evaluation.robustness import (
    build_corruption_specs,
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
    print_device_info,
)


def build_model(
    config,
    device,
):

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

    tile_cfg = (
        config[
            "native_tiles"
        ]
    )

    return (
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


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/base.yaml",
    )

    parser.add_argument(
        "--manifest",
        default=(
            "data/manifests/"
            "sid_test.csv"
        ),
    )

    parser.add_argument(
        "--name",
        default="sid",
    )

    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/"
            "native_tile_best.pt"
        ),
    )

    args = parser.parse_args()

    config = load_config(
        args.config
    )

    device = get_device()

    print_device_info(
        device
    )

    model = build_model(
        config,
        device,
    )

    checkpoint = (
        load_native_tile_checkpoint(

            path=
                args.checkpoint,

            model=
                model,

            device=
                device,
        )
    )

    print(
        f"\nLoaded checkpoint "
        f"epoch: "
        f"{checkpoint['epoch']}"
    )

    dataset = (
        AIGCImageDataset(

            manifest_path=
                args.manifest,

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

            sampling_mode=
                tile_cfg[
                    "evaluation_sampling"
                ],

            seed=
                int(
                    config[
                        "project"
                    ][
                        "seed"
                    ]
                ),
        )
    )

    specs = (
        build_corruption_specs(

            config[
                "robustness"
            ][
                "conditions"
            ]
        )
    )

    run_native_tile_robustness(

        model=
            model,

        base_dataset=
            dataset,

        collator=
            collator,

        specs=
            specs,

        device=
            device,

        batch_size=
            int(
                tile_cfg[
                    "training"
                ][
                    "batch_size"
                ]
            ),

        num_workers=
            int(
                config[
                    "training"
                ][
                    "num_workers"
                ]
            ),

        threshold=
            float(
                config[
                    "evaluation"
                ][
                    "threshold"
                ]
            ),

        seed=
            int(
                config[
                    "project"
                ][
                    "seed"
                ]
            ),

        output_directory=(
            "outputs/evaluation/"
            "native_tile/"
            + args.name
        ),

        output_prefix=
            args.name,
    )


if __name__ == "__main__":
    main()