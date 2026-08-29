import argparse

from torch.optim import (
    AdamW,
)

from torch.utils.data import (
    DataLoader,
)

from src.data.corruption_collate import (
    CorruptionNativeTileCLIPBatchCollator,
)

from src.data.corruption_training_dataset import (
    CorruptionTrainingDataset,
)

from src.data.dataset import (
    AIGCImageDataset,
)

from src.models.corruption_estimator import (
    CorruptionEstimator,
)

from src.models.native_tile_fusion_detector import (
    NativeTileFusionDetector,
)

from src.training.corruption_trainer import (
    train_corruption_estimator,
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

from src.utils.seed import (
    seed_everything,
)


def build_detector(
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

    detector = (
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

    return detector


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/base.yaml",
    )

    parser.add_argument(
        "--train-manifest",
        default=(
            "data/manifests/"
            "unified_train.csv"
        ),
    )

    parser.add_argument(
        "--val-manifest",
        default=(
            "data/manifests/"
            "unified_val.csv"
        ),
    )

    args = parser.parse_args()

    config = load_config(
        args.config
    )

    seed = int(
        config[
            "project"
        ][
            "seed"
        ]
    )

    seed_everything(
        seed
    )

    device = get_device()

    print_device_info(
        device
    )

    estimator_cfg = (
        config[
            "corruption_estimator"
        ]
    )

    tile_cfg = (
        config[
            "native_tiles"
        ]
    )

    # ==================================================
    # BASE DETECTOR
    # ==================================================

    detector = build_detector(
        config,
        device,
    )

    load_native_tile_checkpoint(

        path=
            estimator_cfg[
                "base_checkpoint"
            ],

        model=
            detector,

        device=
            device,
    )

    # ==================================================
    # DATASETS
    # ==================================================

    base_train = (
        AIGCImageDataset(

            manifest_path=
                args.train_manifest,

            return_metadata=
                True,
        )
    )

    base_val = (
        AIGCImageDataset(

            manifest_path=
                args.val_manifest,

            return_metadata=
                True,
        )
    )

    train_dataset = (
        CorruptionTrainingDataset(

            base_dataset=
                base_train,

            seed=
                seed,

            clean_probability=
                float(
                    estimator_cfg[
                        "clean_probability"
                    ]
                ),
        )
    )

    val_dataset = (
        CorruptionTrainingDataset(

            base_dataset=
                base_val,

            seed=
                seed + 1000,

            clean_probability=
                float(
                    estimator_cfg[
                        "clean_probability"
                    ]
                ),
        )
    )

    # ==================================================
    # COLLATE
    # ==================================================

    train_collator = (
        CorruptionNativeTileCLIPBatchCollator(

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

            sampling_mode="random",

            seed=
                seed,
        )
    )

    val_collator = (
        CorruptionNativeTileCLIPBatchCollator(

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

            seed=
                seed + 1000,
        )
    )

    batch_size = int(
        estimator_cfg[
            "training"
        ][
            "batch_size"
        ]
    )

    workers = int(
        config[
            "training"
        ][
            "num_workers"
        ]
    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=
            batch_size,

        shuffle=
            True,

        num_workers=
            workers,

        pin_memory=(
            device.type
            == "cuda"
        ),

        persistent_workers=(
            workers > 0
        ),

        collate_fn=
            train_collator,
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=
            batch_size,

        shuffle=
            False,

        num_workers=
            workers,

        pin_memory=(
            device.type
            == "cuda"
        ),

        persistent_workers=(
            workers > 0
        ),

        collate_fn=
            val_collator,
    )

    # ==================================================
    # ESTIMATOR
    # ==================================================

    estimator = (
        CorruptionEstimator(

            semantic_dim=
                detector.backbone
                .feature_dim,

            forensic_dim=
                int(
                    tile_cfg[
                        "encoder"
                    ][
                        "embedding_dim"
                    ]
                ),

            hidden_dim=
                int(
                    estimator_cfg[
                        "hidden_dim"
                    ]
                ),

            embedding_dim=
                int(
                    estimator_cfg[
                        "embedding_dim"
                    ]
                ),

            num_types=
                int(
                    estimator_cfg[
                        "num_types"
                    ]
                ),

            dropout=
                float(
                    estimator_cfg[
                        "dropout"
                    ]
                ),
        )
        .to(
            device
        )
    )

    optimizer = AdamW(

        estimator.parameters(),

        lr=float(
            estimator_cfg[
                "training"
            ][
                "learning_rate"
            ]
        ),

        weight_decay=float(
            estimator_cfg[
                "training"
            ][
                "weight_decay"
            ]
        ),
    )

    loss_cfg = (
        estimator_cfg[
            "loss"
        ]
    )

    train_corruption_estimator(

        detector=
            detector,

        estimator=
            estimator,

        train_dataloader=
            train_loader,

        val_dataloader=
            val_loader,

        optimizer=
            optimizer,

        device=
            device,

        epochs=
            int(
                estimator_cfg[
                    "training"
                ][
                    "epochs"
                ]
            ),

        checkpoint_directory=
            config[
                "paths"
            ][
                "checkpoints"
            ],

        config=
            config,

        type_weight=
            float(
                loss_cfg[
                    "type_weight"
                ]
            ),

        severity_weight=
            float(
                loss_cfg[
                    "severity_weight"
                ]
            ),
    )


if __name__ == "__main__":
    main()