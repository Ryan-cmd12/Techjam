import argparse

from torch.optim import (
    AdamW,
)

from torch.utils.data import (
    DataLoader,
)

from src.augmentations.paired_views import (
    PairedViewTransform,
)

from src.augmentations.pipeline import (
    CorruptionPipeline,
)

from src.data.balanced_sampler import (
    build_dataset_class_balanced_sampler,
)

from src.data.dataset import (
    AIGCImageDataset,
)

from src.data.native_tile_collate import (
    PairedNativeTileCLIPBatchCollator,
)

from src.models.native_tile_fusion_detector import (
    NativeTileFusionDetector,
)

from src.training.native_tile_checkpoint import (
    warm_start_native_tile_model,
)

from src.training.native_tile_trainer import (
    train_native_tile_model,
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

    training_cfg = (
        config[
            "training"
        ]
    )

    clip_name = (
        model_cfg[
            "clip_model"
        ]
    )

    # ==================================================
    # PAIRED AUGMENTATION
    # ==================================================

    train_transform = (
        PairedViewTransform(

            corruption_pipeline=
                CorruptionPipeline(
                    seed=
                        seed
                ),

            clean_probability=
                float(
                    robust_cfg[
                        "clean_probability"
                    ]
                ),
        )
    )

    val_transform = (
        PairedViewTransform(

            corruption_pipeline=
                CorruptionPipeline(
                    seed=
                        seed + 1000
                ),

            clean_probability=
                0.0,
        )
    )

    # ==================================================
    # DATASETS
    # ==================================================

    train_dataset = (
        AIGCImageDataset(

            manifest_path=
                args.train_manifest,

            paired_transform=
                train_transform,

            return_metadata=
                True,
        )
    )

    val_dataset = (
        AIGCImageDataset(

            manifest_path=
                args.val_manifest,

            paired_transform=
                val_transform,

            return_metadata=
                True,
        )
    )

    # ==================================================
    # BALANCED SAMPLER
    # ==================================================

    sampler = (
        build_dataset_class_balanced_sampler(

            dataset=
                train_dataset,

            seed=
                seed,
        )
    )

    # ==================================================
    # COLLATE
    # ==================================================

    train_collator = (
        PairedNativeTileCLIPBatchCollator(

            model_name=
                clip_name,

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
                    "training_sampling"
                ],

            seed=
                seed,
        )
    )

    val_collator = (
        PairedNativeTileCLIPBatchCollator(

            model_name=
                clip_name,

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
                seed + 1000,
        )
    )

    workers = int(
        training_cfg[
            "num_workers"
        ]
    )

    batch_size = int(
        tile_cfg[
            "training"
        ][
            "batch_size"
        ]
    )

    train_loader = (
        DataLoader(

            train_dataset,

            batch_size=
                batch_size,

            sampler=
                sampler,

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
                train_collator,
        )
    )

    val_loader = (
        DataLoader(

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
    )

    # ==================================================
    # MODEL
    # ==================================================

    encoder_cfg = (
        tile_cfg[
            "encoder"
        ]
    )

    attention_cfg = (
        tile_cfg[
            "attention"
        ]
    )

    fusion_cfg = (
        tile_cfg[
            "fusion"
        ]
    )

    model = (
        NativeTileFusionDetector(

            clip_model_name=
                clip_name,

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
                    encoder_cfg[
                        "embedding_dim"
                    ]
                ),

            forensic_base_channels=
                int(
                    encoder_cfg[
                        "base_channels"
                    ]
                ),

            forensic_dropout=
                float(
                    encoder_cfg[
                        "dropout"
                    ]
                ),

            attention_hidden_dim=
                int(
                    attention_cfg[
                        "hidden_dim"
                    ]
                ),

            attention_dropout=
                float(
                    attention_cfg[
                        "dropout"
                    ]
                ),

            fusion_projection_dim=
                int(
                    fusion_cfg[
                        "projection_dim"
                    ]
                ),

            fusion_hidden_dim=
                int(
                    fusion_cfg[
                        "hidden_dim"
                    ]
                ),

            fusion_dropout=
                float(
                    fusion_cfg[
                        "dropout"
                    ]
                ),
        )
        .to(
            device
        )
    )

    # ==================================================
    # WARM START
    # ==================================================

    warm_start_native_tile_model(

        path=
            tile_cfg[
                "warm_start_checkpoint"
            ],

        model=
            model,

        device=
            device,
    )

    counts = (
        model.count_parameters()
    )

    print(
        "\n========================================"
    )

    print(
        "NATIVE TILE FUSION MODEL"
    )

    print(
        "========================================"
    )

    print(
        f"\nTotal:     "
        f"{counts['total']:,}"
    )

    print(
        f"Trainable: "
        f"{counts['trainable']:,}"
    )

    print(
        f"Frozen:    "
        f"{counts['frozen']:,}"
    )

    # ==================================================
    # OPTIMIZER
    # ==================================================

    tile_training_cfg = (
        tile_cfg[
            "training"
        ]
    )

    optimizer = (
        AdamW(

            [
                {
                    "params":
                        list(
                            model.semantic_parameters()
                        ),

                    "lr":
                        float(
                            tile_training_cfg[
                                "semantic_learning_rate"
                            ]
                        ),
                },

                {
                    "params":
                        list(
                            model.forensic_parameters()
                        ),

                    "lr":
                        float(
                            tile_training_cfg[
                                "forensic_learning_rate"
                            ]
                        ),
                },

                {
                    "params":
                        list(
                            model.attention_parameters()
                        ),

                    "lr":
                        float(
                            tile_training_cfg[
                                "attention_learning_rate"
                            ]
                        ),
                },

                {
                    "params":
                        list(
                            model.fusion_parameters()
                        ),

                    "lr":
                        float(
                            tile_training_cfg[
                                "fusion_learning_rate"
                            ]
                        ),
                },
            ],

            weight_decay=
                float(
                    training_cfg[
                        "weight_decay"
                    ]
                ),
        )
    )

    # ==================================================
    # TRAIN
    # ==================================================

    loss_cfg = (
        tile_cfg[
            "loss"
        ]
    )

    train_native_tile_model(

        model=
            model,

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
                tile_training_cfg[
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

        threshold=
            float(
                config[
                    "evaluation"
                ][
                    "threshold"
                ]
            ),

        auxiliary_branch_weight=
            float(
                loss_cfg[
                    "auxiliary_branch_weight"
                ]
            ),

        feature_consistency_weight=
            float(
                loss_cfg[
                    "feature_consistency_weight"
                ]
            ),

        prediction_consistency_weight=
            float(
                loss_cfg[
                    "prediction_consistency_weight"
                ]
            ),

        attention_entropy_weight=
            float(
                loss_cfg[
                    "attention_entropy_weight"
                ]
            ),

        gradient_clip_norm=
            float(
                training_cfg[
                    "gradient_clip_norm"
                ]
            ),

        use_amp=
            bool(
                training_cfg[
                    "use_amp"
                ]
            ),
    )


if __name__ == "__main__":
    main()