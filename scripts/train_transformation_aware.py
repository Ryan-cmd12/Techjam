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

from src.models.transformation_aware_detector import (
    TransformationAwareDetector,
)

from src.training.transformation_aware_checkpoint import (
    warm_start_transformation_aware_model,
)

from src.training.transformation_aware_trainer import (
    train_transformation_aware_model,
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


def build_model(
    config,
):

    model_cfg = config[
        "model"
    ]

    robust_cfg = config[
        "robust_training"
    ]

    tile_cfg = config[
        "native_tiles"
    ]

    corruption_cfg = config[
        "corruption_estimator"
    ]

    adaptive_cfg = config[
        "transformation_aware"
    ]

    return TransformationAwareDetector(

        clip_model_name=
            model_cfg[
                "clip_model"
            ],

        semantic_hidden_dim=int(
            model_cfg[
                "classifier"
            ][
                "hidden_dim"
            ]
        ),

        semantic_dropout=float(
            model_cfg[
                "classifier"
            ][
                "dropout"
            ]
        ),

        adapter_bottleneck_dim=int(
            robust_cfg[
                "adapter"
            ][
                "bottleneck_dim"
            ]
        ),

        adapter_dropout=float(
            robust_cfg[
                "adapter"
            ][
                "dropout"
            ]
        ),

        forensic_embedding_dim=int(
            tile_cfg[
                "encoder"
            ][
                "embedding_dim"
            ]
        ),

        forensic_base_channels=int(
            tile_cfg[
                "encoder"
            ][
                "base_channels"
            ]
        ),

        forensic_dropout=float(
            tile_cfg[
                "encoder"
            ][
                "dropout"
            ]
        ),

        attention_hidden_dim=int(
            tile_cfg[
                "attention"
            ][
                "hidden_dim"
            ]
        ),

        attention_dropout=float(
            tile_cfg[
                "attention"
            ][
                "dropout"
            ]
        ),

        fusion_projection_dim=int(
            tile_cfg[
                "fusion"
            ][
                "projection_dim"
            ]
        ),

        fusion_hidden_dim=int(
            tile_cfg[
                "fusion"
            ][
                "hidden_dim"
            ]
        ),

        fusion_dropout=float(
            tile_cfg[
                "fusion"
            ][
                "dropout"
            ]
        ),

        corruption_hidden_dim=int(
            corruption_cfg[
                "hidden_dim"
            ]
        ),

        corruption_embedding_dim=int(
            corruption_cfg[
                "embedding_dim"
            ]
        ),

        num_corruption_types=int(
            corruption_cfg[
                "num_types"
            ]
        ),

        corruption_dropout=float(
            corruption_cfg[
                "dropout"
            ]
        ),

        gate_hidden_dim=int(
            adaptive_cfg[
                "gate"
            ][
                "hidden_dim"
            ]
        ),

        gate_dropout=float(
            adaptive_cfg[
                "gate"
            ][
                "dropout"
            ]
        ),

        initial_residual_scale=float(
            adaptive_cfg[
                "gate"
            ][
                "initial_residual_scale"
            ]
        ),
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

    model = build_model(
        config
    ).to(
        device
    )

    adaptive_cfg = config[
        "transformation_aware"
    ]

    warm_start_transformation_aware_model(

        model=
            model,

        native_tile_checkpoint=
            adaptive_cfg[
                "native_tile_checkpoint"
            ],

        corruption_checkpoint=
            adaptive_cfg[
                "corruption_checkpoint"
            ],

        device=
            device,

        freeze_corruption_estimator=
            bool(
                adaptive_cfg[
                    "freeze_corruption_estimator"
                ]
            ),
    )

    # ==================================================
    # DATA
    # ==================================================

    train_transform = (
        PairedViewTransform(

            corruption_pipeline=
                CorruptionPipeline(
                    seed=seed
                ),

            clean_probability=float(
                config[
                    "robust_training"
                ][
                    "clean_probability"
                ]
            ),
        )
    )

    val_transform = (
        PairedViewTransform(

            corruption_pipeline=
                CorruptionPipeline(
                    seed=seed + 1000
                ),

            clean_probability=0.0,
        )
    )

    train_dataset = (
        AIGCImageDataset(

            args.train_manifest,

            paired_transform=
                train_transform,

            return_metadata=True,
        )
    )

    val_dataset = (
        AIGCImageDataset(

            args.val_manifest,

            paired_transform=
                val_transform,

            return_metadata=True,
        )
    )

    sampler = (
        build_dataset_class_balanced_sampler(
            train_dataset,
            seed,
        )
    )

    tile_cfg = config[
        "native_tiles"
    ]

    train_collator = (
        PairedNativeTileCLIPBatchCollator(

            model_name=
                config[
                    "model"
                ][
                    "clip_model"
                ],

            tile_size=int(
                tile_cfg[
                    "tile_size"
                ]
            ),

            max_tiles=int(
                tile_cfg[
                    "max_tiles"
                ]
            ),

            feature_map_size=int(
                tile_cfg[
                    "feature_map_size"
                ]
            ),

            sampling_mode=
                tile_cfg[
                    "training_sampling"
                ],

            seed=seed,
        )
    )

    val_collator = (
        PairedNativeTileCLIPBatchCollator(

            model_name=
                config[
                    "model"
                ][
                    "clip_model"
                ],

            tile_size=int(
                tile_cfg[
                    "tile_size"
                ]
            ),

            max_tiles=int(
                tile_cfg[
                    "max_tiles"
                ]
            ),

            feature_map_size=int(
                tile_cfg[
                    "feature_map_size"
                ]
            ),

            sampling_mode=
                tile_cfg[
                    "evaluation_sampling"
                ],

            seed=seed + 1000,
        )
    )

    batch_size = int(
        adaptive_cfg[
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

        sampler=
            sampler,

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

        shuffle=False,

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
    # OPTIMIZER
    # ==================================================

    training_cfg = adaptive_cfg[
        "training"
    ]

    optimizer = AdamW(
        [
            {
                "params":
                    list(
                        model.semantic_parameters()
                    ),

                "lr":
                    float(
                        training_cfg[
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
                        training_cfg[
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
                        training_cfg[
                            "attention_learning_rate"
                        ]
                    ),
            },

            {
                "params":
                    list(
                        model.gate_parameters()
                    ),

                "lr":
                    float(
                        training_cfg[
                            "gate_learning_rate"
                        ]
                    ),
            },

            {
                "params":
                    list(
                        model.adaptive_parameters()
                    ),

                "lr":
                    float(
                        training_cfg[
                            "adaptive_learning_rate"
                        ]
                    ),
            },
        ],

        weight_decay=float(
            config[
                "training"
            ][
                "weight_decay"
            ]
        ),
    )

    loss_cfg = adaptive_cfg[
        "loss"
    ]

    train_transformation_aware_model(

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

        epochs=int(
            training_cfg[
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

        threshold=float(
            config[
                "evaluation"
            ][
                "threshold"
            ]
        ),

        auxiliary_branch_weight=float(
            loss_cfg[
                "auxiliary_branch_weight"
            ]
        ),

        feature_consistency_weight=float(
            loss_cfg[
                "feature_consistency_weight"
            ]
        ),

        prediction_consistency_weight=float(
            loss_cfg[
                "prediction_consistency_weight"
            ]
        ),

        gate_reliability_weight=float(
            loss_cfg[
                "gate_reliability_weight"
            ]
        ),

        gate_target_temperature=float(
            loss_cfg[
                "gate_target_temperature"
            ]
        ),

        gradient_clip_norm=float(
            config[
                "training"
            ][
                "gradient_clip_norm"
            ]
        ),

        use_amp=bool(
            config[
                "training"
            ][
                "use_amp"
            ]
        ),
    )


if __name__ == "__main__":
    main()