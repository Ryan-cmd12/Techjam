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

from src.data.dataset import (
    AIGCImageDataset,
)

from src.data.paired_collate import (
    PairedCLIPBatchCollator,
)

from src.models.robust_detector import (
    RobustAIGCDetector,
)

from src.training.robust_trainer import (
    train_robust_model,
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
        type=str,
        default="configs/base.yaml",
    )

    parser.add_argument(
        "--train-manifest",
        type=str,
        default=(
            "data/manifests/"
            "cifake_train.csv"
        ),
    )

    parser.add_argument(
        "--val-manifest",
        type=str,
        default=(
            "data/manifests/"
            "cifake_val.csv"
        ),
    )

    args = parser.parse_args()

    config = (
        load_config(
            args.config
        )
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

    device = (
        get_device()
    )

    print_device_info(
        device
    )

    model_config = (
        config[
            "model"
        ]
    )

    robust_config = (
        config[
            "robust_training"
        ]
    )

    training_config = (
        config[
            "training"
        ]
    )

    evaluation_config = (
        config[
            "evaluation"
        ]
    )

    clip_model_name = (
        model_config[
            "clip_model"
        ]
    )

    # Different seeds so train/validation don't
    # share the exact same corruption RNG stream.
    training_pipeline = (
        CorruptionPipeline(
            seed=seed
        )
    )

    validation_pipeline = (
        CorruptionPipeline(
            seed=seed + 1000
        )
    )

    training_paired_transform = (
        PairedViewTransform(
            corruption_pipeline=
                training_pipeline,

            clean_transform=None,

            corrupted_transform=None,

            clean_probability=
                float(
                    robust_config[
                        "clean_probability"
                    ]
                ),
        )
    )

    validation_paired_transform = (
        PairedViewTransform(
            corruption_pipeline=
                validation_pipeline,

            clean_transform=None,

            corrupted_transform=None,

            clean_probability=
                0.0,
        )
    )

    train_dataset = (
        AIGCImageDataset(
            manifest_path=
                args.train_manifest,

            paired_transform=
                training_paired_transform,

            return_metadata=
                True,
        )
    )

    val_dataset = (
        AIGCImageDataset(
            manifest_path=
                args.val_manifest,

            paired_transform=
                validation_paired_transform,

            return_metadata=
                True,
        )
    )

    collator = (
        PairedCLIPBatchCollator(
            model_name=
                clip_model_name
        )
    )

    batch_size = int(
        training_config[
            "batch_size"
        ]
    )

    num_workers = int(
        training_config[
            "num_workers"
        ]
    )

    train_dataloader = (
        DataLoader(
            train_dataset,

            batch_size=
                batch_size,

            shuffle=
                True,

            num_workers=
                num_workers,

            pin_memory=(
                device.type
                == "cuda"
            ),

            persistent_workers=(
                num_workers > 0
            ),

            collate_fn=
                collator,
        )
    )

    val_dataloader = (
        DataLoader(
            val_dataset,

            batch_size=
                batch_size,

            shuffle=
                False,

            num_workers=
                num_workers,

            pin_memory=(
                device.type
                == "cuda"
            ),

            persistent_workers=(
                num_workers > 0
            ),

            collate_fn=
                collator,
        )
    )

    model = (
        RobustAIGCDetector(
            clip_model_name=
                clip_model_name,

            classifier_hidden_dim=
                int(
                    model_config[
                        "classifier"
                    ][
                        "hidden_dim"
                    ]
                ),

            classifier_dropout=
                float(
                    model_config[
                        "classifier"
                    ][
                        "dropout"
                    ]
                ),

            adapter_bottleneck_dim=
                int(
                    robust_config[
                        "adapter"
                    ][
                        "bottleneck_dim"
                    ]
                ),

            adapter_dropout=
                float(
                    robust_config[
                        "adapter"
                    ][
                        "dropout"
                    ]
                ),

            freeze_backbone=
                bool(
                    model_config[
                        "freeze_backbone"
                    ]
                ),

            normalize_embeddings=
                bool(
                    model_config[
                        "normalize_embeddings"
                    ]
                ),
        )
    )

    model = (
        model.to(
            device
        )
    )

    counts = (
        model.count_parameters()
    )

    print(
        "\n=============================="
    )

    print(
        "ROBUST MODEL"
    )

    print(
        "=============================="
    )

    print(
        f"\nTotal parameters: "
        f"{counts['total']:,}"
    )

    print(
        f"Trainable: "
        f"{counts['trainable']:,}"
    )

    print(
        f"Frozen: "
        f"{counts['frozen']:,}"
    )

    print(
        f"Trainable fraction: "
        f"{100 * counts['trainable'] / counts['total']:.4f}%"
    )

    optimizer = (
        AdamW(
            model.get_trainable_parameters(),

            lr=float(
                training_config[
                    "learning_rate"
                ]
            ),

            weight_decay=float(
                training_config[
                    "weight_decay"
                ]
            ),
        )
    )

    train_robust_model(
        model=
            model,

        train_dataloader=
            train_dataloader,

        val_dataloader=
            val_dataloader,

        optimizer=
            optimizer,

        device=
            device,

        epochs=
            int(
                training_config[
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
                evaluation_config[
                    "threshold"
                ]
            ),

        feature_consistency_weight=
            float(
                robust_config[
                    "loss"
                ][
                    "feature_consistency_weight"
                ]
            ),

        prediction_consistency_weight=
            float(
                robust_config[
                    "loss"
                ][
                    "prediction_consistency_weight"
                ]
            ),

        gradient_clip_norm=
            float(
                training_config[
                    "gradient_clip_norm"
                ]
            ),

        use_amp=
            bool(
                training_config[
                    "use_amp"
                ]
            ),
    )


if __name__ == "__main__":
    main()