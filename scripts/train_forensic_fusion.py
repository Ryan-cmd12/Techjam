import argparse

from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.augmentations.paired_views import (
    PairedViewTransform,
)

from src.augmentations.pipeline import (
    CorruptionPipeline,
)

from src.data.dataset import (
    AIGCImageDataset,
)

from src.data.multisignal_collate import (
    PairedMultiSignalCLIPBatchCollator,
)

from src.models.forensic_fusion_detector import (
    ForensicFusionDetector,
)

from src.training.forensic_checkpoint import (
    warm_start_from_robust_checkpoint,
)

from src.training.forensic_trainer import (
    train_forensic_model,
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
            "cifake_train.csv"
        ),
    )

    parser.add_argument(
        "--val-manifest",
        default=(
            "data/manifests/"
            "cifake_val.csv"
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

    model_cfg = config[
        "model"
    ]

    robust_cfg = config[
        "robust_training"
    ]

    forensic_cfg = config[
        "forensic_fusion"
    ]

    train_cfg = config[
        "training"
    ]

    clip_name = model_cfg[
        "clip_model"
    ]

    train_transform = (
        PairedViewTransform(
            corruption_pipeline=
                CorruptionPipeline(
                    seed=seed
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

    collator = (
        PairedMultiSignalCLIPBatchCollator(
            clip_name
        )
    )

    workers = int(
        train_cfg[
            "num_workers"
        ]
    )

    train_loader = DataLoader(
        train_dataset,

        batch_size=int(
            train_cfg[
                "batch_size"
            ]
        ),

        shuffle=True,

        num_workers=workers,

        pin_memory=(
            device.type == "cuda"
        ),

        persistent_workers=(
            workers > 0
        ),

        collate_fn=
            collator,
    )

    val_loader = DataLoader(
        val_dataset,

        batch_size=int(
            train_cfg[
                "batch_size"
            ]
        ),

        shuffle=False,

        num_workers=workers,

        pin_memory=(
            device.type == "cuda"
        ),

        persistent_workers=(
            workers > 0
        ),

        collate_fn=
            collator,
    )

    model = ForensicFusionDetector(
        clip_model_name=
            clip_name,

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
            forensic_cfg[
                "forensic"
            ][
                "embedding_dim"
            ]
        ),

        forensic_base_channels=int(
            forensic_cfg[
                "forensic"
            ][
                "base_channels"
            ]
        ),

        forensic_dropout=float(
            forensic_cfg[
                "forensic"
            ][
                "dropout"
            ]
        ),

        fusion_projection_dim=int(
            forensic_cfg[
                "fusion"
            ][
                "projection_dim"
            ]
        ),

        fusion_hidden_dim=int(
            forensic_cfg[
                "fusion"
            ][
                "hidden_dim"
            ]
        ),

        fusion_dropout=float(
            forensic_cfg[
                "fusion"
            ][
                "dropout"
            ]
        ),
    ).to(
        device
    )

    warm_start_from_robust_checkpoint(
        forensic_cfg[
            "warm_start_checkpoint"
        ],
        model,
        device,
    )

    counts = model.count_parameters()

    print(
        "\n=============================="
    )

    print(
        "FORENSIC FUSION MODEL"
    )

    print(
        "=============================="
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

    semantic_lr = float(
        forensic_cfg[
            "training"
        ][
            "semantic_learning_rate"
        ]
    )

    new_lr = float(
        forensic_cfg[
            "training"
        ][
            "new_branch_learning_rate"
        ]
    )

    optimizer = AdamW(
        [
            {
                "params":
                    list(
                        model.semantic_parameters()
                    ),

                "lr":
                    semantic_lr,
            },
            {
                "params":
                    list(
                        model.new_branch_parameters()
                    ),

                "lr":
                    new_lr,
            },
        ],

        weight_decay=float(
            train_cfg[
                "weight_decay"
            ]
        ),
    )

    loss_cfg = forensic_cfg[
        "loss"
    ]

    train_forensic_model(
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
            train_cfg[
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

        gradient_clip_norm=float(
            train_cfg[
                "gradient_clip_norm"
            ]
        ),

        use_amp=bool(
            train_cfg[
                "use_amp"
            ]
        ),
    )


if __name__ == "__main__":
    main()