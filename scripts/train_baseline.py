import argparse

import torch

from torch.optim import (
    AdamW,
)

from torch.utils.data import (
    DataLoader,
)

from src.data.collate import (
    CLIPBatchCollator,
)

from src.data.dataset import (
    AIGCImageDataset,
)

from src.models.baseline_detector import (
    BaselineAIGCDetector,
)

from src.training.trainer import (
    train_model,
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

    parser = argparse.ArgumentParser(
        description=(
            "Train frozen-CLIP baseline "
            "AIGC detector."
        )
    )

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

    collator = (
        CLIPBatchCollator(
            model_name=
                clip_model_name
        )
    )

    train_dataset = (
        AIGCImageDataset(
            manifest_path=
                args.train_manifest,

            transform=None,

            paired_transform=None,

            return_metadata=True,
        )
    )

    val_dataset = (
        AIGCImageDataset(
            manifest_path=
                args.val_manifest,

            transform=None,

            paired_transform=None,

            return_metadata=True,
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

    pin_memory = (
        device.type == "cuda"
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

            pin_memory=
                pin_memory,

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

            pin_memory=
                pin_memory,

            persistent_workers=(
                num_workers > 0
            ),

            collate_fn=
                collator,
        )
    )

    model = (
        BaselineAIGCDetector(
            clip_model_name=
                clip_model_name,

            hidden_dim=
                int(
                    model_config[
                        "classifier"
                    ][
                        "hidden_dim"
                    ]
                ),

            dropout=
                float(
                    model_config[
                        "classifier"
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

    model = model.to(
        device
    )

    parameter_counts = (
        model.count_parameters()
    )

    print(
        "\n=============================="
    )

    print(
        "MODEL"
    )

    print(
        "=============================="
    )

    print(
        f"\nTotal parameters:     "
        f"{parameter_counts['total']:,}"
    )

    print(
        f"Trainable parameters: "
        f"{parameter_counts['trainable']:,}"
    )

    print(
        f"Frozen parameters:    "
        f"{parameter_counts['frozen']:,}"
    )

    trainable_ratio = (
        parameter_counts[
            "trainable"
        ]
        / parameter_counts[
            "total"
        ]
        * 100
    )

    print(
        f"Trainable fraction:   "
        f"{trainable_ratio:.4f}%"
    )

    optimizer = AdamW(
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

    train_model(
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

        epochs=int(
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

        threshold=float(
            evaluation_config[
                "threshold"
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