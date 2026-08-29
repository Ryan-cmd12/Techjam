from __future__ import annotations

import argparse
import json

from pathlib import Path

import pandas as pd
import torch

from torch import nn
from torch.utils.data import (
    DataLoader,
)

from src.data.collate import (
    CLIPBatchCollator,
)

from src.data.dataset import (
    AIGCImageDataset,
)

from src.models.robust_detector import (
    RobustAIGCDetector,
)

from src.training.robust_checkpoint import (
    load_robust_checkpoint,
)

from src.training.trainer import (
    evaluate_model,
)

from src.training.metrics import (
    print_metrics,
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
            "Evaluate CIFAKE-trained "
            "robust detector on unseen "
            "SID_Set / FLUX images."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/base.yaml",
    )

    parser.add_argument(
        "--manifest",
        default=(
            "data/manifests/"
            "sid_val.csv"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/"
            "robust_best.pt"
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

    dataset = AIGCImageDataset(
        manifest_path=
            args.manifest,

        return_metadata=
            True,
    )

    collator = CLIPBatchCollator(
        model_name=
            clip_model_name
    )

    workers = int(
        training_config[
            "num_workers"
        ]
    )

    dataloader = DataLoader(
        dataset,

        batch_size=int(
            training_config[
                "batch_size"
            ]
        ),

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
            collator,
    )

    model = RobustAIGCDetector(
        clip_model_name=
            clip_model_name,

        classifier_hidden_dim=int(
            model_config[
                "classifier"
            ][
                "hidden_dim"
            ]
        ),

        classifier_dropout=float(
            model_config[
                "classifier"
            ][
                "dropout"
            ]
        ),

        adapter_bottleneck_dim=int(
            robust_config[
                "adapter"
            ][
                "bottleneck_dim"
            ]
        ),

        adapter_dropout=float(
            robust_config[
                "adapter"
            ][
                "dropout"
            ]
        ),

        freeze_backbone=
            True,

        normalize_embeddings=bool(
            model_config[
                "normalize_embeddings"
            ]
        ),
    ).to(
        device
    )

    checkpoint = (
        load_robust_checkpoint(
            path=
                args.checkpoint,

            model=
                model,

            device=
                device,
        )
    )

    print(
        "\n========================================"
    )

    print(
        "SID CROSS-DATASET OOD TEST"
    )

    print(
        "========================================"
    )

    print(
        "\nTraining domain:"
    )

    print(
        "CIFAKE"
    )

    print(
        "\nEvaluation domain:"
    )

    print(
        "SID_Set"
    )

    print(
        "\nUnseen synthetic generator:"
    )

    print(
        "FLUX"
    )

    print(
        f"\nCheckpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    (
        loss,
        metrics,
        raw_results,
    ) = evaluate_model(
        model=
            model,

        dataloader=
            dataloader,

        criterion=
            criterion,

        device=
            device,

        threshold=float(
            evaluation_config[
                "threshold"
            ]
        ),

        description=(
            "CIFAKE → SID OOD"
        ),
    )

    print(
        f"\nLoss: "
        f"{loss:.6f}"
    )

    print_metrics(
        metrics,
        title=(
            "CIFAKE → SID_SET OOD"
        ),
    )

    output_directory = Path(
        config[
            "paths"
        ][
            "outputs"
        ]
    ) / "evaluation" / "cross_dataset"

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        output_directory
        / "cifake_to_sid_metrics.json"
    )

    predictions_path = (
        output_directory
        / "cifake_to_sid_predictions.csv"
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "train_dataset":
                    "cifake",

                "test_dataset":
                    "sid_set",

                "unseen_generator":
                    "flux",

                "loss":
                    float(
                        loss
                    ),

                **metrics,
            },

            file,
            indent=4,
        )

    predictions = pd.DataFrame(
        {
            "image_path":
                raw_results[
                    "image_paths"
                ],

            "label":
                raw_results[
                    "labels"
                ].astype(
                    int
                ),

            "pred":
                raw_results[
                    "probabilities"
                ],
        }
    )

    predictions[
        "prediction"
    ] = (
        predictions[
            "pred"
        ]
        >= float(
            evaluation_config[
                "threshold"
            ]
        )
    ).astype(
        int
    )

    predictions.to_csv(
        predictions_path,
        index=False,
    )

    print(
        "\nSaved:"
    )

    print(
        metrics_path
    )

    print(
        predictions_path
    )


if __name__ == "__main__":
    main()