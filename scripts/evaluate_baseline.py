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

from src.models.baseline_detector import (
    BaselineAIGCDetector,
)

from src.training.checkpoint import (
    load_baseline_checkpoint,
)

from src.training.metrics import (
    print_metrics,
)

from src.training.trainer import (
    evaluate_model,
)

from src.utils.config import (
    load_config,
)

from src.utils.device import (
    get_device,
    print_device_info,
)


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the clean CLIP "
            "baseline on CIFAKE test."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
    )

    parser.add_argument(
        "--manifest",
        type=str,
        default=(
            "data/manifests/"
            "cifake_test.csv"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "checkpoints/"
            "baseline_best.pt"
        ),
    )

    args = parser.parse_args()

    config = (
        load_config(
            args.config
        )
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

    dataset = (
        AIGCImageDataset(
            manifest_path=
                args.manifest,

            return_metadata=
                True,
        )
    )

    collator = (
        CLIPBatchCollator(
            model_name=
                clip_model_name
        )
    )

    dataloader = (
        DataLoader(
            dataset,

            batch_size=
                int(
                    training_config[
                        "batch_size"
                    ]
                ),

            shuffle=
                False,

            num_workers=
                int(
                    training_config[
                        "num_workers"
                    ]
                ),

            pin_memory=(
                device.type
                == "cuda"
            ),

            persistent_workers=(
                int(
                    training_config[
                        "num_workers"
                    ]
                )
                > 0
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
                True,

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

    checkpoint = (
        load_baseline_checkpoint(
            path=
                args.checkpoint,

            model=
                model,

            device=
                device,
        )
    )

    print(
        "\nLoaded checkpoint:"
    )

    print(
        args.checkpoint
    )

    print(
        f"\nCheckpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    (
        test_loss,
        metrics,
        raw_results,
    ) = (
        evaluate_model(
            model=
                model,

            dataloader=
                dataloader,

            criterion=
                criterion,

            device=
                device,

            threshold=
                float(
                    evaluation_config[
                        "threshold"
                    ]
                ),

            description=
                "Clean CIFAKE Test",
        )
    )

    print(
        f"\nTest loss: "
        f"{test_loss:.6f}"
    )

    print_metrics(
        metrics,
        title=(
            "Clean CIFAKE Test"
        ),
    )

    output_directory = Path(
        config[
            "paths"
        ][
            "evaluation"
        ]
        if "evaluation"
        in config[
            "paths"
        ]
        else (
            Path(
                config[
                    "paths"
                ][
                    "outputs"
                ]
            )
            / "evaluation"
        )
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_output = (
        output_directory
        / "baseline_clean_metrics.json"
    )

    with metrics_output.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "loss":
                    test_loss,

                **metrics,
            },
            file,
            indent=4,
        )

    prediction_dataframe = (
        pd.DataFrame(
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
    )

    prediction_dataframe[
        "prediction"
    ] = (
        prediction_dataframe[
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

    prediction_output = (
        output_directory
        / "baseline_clean_predictions.csv"
    )

    prediction_dataframe.to_csv(
        prediction_output,
        index=False,
    )

    print(
        "\nSaved metrics:"
    )

    print(
        metrics_output
    )

    print(
        "\nSaved predictions:"
    )

    print(
        prediction_output
    )


if __name__ == "__main__":
    main()