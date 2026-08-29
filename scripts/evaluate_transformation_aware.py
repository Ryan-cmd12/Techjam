from __future__ import annotations

import argparse
import json

from pathlib import Path

import pandas as pd
import torch

from torch.utils.data import (
    DataLoader,
)

from tqdm import tqdm

from src.data.dataset import (
    AIGCImageDataset,
)

from src.data.native_tile_collate import (
    NativeTileCLIPBatchCollator,
)

from src.evaluation.corruption_dataset import (
    CorruptedEvaluationDataset,
)

from src.evaluation.robustness import (
    add_robustness_columns,
    build_corruption_specs,
    build_summary,
)

from src.training.metrics import (
    compute_binary_metrics,
)

from src.training.transformation_aware_checkpoint import (
    load_transformation_aware_checkpoint,
)

from src.utils.config import (
    load_config,
)

from src.utils.device import (
    get_device,
)

from scripts.train_transformation_aware import (
    build_model,
)


@torch.no_grad()
def evaluate_condition(
    model,
    dataloader,
    device,
    threshold,
    spec,
):

    model.eval()

    labels = []
    probabilities = []

    semantic_weights = []
    forensic_weights = []

    severities = []

    predicted_types = []

    for batch in tqdm(
        dataloader,
        desc=spec.name,
    ):

        output = model.forward_with_details(

            pixel_values=
                batch[
                    "pixel_values"
                ].to(
                    device
                ),

            forensic_tiles=
                batch[
                    "forensic_tiles"
                ].to(
                    device
                ),

            tile_mask=
                batch[
                    "tile_mask"
                ].to(
                    device
                ),
        )

        probabilities.extend(
            torch.sigmoid(
                output[
                    "logits"
                ]
            )
            .cpu()
            .numpy()
            .tolist()
        )

        labels.extend(
            batch[
                "labels"
            ]
            .numpy()
            .tolist()
        )

        semantic_weights.extend(
            output[
                "semantic_weight"
            ]
            .cpu()
            .numpy()
            .tolist()
        )

        forensic_weights.extend(
            output[
                "forensic_weight"
            ]
            .cpu()
            .numpy()
            .tolist()
        )

        severities.extend(
            output[
                "corruption_severity"
            ]
            .cpu()
            .numpy()
            .tolist()
        )

        predicted_types.extend(
            output[
                "corruption_type_probabilities"
            ]
            .argmax(
                dim=-1
            )
            .cpu()
            .numpy()
            .tolist()
        )

    metrics = compute_binary_metrics(
        labels,
        probabilities,
        threshold,
    )

    return {
        **metrics,

        "semantic_weight":
            float(
                sum(
                    semantic_weights
                )
                / len(
                    semantic_weights
                )
            ),

        "forensic_weight":
            float(
                sum(
                    forensic_weights
                )
                / len(
                    forensic_weights
                )
            ),

        "predicted_severity":
            float(
                sum(
                    severities
                )
                / len(
                    severities
                )
            ),
    }


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

    args = parser.parse_args()

    config = load_config(
        args.config
    )

    device = get_device()

    model = build_model(
        config
    ).to(
        device
    )

    load_transformation_aware_checkpoint(

        (
            "checkpoints/"
            "transformation_aware_best.pt"
        ),

        model,
        device,
    )

    base_dataset = AIGCImageDataset(
        args.manifest,
        return_metadata=True,
    )

    tile_cfg = config[
        "native_tiles"
    ]

    collator = NativeTileCLIPBatchCollator(

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

        seed=int(
            config[
                "project"
            ][
                "seed"
            ]
        ),
    )

    specs = build_corruption_specs(
        config[
            "robustness"
        ][
            "conditions"
        ]
    )

    rows = []

    for spec in specs:

        dataset = CorruptedEvaluationDataset(

            base_dataset=
                base_dataset,

            corruption_type=
                spec.corruption_type,

            severity=
                spec.severity,

            seed=int(
                config[
                    "project"
                ][
                    "seed"
                ]
            ),
        )

        loader = DataLoader(

            dataset,

            batch_size=int(
                config[
                    "transformation_aware"
                ][
                    "training"
                ][
                    "batch_size"
                ]
            ),

            shuffle=False,

            num_workers=int(
                config[
                    "training"
                ][
                    "num_workers"
                ]
            ),

            collate_fn=
                collator,
        )

        metrics = evaluate_condition(

            model,
            loader,
            device,

            float(
                config[
                    "evaluation"
                ][
                    "threshold"
                ]
            ),

            spec,
        )

        rows.append(
            {
                "condition_key":
                    spec.key,

                "condition_name":
                    spec.name,

                "corruption_type":
                    spec.corruption_type,

                "severity":
                    spec.severity,

                **metrics,
            }
        )

        print(
            f"\n{spec.name}"
        )

        print(
            f"AUROC:     "
            f"{metrics['auroc']:.4f}"
        )

        print(
            f"Semantic:  "
            f"{metrics['semantic_weight']:.4f}"
        )

        print(
            f"Forensic:  "
            f"{metrics['forensic_weight']:.4f}"
        )

    dataframe = pd.DataFrame(
        rows
    )

    dataframe = add_robustness_columns(
        dataframe
    )

    output_directory = Path(
        "outputs/evaluation/"
        "transformation_aware/"
        + args.name
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(

        output_directory
        / "gating_robustness.csv",

        index=False,
    )

    summary = build_summary(
        dataframe
    )

    with (
        output_directory
        / "gating_summary.json"
    ).open(
        "w"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )


if __name__ == "__main__":
    main()