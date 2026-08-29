from __future__ import annotations

import argparse

import numpy as np

from torch.utils.data import (
    Subset,
)

from src.data.dataset import (
    AIGCImageDataset,
)

from src.data.native_tile_collate import (
    NativeTileCLIPBatchCollator,
)

from src.evaluation.laundering import (
    run_laundering_benchmark,
)

from src.evaluation.laundering_dataset import (
    build_laundering_specs,
)

from src.training.transformation_aware_checkpoint import (
    load_transformation_aware_checkpoint,
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

from scripts.train_transformation_aware import (
    build_model,
)


def build_balanced_subset(
    dataset: AIGCImageDataset,
    max_samples: int | None,
    seed: int,
):

    if (
        max_samples is None
        or max_samples >= len(
            dataset
        )
    ):

        return dataset

    dataframe = dataset.dataframe

    labels = sorted(
        dataframe[
            "label"
        ]
        .unique()
        .tolist()
    )

    rng = (
        np.random.default_rng(
            seed
        )
    )

    samples_per_class = (
        max_samples
        // len(
            labels
        )
    )

    remainder = (
        max_samples
        % len(
            labels
        )
    )

    selected = []

    for label_index, label in enumerate(
        labels
    ):

        indices = (
            dataframe.index[
                dataframe[
                    "label"
                ]
                == label
            ]
            .to_numpy()
        )

        count = (
            samples_per_class
            + (
                1
                if label_index
                < remainder
                else 0
            )
        )

        count = min(
            count,
            len(
                indices
            ),
        )

        chosen = rng.choice(
            indices,
            size=count,
            replace=False,
        )

        selected.extend(
            chosen.tolist()
        )

    rng.shuffle(
        selected
    )

    print(
        f"\nUsing balanced subset: "
        f"{len(selected):,}/"
        f"{len(dataset):,}"
    )

    return Subset(
        dataset,
        selected,
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
            "transformation_aware_best.pt"
        ),
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
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

    # ==================================================
    # MODEL
    # ==================================================

    model = build_model(
        config
    ).to(
        device
    )

    checkpoint = (
        load_transformation_aware_checkpoint(
            path=
                args.checkpoint,
            model=
                model,
            device=
                device,
        )
    )

    print(
        f"\nLoaded epoch: "
        f"{checkpoint['epoch']}"
    )

    # ==================================================
    # DATA
    # ==================================================

    base_dataset = (
        AIGCImageDataset(
            manifest_path=
                args.manifest,
            return_metadata=
                True,
        )
    )

    base_dataset = (
        build_balanced_subset(
            dataset=
                base_dataset,
            max_samples=
                args.max_samples,
            seed=
                seed,
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
                seed,
        )
    )

    specs = (
        build_laundering_specs(
            config[
                "laundering"
            ][
                "pipelines"
            ]
        )
    )

    output_directory = (
        "outputs/evaluation/"
        "laundering/"
        + args.name
    )

    run_laundering_benchmark(
        model=
            model,

        base_dataset=
            base_dataset,

        collator=
            collator,

        specs=
            specs,

        device=
            device,

        batch_size=
            int(
                config[
                    "transformation_aware"
                ][
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
            seed,

        output_directory=
            output_directory,
    )


if __name__ == "__main__":
    main()