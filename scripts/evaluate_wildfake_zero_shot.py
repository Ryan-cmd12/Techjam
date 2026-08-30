from __future__ import annotations

import argparse
import json

from pathlib import Path

import pandas as pd

from src.data.dataset import (
    AIGCImageDataset,
)

from src.data.native_tile_collate import (
    NativeTileCLIPBatchCollator,
)

from src.evaluation.corruption_dataset import (
    CorruptedEvaluationDataset,
)

from src.evaluation.wildfake_generalization import (
    build_family_metrics,
    build_generator_metrics,
    collect_wildfake_predictions,
    summarize_condition,
)

from src.inference.predictor import (
    build_transformation_aware_model,
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
            "wildfake_test.csv"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/"
            "transformation_aware_best.pt"
        ),
    )

    parser.add_argument(
        "--calibration",
        default=(
            "outputs/calibration/"
            "calibration.json"
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

    # ==================================================
    # CALIBRATION
    # ==================================================

    with open(
        args.calibration,
        "r",
        encoding="utf-8",
    ) as file:

        calibration = json.load(
            file
        )

    temperature = float(
        calibration[
            "temperature"
        ]
    )

    print(
        f"\nCalibration temperature: "
        f"{temperature:.4f}"
    )

    # ==================================================
    # MODEL
    # ==================================================

    model = (
        build_transformation_aware_model(
            config
        )
        .to(
            device
        )
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
        f"Loaded detector epoch: "
        f"{checkpoint['epoch']}"
    )

    # ==================================================
    # DATA
    # ==================================================

    manifest = pd.read_csv(
        args.manifest
    )

    base_dataset = (
        AIGCImageDataset(

            manifest_path=
                args.manifest,

            return_metadata=
                True,
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

            seed=
                seed,
        )
    )

    cfg = config[
        "wildfake"
    ]

    conditions = (
        cfg[
            "evaluation"
        ][
            "conditions"
        ]
    )

    output_directory = Path(
        "outputs/evaluation/"
        "wildfake_zero_shot"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_frames = []

    all_generator_metrics = []
    all_family_metrics = []

    summaries = {}

    for condition in conditions:

        key = str(
            condition[
                "key"
            ]
        )

        name = str(
            condition[
                "name"
            ]
        )

        corruption_type = str(
            condition[
                "type"
            ]
        )

        severity = (
            condition.get(
                "severity"
            )
        )

        print(
            "\n========================================"
        )

        print(
            f"WILDFAKE ZERO-SHOT — "
            f"{name}"
        )

        print(
            "========================================"
        )

        dataset = (
            CorruptedEvaluationDataset(

                base_dataset=
                    base_dataset,

                corruption_type=
                    corruption_type,

                severity=
                    severity,

                seed=
                    seed,
            )
        )

        predictions = (
            collect_wildfake_predictions(

                model=
                    model,

                dataset=
                    dataset,

                collator=
                    collator,

                device=
                    device,

                batch_size=int(
                    config[
                        "transformation_aware"
                    ][
                        "training"
                    ][
                        "batch_size"
                    ]
                ),

                num_workers=int(
                    config[
                        "training"
                    ][
                        "num_workers"
                    ]
                ),

                temperature=
                    temperature,

                condition_key=
                    key,

                condition_name=
                    name,
            )
        )

        # Add WildFake hierarchy metadata.
        metadata = (
            manifest[
                [
                    "content_hash",
                    "generation_family",
                    "subcategory",
                    "generator",
                    "source",
                    "wildfake_relative_path",
                ]
            ]
            .drop_duplicates(
                subset=[
                    "content_hash"
                ]
            )
        )

        predictions = (
            predictions.merge(
                metadata,

                on="content_hash",

                how="left",
            )
        )

        generator_metrics = (
            build_generator_metrics(

                dataframe=
                    predictions,

                min_samples=int(
                    cfg.get(
                        "min_generator_samples",
                        25,
                    )
                ),

                seed=
                    seed,
            )
        )

        family_metrics = (
            build_family_metrics(

                dataframe=
                    predictions,

                min_samples=int(
                    cfg.get(
                        "min_generator_samples",
                        25,
                    )
                ),

                seed=
                    seed,
            )
        )

        summary = (
            summarize_condition(

                predictions=
                    predictions,

                generator_metrics=
                    generator_metrics,
            )
        )

        summaries[
            key
        ] = summary

        predictions[
            "condition_key"
        ] = key

        predictions[
            "condition_name"
        ] = name

        generator_metrics[
            "condition_key"
        ] = key

        generator_metrics[
            "condition_name"
        ] = name

        family_metrics[
            "condition_key"
        ] = key

        family_metrics[
            "condition_name"
        ] = name

        prediction_frames.append(
            predictions
        )

        all_generator_metrics.append(
            generator_metrics
        )

        all_family_metrics.append(
            family_metrics
        )

        overall = summary[
            "overall"
        ]

        print(
            "\nOverall:"
        )

        print(
            f"AUROC:        "
            f"{overall['auroc']:.4f}"
        )

        print(
            f"Accuracy:     "
            f"{overall['accuracy']:.4f}"
        )

        print(
            f"F1:           "
            f"{overall['f1']:.4f}"
        )

        print(
            f"Real spec.:   "
            f"{summary['real_specificity']:.4f}"
        )

        print(
            f"Fake recall:  "
            f"{summary['fake_recall']:.4f}"
        )

        if (
            "mean_generator_auroc"
            in summary
        ):

            print(
                f"\nMean generator AUROC: "
                f"{summary['mean_generator_auroc']:.4f}"
            )

            print(
                f"Worst generator: "
                f"{summary['worst_generator']}"
            )

            print(
                f"Worst generator AUROC: "
                f"{summary['worst_generator_auroc']:.4f}"
            )

    # ==================================================
    # SAVE
    # ==================================================

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    generator_metrics = pd.concat(
        all_generator_metrics,
        ignore_index=True,
    )

    family_metrics = pd.concat(
        all_family_metrics,
        ignore_index=True,
    )

    predictions.to_csv(
        output_directory
        / "predictions.csv",

        index=False,
    )

    generator_metrics.to_csv(
        output_directory
        / "generator_metrics.csv",

        index=False,
    )

    family_metrics.to_csv(
        output_directory
        / "family_metrics.csv",

        index=False,
    )

    with (
        output_directory
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summaries,
            file,
            indent=4,
        )

    # ==================================================
    # FINAL TABLE
    # ==================================================

    print(
        "\n========================================"
    )

    print(
        "WILDFAKE DOUBLE-OOD SUMMARY"
    )

    print(
        "========================================"
    )

    for condition in conditions:

        key = condition[
            "key"
        ]

        summary = summaries[
            key
        ]

        print(
            f"\n{condition['name']}"
        )

        print(
            f"Overall AUROC:       "
            f"{summary['overall']['auroc']:.4f}"
        )

        if (
            "mean_generator_auroc"
            in summary
        ):

            print(
                f"Mean generator AUROC:"
                f" {summary['mean_generator_auroc']:.4f}"
            )

            print(
                f"Worst generator:     "
                f"{summary['worst_generator']} "
                f""
                f"({summary['worst_generator_auroc']:.4f})"
            )

    print(
        "\nSaved:"
    )

    print(
        output_directory.resolve()
    )


if __name__ == "__main__":

    main()