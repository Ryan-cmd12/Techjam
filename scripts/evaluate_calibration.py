from __future__ import annotations

import argparse
import json

from pathlib import Path

import numpy as np

from sklearn.metrics import (
    roc_auc_score,
)

from src.calibration.metrics import (
    calibration_metrics,
)

from src.calibration.reliability import (
    ReliabilityCalibrator,
    build_reliability_features,
    sigmoid_numpy,
)

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
    build_corruption_specs,
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

from scripts.fit_calibration import (
    collect_outputs,
)

from scripts.train_transformation_aware import (
    build_model,
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

    device = get_device()

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

    reliability_model = (
        ReliabilityCalibrator.from_dict(

            calibration[
                "reliability"
            ]
        )
    )

    model = build_model(
        config
    ).to(
        device
    )

    load_transformation_aware_checkpoint(

        path=(
            "checkpoints/"
            "transformation_aware_best.pt"
        ),

        model=
            model,

        device=
            device,
    )

    dataset = AIGCImageDataset(

        manifest_path=
            args.manifest,

        return_metadata=
            True,
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

            seed=int(
                config[
                    "project"
                ][
                    "seed"
                ]
            ),
        )
    )

    specs = build_corruption_specs(

        config[
            "calibration"
        ][
            "reliability"
        ][
            "probes"
        ]
    )

    condition_frames = {}

    for spec in specs:

        condition_dataset = (
            CorruptedEvaluationDataset(

                base_dataset=
                    dataset,

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
        )

        condition_frames[
            spec.key
        ] = collect_outputs(

            model=
                model,

            dataset=
                condition_dataset,

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

            condition_name=
                spec.name,
        )

    reliability_data = (
        build_reliability_features(

            condition_frames=
                condition_frames,

            temperature=
                temperature,
        )
    )

    clean_frame = (
        condition_frames[
            "clean"
        ]
    )

    labels = (
        clean_frame[
            "label"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    logits = (
        clean_frame[
            "logit"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    before_probability = (
        sigmoid_numpy(
            logits
        )
    )

    calibrated_probability = (
        sigmoid_numpy(
            logits
            / temperature
        )
    )

    before_metrics = (
        calibration_metrics(
            labels=
                labels,

            probabilities=
                before_probability,
        )
    )

    after_metrics = (
        calibration_metrics(
            labels=
                labels,

            probabilities=
                calibrated_probability,
        )
    )

    reliability_probability = (
        reliability_model.predict_proba(

            reliability_data[
                "features"
            ]
        )
    )

    reliability_targets = (
        reliability_data[
            "robust_correctness"
        ]
    )

    reliability_metrics = (
        calibration_metrics(

            labels=
                reliability_targets,

            probabilities=
                reliability_probability,
        )
    )

    if (
        len(
            np.unique(
                reliability_targets
            )
        )
        == 2
    ):

        reliability_auroc = float(
            roc_auc_score(
                reliability_targets,
                reliability_probability,
            )
        )

    else:

        reliability_auroc = float(
            "nan"
        )

    print(
        "\n========================================"
    )

    print(
        f"{args.name.upper()} CALIBRATION"
    )

    print(
        "========================================"
    )

    print(
        f"\nTemperature: "
        f"{temperature:.4f}"
    )

    print(
        "\nBefore calibration:"
    )

    for key, value in (
        before_metrics.items()
    ):

        print(
            f"{key:12s} "
            f"{value:.6f}"
        )

    print(
        "\nAfter calibration:"
    )

    for key, value in (
        after_metrics.items()
    ):

        print(
            f"{key:12s} "
            f"{value:.6f}"
        )

    print(
        "\nReliability:"
    )

    print(
        f"Robust correctness: "
        f"{reliability_targets.mean():.4f}"
    )

    print(
        f"Reliability AUROC:   "
        f"{reliability_auroc:.4f}"
    )

    print(
        f"Reliability Brier:   "
        f"{reliability_metrics['brier']:.4f}"
    )

    print(
        f"Reliability ECE:     "
        f"{reliability_metrics['ece']:.4f}"
    )

    output_directory = Path(
        "outputs/evaluation/"
        "calibration/"
        + args.name
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        output_directory
        / "calibration_metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "temperature":
                    temperature,

                "before":
                    before_metrics,

                "after":
                    after_metrics,

                "robust_correctness_rate":
                    float(
                        reliability_targets.mean()
                    ),

                "reliability_auroc":
                    reliability_auroc,

                "reliability_metrics":
                    reliability_metrics,
            },

            file,
            indent=4,
        )


if __name__ == "__main__":
    main()