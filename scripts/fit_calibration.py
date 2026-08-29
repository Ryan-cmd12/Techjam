from __future__ import annotations

import argparse
import json
import math

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import (
    DataLoader,
    Subset,
)

from tqdm import tqdm

from src.calibration.metrics import (
    calibration_metrics,
)

from src.calibration.reliability import (
    ReliabilityCalibrator,
    build_reliability_features,
)

from src.calibration.temperature_scaling import (
    TemperatureScaler,
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
    print_device_info,
)

from src.utils.seed import (
    seed_everything,
)

from scripts.train_transformation_aware import (
    build_model,
)


def build_balanced_dataset_subset(
    dataset,
    maximum_samples,
    seed,
):

    if (
        maximum_samples is None
        or maximum_samples
        >= len(
            dataset
        )
    ):

        return dataset

    dataframe = (
        dataset.dataframe
    )

    strata = (
        dataframe[
            [
                "dataset",
                "label",
            ]
        ]
        .astype(
            {
                "dataset":
                    str,

                "label":
                    int,
            }
        )
    )

    unique_strata = (
        strata
        .drop_duplicates()
        .itertuples(
            index=False,
            name=None,
        )
    )

    unique_strata = list(
        unique_strata
    )

    rng = np.random.default_rng(
        seed
    )

    per_stratum = max(
        1,
        maximum_samples
        // len(
            unique_strata
        ),
    )

    selected = []

    for (
        dataset_name,
        label,
    ) in unique_strata:

        mask = (
            (
                dataframe[
                    "dataset"
                ].astype(
                    str
                )
                == dataset_name
            )
            & (
                dataframe[
                    "label"
                ].astype(
                    int
                )
                == label
            )
        )

        indices = (
            dataframe.index[
                mask
            ]
            .to_numpy()
        )

        count = min(
            per_stratum,
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

    if (
        len(
            selected
        )
        > maximum_samples
    ):

        selected = selected[
            :maximum_samples
        ]

    print(
        f"\nCalibration subset: "
        f"{len(selected):,}/"
        f"{len(dataset):,}"
    )

    return Subset(
        dataset,
        selected,
    )


def attention_entropy(
    attention,
    mask,
):

    attention = (
        attention.clamp_min(
            1e-8
        )
    )

    entropy = -(
        attention
        * torch.log(
            attention
        )
        * mask.float()
    ).sum(
        dim=1
    )

    counts = (
        mask.sum(
            dim=1
        )
        .float()
    )

    maximum_entropy = (
        torch.log(
            counts.clamp_min(
                1.0
            )
        )
    )

    result = torch.zeros_like(
        entropy
    )

    multi_tile = (
        counts > 1
    )

    result[
        multi_tile
    ] = (
        entropy[
            multi_tile
        ]
        / maximum_entropy[
            multi_tile
        ]
    )

    return result


@torch.no_grad()
def collect_outputs(
    model,
    dataset,
    collator,
    device,
    batch_size,
    num_workers,
    condition_name,
):

    loader = DataLoader(

        dataset,

        batch_size=
            batch_size,

        shuffle=False,

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

    records = []

    model.eval()

    for batch in tqdm(
        loader,
        desc=condition_name,
    ):

        pixel_values = (
            batch[
                "pixel_values"
            ].to(
                device,
                non_blocking=True,
            )
        )

        forensic_tiles = (
            batch[
                "forensic_tiles"
            ].to(
                device,
                non_blocking=True,
            )
        )

        tile_mask = (
            batch[
                "tile_mask"
            ].to(
                device,
                non_blocking=True,
            )
        )

        output = (
            model.forward_with_details(

                pixel_values=
                    pixel_values,

                forensic_tiles=
                    forensic_tiles,

                tile_mask=
                    tile_mask,
            )
        )

        entropy = attention_entropy(
            output[
                "attention"
            ],
            tile_mask,
        )

        semantic_probability = (
            torch.sigmoid(
                output[
                    "semantic_logits"
                ]
            )
        )

        forensic_probability = (
            torch.sigmoid(
                output[
                    "forensic_logits"
                ]
            )
        )

        labels = (
            batch[
                "labels"
            ]
            .cpu()
            .numpy()
        )

        logits = (
            output[
                "logits"
            ]
            .cpu()
            .numpy()
        )

        semantic_probability = (
            semantic_probability
            .cpu()
            .numpy()
        )

        forensic_probability = (
            forensic_probability
            .cpu()
            .numpy()
        )

        semantic_weight = (
            output[
                "semantic_weight"
            ]
            .cpu()
            .numpy()
        )

        forensic_weight = (
            output[
                "forensic_weight"
            ]
            .cpu()
            .numpy()
        )

        severity = (
            output[
                "corruption_severity"
            ]
            .cpu()
            .numpy()
        )

        entropy = (
            entropy
            .cpu()
            .numpy()
        )

        for index in range(
            len(
                labels
            )
        ):

            records.append(
                {
                    "content_hash":
                        batch[
                            "content_hash"
                        ][
                            index
                        ],

                    "image_path":
                        batch[
                            "image_path"
                        ][
                            index
                        ],

                    "label":
                        int(
                            labels[
                                index
                            ]
                        ),

                    "logit":
                        float(
                            logits[
                                index
                            ]
                        ),

                    "semantic_probability":
                        float(
                            semantic_probability[
                                index
                            ]
                        ),

                    "forensic_probability":
                        float(
                            forensic_probability[
                                index
                            ]
                        ),

                    "semantic_weight":
                        float(
                            semantic_weight[
                                index
                            ]
                        ),

                    "forensic_weight":
                        float(
                            forensic_weight[
                                index
                            ]
                        ),

                    "predicted_severity":
                        float(
                            severity[
                                index
                            ]
                        ),

                    "attention_entropy":
                        float(
                            entropy[
                                index
                            ]
                        ),
                }
            )

    return pd.DataFrame(
        records
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
            "unified_val.csv"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/"
            "transformation_aware_best.pt"
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
    # DATASET
    # ==================================================

    base_dataset = (
        AIGCImageDataset(

            manifest_path=
                args.manifest,

            return_metadata=
                True,
        )
    )

    calibration_cfg = (
        config[
            "calibration"
        ]
    )

    max_samples = (
        calibration_cfg[
            "reliability"
        ].get(
            "max_samples"
        )
    )

    base_dataset = (
        build_balanced_dataset_subset(

            dataset=
                base_dataset,

            maximum_samples=
                max_samples,

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

    specs = (
        build_corruption_specs(

            calibration_cfg[
                "reliability"
            ][
                "probes"
            ]
        )
    )

    condition_frames = {}

    for spec in specs:

        condition_dataset = (
            CorruptedEvaluationDataset(

                base_dataset=
                    base_dataset,

                corruption_type=
                    spec.corruption_type,

                severity=
                    spec.severity,

                seed=
                    seed,
            )
        )

        dataframe = collect_outputs(

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

        condition_frames[
            spec.key
        ] = dataframe

    # ==================================================
    # TEMPERATURE FIT
    # ==================================================

    all_logits = []

    all_labels = []

    for dataframe in (
        condition_frames.values()
    ):

        all_logits.extend(
            dataframe[
                "logit"
            ].tolist()
        )

        all_labels.extend(
            dataframe[
                "label"
            ].tolist()
        )

    logits_tensor = torch.tensor(
        all_logits,
        dtype=torch.float32,
    )

    labels_tensor = torch.tensor(
        all_labels,
        dtype=torch.float32,
    )

    uncalibrated_probability = (
        torch.sigmoid(
            logits_tensor
        )
        .numpy()
    )

    before_metrics = (
        calibration_metrics(
            labels=
                all_labels,

            probabilities=
                uncalibrated_probability,
        )
    )

    scaler = (
        TemperatureScaler()
    )

    temperature = scaler.fit(

        logits=
            logits_tensor,

        labels=
            labels_tensor,

        max_iterations=int(
            calibration_cfg[
                "temperature"
            ][
                "max_iterations"
            ]
        ),
    )

    calibrated_probability = (
        scaler.probabilities(
            logits_tensor
        )
        .detach()
        .numpy()
    )

    after_metrics = (
        calibration_metrics(
            labels=
                all_labels,

            probabilities=
                calibrated_probability,
        )
    )

    print(
        "\n========================================"
    )

    print(
        "TEMPERATURE CALIBRATION"
    )

    print(
        "========================================"
    )

    print(
        f"\nTemperature: "
        f"{temperature:.4f}"
    )

    print(
        "\nBefore:"
    )

    for key, value in (
        before_metrics.items()
    ):

        print(
            f"{key:12s} "
            f"{value:.6f}"
        )

    print(
        "\nAfter:"
    )

    for key, value in (
        after_metrics.items()
    ):

        print(
            f"{key:12s} "
            f"{value:.6f}"
        )

    # ==================================================
    # RELIABILITY
    # ==================================================

    reliability_data = (
        build_reliability_features(

            condition_frames=
                condition_frames,

            temperature=
                temperature,
        )
    )

    reliability_model = (
        ReliabilityCalibrator.fit(

            features=
                reliability_data[
                    "features"
                ],

            targets=
                reliability_data[
                    "robust_correctness"
                ],

            feature_names=
                reliability_data[
                    "feature_names"
                ],
        )
    )

    reliability_probability = (
        reliability_model.predict_proba(

            reliability_data[
                "features"
            ]
        )
    )

    reliability_metrics = (
        calibration_metrics(

            labels=
                reliability_data[
                    "robust_correctness"
                ],

            probabilities=
                reliability_probability,
        )
    )

    robust_rate = float(
        reliability_data[
            "robust_correctness"
        ].mean()
    )

    print(
        "\n========================================"
    )

    print(
        "ROBUST RELIABILITY CALIBRATION"
    )

    print(
        "========================================"
    )

    print(
        f"\nProbe-robust correctness rate: "
        f"{robust_rate:.4f}"
    )

    print(
        "\nReliability calibration:"
    )

    for key, value in (
        reliability_metrics.items()
    ):

        print(
            f"{key:12s} "
            f"{value:.6f}"
        )

    # ==================================================
    # FEATURE IMPORTANCE
    # ==================================================

    if (
        reliability_model.coefficients
        is not None
    ):

        print(
            "\nReliability feature weights:"
        )

        pairs = list(
            zip(
                reliability_model.feature_names,
                reliability_model.coefficients,
            )
        )

        pairs.sort(
            key=lambda item:
                abs(
                    item[
                        1
                    ]
                ),
            reverse=True,
        )

        for name, coefficient in pairs:

            print(
                f"{name:25s} "
                f"{coefficient:+.4f}"
            )

    # ==================================================
    # SAVE
    # ==================================================

    output_directory = Path(
        "outputs/calibration"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact = {
        "checkpoint":
            args.checkpoint,

        "temperature":
            temperature,

        "temperature_metrics_before":
            before_metrics,

        "temperature_metrics_after":
            after_metrics,

        "reliability":
            reliability_model.to_dict(),

        "reliability_metrics":
            reliability_metrics,

        "probe_robust_correctness_rate":
            robust_rate,

        "probe_conditions":
            reliability_data[
                "condition_names"
            ],
    }

    artifact_path = (
        output_directory
        / "calibration.json"
    )

    with artifact_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            artifact,
            file,
            indent=4,
        )

    feature_dataframe = pd.DataFrame(

        reliability_data[
            "features"
        ],

        columns=
            reliability_data[
                "feature_names"
            ],
    )

    feature_dataframe[
        "content_hash"
    ] = reliability_data[
        "content_hashes"
    ]

    feature_dataframe[
        "robust_correct"
    ] = reliability_data[
        "robust_correctness"
    ]

    feature_dataframe[
        "reliability"
    ] = reliability_probability

    feature_dataframe.to_csv(

        output_directory
        / "calibration_samples.csv",

        index=False,
    )

    print(
        "\nSaved:"
    )

    print(
        artifact_path
    )


if __name__ == "__main__":
    main()