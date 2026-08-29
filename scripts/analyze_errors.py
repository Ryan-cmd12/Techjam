from __future__ import annotations

import argparse
import json

from pathlib import Path

import numpy as np
import pandas as pd

from torch.utils.data import (
    Subset,
)

from src.calibration.reliability import (
    ReliabilityCalibrator,
    build_reliability_features,
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

from src.evaluation.error_analysis import (
    attach_reliability,
    build_condition_summary,
    build_group_summary,
    build_image_stability,
    build_report,
    collect_diagnostic_outputs,
    copy_example_images,
    create_contact_sheet,
    select_error_categories,
)

from src.evaluation.laundering_dataset import (
    LaunderingEvaluationDataset,
    build_laundering_specs,
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


# ============================================================
# BALANCED SUBSET
# ============================================================


def build_balanced_subset(
    dataset,
    max_samples,
    seed,
):
    """
    Build a deterministic, class-balanced analysis subset
    while guaranteeing that each SHA256 content hash appears
    at most once.

    Exact duplicate files should never be counted twice in
    reliability/stability analysis because content_hash is
    used as the image identity across corruption conditions.
    """

    dataframe = (
        dataset.dataframe
        .reset_index(
            drop=True
        )
    )

    # ==================================================
    # REMOVE EXACT DUPLICATES FIRST
    # ==================================================

    duplicate_mask = (
        dataframe[
            "content_hash"
        ]
        .astype(
            str
        )
        .duplicated(
            keep="first"
        )
    )

    duplicate_count = int(
        duplicate_mask.sum()
    )

    if duplicate_count > 0:

        print(
            "\n[DATA CLEANUP]"
        )

        print(
            f"Removing "
            f"{duplicate_count:,} exact "
            f"duplicate image rows before "
            f"error analysis."
        )

    # --------------------------------------------------
    # Check that duplicate images do not have
    # contradictory labels.
    # --------------------------------------------------

    label_counts = (
        dataframe
        .groupby(
            "content_hash"
        )[
            "label"
        ]
        .nunique()
    )

    conflicting_hashes = (
        label_counts[
            label_counts > 1
        ]
        .index
        .tolist()
    )

    if conflicting_hashes:

        raise RuntimeError(
            "Exact duplicate images have "
            "conflicting labels in the "
            "analysis manifest.\n"
            f"Conflicting hashes: "
            f"{len(conflicting_hashes):,}"
        )

    # These are POSITIONAL indices into the original
    # AIGCImageDataset.
    unique_positions = (
        np.flatnonzero(
            ~duplicate_mask
            .to_numpy()
        )
    )

    unique_dataframe = (
        dataframe.iloc[
            unique_positions
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    print(
        f"\nUnique analysis images: "
        f"{len(unique_dataframe):,}/"
        f"{len(dataframe):,}"
    )

    # ==================================================
    # IF NO SUBSAMPLING IS REQUIRED
    # ==================================================

    if (
        max_samples is None
        or max_samples
        >= len(
            unique_dataframe
        )
    ):

        return Subset(
            dataset,
            unique_positions.tolist(),
        )

    # ==================================================
    # BALANCE BY LABEL
    # ==================================================

    labels = sorted(
        unique_dataframe[
            "label"
        ]
        .astype(
            int
        )
        .unique()
        .tolist()
    )

    rng = np.random.default_rng(
        seed
    )

    samples_per_label = (
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

    selected_unique_positions = []

    for label_index, label in enumerate(
        labels
    ):

        candidate_positions = (
            np.flatnonzero(
                (
                    unique_dataframe[
                        "label"
                    ]
                    .astype(
                        int
                    )
                    .to_numpy()
                )
                == label
            )
        )

        requested_count = (
            samples_per_label
            + (
                1
                if label_index
                < remainder
                else 0
            )
        )

        requested_count = min(
            requested_count,
            len(
                candidate_positions
            ),
        )

        chosen = rng.choice(
            candidate_positions,
            size=
                requested_count,
            replace=False,
        )

        selected_unique_positions.extend(
            chosen.tolist()
        )

    rng.shuffle(
        selected_unique_positions
    )

    # Convert positions in unique_dataframe back into
    # positions in the original dataset.
    original_positions = [
        int(
            unique_positions[
                position
            ]
        )

        for position
        in selected_unique_positions
    ]

    print(
        f"Error analysis subset: "
        f"{len(original_positions):,}/"
        f"{len(unique_dataframe):,} "
        f"unique images"
    )

    return Subset(
        dataset,
        original_positions,
    )


# ============================================================
# MAIN
# ============================================================


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
        "--calibration",
        default=(
            "outputs/calibration/"
            "calibration.json"
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

    analysis_cfg = (
        config[
            "error_analysis"
        ]
    )

    # ==================================================
    # CALIBRATION
    # ==================================================

    calibration_path = Path(
        args.calibration
    )

    if calibration_path.exists():

        with calibration_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            calibration_payload = (
                json.load(
                    file
                )
            )

        temperature = float(
            calibration_payload[
                "temperature"
            ]
        )

        reliability_model = (
            ReliabilityCalibrator.from_dict(

                calibration_payload[
                    "reliability"
                ]
            )
        )

        print(
            f"\nLoaded calibration."
        )

        print(
            f"Temperature: "
            f"{temperature:.4f}"
        )

    else:

        temperature = 1.0
        reliability_model = None

        print(
            "\n[WARNING] "
            "Calibration file not found."
        )

        print(
            "Reliability analysis "
            "will be skipped."
        )

    # ==================================================
    # MODEL
    # ==================================================

    model = (
        build_model(
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
        f"\nLoaded detector epoch: "
        f"{checkpoint['epoch']}"
    )

    # ==================================================
    # BASE DATASET
    # ==================================================

    base_dataset = (
        AIGCImageDataset(

            manifest_path=
                args.manifest,

            return_metadata=True,
        )
    )

    requested_max = (
        args.max_samples

        if args.max_samples
        is not None

        else analysis_cfg.get(
            "max_samples"
        )
    )

    base_dataset = (
        build_balanced_subset(

            dataset=
                base_dataset,

            max_samples=
                requested_max,

            seed=
                seed,
        )
    )

    # ==================================================
    # COLLATOR
    # ==================================================

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

    batch_size = int(
        config[
            "transformation_aware"
        ][
            "training"
        ][
            "batch_size"
        ]
    )

    num_workers = int(
        config[
            "training"
        ][
            "num_workers"
        ]
    )

    # ==================================================
    # RELIABILITY PROBES
    # ==================================================

    probe_frames = {}

    probe_specs = (
        build_corruption_specs(

            config[
                "calibration"
            ][
                "reliability"
            ][
                "probes"
            ]
        )
    )

    print(
        "\n========================================"
    )

    print(
        "RELIABILITY PROBES"
    )

    print(
        "========================================"
    )

    for spec in probe_specs:

        probe_dataset = (
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

        dataframe = (
            collect_diagnostic_outputs(

                model=
                    model,

                dataset=
                    probe_dataset,

                collator=
                    collator,

                device=
                    device,

                batch_size=
                    batch_size,

                num_workers=
                    num_workers,

                condition_key=
                    spec.key,

                condition_name=
                    spec.name,

                temperature=
                    temperature,
            )
        )

        probe_frames[
            spec.key
        ] = dataframe

    # ==================================================
    # RELIABILITY SCORE
    # ==================================================

    reliability_hashes = []
    reliability_probabilities = []

    if (
        reliability_model
        is not None
    ):

        reliability_data = (
            build_reliability_features(

                condition_frames=
                    probe_frames,

                temperature=
                    temperature,
            )
        )

        reliability_hashes = (
            reliability_data[
                "content_hashes"
            ]
        )

        reliability_probabilities = (
            reliability_model
            .predict_proba(

                reliability_data[
                    "features"
                ]
            )
        )

    # ==================================================
    # LAUNDERING SUITE
    # ==================================================

    laundering_specs = (
        build_laundering_specs(

            config[
                "laundering"
            ][
                "pipelines"
            ]
        )
    )

    laundering_frames = []

    print(
        "\n========================================"
    )

    print(
        "LAUNDERING DIAGNOSTICS"
    )

    print(
        "========================================"
    )

    for spec in laundering_specs:

        laundering_dataset = (
            LaunderingEvaluationDataset(

                base_dataset=
                    base_dataset,

                spec=
                    spec,

                seed=
                    seed,
            )
        )

        dataframe = (
            collect_diagnostic_outputs(

                model=
                    model,

                dataset=
                    laundering_dataset,

                collator=
                    collator,

                device=
                    device,

                batch_size=
                    batch_size,

                num_workers=
                    num_workers,

                condition_key=
                    spec.key,

                condition_name=
                    spec.name,

                temperature=
                    temperature,
            )
        )

        laundering_frames.append(
            dataframe
        )

    predictions = (
        pd.concat(
            laundering_frames,
            ignore_index=True,
        )
    )

    # ==================================================
    # ATTACH RELIABILITY
    # ==================================================

    if reliability_hashes:

        predictions = (
            attach_reliability(

                dataframe=
                    predictions,

                reliability_hashes=
                    reliability_hashes,

                reliability_probabilities=
                    reliability_probabilities,
            )
        )

    else:

        predictions[
            "reliability"
        ] = np.nan

    # ==================================================
    # ANALYSIS
    # ==================================================

    condition_summary = (
        build_condition_summary(
            predictions
        )
    )

    stability = (
        build_image_stability(
            predictions
        )
    )

    categories = (
        select_error_categories(

            predictions=
                predictions,

            stability=
                stability,

            confident_threshold=float(
                analysis_cfg[
                    "confident_threshold"
                ]
            ),

            disagreement_threshold=float(
                analysis_cfg[
                    "branch_disagreement_threshold"
                ]
            ),

            low_reliability_threshold=float(
                analysis_cfg[
                    "low_reliability_threshold"
                ]
            ),
        )
    )

    dataset_summary = (
        build_group_summary(

            dataframe=
                predictions,

            group_column=
                "dataset",
        )
    )

    generator_summary = (
        build_group_summary(

            dataframe=
                predictions,

            group_column=
                "generator",
        )
    )

    report = (
        build_report(

            predictions=
                predictions,

            stability=
                stability,

            categories=
                categories,
        )
    )

    # ==================================================
    # OUTPUT
    # ==================================================

    output_directory = (
        Path(
            "outputs/evaluation/"
            "error_analysis"
        )
        / args.name
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(

        output_directory
        / "all_diagnostics.csv",

        index=False,
    )

    condition_summary.to_csv(

        output_directory
        / "condition_summary.csv",

        index=False,
    )

    stability.to_csv(

        output_directory
        / "image_stability.csv",

        index=False,
    )

    if not dataset_summary.empty:

        dataset_summary.to_csv(

            output_directory
            / "dataset_summary.csv",

            index=False,
        )

    if not generator_summary.empty:

        generator_summary.to_csv(

            output_directory
            / "generator_summary.csv",

            index=False,
        )

    # ==================================================
    # CATEGORY CSVs + IMAGES
    # ==================================================

    top_k = int(
        analysis_cfg[
            "top_k_examples"
        ]
    )

    contact_cfg = (
        analysis_cfg[
            "contact_sheet"
        ]
    )

    for (
        category_name,
        dataframe,
    ) in categories.items():

        category_directory = (
            output_directory
            / "categories"
            / category_name
        )

        category_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataframe.to_csv(

            category_directory
            / "examples.csv",

            index=False,
        )

        copy_example_images(

            dataframe=
                dataframe,

            output_directory=
                category_directory
                / "images",

            top_k=
                top_k,
        )

        create_contact_sheet(

            dataframe=
                dataframe,

            output_path=(
                output_directory
                / "contact_sheets"
                / (
                    category_name
                    + ".png"
                )
            ),

            title=(
                category_name
                .replace(
                    "_",
                    " "
                )
                .title()
            ),

            top_k=
                top_k,

            columns=int(
                contact_cfg[
                    "columns"
                ]
            ),

            cell_width=int(
                contact_cfg[
                    "cell_width"
                ]
            ),

            cell_height=int(
                contact_cfg[
                    "cell_height"
                ]
            ),
        )

    # ==================================================
    # REPORT JSON
    # ==================================================

    with (
        output_directory
        / "error_report.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    # ==================================================
    # PRINT
    # ==================================================

    print(
        "\n========================================"
    )

    print(
        "ERROR ANALYSIS SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        f"\nImages analyzed: "
        f"{report['num_images']:,}"
    )

    print(
        f"Clean errors: "
        f"{report['clean_errors']:,}"
    )

    print(
        f"Confident false positives: "
        f"{report['confident_false_positives']:,}"
    )

    print(
        f"Confident false negatives: "
        f"{report['confident_false_negatives']:,}"
    )

    print(
        f"\nTransformation failures: "
        f"{report['transformation_failures']:,}"
    )

    print(
        f"Images with prediction flips: "
        f"{report['images_with_prediction_flip']:,}"
    )

    print(
        "\nAdaptive gating:"
    )

    print(
        f"Gate helped: "
        f"{report['gate_helped_rows']:,}"
    )

    print(
        f"Gate hurt:   "
        f"{report['gate_hurt_rows']:,}"
    )

    print(
        f"Net fixes:   "
        f"{report['net_gate_corrections']:+,}"
    )

    print(
        "\nBranch disagreement:"
    )

    print(
        f"Clean:      "
        f"{report['mean_clean_branch_disagreement']:.4f}"
    )

    print(
        f"Laundered: "
        f"{report['mean_transformed_branch_disagreement']:.4f}"
    )

    if (
        report[
            "mean_clean_reliability"
        ]
        is not None
    ):

        print(
            f"\nMean clean reliability: "
            f"{report['mean_clean_reliability']:.4f}"
        )

    print(
        "\nSaved:"
    )

    print(
        output_directory.resolve()
    )


if __name__ == "__main__":

    main()