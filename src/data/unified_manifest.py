from __future__ import annotations

import json

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "image_path",
    "label",
    "class_name",
    "dataset",
    "source",
    "generator",
    "original_split",
    "split",
    "content_hash",
}


# ============================================================
# LOAD / VALIDATE
# ============================================================


def validate_manifest(
    dataframe: pd.DataFrame,
    name: str,
) -> None:

    missing = (
        REQUIRED_COLUMNS
        - set(
            dataframe.columns
        )
    )

    if missing:

        raise ValueError(
            f"{name} is missing "
            f"required columns: "
            f"{sorted(missing)}"
        )


def load_manifest(
    path: str | Path,
) -> pd.DataFrame:

    path = Path(
        path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Manifest not found: "
            f"{path}"
        )

    dataframe = (
        pd.read_csv(
            path
        )
    )

    validate_manifest(
        dataframe=
            dataframe,

        name=
            str(
                path
            ),
    )

    # Useful while diagnosing where duplicates came from.
    dataframe[
        "_manifest_source"
    ] = path.name

    return dataframe


# ============================================================
# SAME-SPLIT DUPLICATES
# ============================================================


def remove_same_split_duplicates(
    dataframe: pd.DataFrame,
    split_name: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    duplicate_mask = (
        dataframe[
            "content_hash"
        ]
        .duplicated(
            keep="first"
        )
    )

    duplicates = (
        dataframe[
            duplicate_mask
        ]
        .copy()
    )

    cleaned = (
        dataframe[
            ~duplicate_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    print(
        f"\n{split_name} "
        f"same-split duplicates removed: "
        f"{len(duplicates):,}"
    )

    return (
        cleaned,
        duplicates,
    )


# ============================================================
# LABEL CONFLICT CHECKING
# ============================================================


def find_label_conflicts(
    train_dataframe: pd.DataFrame,
    val_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
) -> pd.DataFrame:

    combined = pd.concat(
        [
            train_dataframe.assign(
                _unified_split="train"
            ),

            val_dataframe.assign(
                _unified_split="val"
            ),

            test_dataframe.assign(
                _unified_split="test"
            ),
        ],

        ignore_index=True,
    )

    label_counts = (
        combined
        .groupby(
            "content_hash"
        )[
            "label"
        ]
        .nunique()
    )

    conflicting_hashes = set(
        label_counts[
            label_counts > 1
        ].index
    )

    if not conflicting_hashes:

        return pd.DataFrame()

    conflicts = (
        combined[
            combined[
                "content_hash"
            ]
            .isin(
                conflicting_hashes
            )
        ]
        .copy()
        .sort_values(
            [
                "content_hash",
                "_unified_split",
            ]
        )
    )

    return conflicts


# ============================================================
# LEAKAGE REPORTING
# ============================================================


def calculate_split_overlaps(
    train_dataframe: pd.DataFrame,
    val_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
) -> dict[str, set[str]]:

    train_hashes = set(
        train_dataframe[
            "content_hash"
        ]
    )

    val_hashes = set(
        val_dataframe[
            "content_hash"
        ]
    )

    test_hashes = set(
        test_dataframe[
            "content_hash"
        ]
    )

    return {
        "train_val":
            train_hashes
            & val_hashes,

        "train_test":
            train_hashes
            & test_hashes,

        "val_test":
            val_hashes
            & test_hashes,
    }


def print_split_overlaps(
    overlaps: dict[
        str,
        set[str],
    ],
    title: str,
) -> None:

    print(
        "\n========================================"
    )

    print(
        title
    )

    print(
        "========================================"
    )

    print(
        f"\ntrain ↔ val overlap: "
        f"{len(overlaps['train_val']):,}"
    )

    print(
        f"\ntrain ↔ test overlap: "
        f"{len(overlaps['train_test']):,}"
    )

    print(
        f"\nval ↔ test overlap: "
        f"{len(overlaps['val_test']):,}"
    )


def build_cross_split_report(
    train_dataframe: pd.DataFrame,
    val_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
) -> pd.DataFrame:

    overlaps = (
        calculate_split_overlaps(
            train_dataframe=
                train_dataframe,

            val_dataframe=
                val_dataframe,

            test_dataframe=
                test_dataframe,
        )
    )

    leaking_hashes = (
        overlaps[
            "train_val"
        ]
        | overlaps[
            "train_test"
        ]
        | overlaps[
            "val_test"
        ]
    )

    if not leaking_hashes:

        return pd.DataFrame()

    frames = []

    for split_name, dataframe in [
        (
            "train",
            train_dataframe,
        ),
        (
            "val",
            val_dataframe,
        ),
        (
            "test",
            test_dataframe,
        ),
    ]:

        leaked_rows = (
            dataframe[
                dataframe[
                    "content_hash"
                ]
                .isin(
                    leaking_hashes
                )
            ]
            .copy()
        )

        leaked_rows[
            "_unified_split"
        ] = split_name

        frames.append(
            leaked_rows
        )

    report = (
        pd.concat(
            frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "content_hash",
                "_unified_split",
            ]
        )
    )

    return report


def print_overlap_origins(
    report: pd.DataFrame,
) -> None:

    if report.empty:

        return

    print(
        "\n========================================"
    )

    print(
        "LEAKAGE ORIGINS"
    )

    print(
        "========================================"
    )

    print(
        "\nDataset / split rows involved "
        "in cross-split leakage:"
    )

    counts = (
        report
        .groupby(
            [
                "_unified_split",
                "dataset",
            ]
        )
        .size()
        .sort_values(
            ascending=False
        )
    )

    print(
        counts
    )

    if (
        "_manifest_source"
        in report.columns
    ):

        print(
            "\nSource manifests involved:"
        )

        manifest_counts = (
            report
            .groupby(
                [
                    "_unified_split",
                    "_manifest_source",
                ]
            )
            .size()
            .sort_values(
                ascending=False
            )
        )

        print(
            manifest_counts
        )


# ============================================================
# CROSS-SPLIT SANITIZATION
# ============================================================


def sanitize_cross_split_leakage(
    train_dataframe: pd.DataFrame,
    val_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Resolve exact-image leakage using this priority:

        TEST > VALIDATION > TRAIN

    Rules:

    1. Test is NEVER modified.

    2. Remove validation images whose hashes occur in test.

    3. Remove training images whose hashes occur in either:
           - test
           - cleaned validation

    This preserves the integrity of held-out evaluation data.
    """

    test_hashes = set(
        test_dataframe[
            "content_hash"
        ]
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    val_test_mask = (
        val_dataframe[
            "content_hash"
        ]
        .isin(
            test_hashes
        )
    )

    removed_val = (
        val_dataframe[
            val_test_mask
        ]
        .copy()
    )

    removed_val[
        "removed_from_split"
    ] = "val"

    removed_val[
        "kept_in_split"
    ] = "test"

    removed_val[
        "removal_reason"
    ] = (
        "exact duplicate exists in test"
    )

    cleaned_val = (
        val_dataframe[
            ~val_test_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    cleaned_val_hashes = set(
        cleaned_val[
            "content_hash"
        ]
    )

    train_test_mask = (
        train_dataframe[
            "content_hash"
        ]
        .isin(
            test_hashes
        )
    )

    train_val_mask = (
        train_dataframe[
            "content_hash"
        ]
        .isin(
            cleaned_val_hashes
        )
    )

    train_remove_mask = (
        train_test_mask
        | train_val_mask
    )

    removed_train = (
        train_dataframe[
            train_remove_mask
        ]
        .copy()
    )

    removed_train[
        "removed_from_split"
    ] = "train"

    removed_train[
        "kept_in_split"
    ] = ""

    removed_train[
        "removal_reason"
    ] = ""

    removed_train.loc[
        train_test_mask[
            train_remove_mask
        ].values,
        "kept_in_split",
    ] = "test"

    removed_train.loc[
        train_test_mask[
            train_remove_mask
        ].values,
        "removal_reason",
    ] = (
        "exact duplicate exists in test"
    )

    # Rows duplicated only with validation.
    val_only_mask = (
        ~train_test_mask[
            train_remove_mask
        ].values
    )

    removed_train.loc[
        val_only_mask,
        "kept_in_split",
    ] = "val"

    removed_train.loc[
        val_only_mask,
        "removal_reason",
    ] = (
        "exact duplicate exists in validation"
    )

    cleaned_train = (
        train_dataframe[
            ~train_remove_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    cleaned_test = (
        test_dataframe
        .copy()
        .reset_index(
            drop=True
        )
    )

    removed = pd.concat(
        [
            removed_train,
            removed_val,
        ],
        ignore_index=True,
    )

    print(
        "\n========================================"
    )

    print(
        "LEAKAGE SANITIZATION"
    )

    print(
        "========================================"
    )

    print(
        f"\nRemoved from training:   "
        f"{len(removed_train):,}"
    )

    print(
        f"Removed from validation: "
        f"{len(removed_val):,}"
    )

    print(
        f"Removed from test:       "
        f"0"
    )

    print(
        "\nPriority policy:"
    )

    print(
        "TEST > VALIDATION > TRAIN"
    )

    return (
        cleaned_train,
        cleaned_val,
        cleaned_test,
        removed,
    )


# ============================================================
# FINAL ASSERTION
# ============================================================


def assert_no_cross_split_leakage(
    train_dataframe: pd.DataFrame,
    val_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
) -> None:

    overlaps = (
        calculate_split_overlaps(
            train_dataframe=
                train_dataframe,

            val_dataframe=
                val_dataframe,

            test_dataframe=
                test_dataframe,
        )
    )

    print_split_overlaps(
        overlaps=
            overlaps,

        title=
            "POST-SANITIZATION LEAKAGE CHECK",
    )

    total_overlap = (
        len(
            overlaps[
                "train_val"
            ]
        )
        + len(
            overlaps[
                "train_test"
            ]
        )
        + len(
            overlaps[
                "val_test"
            ]
        )
    )

    if total_overlap != 0:

        raise RuntimeError(
            "Cross-split leakage still "
            "exists after sanitization."
        )

    print(
        "\n✓ No exact cross-split "
        "image leakage remains."
    )


# ============================================================
# SUMMARY
# ============================================================


def print_dataset_summary(
    dataframe: pd.DataFrame,
    split_name: str,
) -> None:

    print(
        "\n========================================"
    )

    print(
        f"UNIFIED "
        f"{split_name.upper()}"
    )

    print(
        "========================================"
    )

    print(
        f"\nImages: "
        f"{len(dataframe):,}"
    )

    print(
        "\nDatasets:"
    )

    print(
        dataframe[
            "dataset"
        ]
        .value_counts()
    )

    print(
        "\nClasses:"
    )

    print(
        dataframe[
            "class_name"
        ]
        .value_counts()
    )

    if (
        "generator"
        in dataframe.columns
    ):

        print(
            "\nGenerators:"
        )

        print(
            dataframe[
                "generator"
            ]
            .value_counts()
            .head(
                20
            )
        )


# ============================================================
# BUILD
# ============================================================


def build_unified_manifests(
    output_directory: str | Path,
    train_paths: list[
        str | Path
    ],
    val_paths: list[
        str | Path
    ],
    test_paths: list[
        str | Path
    ],
) -> None:

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    train_frames = [
        load_manifest(
            path
        )
        for path in train_paths
    ]

    val_frames = [
        load_manifest(
            path
        )
        for path in val_paths
    ]

    test_frames = [
        load_manifest(
            path
        )
        for path in test_paths
    ]

    train_dataframe = (
        pd.concat(
            train_frames,
            ignore_index=True,
        )
    )

    val_dataframe = (
        pd.concat(
            val_frames,
            ignore_index=True,
        )
    )

    test_dataframe = (
        pd.concat(
            test_frames,
            ignore_index=True,
        )
    )

    train_dataframe[
        "split"
    ] = "train"

    val_dataframe[
        "split"
    ] = "val"

    test_dataframe[
        "split"
    ] = "test"

    original_counts = {
        "train":
            len(
                train_dataframe
            ),

        "val":
            len(
                val_dataframe
            ),

        "test":
            len(
                test_dataframe
            ),
    }

    # --------------------------------------------------------
    # LABEL CONFLICTS
    # --------------------------------------------------------

    conflicts = (
        find_label_conflicts(
            train_dataframe=
                train_dataframe,

            val_dataframe=
                val_dataframe,

            test_dataframe=
                test_dataframe,
        )
    )

    if not conflicts.empty:

        conflict_path = (
            output_directory
            / "unified_label_conflicts.csv"
        )

        conflicts.to_csv(
            conflict_path,
            index=False,
        )

        print(
            "\n[ERROR] "
            "Exact same images have "
            "conflicting REAL/FAKE labels."
        )

        print(
            f"\nReport:"
            f"\n{conflict_path}"
        )

        raise RuntimeError(
            "Label conflicts detected "
            "between exact duplicate images."
        )

    # --------------------------------------------------------
    # REMOVE SAME-SPLIT DUPLICATES
    # --------------------------------------------------------

    (
        train_dataframe,
        train_duplicates,
    ) = (
        remove_same_split_duplicates(
            dataframe=
                train_dataframe,

            split_name=
                "train",
        )
    )

    (
        val_dataframe,
        val_duplicates,
    ) = (
        remove_same_split_duplicates(
            dataframe=
                val_dataframe,

            split_name=
                "val",
        )
    )

    (
        test_dataframe,
        test_duplicates,
    ) = (
        remove_same_split_duplicates(
            dataframe=
                test_dataframe,

            split_name=
                "test",
        )
    )

    same_split_duplicates = (
        pd.concat(
            [
                train_duplicates.assign(
                    removed_from_split=
                        "train"
                ),

                val_duplicates.assign(
                    removed_from_split=
                        "val"
                ),

                test_duplicates.assign(
                    removed_from_split=
                        "test"
                ),
            ],

            ignore_index=True,
        )
    )

    if not same_split_duplicates.empty:

        same_split_duplicates.to_csv(
            output_directory
            / (
                "unified_same_split_"
                "duplicates_removed.csv"
            ),

            index=False,
        )

    # --------------------------------------------------------
    # REPORT CROSS-SPLIT LEAKAGE BEFORE CLEANUP
    # --------------------------------------------------------

    overlaps_before = (
        calculate_split_overlaps(
            train_dataframe=
                train_dataframe,

            val_dataframe=
                val_dataframe,

            test_dataframe=
                test_dataframe,
        )
    )

    print_split_overlaps(
        overlaps=
            overlaps_before,

        title=
            "PRE-SANITIZATION LEAKAGE CHECK",
    )

    leakage_report = (
        build_cross_split_report(
            train_dataframe=
                train_dataframe,

            val_dataframe=
                val_dataframe,

            test_dataframe=
                test_dataframe,
        )
    )

    if not leakage_report.empty:

        leakage_path = (
            output_directory
            / (
                "unified_cross_split_"
                "leakage_before_cleanup.csv"
            )
        )

        leakage_report.to_csv(
            leakage_path,
            index=False,
        )

        print_overlap_origins(
            leakage_report
        )

        print(
            "\nLeakage report saved:"
        )

        print(
            leakage_path
        )

    # --------------------------------------------------------
    # SANITIZE
    # --------------------------------------------------------

    (
        train_dataframe,
        val_dataframe,
        test_dataframe,
        removed_cross_split,
    ) = (
        sanitize_cross_split_leakage(
            train_dataframe=
                train_dataframe,

            val_dataframe=
                val_dataframe,

            test_dataframe=
                test_dataframe,
        )
    )

    if not removed_cross_split.empty:

        removed_cross_split.to_csv(
            output_directory
            / (
                "unified_cross_split_"
                "duplicates_removed.csv"
            ),

            index=False,
        )

    # --------------------------------------------------------
    # FINAL ASSERTION
    # --------------------------------------------------------

    assert_no_cross_split_leakage(
        train_dataframe=
            train_dataframe,

        val_dataframe=
            val_dataframe,

        test_dataframe=
            test_dataframe,
    )

    # --------------------------------------------------------
    # REMOVE INTERNAL DIAGNOSTIC COLUMN
    # --------------------------------------------------------

    for dataframe in [
        train_dataframe,
        val_dataframe,
        test_dataframe,
    ]:

        if (
            "_manifest_source"
            in dataframe.columns
        ):

            dataframe.drop(
                columns=[
                    "_manifest_source"
                ],
                inplace=True,
            )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    train_path = (
        output_directory
        / "unified_train.csv"
    )

    val_path = (
        output_directory
        / "unified_val.csv"
    )

    test_path = (
        output_directory
        / "unified_test.csv"
    )

    train_dataframe.to_csv(
        train_path,
        index=False,
    )

    val_dataframe.to_csv(
        val_path,
        index=False,
    )

    test_dataframe.to_csv(
        test_path,
        index=False,
    )

    all_dataframe = (
        pd.concat(
            [
                train_dataframe,
                val_dataframe,
                test_dataframe,
            ],
            ignore_index=True,
        )
    )

    all_dataframe.to_csv(
        output_directory
        / "unified_all.csv",
        index=False,
    )

    # --------------------------------------------------------
    # SUMMARY JSON
    # --------------------------------------------------------

    final_counts = {
        "train":
            len(
                train_dataframe
            ),

        "val":
            len(
                val_dataframe
            ),

        "test":
            len(
                test_dataframe
            ),
    }

    summary = {
        "original_counts":
            original_counts,

        "final_counts":
            final_counts,

        "same_split_duplicates_removed":
            {
                "train":
                    len(
                        train_duplicates
                    ),

                "val":
                    len(
                        val_duplicates
                    ),

                "test":
                    len(
                        test_duplicates
                    ),
            },

        "cross_split_unique_hash_overlaps_before":
            {
                "train_val":
                    len(
                        overlaps_before[
                            "train_val"
                        ]
                    ),

                "train_test":
                    len(
                        overlaps_before[
                            "train_test"
                        ]
                    ),

                "val_test":
                    len(
                        overlaps_before[
                            "val_test"
                        ]
                    ),
            },

        "cross_split_rows_removed":
            int(
                len(
                    removed_cross_split
                )
            ),

        "priority_policy":
            "test > validation > train",
    }

    with (
        output_directory
        / "unified_manifest_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print_dataset_summary(
        train_dataframe,
        "train",
    )

    print_dataset_summary(
        val_dataframe,
        "validation",
    )

    print_dataset_summary(
        test_dataframe,
        "test",
    )

    print(
        "\n========================================"
    )

    print(
        "UNIFIED MANIFEST COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        "\nSaved:"
    )

    print(
        train_path
    )

    print(
        val_path
    )

    print(
        test_path
    )

    print(
        "\nLeakage policy:"
    )

    print(
        "TEST > VALIDATION > TRAIN"
    )