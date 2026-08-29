from __future__ import annotations

import json

from pathlib import Path

import pandas as pd

from sklearn.model_selection import (
    train_test_split,
)


def load_manifest(
    path: str | Path,
) -> pd.DataFrame:

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"Manifest not found: {path}"
        )

    return pd.read_csv(path)


def remove_internal_duplicates(
    dataframe: pd.DataFrame,
    name: str,
) -> pd.DataFrame:

    before = len(dataframe)

    dataframe = (
        dataframe
        .drop_duplicates(
            subset=[
                "content_hash"
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    removed = (
        before
        - len(dataframe)
    )

    print(
        f"{name} duplicates removed: "
        f"{removed:,}"
    )

    return dataframe


def check_label_conflicts(
    train_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
) -> None:

    combined = pd.concat(
        [
            train_dataframe,
            test_dataframe,
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

    conflicts = label_counts[
        label_counts > 1
    ]

    if len(conflicts) > 0:

        raise RuntimeError(
            f"Found {len(conflicts):,} "
            f"SID image hashes with "
            f"conflicting labels."
        )


def build_sid_model_partitions(
    sid_train_path: str | Path,
    sid_official_validation_path: str | Path,
    output_directory: str | Path,
    dev_fraction: float = 0.10,
    seed: int = 42,
) -> None:

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_train = load_manifest(
        sid_train_path
    )

    official_test = load_manifest(
        sid_official_validation_path
    )

    print(
        "\n========================================"
    )

    print(
        "SID MODEL PARTITION"
    )

    print(
        "========================================"
    )

    print(
        f"\nSource SID train: "
        f"{len(source_train):,}"
    )

    print(
        f"Official SID validation: "
        f"{len(official_test):,}"
    )

    source_train = (
        remove_internal_duplicates(
            source_train,
            "SID train",
        )
    )

    official_test = (
        remove_internal_duplicates(
            official_test,
            "SID official validation",
        )
    )

    check_label_conflicts(
        train_dataframe=
            source_train,

        test_dataframe=
            official_test,
    )

    # --------------------------------------------------
    # Protect official validation
    # --------------------------------------------------

    official_test_hashes = set(
        official_test[
            "content_hash"
        ]
    )

    leakage_mask = (
        source_train[
            "content_hash"
        ]
        .isin(
            official_test_hashes
        )
    )

    leaked_training_rows = (
        source_train[
            leakage_mask
        ]
        .copy()
    )

    source_train = (
        source_train[
            ~leakage_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    print(
        f"\nSID train rows removed "
        f"because they duplicate the "
        f"official validation set: "
        f"{len(leaked_training_rows):,}"
    )

    if len(
        leaked_training_rows
    ) > 0:

        leaked_training_rows.to_csv(
            output_directory
            / (
                "sid_train_duplicates_"
                "official_test.csv"
            ),
            index=False,
        )

    # --------------------------------------------------
    # Internal development split
    # --------------------------------------------------

    model_train, model_dev = (
        train_test_split(
            source_train,

            test_size=
                dev_fraction,

            random_state=
                seed,

            stratify=
                source_train[
                    "label"
                ],
        )
    )

    model_train = (
        model_train
        .copy()
        .reset_index(
            drop=True
        )
    )

    model_dev = (
        model_dev
        .copy()
        .reset_index(
            drop=True
        )
    )

    official_test = (
        official_test
        .copy()
        .reset_index(
            drop=True
        )
    )

    model_train[
        "split"
    ] = "train"

    model_dev[
        "split"
    ] = "val"

    # Important:
    #
    # Original split remains "validation",
    # but our experimental role is test.
    official_test[
        "split"
    ] = "test"

    # --------------------------------------------------
    # Final leakage check
    # --------------------------------------------------

    train_hashes = set(
        model_train[
            "content_hash"
        ]
    )

    dev_hashes = set(
        model_dev[
            "content_hash"
        ]
    )

    test_hashes = set(
        official_test[
            "content_hash"
        ]
    )

    train_dev_overlap = (
        train_hashes
        & dev_hashes
    )

    train_test_overlap = (
        train_hashes
        & test_hashes
    )

    dev_test_overlap = (
        dev_hashes
        & test_hashes
    )

    print(
        "\nFinal SID overlaps:"
    )

    print(
        f"train ↔ dev:  "
        f"{len(train_dev_overlap):,}"
    )

    print(
        f"train ↔ test: "
        f"{len(train_test_overlap):,}"
    )

    print(
        f"dev ↔ test:   "
        f"{len(dev_test_overlap):,}"
    )

    if (
        train_dev_overlap
        or train_test_overlap
        or dev_test_overlap
    ):

        raise RuntimeError(
            "SID leakage remains after "
            "partitioning."
        )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    train_path = (
        output_directory
        / "sid_model_train.csv"
    )

    dev_path = (
        output_directory
        / "sid_model_val.csv"
    )

    test_path = (
        output_directory
        / "sid_test.csv"
    )

    model_train.to_csv(
        train_path,
        index=False,
    )

    model_dev.to_csv(
        dev_path,
        index=False,
    )

    official_test.to_csv(
        test_path,
        index=False,
    )

    summary = {
        "source_train":
            int(
                len(
                    source_train
                )
            ),

        "model_train":
            int(
                len(
                    model_train
                )
            ),

        "model_validation":
            int(
                len(
                    model_dev
                )
            ),

        "official_test":
            int(
                len(
                    official_test
                )
            ),

        "removed_train_test_duplicates":
            int(
                len(
                    leaked_training_rows
                )
            ),

        "development_fraction":
            float(
                dev_fraction
            ),

        "seed":
            int(
                seed
            ),
    }

    with (
        output_directory
        / "sid_partition_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    print(
        "\n========================================"
    )

    print(
        "SID PARTITION COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        f"\nModel train: "
        f"{len(model_train):,}"
    )

    print(
        f"Model validation: "
        f"{len(model_dev):,}"
    )

    print(
        f"Untouched test: "
        f"{len(official_test):,}"
    )