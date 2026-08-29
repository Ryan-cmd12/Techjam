from __future__ import annotations

import argparse

from pathlib import Path

import pandas as pd


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        default=(
            "data/manifests/"
            "sid_val.csv"
        ),
    )

    args = parser.parse_args()

    dataframe = pd.read_csv(
        args.manifest
    )

    print(
        "\n========================================"
    )

    print(
        "SID MANIFEST INSPECTION"
    )

    print(
        "========================================"
    )

    print(
        f"\nImages: "
        f"{len(dataframe):,}"
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

    print(
        "\nScopes:"
    )

    print(
        dataframe[
            "scope"
        ]
        .value_counts()
    )

    print(
        "\nGenerators:"
    )

    print(
        dataframe[
            "generator"
        ]
        .value_counts()
    )

    print(
        "\nSources:"
    )

    print(
        dataframe[
            "source"
        ]
        .value_counts()
    )

    print(
        "\nSplits:"
    )

    print(
        dataframe[
            "split"
        ]
        .value_counts()
    )

    print(
        "\nWidth:"
    )

    print(
        dataframe[
            "width"
        ].describe()
    )

    print(
        "\nHeight:"
    )

    print(
        dataframe[
            "height"
        ].describe()
    )

    missing = 0

    for image_path in (
        dataframe[
            "image_path"
        ]
    ):

        path = Path(
            image_path
        )

        if not path.is_absolute():

            path = (
                Path.cwd()
                / path
            )

        if not path.exists():

            missing += 1

    print(
        f"\nMissing files: "
        f"{missing}"
    )

    duplicate_count = (
        dataframe[
            "content_hash"
        ]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate hashes: "
        f"{duplicate_count}"
    )

    print(
        "\n========================================"
    )

    print(
        "SID MANIFEST VALID"
        if missing == 0
        else "SID MANIFEST HAS ERRORS"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()