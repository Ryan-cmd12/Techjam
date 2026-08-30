from __future__ import annotations

import argparse

import pandas as pd


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        default=(
            "data/manifests/"
            "wildfake_test.csv"
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
        "WILDFAKE MANIFEST"
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
        ].value_counts()
    )

    print(
        "\nFamilies:"
    )

    print(
        dataframe[
            "generation_family"
        ].value_counts()
    )

    print(
        "\nGenerators:"
    )

    fake = dataframe[
        dataframe[
            "label"
        ] == 1
    ]

    print(
        fake[
            "generator"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nSubcategories:"
    )

    print(
        fake[
            "subcategory"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nDuplicate hashes:"
    )

    print(
        dataframe[
            "content_hash"
        ]
        .duplicated()
        .sum()
    )

    print(
        "\nMissing files:"
    )

    missing = (
        ~dataframe[
            "image_path"
        ]
        .map(
            lambda path:
                __import__(
                    "pathlib"
                )
                .Path(
                    path
                )
                .exists()
        )
    )

    print(
        int(
            missing.sum()
        )
    )

    print(
        "\nDimensions:"
    )

    print(
        dataframe[
            [
                "width",
                "height",
            ]
        ].describe()
    )


if __name__ == "__main__":

    main()