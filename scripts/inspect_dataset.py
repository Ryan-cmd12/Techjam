import argparse
from pathlib import Path

import pandas as pd

from PIL import Image

from src.data.image_utils import (
    load_rgb_image,
)


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Inspect an AIGC dataset manifest."
        )
    )

    parser.add_argument(
        "manifest",
        type=str,
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=100,
        help=(
            "Number of images to inspect for "
            "resolution statistics."
        ),
    )

    args = parser.parse_args()

    manifest_path = Path(
        args.manifest
    )

    dataframe = pd.read_csv(
        manifest_path
    )

    print(
        "\n=============================="
    )

    print(
        "DATASET INSPECTION"
    )

    print(
        "=============================="
    )

    print(
        f"\nManifest: "
        f"{manifest_path}"
    )

    print(
        f"Images: "
        f"{len(dataframe)}"
    )

    print(
        "\nLabels:"
    )

    print(
        dataframe[
            "class_name"
        ]
        .value_counts()
    )

    print(
        "\nSources:"
    )

    source_counts = (
        dataframe
        .groupby(
            [
                "class_name",
                "source",
            ]
        )
        .size()
        .sort_values(
            ascending=False
        )
    )

    print(
        source_counts
    )

    if "split" in dataframe.columns:

        print(
            "\nSplits:"
        )

        print(
            dataframe[
                "split"
            ]
            .value_counts()
        )

    sample_count = min(
        args.sample,
        len(dataframe),
    )

    sample_df = (
        dataframe
        .sample(
            n=sample_count,
            random_state=42,
        )
    )

    widths = []
    heights = []
    failures = 0

    print(
        f"\nInspecting "
        f"{sample_count} images..."
    )

    for _, row in sample_df.iterrows():

        path = Path(
            row["image_path"]
        )

        try:

            image = load_rgb_image(
                path
            )

            width, height = (
                image.size
            )

            widths.append(
                width
            )

            heights.append(
                height
            )

        except Exception as error:

            failures += 1

            print(
                f"[ERROR] "
                f"{path}: "
                f"{error}"
            )

    if widths:

        width_series = pd.Series(
            widths
        )

        height_series = pd.Series(
            heights
        )

        print(
            "\nWidth statistics:"
        )

        print(
            width_series.describe()
        )

        print(
            "\nHeight statistics:"
        )

        print(
            height_series.describe()
        )

    print(
        f"\nFailed reads: "
        f"{failures}"
    )


if __name__ == "__main__":
    main()