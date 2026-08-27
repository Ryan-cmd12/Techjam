import argparse

from pathlib import Path

import pandas as pd

from src.augmentations.pipeline import (
    CorruptionPipeline,
)

from src.data.image_utils import (
    load_rgb_image,
)


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Smoke test all corruption "
            "operations."
        )
    )

    parser.add_argument(
        "--manifest",
        type=str,
        default=(
            "data/manifests/"
            "cifake_train.csv"
        ),
    )

    args = parser.parse_args()

    dataframe = pd.read_csv(
        args.manifest
    )

    if len(dataframe) == 0:

        raise RuntimeError(
            "Manifest is empty."
        )

    image_path = Path(
        dataframe.iloc[
            0
        ][
            "image_path"
        ]
    )

    if not image_path.is_absolute():

        image_path = (
            Path.cwd()
            / image_path
        )

    image = (
        load_rgb_image(
            image_path
        )
    )

    pipeline = (
        CorruptionPipeline(
            seed=42
        )
    )

    print(
        "\n=============================="
    )

    print(
        "CORRUPTION TEST"
    )

    print(
        "=============================="
    )

    print(
        f"\nImage:"
        f"\n{image_path}"
    )

    print(
        f"\nOriginal size: "
        f"{image.size}"
    )

    tests = [
        (
            "jpeg",
            90,
        ),
        (
            "jpeg",
            70,
        ),
        (
            "jpeg",
            50,
        ),
        (
            "jpeg",
            30,
        ),

        (
            "blur",
            0.5,
        ),
        (
            "blur",
            1.0,
        ),
        (
            "blur",
            2.0,
        ),

        (
            "resize",
            0.5,
        ),
        (
            "resize",
            0.25,
        ),

        (
            "noise",
            0.02,
        ),
        (
            "noise",
            0.05,
        ),
        (
            "noise",
            0.10,
        ),

        (
            "color_jitter",
            0.20,
        ),

        (
            "crop",
            0.80,
        ),
    ]

    for (
        corruption_type,
        severity,
    ) in tests:

        (
            corrupted,
            metadata,
        ) = (
            pipeline.apply_specific(
                image=image,

                corruption_type=
                    corruption_type,

                severity=
                    severity,
            )
        )

        print(
            "\n------------------------------"
        )

        print(
            f"Type: "
            f"{corruption_type}"
        )

        print(
            f"Severity: "
            f"{severity}"
        )

        print(
            f"Output size: "
            f"{corrupted.size}"
        )

        print(
            "Metadata:"
        )

        print(
            metadata.to_dict()
        )

        if (
            corrupted.size
            != image.size
        ):

            raise AssertionError(
                "Corruption changed "
                "output dimensions."
            )

    print(
        "\n=============================="
    )

    print(
        "ALL CORRUPTIONS WORKING"
    )

    print(
        "=============================="
    )


if __name__ == "__main__":
    main()