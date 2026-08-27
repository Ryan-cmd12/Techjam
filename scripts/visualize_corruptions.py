import argparse

from pathlib import Path

import matplotlib.pyplot as plt
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
            "Visualize CIFAKE corruption "
            "examples."
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

    parser.add_argument(
        "--index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "outputs/figures/"
            "corruption_examples.png"
        ),
    )

    args = parser.parse_args()

    dataframe = pd.read_csv(
        args.manifest
    )

    if not (
        0
        <= args.index
        < len(dataframe)
    ):

        raise IndexError(
            f"Index {args.index} "
            f"is outside dataset."
        )

    image_path = Path(
        dataframe.iloc[
            args.index
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

    examples = [
        (
            "Clean",
            image,
        ),
    ]

    specs = [
        (
            "JPEG 90",
            "jpeg",
            90,
        ),
        (
            "JPEG 70",
            "jpeg",
            70,
        ),
        (
            "JPEG 50",
            "jpeg",
            50,
        ),
        (
            "JPEG 30",
            "jpeg",
            30,
        ),

        (
            "Blur 0.5",
            "blur",
            0.5,
        ),
        (
            "Blur 1.0",
            "blur",
            1.0,
        ),
        (
            "Blur 2.0",
            "blur",
            2.0,
        ),

        (
            "Resize 0.5",
            "resize",
            0.5,
        ),
        (
            "Resize 0.25",
            "resize",
            0.25,
        ),

        (
            "Noise 0.02",
            "noise",
            0.02,
        ),
        (
            "Noise 0.05",
            "noise",
            0.05,
        ),
        (
            "Noise 0.10",
            "noise",
            0.10,
        ),

        (
            "Color ±20%",
            "color_jitter",
            0.20,
        ),

        (
            "Crop 80%",
            "crop",
            0.80,
        ),
    ]

    for (
        title,
        corruption_type,
        severity,
    ) in specs:

        (
            corrupted,
            _,
        ) = (
            pipeline.apply_specific(
                image=image,

                corruption_type=
                    corruption_type,

                severity=
                    severity,
            )
        )

        examples.append(
            (
                title,
                corrupted,
            )
        )

    columns = 4

    rows = (
        len(examples)
        + columns
        - 1
    ) // columns

    figure = plt.figure(
        figsize=(
            14,
            rows * 3,
        )
    )

    for index, (
        title,
        example_image,
    ) in enumerate(
        examples,
        start=1,
    ):

        axis = figure.add_subplot(
            rows,
            columns,
            index,
        )

        axis.imshow(
            example_image
        )

        axis.set_title(
            title
        )

        axis.axis(
            "off"
        )

    figure.tight_layout()

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        f"\nSaved corruption grid:"
        f"\n{output_path.resolve()}"
    )


if __name__ == "__main__":
    main()