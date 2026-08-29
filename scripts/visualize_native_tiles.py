from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data.image_utils import (
    load_rgb_image,
)

from src.data.native_tiles import (
    NativeTileSampler,
)


def main():

    manifest = (
        "data/manifests/"
        "sid_model_train.csv"
    )

    dataframe = pd.read_csv(
        manifest
    )

    row = dataframe.iloc[
        0
    ]

    path = Path(
        row[
            "image_path"
        ]
    )

    if not path.is_absolute():

        path = (
            Path.cwd()
            / path
        )

    image = load_rgb_image(
        path
    )

    sampler = NativeTileSampler(
        tile_size=256,
        max_tiles=6,
        mode="grid",
        seed=42,
    )

    boxes = sampler.sample(
        image=
            image,

        sample_key=
            str(
                row[
                    "content_hash"
                ]
            ),
    )

    figure = plt.figure(
        figsize=(
            14,
            8,
        )
    )

    axis = figure.add_subplot(
        2,
        4,
        1,
    )

    axis.imshow(
        image
    )

    axis.set_title(
        f"Original\n"
        f"{image.size[0]}×"
        f"{image.size[1]}"
    )

    axis.axis(
        "off"
    )

    for index, box in enumerate(
        boxes,
        start=2,
    ):

        tile = image.crop(
            box.as_tuple()
        )

        axis = figure.add_subplot(
            2,
            4,
            index,
        )

        axis.imshow(
            tile
        )

        axis.set_title(
            f"Tile {index - 1}\n"
            f"{tile.size[0]}×"
            f"{tile.size[1]}"
        )

        axis.axis(
            "off"
        )

    figure.tight_layout()

    output = Path(
        "outputs/figures/"
        "native_sid_tiles.png"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        "\nSaved:"
    )

    print(
        output.resolve()
    )


if __name__ == "__main__":
    main()