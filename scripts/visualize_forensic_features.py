from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from torchvision.transforms.functional import (
    pil_to_tensor,
)

from src.data.image_utils import (
    load_rgb_image,
)

from src.models.forensic_features import (
    ForensicFeatureExtractor,
)


def normalize_for_display(
    tensor,
):
    tensor = tensor.detach().cpu()

    minimum = tensor.min()
    maximum = tensor.max()

    return (
        tensor - minimum
    ) / (
        maximum
        - minimum
        + 1e-8
    )


def main():

    manifest = (
        "data/manifests/"
        "cifake_train.csv"
    )

    dataframe = pd.read_csv(
        manifest
    )

    image_path = Path(
        dataframe.iloc[0][
            "image_path"
        ]
    )

    if not image_path.is_absolute():
        image_path = (
            Path.cwd()
            / image_path
        )

    image = load_rgb_image(
        image_path
    )

    tensor = (
        pil_to_tensor(
            image
        )
        .float()
        / 255.0
    ).unsqueeze(0)

    extractor = (
        ForensicFeatureExtractor()
    )

    with torch.no_grad():
        maps = extractor(
            tensor
        )

    residual = (
        maps["residual"][0]
        .abs()
        .mean(dim=0)
    )

    fft = maps[
        "fft"
    ][0, 0]

    dct = maps[
        "dct"
    ][0, 0]

    wavelet = (
        maps["wavelet"][0]
        .abs()
        .mean(dim=0)
    )

    figure = plt.figure(
        figsize=(14, 7)
    )

    titles = [
        "Original",
        "High-pass residual",
        "FFT log magnitude",
        "DCT magnitude",
        "Haar wavelet details",
    ]

    displays = [
        image,
        normalize_for_display(
            residual
        ),
        normalize_for_display(
            fft
        ),
        normalize_for_display(
            dct
        ),
        normalize_for_display(
            wavelet
        ),
    ]

    for index, (
        title,
        display,
    ) in enumerate(
        zip(
            titles,
            displays,
        ),
        start=1,
    ):

        axis = figure.add_subplot(
            2,
            3,
            index,
        )

        axis.imshow(
            display,
            cmap=(
                None
                if index == 1
                else "gray"
            ),
        )

        axis.set_title(
            title
        )

        axis.axis(
            "off"
        )

    figure.tight_layout()

    output = Path(
        "outputs/figures/"
        "forensic_features.png"
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
        f"\nSaved:"
        f"\n{output.resolve()}"
    )


if __name__ == "__main__":
    main()