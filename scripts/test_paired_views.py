from torch.utils.data import (
    DataLoader,
)

from torchvision import transforms

from src.augmentations.paired_views import (
    PairedViewTransform,
)

from src.augmentations.pipeline import (
    CorruptionPipeline,
)

from src.data.dataset import (
    AIGCImageDataset,
)


def main():

    tensor_transform = (
        transforms.ToTensor()
    )

    pipeline = (
        CorruptionPipeline(
            seed=42
        )
    )

    paired_transform = (
        PairedViewTransform(
            corruption_pipeline=
                pipeline,

            clean_transform=
                tensor_transform,

            corrupted_transform=
                tensor_transform,

            clean_probability=
                0.0,
        )
    )

    dataset = (
        AIGCImageDataset(
            manifest_path=(
                "data/manifests/"
                "cifake_train.csv"
            ),

            paired_transform=
                paired_transform,

            return_metadata=
                True,
        )
    )

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )

    batch = next(
        iter(
            dataloader
        )
    )

    print(
        "\n=============================="
    )

    print(
        "PAIRED VIEW TEST"
    )

    print(
        "=============================="
    )

    print(
        "\nClean:"
    )

    print(
        batch[
            "clean_image"
        ].shape
    )

    print(
        "\nCorrupted:"
    )

    print(
        batch[
            "corrupted_image"
        ].shape
    )

    print(
        "\nLabels:"
    )

    print(
        batch[
            "label"
        ]
    )

    print(
        "\nCorruption types:"
    )

    print(
        batch[
            "corruption"
        ][
            "corruption_type"
        ]
    )

    print(
        "\nJPEG qualities:"
    )

    print(
        batch[
            "corruption"
        ][
            "jpeg_quality"
        ]
    )

    print(
        "\nBlur sigmas:"
    )

    print(
        batch[
            "corruption"
        ][
            "blur_sigma"
        ]
    )

    print(
        "\nResize scales:"
    )

    print(
        batch[
            "corruption"
        ][
            "resize_scale"
        ]
    )

    print(
        "\nNoise sigmas:"
    )

    print(
        batch[
            "corruption"
        ][
            "noise_sigma"
        ]
    )

    print(
        "\n=============================="
    )

    print(
        "PAIRED VIEWS WORKING"
    )

    print(
        "=============================="
    )


if __name__ == "__main__":
    main()