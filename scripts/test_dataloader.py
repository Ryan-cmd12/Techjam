import argparse

import torch

from torch.utils.data import (
    DataLoader,
)

from torchvision import transforms

from src.data.dataset import (
    AIGCImageDataset,
)


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Smoke test the CIFAKE "
            "PyTorch DataLoader."
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
        "--batch-size",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    dataset = (
        AIGCImageDataset(
            manifest_path=
                args.manifest,

            transform=
                transform,

            return_metadata=
                True,
        )
    )

    dataloader = DataLoader(
        dataset,

        batch_size=
            args.batch_size,

        shuffle=
            True,

        num_workers=
            0,
    )

    print(
        "\n=============================="
    )

    print(
        "DATALOADER TEST"
    )

    print(
        "=============================="
    )

    print(
        f"\nDataset size: "
        f"{len(dataset)}"
    )

    batch = next(
        iter(
            dataloader
        )
    )

    images = (
        batch[
            "image"
        ]
    )

    labels = (
        batch[
            "label"
        ]
    )

    print(
        f"\nImage tensor:"
        f"\n{images.shape}"
    )

    print(
        f"\nImage dtype:"
        f"\n{images.dtype}"
    )

    print(
        f"\nPixel range:"
        f"\n"
        f"{images.min().item():.4f} "
        f"→ "
        f"{images.max().item():.4f}"
    )

    print(
        f"\nLabels:"
        f"\n{labels}"
    )

    print(
        "\nDatasets:"
    )

    print(
        batch[
            "dataset"
        ]
    )

    print(
        "\nSplits:"
    )

    print(
        batch[
            "split"
        ]
    )

    print(
        "\nExample paths:"
    )

    for path in (
        batch[
            "image_path"
        ][:3]
    ):

        print(
            path
        )

    real_count = (
        labels
        == 0
    ).sum().item()

    fake_count = (
        labels
        == 1
    ).sum().item()

    print(
        f"\nReal in batch: "
        f"{real_count}"
    )

    print(
        f"Fake in batch: "
        f"{fake_count}"
    )

    print(
        "\n=============================="
    )

    print(
        "DATALOADER WORKING"
    )

    print(
        "=============================="
    )


if __name__ == "__main__":
    main()