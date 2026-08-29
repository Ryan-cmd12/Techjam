from __future__ import annotations

import hashlib

from pathlib import Path

import torch

from PIL import Image

from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}


def discover_images(
    input_path: str | Path,
) -> list[Path]:

    input_path = Path(
        input_path
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"Input does not exist: "
            f"{input_path}"
        )

    if input_path.is_file():

        if (
            input_path.suffix.lower()
            not in IMAGE_EXTENSIONS
        ):

            raise ValueError(
                f"Unsupported image type: "
                f"{input_path}"
            )

        return [
            input_path
        ]

    paths = [
        path

        for path
        in input_path.rglob("*")

        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ]

    paths.sort(
        key=lambda path:
            str(
                path
            ).lower()
    )

    if not paths:

        raise RuntimeError(
            f"No supported images "
            f"found under: "
            f"{input_path}"
        )

    return paths


def hash_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:

        while True:

            chunk = file.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


class InferenceImageDataset(
    Dataset
):

    def __init__(
        self,
        input_path: str | Path,
    ):

        self.input_path = Path(
            input_path
        )

        self.image_paths = (
            discover_images(
                self.input_path
            )
        )

        print(
            f"\nImages found: "
            f"{len(self.image_paths):,}"
        )


    def __len__(
        self,
    ) -> int:

        return len(
            self.image_paths
        )


    def __getitem__(
        self,
        index: int,
    ):

        path = self.image_paths[
            index
        ]

        with Image.open(
            path
        ) as image:

            image = (
                image
                .convert(
                    "RGB"
                )
                .copy()
            )

        return {
            "image":
                image,

            # Dummy label because the existing
            # collator expects a label tensor.
            "label":
                torch.tensor(
                    0.0,
                    dtype=torch.float32,
                ),

            "image_path":
                str(
                    path
                ),

            "content_hash":
                hash_file(
                    path
                ),

            "class_name":
                "unknown",

            "dataset":
                "inference",

            "source":
                "inference",

            "generator":
                "unknown",

            "original_split":
                "inference",

            "split":
                "inference",
        }