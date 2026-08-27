from __future__ import annotations

from typing import Callable
from typing import Any

from PIL import Image

from src.augmentations.pipeline import (
    CorruptionPipeline,
)


class PairedViewTransform:

    def __init__(
        self,
        corruption_pipeline:
            CorruptionPipeline,

        clean_transform:
            Callable | None = None,

        corrupted_transform:
            Callable | None = None,

        clean_probability: float = 0.0,
    ):

        self.corruption_pipeline = (
            corruption_pipeline
        )

        self.clean_transform = (
            clean_transform
        )

        self.corrupted_transform = (
            corrupted_transform
        )

        self.clean_probability = (
            clean_probability
        )


    def __call__(
        self,
        image: Image.Image,
    ) -> dict[str, Any]:

        clean_image = (
            image.copy()
        )

        (
            corrupted_image,
            corruption_metadata,
        ) = (
            self.corruption_pipeline
            .apply_random(
                image=image,
                clean_probability=
                    self.clean_probability,
            )
        )

        if self.clean_transform is not None:

            clean_output = (
                self.clean_transform(
                    clean_image
                )
            )

        else:

            clean_output = (
                clean_image
            )

        if self.corrupted_transform is not None:

            corrupted_output = (
                self.corrupted_transform(
                    corrupted_image
                )
            )

        else:

            corrupted_output = (
                corrupted_image
            )

        return {
            "clean":
                clean_output,

            "corrupted":
                corrupted_output,

            "corruption":
                corruption_metadata
                .to_dict(),
        }