from __future__ import annotations

import random

import numpy as np

from PIL import Image

from src.augmentations.corruption import (
    CorruptionMetadata,
    apply_center_crop,
    apply_color_jitter,
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_jpeg_compression,
    apply_resize,
)


class CorruptionPipeline:

    def __init__(
        self,
        seed: int | None = None,
    ):

        self.seed = seed

        self.python_rng = (
            random.Random(
                seed
            )
        )

        self.numpy_rng = (
            np.random.default_rng(
                seed
            )
        )

        self.jpeg_qualities = [
            90,
            70,
            50,
            30,
        ]

        self.blur_sigmas = [
            0.5,
            1.0,
            2.0,
        ]

        self.resize_scales = [
            0.5,
            0.25,
        ]

        self.noise_sigmas = [
            0.02,
            0.05,
            0.10,
        ]

        self.crop_ratios = [
            0.80,
        ]

        self.color_factor_min = (
            0.80
        )

        self.color_factor_max = (
            1.20
        )

        self.corruption_types = [
            "jpeg",
            "blur",
            "resize",
            "noise",
            "color_jitter",
            "crop",
        ]


    def clean_metadata(
        self,
    ) -> CorruptionMetadata:

        return (
            CorruptionMetadata()
        )


    def apply_specific(
        self,
        image: Image.Image,
        corruption_type: str,
        severity: float | int | None = None,
    ) -> tuple[
        Image.Image,
        CorruptionMetadata,
    ]:

        corruption_type = (
            corruption_type
            .lower()
            .strip()
        )

        if corruption_type == "clean":

            return (
                image.copy(),
                self.clean_metadata(),
            )

        if corruption_type == "jpeg":

            if severity is None:

                severity = (
                    self.python_rng.choice(
                        self.jpeg_qualities
                    )
                )

            return (
                apply_jpeg_compression(
                    image=image,
                    quality=int(
                        severity
                    ),
                )
            )

        if corruption_type == "blur":

            if severity is None:

                severity = (
                    self.python_rng.choice(
                        self.blur_sigmas
                    )
                )

            return (
                apply_gaussian_blur(
                    image=image,
                    sigma=float(
                        severity
                    ),
                )
            )

        if corruption_type == "resize":

            if severity is None:

                severity = (
                    self.python_rng.choice(
                        self.resize_scales
                    )
                )

            return (
                apply_resize(
                    image=image,
                    scale=float(
                        severity
                    ),
                )
            )

        if corruption_type == "noise":

            if severity is None:

                severity = (
                    self.python_rng.choice(
                        self.noise_sigmas
                    )
                )

            return (
                apply_gaussian_noise(
                    image=image,
                    sigma=float(
                        severity
                    ),
                    rng=
                        self.numpy_rng,
                )
            )

        if corruption_type == "color_jitter":

            # If a single severity is supplied,
            # interpret it as a symmetric fractional
            # adjustment.
            #
            # severity=0.2 means factors in:
            #
            # 0.8 -> 1.2

            if severity is None:

                low = (
                    self.color_factor_min
                )

                high = (
                    self.color_factor_max
                )

            else:

                severity = float(
                    severity
                )

                if not (
                    0.0
                    <= severity
                    < 1.0
                ):

                    raise ValueError(
                        "Color jitter severity "
                        "must be between 0 and 1."
                    )

                low = (
                    1.0
                    - severity
                )

                high = (
                    1.0
                    + severity
                )

            brightness = (
                self.python_rng.uniform(
                    low,
                    high,
                )
            )

            contrast = (
                self.python_rng.uniform(
                    low,
                    high,
                )
            )

            saturation = (
                self.python_rng.uniform(
                    low,
                    high,
                )
            )

            return (
                apply_color_jitter(
                    image=image,

                    brightness_factor=
                        brightness,

                    contrast_factor=
                        contrast,

                    saturation_factor=
                        saturation,
                )
            )

        if corruption_type == "crop":

            if severity is None:

                severity = (
                    self.python_rng.choice(
                        self.crop_ratios
                    )
                )

            return (
                apply_center_crop(
                    image=image,
                    crop_ratio=float(
                        severity
                    ),
                )
            )

        raise ValueError(
            f"Unknown corruption type: "
            f"{corruption_type}"
        )


    def apply_random(
        self,
        image: Image.Image,
        clean_probability: float = 0.10,
    ) -> tuple[
        Image.Image,
        CorruptionMetadata,
    ]:

        if not (
            0.0
            <= clean_probability
            <= 1.0
        ):

            raise ValueError(
                "clean_probability must "
                "be between 0 and 1."
            )

        if (
            self.python_rng.random()
            < clean_probability
        ):

            return (
                image.copy(),
                self.clean_metadata(),
            )

        corruption_type = (
            self.python_rng.choice(
                self.corruption_types
            )
        )

        return (
            self.apply_specific(
                image=image,
                corruption_type=
                    corruption_type,
            )
        )


    def apply_sequence(
        self,
        image: Image.Image,
        corruption_sequence: list[
            tuple[
                str,
                float | int | None,
            ]
        ],
    ) -> tuple[
        Image.Image,
        list[
            CorruptionMetadata
        ],
    ]:

        current_image = (
            image.copy()
        )

        metadata_sequence = []

        for (
            corruption_type,
            severity,
        ) in corruption_sequence:

            (
                current_image,
                metadata,
            ) = (
                self.apply_specific(
                    image=
                        current_image,

                    corruption_type=
                        corruption_type,

                    severity=
                        severity,
                )
            )

            metadata_sequence.append(
                metadata
            )

        return (
            current_image,
            metadata_sequence,
        )