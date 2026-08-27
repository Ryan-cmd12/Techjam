from __future__ import annotations

import io

from dataclasses import asdict
from dataclasses import dataclass

import numpy as np

from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter


@dataclass
class CorruptionMetadata:

    corruption_type: str = "clean"

    jpeg_quality: int = 100

    blur_sigma: float = 0.0

    resize_scale: float = 1.0

    noise_sigma: float = 0.0

    brightness_factor: float = 1.0

    contrast_factor: float = 1.0

    saturation_factor: float = 1.0

    crop_ratio: float = 1.0


    def to_dict(
        self,
    ) -> dict:

        return asdict(
            self
        )


def ensure_rgb(
    image: Image.Image,
) -> Image.Image:

    if image.mode != "RGB":

        image = image.convert(
            "RGB"
        )

    return image


def apply_jpeg_compression(
    image: Image.Image,
    quality: int,
) -> tuple[
    Image.Image,
    CorruptionMetadata,
]:

    image = ensure_rgb(
        image
    )

    if not (
        1 <= quality <= 100
    ):

        raise ValueError(
            "JPEG quality must be "
            "between 1 and 100."
        )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=int(
            quality
        ),
        optimize=False,
    )

    buffer.seek(
        0
    )

    compressed = Image.open(
        buffer
    )

    compressed = (
        compressed
        .convert(
            "RGB"
        )
        .copy()
    )

    buffer.close()

    metadata = (
        CorruptionMetadata(
            corruption_type="jpeg",
            jpeg_quality=int(
                quality
            ),
        )
    )

    return (
        compressed,
        metadata,
    )


def apply_gaussian_blur(
    image: Image.Image,
    sigma: float,
) -> tuple[
    Image.Image,
    CorruptionMetadata,
]:

    image = ensure_rgb(
        image
    )

    if sigma < 0:

        raise ValueError(
            "Gaussian blur sigma "
            "cannot be negative."
        )

    blurred = (
        image.filter(
            ImageFilter.GaussianBlur(
                radius=float(
                    sigma
                )
            )
        )
    )

    metadata = (
        CorruptionMetadata(
            corruption_type="blur",
            blur_sigma=float(
                sigma
            ),
        )
    )

    return (
        blurred,
        metadata,
    )


def apply_resize(
    image: Image.Image,
    scale: float,
) -> tuple[
    Image.Image,
    CorruptionMetadata,
]:

    image = ensure_rgb(
        image
    )

    if not (
        0.0 < scale <= 1.0
    ):

        raise ValueError(
            "Resize scale must be "
            "greater than 0 and <= 1."
        )

    original_width, original_height = (
        image.size
    )

    scaled_width = max(
        1,
        int(
            round(
                original_width
                * scale
            )
        ),
    )

    scaled_height = max(
        1,
        int(
            round(
                original_height
                * scale
            )
        ),
    )

    downscaled = image.resize(
        (
            scaled_width,
            scaled_height,
        ),
        resample=Image.Resampling.BICUBIC,
    )

    restored = downscaled.resize(
        (
            original_width,
            original_height,
        ),
        resample=Image.Resampling.BICUBIC,
    )

    metadata = (
        CorruptionMetadata(
            corruption_type="resize",
            resize_scale=float(
                scale
            ),
        )
    )

    return (
        restored,
        metadata,
    )


def apply_gaussian_noise(
    image: Image.Image,
    sigma: float,
    rng: np.random.Generator | None = None,
) -> tuple[
    Image.Image,
    CorruptionMetadata,
]:

    image = ensure_rgb(
        image
    )

    if sigma < 0:

        raise ValueError(
            "Gaussian noise sigma "
            "cannot be negative."
        )

    if rng is None:

        rng = (
            np.random.default_rng()
        )

    image_array = (
        np.asarray(
            image,
            dtype=np.float32,
        )
        / 255.0
    )

    noise = rng.normal(
        loc=0.0,
        scale=float(
            sigma
        ),
        size=image_array.shape,
    ).astype(
        np.float32
    )

    noisy_array = (
        image_array
        + noise
    )

    noisy_array = np.clip(
        noisy_array,
        0.0,
        1.0,
    )

    noisy_array = (
        noisy_array
        * 255.0
    ).round().astype(
        np.uint8
    )

    noisy_image = Image.fromarray(
        noisy_array,
        mode="RGB",
    )

    metadata = (
        CorruptionMetadata(
            corruption_type="noise",
            noise_sigma=float(
                sigma
            ),
        )
    )

    return (
        noisy_image,
        metadata,
    )


def apply_color_jitter(
    image: Image.Image,
    brightness_factor: float = 1.0,
    contrast_factor: float = 1.0,
    saturation_factor: float = 1.0,
) -> tuple[
    Image.Image,
    CorruptionMetadata,
]:

    image = ensure_rgb(
        image
    )

    factors = {
        "brightness_factor":
            brightness_factor,

        "contrast_factor":
            contrast_factor,

        "saturation_factor":
            saturation_factor,
    }

    for name, value in factors.items():

        if value <= 0:

            raise ValueError(
                f"{name} must be "
                f"greater than zero."
            )

    result = (
        ImageEnhance.Brightness(
            image
        )
        .enhance(
            float(
                brightness_factor
            )
        )
    )

    result = (
        ImageEnhance.Contrast(
            result
        )
        .enhance(
            float(
                contrast_factor
            )
        )
    )

    result = (
        ImageEnhance.Color(
            result
        )
        .enhance(
            float(
                saturation_factor
            )
        )
    )

    metadata = (
        CorruptionMetadata(
            corruption_type=
                "color_jitter",

            brightness_factor=
                float(
                    brightness_factor
                ),

            contrast_factor=
                float(
                    contrast_factor
                ),

            saturation_factor=
                float(
                    saturation_factor
                ),
        )
    )

    return (
        result,
        metadata,
    )


def apply_center_crop(
    image: Image.Image,
    crop_ratio: float,
) -> tuple[
    Image.Image,
    CorruptionMetadata,
]:

    image = ensure_rgb(
        image
    )

    if not (
        0.0 < crop_ratio <= 1.0
    ):

        raise ValueError(
            "Crop ratio must be "
            "greater than 0 and <= 1."
        )

    original_width, original_height = (
        image.size
    )

    crop_width = max(
        1,
        int(
            round(
                original_width
                * crop_ratio
            )
        ),
    )

    crop_height = max(
        1,
        int(
            round(
                original_height
                * crop_ratio
            )
        ),
    )

    left = (
        original_width
        - crop_width
    ) // 2

    top = (
        original_height
        - crop_height
    ) // 2

    right = (
        left
        + crop_width
    )

    bottom = (
        top
        + crop_height
    )

    cropped = image.crop(
        (
            left,
            top,
            right,
            bottom,
        )
    )

    # Restore original dimensions.
    #
    # This makes comparisons easier because every corruption
    # returns the same image dimensions as the input.
    restored = cropped.resize(
        (
            original_width,
            original_height,
        ),
        resample=Image.Resampling.BICUBIC,
    )

    metadata = (
        CorruptionMetadata(
            corruption_type="crop",
            crop_ratio=float(
                crop_ratio
            ),
        )
    )

    return (
        restored,
        metadata,
    )