from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from torchvision.transforms.functional import (
    pil_to_tensor,
)

from transformers import (
    AutoProcessor,
)

from src.data.native_tiles import (
    NativeTileSampler,
    TileBox,
    crop_tiles,
)

from src.models.native_forensic_features import (
    NativeForensicFeatureExtractor,
)


def pil_to_float_tensor(
    image,
) -> torch.Tensor:

    return (
        pil_to_tensor(
            image
        )
        .float()
        / 255.0
    )


class NativeTileFeatureBuilder:

    def __init__(
        self,
        feature_map_size: int = 64,
    ):

        self.feature_map_size = (
            feature_map_size
        )

        self.extractor = (
            NativeForensicFeatureExtractor()
        )

        self.extractor.eval()

    @torch.no_grad()
    def build(
        self,
        image,
        boxes: list[
            TileBox
        ],
    ) -> torch.Tensor:

        tiles = crop_tiles(
            image,
            boxes,
        )

        feature_maps = []

        for tile in tiles:

            tensor = (
                pil_to_float_tensor(
                    tile
                )
                .unsqueeze(0)
            )

            forensic = (
                self.extractor(
                    tensor
                )[
                    "combined"
                ]
            )

            forensic = (
                F.interpolate(
                    forensic,

                    size=(
                        self.feature_map_size,
                        self.feature_map_size,
                    ),

                    mode="bilinear",

                    align_corners=False,
                )
            )

            feature_maps.append(
                forensic.squeeze(
                    0
                )
            )

        return torch.stack(
            feature_maps,
            dim=0,
        )


def pad_tile_features(
    tile_features: torch.Tensor,
    boxes: list[
        TileBox
    ],
    image_size: tuple[
        int,
        int
    ],
    max_tiles: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:

    tile_count = (
        tile_features.shape[0]
    )

    if tile_count > max_tiles:

        raise ValueError(
            "tile_count exceeds max_tiles"
        )

    channels = (
        tile_features.shape[1]
    )

    height = (
        tile_features.shape[2]
    )

    width = (
        tile_features.shape[3]
    )

    padded_features = torch.zeros(
        (
            max_tiles,
            channels,
            height,
            width,
        ),
        dtype=
            tile_features.dtype,
    )

    padded_features[
        :tile_count
    ] = tile_features

    tile_mask = torch.zeros(
        max_tiles,
        dtype=torch.bool,
    )

    tile_mask[
        :tile_count
    ] = True

    normalized_boxes = torch.zeros(
        (
            max_tiles,
            4,
        ),
        dtype=torch.float32,
    )

    image_width, image_height = (
        image_size
    )

    for index, box in enumerate(
        boxes
    ):

        normalized_boxes[
            index
        ] = torch.tensor(
            box.normalized(
                image_width=
                    image_width,

                image_height=
                    image_height,
            ),

            dtype=torch.float32,
        )

    return (
        padded_features,
        tile_mask,
        normalized_boxes,
    )


class NativeTileCLIPBatchCollator:

    def __init__(
        self,
        model_name: str,
        tile_size: int = 256,
        max_tiles: int = 6,
        feature_map_size: int = 64,
        sampling_mode: str = "grid",
        seed: int = 42,
    ):

        self.processor = (
            AutoProcessor
            .from_pretrained(
                model_name
            )
        )

        self.sampler = (
            NativeTileSampler(
                tile_size=
                    tile_size,

                max_tiles=
                    max_tiles,

                mode=
                    sampling_mode,

                seed=
                    seed,
            )
        )

        self.builder = (
            NativeTileFeatureBuilder(
                feature_map_size=
                    feature_map_size
            )
        )

        self.max_tiles = (
            max_tiles
        )

        self.batch_counter = 0

    def __call__(
        self,
        samples: list[
            dict[str, Any]
        ],
    ) -> dict[str, Any]:

        images = [
            sample[
                "image"
            ]
            for sample
            in samples
        ]

        processed = self.processor(
            images=images,
            return_tensors="pt",
        )

        forensic_batches = []

        masks = []

        box_batches = []

        for sample, image in zip(
            samples,
            images,
        ):

            sample_key = str(
                sample.get(
                    "content_hash",
                    sample.get(
                        "image_path",
                        "",
                    ),
                )
            )

            boxes = (
                self.sampler.sample(
                    image=
                        image,

                    sample_key=
                        sample_key,

                    sampling_token=
                        self.batch_counter,
                )
            )

            features = (
                self.builder.build(
                    image=
                        image,

                    boxes=
                        boxes,
                )
            )

            (
                features,
                mask,
                normalized_boxes,
            ) = pad_tile_features(
                tile_features=
                    features,

                boxes=
                    boxes,

                image_size=
                    image.size,

                max_tiles=
                    self.max_tiles,
            )

            forensic_batches.append(
                features
            )

            masks.append(
                mask
            )

            box_batches.append(
                normalized_boxes
            )

        self.batch_counter += 1

        batch = {
            "pixel_values":
                processed[
                    "pixel_values"
                ],

            "forensic_tiles":
                torch.stack(
                    forensic_batches
                ),

            "tile_mask":
                torch.stack(
                    masks
                ),

            "tile_boxes":
                torch.stack(
                    box_batches
                ),

            "labels":
                torch.stack(
                    [
                        sample[
                            "label"
                        ]
                        for sample
                        in samples
                    ]
                ),
        }

        metadata_keys = [
            "image_path",
            "class_name",
            "dataset",
            "source",
            "generator",
            "original_split",
            "split",
            "content_hash",
        ]

        for key in metadata_keys:

            if key in samples[0]:

                batch[key] = [
                    sample[
                        key
                    ]
                    for sample
                    in samples
                ]

        return batch


class PairedNativeTileCLIPBatchCollator:

    def __init__(
        self,
        model_name: str,
        tile_size: int = 256,
        max_tiles: int = 6,
        feature_map_size: int = 64,
        sampling_mode: str = "random",
        seed: int = 42,
    ):

        self.processor = (
            AutoProcessor
            .from_pretrained(
                model_name
            )
        )

        self.sampler = (
            NativeTileSampler(
                tile_size=
                    tile_size,

                max_tiles=
                    max_tiles,

                mode=
                    sampling_mode,

                seed=
                    seed,
            )
        )

        self.builder = (
            NativeTileFeatureBuilder(
                feature_map_size=
                    feature_map_size
            )
        )

        self.max_tiles = (
            max_tiles
        )

        self.batch_counter = 0

    def __call__(
        self,
        samples: list[
            dict[str, Any]
        ],
    ) -> dict[str, Any]:

        clean_images = [
            sample[
                "clean_image"
            ]
            for sample
            in samples
        ]

        corrupted_images = [
            sample[
                "corrupted_image"
            ]
            for sample
            in samples
        ]

        combined = (
            clean_images
            + corrupted_images
        )

        processed = self.processor(
            images=
                combined,

            return_tensors=
                "pt",
        )

        batch_size = (
            len(
                samples
            )
        )

        clean_tile_batches = []
        corrupted_tile_batches = []

        masks = []
        box_batches = []

        for (
            sample,
            clean_image,
            corrupted_image,
        ) in zip(
            samples,
            clean_images,
            corrupted_images,
        ):

            sample_key = str(
                sample.get(
                    "content_hash",
                    sample.get(
                        "image_path",
                        "",
                    ),
                )
            )

            # Same coordinates for clean/corrupted views.
            boxes = (
                self.sampler.sample(
                    image=
                        clean_image,

                    sample_key=
                        sample_key,

                    sampling_token=
                        self.batch_counter,
                )
            )

            clean_features = (
                self.builder.build(
                    image=
                        clean_image,

                    boxes=
                        boxes,
                )
            )

            corrupted_features = (
                self.builder.build(
                    image=
                        corrupted_image,

                    boxes=
                        boxes,
                )
            )

            (
                clean_features,
                mask,
                normalized_boxes,
            ) = pad_tile_features(
                tile_features=
                    clean_features,

                boxes=
                    boxes,

                image_size=
                    clean_image.size,

                max_tiles=
                    self.max_tiles,
            )

            (
                corrupted_features,
                _,
                _,
            ) = pad_tile_features(
                tile_features=
                    corrupted_features,

                boxes=
                    boxes,

                image_size=
                    corrupted_image.size,

                max_tiles=
                    self.max_tiles,
            )

            clean_tile_batches.append(
                clean_features
            )

            corrupted_tile_batches.append(
                corrupted_features
            )

            masks.append(
                mask
            )

            box_batches.append(
                normalized_boxes
            )

        self.batch_counter += 1

        combined_pixels = (
            processed[
                "pixel_values"
            ]
        )

        batch = {
            "clean_pixel_values":
                combined_pixels[
                    :batch_size
                ],

            "corrupted_pixel_values":
                combined_pixels[
                    batch_size:
                ],

            "clean_forensic_tiles":
                torch.stack(
                    clean_tile_batches
                ),

            "corrupted_forensic_tiles":
                torch.stack(
                    corrupted_tile_batches
                ),

            "tile_mask":
                torch.stack(
                    masks
                ),

            "tile_boxes":
                torch.stack(
                    box_batches
                ),

            "labels":
                torch.stack(
                    [
                        sample[
                            "label"
                        ]
                        for sample
                        in samples
                    ]
                ),

            "corruption_type":
                [
                    sample[
                        "corruption"
                    ][
                        "corruption_type"
                    ]
                    for sample
                    in samples
                ],
        }

        return batch