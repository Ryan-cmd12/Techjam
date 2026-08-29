from __future__ import annotations

from typing import Any

import torch

from src.data.native_tile_collate import (
    NativeTileCLIPBatchCollator,
)


class CorruptionNativeTileCLIPBatchCollator:

    def __init__(
        self,
        model_name: str,
        tile_size: int = 256,
        max_tiles: int = 6,
        feature_map_size: int = 64,
        sampling_mode: str = "grid",
        seed: int = 42,
    ):

        self.base_collator = (
            NativeTileCLIPBatchCollator(

                model_name=
                    model_name,

                tile_size=
                    tile_size,

                max_tiles=
                    max_tiles,

                feature_map_size=
                    feature_map_size,

                sampling_mode=
                    sampling_mode,

                seed=
                    seed,
            )
        )


    def __call__(
        self,
        samples: list[
            dict[str, Any]
        ],
    ) -> dict[str, Any]:

        batch = (
            self.base_collator(
                samples
            )
        )

        batch[
            "corruption_type"
        ] = [
            sample[
                "corruption"
            ][
                "corruption_type"
            ]
            for sample
            in samples
        ]

        numeric_keys = [
            "jpeg_quality",
            "blur_sigma",
            "resize_scale",
            "noise_sigma",
            "brightness_factor",
            "contrast_factor",
            "saturation_factor",
            "crop_ratio",
        ]

        for key in numeric_keys:

            batch[
                key
            ] = torch.tensor(

                [
                    float(
                        sample[
                            "corruption"
                        ][
                            key
                        ]
                    )

                    for sample
                    in samples
                ],

                dtype=
                    torch.float32,
            )

        return batch