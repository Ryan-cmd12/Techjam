from __future__ import annotations

from typing import Any

import torch

from transformers import (
    AutoProcessor,
)


class PairedCLIPBatchCollator:

    def __init__(
        self,
        model_name: str,
    ):

        print(
            f"Loading paired CLIP processor: "
            f"{model_name}"
        )

        self.processor = (
            AutoProcessor.from_pretrained(
                model_name
            )
        )


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

        # Process everything together.
        #
        # Instead of making two separate processor calls:
        #
        # clean → processor
        # corrupt → processor
        #
        # we concatenate them and split afterwards.
        combined_images = (
            clean_images
            + corrupted_images
        )

        processed = (
            self.processor(
                images=
                    combined_images,

                return_tensors=
                    "pt",
            )
        )

        combined_pixel_values = (
            processed[
                "pixel_values"
            ]
        )

        batch_size = (
            len(
                samples
            )
        )

        clean_pixel_values = (
            combined_pixel_values[
                :batch_size
            ]
        )

        corrupted_pixel_values = (
            combined_pixel_values[
                batch_size:
            ]
        )

        labels = torch.stack(
            [
                sample[
                    "label"
                ]
                for sample
                in samples
            ]
        )

        batch = {
            "clean_pixel_values":
                clean_pixel_values,

            "corrupted_pixel_values":
                corrupted_pixel_values,

            "labels":
                labels,
        }

        # --------------------------------------------------
        # Corruption metadata
        # --------------------------------------------------

        corruption_types = [
            sample[
                "corruption"
            ][
                "corruption_type"
            ]
            for sample
            in samples
        ]

        batch[
            "corruption_type"
        ] = corruption_types

        numeric_corruption_keys = [
            "jpeg_quality",
            "blur_sigma",
            "resize_scale",
            "noise_sigma",
            "brightness_factor",
            "contrast_factor",
            "saturation_factor",
            "crop_ratio",
        ]

        for key in numeric_corruption_keys:

            values = [
                float(
                    sample[
                        "corruption"
                    ][
                        key
                    ]
                )
                for sample
                in samples
            ]

            batch[key] = (
                torch.tensor(
                    values,
                    dtype=torch.float32,
                )
            )

        # --------------------------------------------------
        # Original dataset metadata
        # --------------------------------------------------

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