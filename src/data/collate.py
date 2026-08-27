from __future__ import annotations

from typing import Any

import torch

from transformers import (
    AutoProcessor,
)


class CLIPBatchCollator:

    def __init__(
        self,
        model_name: str,
    ):

        print(
            f"Loading CLIP processor: "
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

        images = [
            sample["image"]
            for sample
            in samples
        ]

        labels = torch.stack(
            [
                sample["label"]
                for sample
                in samples
            ]
        )

        processed = (
            self.processor(
                images=images,
                return_tensors="pt",
            )
        )

        batch = {
            "pixel_values":
                processed[
                    "pixel_values"
                ],

            "labels":
                labels,
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
                    sample[key]
                    for sample
                    in samples
                ]

        return batch