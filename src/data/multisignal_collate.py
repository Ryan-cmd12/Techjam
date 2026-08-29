from __future__ import annotations

from typing import Any

import torch

from torchvision.transforms.functional import pil_to_tensor
from transformers import AutoProcessor


def pil_to_float_tensor(image) -> torch.Tensor:
    tensor = pil_to_tensor(image).float() / 255.0
    return tensor


def stack_forensic_images(images) -> torch.Tensor:
    tensors = [
        pil_to_float_tensor(image)
        for image in images
    ]

    shapes = {
        tuple(tensor.shape)
        for tensor in tensors
    }

    if len(shapes) != 1:
        raise ValueError(
            "Forensic images in a batch must currently have the "
            "same dimensions. This is fine for CIFAKE. "
            "Variable-resolution datasets will use the native-tile "
            "pipeline in a later step."
        )

    return torch.stack(
        tensors,
        dim=0,
    )


class MultiSignalCLIPBatchCollator:

    def __init__(
        self,
        model_name: str,
    ):
        print(
            f"Loading multi-signal CLIP processor: {model_name}"
        )

        self.processor = AutoProcessor.from_pretrained(
            model_name
        )

    def __call__(
        self,
        samples: list[dict[str, Any]],
    ) -> dict[str, Any]:

        images = [
            sample["image"]
            for sample in samples
        ]

        processed = self.processor(
            images=images,
            return_tensors="pt",
        )

        labels = torch.stack(
            [
                sample["label"]
                for sample in samples
            ]
        )

        batch = {
            "pixel_values":
                processed["pixel_values"],

            "forensic_images":
                stack_forensic_images(images),

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
                    for sample in samples
                ]

        return batch


class PairedMultiSignalCLIPBatchCollator:

    def __init__(
        self,
        model_name: str,
    ):
        print(
            f"Loading paired multi-signal CLIP processor: "
            f"{model_name}"
        )

        self.processor = AutoProcessor.from_pretrained(
            model_name
        )

    def __call__(
        self,
        samples: list[dict[str, Any]],
    ) -> dict[str, Any]:

        clean_images = [
            sample["clean_image"]
            for sample in samples
        ]

        corrupted_images = [
            sample["corrupted_image"]
            for sample in samples
        ]

        combined_images = (
            clean_images
            + corrupted_images
        )

        processed = self.processor(
            images=combined_images,
            return_tensors="pt",
        )

        batch_size = len(samples)

        combined_pixels = processed[
            "pixel_values"
        ]

        clean_pixels = combined_pixels[
            :batch_size
        ]

        corrupted_pixels = combined_pixels[
            batch_size:
        ]

        labels = torch.stack(
            [
                sample["label"]
                for sample in samples
            ]
        )

        batch = {
            "clean_pixel_values":
                clean_pixels,

            "corrupted_pixel_values":
                corrupted_pixels,

            "clean_forensic_images":
                stack_forensic_images(
                    clean_images
                ),

            "corrupted_forensic_images":
                stack_forensic_images(
                    corrupted_images
                ),

            "labels":
                labels,

            "corruption_type":
                [
                    sample["corruption"][
                        "corruption_type"
                    ]
                    for sample in samples
                ],
        }

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
            batch[key] = torch.tensor(
                [
                    float(
                        sample["corruption"][key]
                    )
                    for sample in samples
                ],
                dtype=torch.float32,
            )

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
                    for sample in samples
                ]

        return batch