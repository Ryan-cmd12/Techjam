from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset

from src.augmentations.pipeline import (
    CorruptionPipeline,
)


class CorruptionTrainingDataset(
    Dataset
):

    def __init__(
        self,
        base_dataset: Dataset,
        seed: int = 42,
        clean_probability: float = 1.0 / 7.0,
    ):

        self.base_dataset = (
            base_dataset
        )

        self.pipeline = (
            CorruptionPipeline(
                seed=seed
            )
        )

        self.clean_probability = (
            clean_probability
        )


    def __len__(
        self,
    ) -> int:

        return len(
            self.base_dataset
        )


    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:

        sample = dict(
            self.base_dataset[
                index
            ]
        )

        image = sample[
            "image"
        ]

        (
            corrupted,
            metadata,
        ) = (
            self.pipeline.apply_random(

                image=image,

                clean_probability=
                    self.clean_probability,
            )
        )

        sample[
            "image"
        ] = corrupted

        sample[
            "corruption"
        ] = (
            metadata.to_dict()
        )

        return sample