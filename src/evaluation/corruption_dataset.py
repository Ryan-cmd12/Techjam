from __future__ import annotations

import hashlib

from typing import Any

from torch.utils.data import (
    Dataset,
)

from src.augmentations.pipeline import (
    CorruptionPipeline,
)


class CorruptedEvaluationDataset(
    Dataset
):

    def __init__(
        self,
        base_dataset: Dataset,
        corruption_type: str,
        severity: float | int | None,
        seed: int = 42,
    ):

        self.base_dataset = (
            base_dataset
        )

        self.corruption_type = (
            corruption_type
        )

        self.severity = (
            severity
        )

        self.seed = (
            seed
        )


    def __len__(
        self,
    ) -> int:

        return len(
            self.base_dataset
        )


    def _build_item_seed(
        self,
        sample: dict[str, Any],
        index: int,
    ) -> int:

        content_hash = str(
            sample.get(
                "content_hash",
                index,
            )
        )

        seed_string = (
            f"{self.seed}|"
            f"{content_hash}|"
            f"{self.corruption_type}|"
            f"{self.severity}"
        )

        digest = (
            hashlib.sha256(
                seed_string.encode(
                    "utf-8"
                )
            )
            .hexdigest()
        )

        # Use the first 8 hexadecimal digits.
        #
        # This gives a stable 32-bit seed.
        item_seed = int(
            digest[:8],
            16,
        )

        return item_seed


    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:

        base_sample = (
            self.base_dataset[
                index
            ]
        )

        sample = dict(
            base_sample
        )

        image = sample[
            "image"
        ]

        item_seed = (
            self._build_item_seed(
                sample=sample,
                index=index,
            )
        )

        pipeline = (
            CorruptionPipeline(
                seed=item_seed
            )
        )

        (
            corrupted_image,
            corruption_metadata,
        ) = (
            pipeline.apply_specific(
                image=image,

                corruption_type=
                    self.corruption_type,

                severity=
                    self.severity,
            )
        )

        sample[
            "image"
        ] = (
            corrupted_image
        )

        sample[
            "corruption"
        ] = (
            corruption_metadata
            .to_dict()
        )

        sample[
            "evaluation_corruption"
        ] = (
            self.corruption_type
        )

        sample[
            "evaluation_severity"
        ] = (
            self.severity
        )

        return sample