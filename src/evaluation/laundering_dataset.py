from __future__ import annotations

import hashlib

from dataclasses import dataclass
from typing import Any

from torch.utils.data import Dataset

from src.augmentations.pipeline import (
    CorruptionPipeline,
)


@dataclass(
    frozen=True
)
class LaunderingStep:

    corruption_type: str
    severity: float | int | None


@dataclass(
    frozen=True
)
class LaunderingSpec:

    key: str
    name: str
    steps: tuple[LaunderingStep, ...]


def build_laundering_specs(
    config_entries: list[dict],
) -> list[LaunderingSpec]:

    specs = []

    for entry in config_entries:

        steps = []

        for step in entry.get(
            "steps",
            [],
        ):

            steps.append(
                LaunderingStep(
                    corruption_type=str(
                        step["type"]
                    ),
                    severity=step.get(
                        "severity"
                    ),
                )
            )

        specs.append(
            LaunderingSpec(
                key=str(
                    entry["key"]
                ),
                name=str(
                    entry["name"]
                ),
                steps=tuple(
                    steps
                ),
            )
        )

    return specs


class LaunderingEvaluationDataset(
    Dataset
):

    def __init__(
        self,
        base_dataset: Dataset,
        spec: LaunderingSpec,
        seed: int = 42,
    ):

        self.base_dataset = (
            base_dataset
        )

        self.spec = spec
        self.seed = seed


    def __len__(
        self,
    ) -> int:

        return len(
            self.base_dataset
        )


    def _build_seed(
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

        description = "|".join(
            (
                f"{step.corruption_type}:"
                f"{step.severity}"
            )
            for step in self.spec.steps
        )

        seed_string = (
            f"{self.seed}|"
            f"{content_hash}|"
            f"{self.spec.key}|"
            f"{description}"
        )

        digest = hashlib.sha256(
            seed_string.encode(
                "utf-8"
            )
        ).hexdigest()

        return int(
            digest[:8],
            16,
        )


    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:

        base_sample = dict(
            self.base_dataset[
                index
            ]
        )

        image = base_sample[
            "image"
        ]

        item_seed = (
            self._build_seed(
                sample=base_sample,
                index=index,
            )
        )

        pipeline = (
            CorruptionPipeline(
                seed=item_seed
            )
        )

        metadata_sequence = []

        transformed = image

        for step in self.spec.steps:

            (
                transformed,
                metadata,
            ) = pipeline.apply_specific(
                image=transformed,
                corruption_type=
                    step.corruption_type,
                severity=
                    step.severity,
            )

            metadata_sequence.append(
                metadata.to_dict()
            )

        base_sample[
            "image"
        ] = transformed

        base_sample[
            "laundering_key"
        ] = self.spec.key

        base_sample[
            "laundering_name"
        ] = self.spec.name

        base_sample[
            "laundering_steps"
        ] = metadata_sequence

        return base_sample