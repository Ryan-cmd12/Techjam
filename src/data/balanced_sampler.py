from __future__ import annotations

import torch

from torch.utils.data import (
    WeightedRandomSampler,
)


def build_dataset_class_balanced_sampler(
    dataset,
    seed: int = 42,
) -> WeightedRandomSampler:

    dataframe = (
        dataset.dataframe
    )

    strata = list(
        zip(
            dataframe[
                "dataset"
            ].astype(
                str
            ),

            dataframe[
                "label"
            ].astype(
                int
            ),
        )
    )

    counts = {}

    for stratum in strata:

        counts[
            stratum
        ] = (
            counts.get(
                stratum,
                0,
            )
            + 1
        )

    print(
        "\nTraining strata:"
    )

    for (
        stratum,
        count,
    ) in sorted(
        counts.items()
    ):

        print(
            f"{stratum}: "
            f"{count:,}"
        )

    weights = [
        1.0
        / counts[
            stratum
        ]

        for stratum
        in strata
    ]

    generator = torch.Generator()

    generator.manual_seed(
        seed
    )

    sampler = (
        WeightedRandomSampler(
            weights=
                weights,

            num_samples=
                len(
                    dataset
                ),

            replacement=
                True,

            generator=
                generator,
        )
    )

    return sampler