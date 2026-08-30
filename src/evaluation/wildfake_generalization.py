from __future__ import annotations

import json

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import (
    DataLoader,
)

from tqdm import tqdm

from src.training.metrics import (
    compute_binary_metrics,
)


@torch.no_grad()
def collect_wildfake_predictions(
    model,
    dataset,
    collator,
    device,
    batch_size,
    num_workers,
    temperature,
    condition_key,
    condition_name,
):

    dataloader = DataLoader(

        dataset,

        batch_size=
            batch_size,

        shuffle=
            False,

        num_workers=
            num_workers,

        pin_memory=(
            device.type
            == "cuda"
        ),

        persistent_workers=(
            num_workers > 0
        ),

        collate_fn=
            collator,
    )

    model.eval()

    rows = []

    for batch in tqdm(
        dataloader,
        desc=condition_name,
    ):

        output = (
            model.forward_with_details(

                pixel_values=
                    batch[
                        "pixel_values"
                    ].to(
                        device,
                        non_blocking=True,
                    ),

                forensic_tiles=
                    batch[
                        "forensic_tiles"
                    ].to(
                        device,
                        non_blocking=True,
                    ),

                tile_mask=
                    batch[
                        "tile_mask"
                    ].to(
                        device,
                        non_blocking=True,
                    ),
            )
        )

        probability = (
            torch.sigmoid(
                output[
                    "logits"
                ]
                / temperature
            )
            .cpu()
            .numpy()
        )

        base_probability = (
            torch.sigmoid(
                output[
                    "base_logits"
                ]
                / temperature
            )
            .cpu()
            .numpy()
        )

        semantic_probability = (
            torch.sigmoid(
                output[
                    "semantic_logits"
                ]
            )
            .cpu()
            .numpy()
        )

        forensic_probability = (
            torch.sigmoid(
                output[
                    "forensic_logits"
                ]
            )
            .cpu()
            .numpy()
        )

        semantic_weight = (
            output[
                "semantic_weight"
            ]
            .cpu()
            .numpy()
        )

        forensic_weight = (
            output[
                "forensic_weight"
            ]
            .cpu()
            .numpy()
        )

        severity = (
            output[
                "corruption_severity"
            ]
            .cpu()
            .numpy()
        )

        labels = (
            batch[
                "labels"
            ]
            .cpu()
            .numpy()
            .astype(
                np.int64
            )
        )

        for index in range(
            len(
                labels
            )
        ):

            pred = float(
                probability[
                    index
                ]
            )

            rows.append(
                {
                    "content_hash":
                        batch[
                            "content_hash"
                        ][
                            index
                        ],

                    "image_path":
                        batch[
                            "image_path"
                        ][
                            index
                        ],

                    "label":
                        int(
                            labels[
                                index
                            ]
                        ),

                    "pred":
                        pred,

                    "prediction":
                        int(
                            pred >= 0.5
                        ),

                    "base_pred":
                        float(
                            base_probability[
                                index
                            ]
                        ),

                    "semantic_pred":
                        float(
                            semantic_probability[
                                index
                            ]
                        ),

                    "forensic_pred":
                        float(
                            forensic_probability[
                                index
                            ]
                        ),

                    "semantic_weight":
                        float(
                            semantic_weight[
                                index
                            ]
                        ),

                    "forensic_weight":
                        float(
                            forensic_weight[
                                index
                            ]
                        ),

                    "predicted_severity":
                        float(
                            severity[
                                index
                            ]
                        ),

                    "condition_key":
                        condition_key,

                    "condition_name":
                        condition_name,
                }
            )

    return pd.DataFrame(
        rows
    )


def balanced_real_fake_frame(
    real: pd.DataFrame,
    fake: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:

    count = min(
        len(
            real
        ),
        len(
            fake
        ),
    )

    if count == 0:

        return pd.DataFrame()

    selected_real = (
        real.sample(
            n=count,
            random_state=
                seed,
            replace=False,
        )
    )

    selected_fake = (
        fake.sample(
            n=count,
            random_state=
                seed + 1,
            replace=False,
        )
    )

    return pd.concat(
        [
            selected_real,
            selected_fake,
        ],
        ignore_index=True,
    )


def evaluate_binary_frame(
    dataframe: pd.DataFrame,
) -> dict:

    if dataframe.empty:

        return {}

    labels = (
        dataframe[
            "label"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    if len(
        np.unique(
            labels
        )
    ) < 2:

        return {}

    return (
        compute_binary_metrics(

            labels=
                labels,

            probabilities=
                dataframe[
                    "pred"
                ]
                .to_numpy(
                    dtype=np.float64
                ),

            threshold=
                0.5,
        )
    )


def build_generator_metrics(
    dataframe: pd.DataFrame,
    min_samples: int,
    seed: int,
) -> pd.DataFrame:

    real = dataframe[
        dataframe[
            "label"
        ] == 0
    ]

    fake = dataframe[
        dataframe[
            "label"
        ] == 1
    ]

    rows = []

    generators = sorted(
        fake[
            "generator"
        ]
        .dropna()
        .astype(
            str
        )
        .unique()
        .tolist()
    )

    for index, generator in enumerate(
        generators
    ):

        generator_fake = (
            fake[
                fake[
                    "generator"
                ]
                .astype(
                    str
                )
                == generator
            ]
        )

        if len(
            generator_fake
        ) < min_samples:

            continue

        comparison = (
            balanced_real_fake_frame(

                real=
                    real,

                fake=
                    generator_fake,

                seed=
                    seed
                    + index
                    * 101,
            )
        )

        metrics = (
            evaluate_binary_frame(
                comparison
            )
        )

        if not metrics:

            continue

        fake_accuracy = float(
            (
                generator_fake[
                    "pred"
                ]
                >= 0.5
            ).mean()
        )

        rows.append(
            {
                "generator":
                    generator,

                "family":
                    generator_fake[
                        "generation_family"
                    ].mode().iloc[
                        0
                    ],

                "subcategory":
                    generator_fake[
                        "subcategory"
                    ].mode().iloc[
                        0
                    ],

                "fake_samples":
                    len(
                        generator_fake
                    ),

                "comparison_samples":
                    len(
                        comparison
                    ),

                "fake_recall":
                    fake_accuracy,

                "mean_fake_probability":
                    float(
                        generator_fake[
                            "pred"
                        ].mean()
                    ),

                "mean_semantic_weight":
                    float(
                        generator_fake[
                            "semantic_weight"
                        ].mean()
                    ),

                "mean_forensic_weight":
                    float(
                        generator_fake[
                            "forensic_weight"
                        ].mean()
                    ),

                **metrics,
            }
        )

    dataframe = pd.DataFrame(
        rows
    )

    if not dataframe.empty:

        dataframe = (
            dataframe.sort_values(
                "auroc",
                ascending=True,
            )
            .reset_index(
                drop=True
            )
        )

    return dataframe


def build_family_metrics(
    dataframe: pd.DataFrame,
    min_samples: int,
    seed: int,
) -> pd.DataFrame:

    real = dataframe[
        dataframe[
            "label"
        ] == 0
    ]

    fake = dataframe[
        dataframe[
            "label"
        ] == 1
    ]

    rows = []

    families = sorted(
        fake[
            "generation_family"
        ]
        .dropna()
        .astype(
            str
        )
        .unique()
        .tolist()
    )

    for index, family in enumerate(
        families
    ):

        family_fake = (
            fake[
                fake[
                    "generation_family"
                ]
                .astype(
                    str
                )
                == family
            ]
        )

        if len(
            family_fake
        ) < min_samples:

            continue

        comparison = (
            balanced_real_fake_frame(

                real=
                    real,

                fake=
                    family_fake,

                seed=
                    seed
                    + index
                    * 997,
            )
        )

        metrics = (
            evaluate_binary_frame(
                comparison
            )
        )

        if not metrics:

            continue

        rows.append(
            {
                "generation_family":
                    family,

                "fake_samples":
                    len(
                        family_fake
                    ),

                "fake_recall":
                    float(
                        (
                            family_fake[
                                "pred"
                            ]
                            >= 0.5
                        ).mean()
                    ),

                "mean_fake_probability":
                    float(
                        family_fake[
                            "pred"
                        ].mean()
                    ),

                "mean_semantic_weight":
                    float(
                        family_fake[
                            "semantic_weight"
                        ].mean()
                    ),

                "mean_forensic_weight":
                    float(
                        family_fake[
                            "forensic_weight"
                        ].mean()
                    ),

                **metrics,
            }
        )

    return pd.DataFrame(
        rows
    )


def summarize_condition(
    predictions: pd.DataFrame,
    generator_metrics: pd.DataFrame,
) -> dict:

    overall = (
        evaluate_binary_frame(
            predictions
        )
    )

    real = predictions[
        predictions[
            "label"
        ] == 0
    ]

    fake = predictions[
        predictions[
            "label"
        ] == 1
    ]

    summary = {
        "images":
            len(
                predictions
            ),

        "real_images":
            len(
                real
            ),

        "fake_images":
            len(
                fake
            ),

        "real_specificity":
            float(
                (
                    real[
                        "pred"
                    ]
                    < 0.5
                ).mean()
            ),

        "fake_recall":
            float(
                (
                    fake[
                        "pred"
                    ]
                    >= 0.5
                ).mean()
            ),

        "overall":
            overall,
    }

    if not generator_metrics.empty:

        worst = (
            generator_metrics.iloc[
                0
            ]
        )

        summary[
            "mean_generator_auroc"
        ] = float(
            generator_metrics[
                "auroc"
            ].mean()
        )

        summary[
            "worst_generator"
        ] = str(
            worst[
                "generator"
            ]
        )

        summary[
            "worst_generator_auroc"
        ] = float(
            worst[
                "auroc"
            ]
        )

    return summary