from __future__ import annotations

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    log_loss,
    brier_score_loss,
)


def expected_calibration_error(
    labels,
    probabilities,
    num_bins: int = 15,
) -> float:

    labels = np.asarray(
        labels,
        dtype=np.float64,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    predictions = (
        probabilities >= 0.5
    ).astype(
        np.int64
    )

    confidences = np.maximum(
        probabilities,
        1.0 - probabilities,
    )

    correctness = (
        predictions
        == labels
    ).astype(
        np.float64
    )

    bin_edges = np.linspace(
        0.0,
        1.0,
        num_bins + 1,
    )

    ece = 0.0

    total = len(
        labels
    )

    for index in range(
        num_bins
    ):

        lower = (
            bin_edges[
                index
            ]
        )

        upper = (
            bin_edges[
                index + 1
            ]
        )

        if index == 0:

            mask = (
                confidences >= lower
            ) & (
                confidences <= upper
            )

        else:

            mask = (
                confidences > lower
            ) & (
                confidences <= upper
            )

        count = int(
            mask.sum()
        )

        if count == 0:

            continue

        average_confidence = (
            confidences[
                mask
            ].mean()
        )

        average_accuracy = (
            correctness[
                mask
            ].mean()
        )

        ece += (
            count
            / total
        ) * abs(
            average_confidence
            - average_accuracy
        )

    return float(
        ece
    )


def calibration_metrics(
    labels,
    probabilities,
    num_bins: int = 15,
) -> dict[str, float]:

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    probabilities = np.clip(
        probabilities,
        1e-7,
        1.0 - 1e-7,
    )

    predictions = (
        probabilities >= 0.5
    ).astype(
        np.int64
    )

    return {
        "accuracy":
            float(
                accuracy_score(
                    labels,
                    predictions,
                )
            ),

        "nll":
            float(
                log_loss(
                    labels,
                    probabilities,
                    labels=[
                        0,
                        1,
                    ],
                )
            ),

        "brier":
            float(
                brier_score_loss(
                    labels,
                    probabilities,
                )
            ),

        "ece":
            expected_calibration_error(
                labels=
                    labels,

                probabilities=
                    probabilities,

                num_bins=
                    num_bins,
            ),
    }