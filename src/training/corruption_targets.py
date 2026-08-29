from __future__ import annotations

import torch


CORRUPTION_TYPES = [
    "clean",
    "jpeg",
    "blur",
    "resize",
    "noise",
    "color_jitter",
    "crop",
]


CORRUPTION_TO_ID = {
    name:
        index

    for index, name
    in enumerate(
        CORRUPTION_TYPES
    )
}


ID_TO_CORRUPTION = {
    index:
        name

    for name, index
    in CORRUPTION_TO_ID.items()
}


def build_type_targets(
    corruption_types:
        list[str],

    device:
        torch.device,
) -> torch.Tensor:

    ids = []

    for corruption_type in (
        corruption_types
    ):

        if (
            corruption_type
            not in CORRUPTION_TO_ID
        ):

            raise ValueError(
                "Unknown corruption type: "
                f"{corruption_type}"
            )

        ids.append(
            CORRUPTION_TO_ID[
                corruption_type
            ]
        )

    return torch.tensor(
        ids,
        dtype=torch.long,
        device=device,
    )


def build_severity_targets(
    batch: dict,
    device: torch.device,
) -> torch.Tensor:
    """
    Convert each corruption parameter into a roughly
    normalized 0–1 severity.

    clean:
        0

    jpeg:
        q90 ≈ 0.14
        q30 = 1

    blur:
        sigma 2 = 1

    resize:
        scale .25 = 1

    noise:
        sigma .10 = 1

    color:
        max ±20% = 1

    crop:
        80% = 1
    """

    corruption_types = (
        batch[
            "corruption_type"
        ]
    )

    targets = []

    for index, corruption_type in enumerate(
        corruption_types
    ):

        if corruption_type == "clean":

            severity = 0.0

        elif corruption_type == "jpeg":

            quality = float(
                batch[
                    "jpeg_quality"
                ][
                    index
                ]
            )

            severity = (
                100.0
                - quality
            ) / 70.0

        elif corruption_type == "blur":

            sigma = float(
                batch[
                    "blur_sigma"
                ][
                    index
                ]
            )

            severity = (
                sigma
                / 2.0
            )

        elif corruption_type == "resize":

            scale = float(
                batch[
                    "resize_scale"
                ][
                    index
                ]
            )

            severity = (
                1.0
                - scale
            ) / 0.75

        elif corruption_type == "noise":

            sigma = float(
                batch[
                    "noise_sigma"
                ][
                    index
                ]
            )

            severity = (
                sigma
                / 0.10
            )

        elif (
            corruption_type
            == "color_jitter"
        ):

            brightness = abs(
                float(
                    batch[
                        "brightness_factor"
                    ][
                        index
                    ]
                )
                - 1.0
            )

            contrast = abs(
                float(
                    batch[
                        "contrast_factor"
                    ][
                        index
                    ]
                )
                - 1.0
            )

            saturation = abs(
                float(
                    batch[
                        "saturation_factor"
                    ][
                        index
                    ]
                )
                - 1.0
            )

            severity = (
                max(
                    brightness,
                    contrast,
                    saturation,
                )
                / 0.20
            )

        elif corruption_type == "crop":

            crop_ratio = float(
                batch[
                    "crop_ratio"
                ][
                    index
                ]
            )

            severity = (
                1.0
                - crop_ratio
            ) / 0.20

        else:

            raise ValueError(
                "Unknown corruption type: "
                f"{corruption_type}"
            )

        severity = max(
            0.0,
            min(
                1.0,
                severity,
            ),
        )

        targets.append(
            severity
        )

    return torch.tensor(
        targets,
        dtype=torch.float32,
        device=device,
    )