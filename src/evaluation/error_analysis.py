from __future__ import annotations

import json
import math
import shutil

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from PIL import (
    Image,
    ImageDraw,
)

from torch.utils.data import (
    DataLoader,
)

from tqdm import tqdm

from src.training.corruption_targets import (
    ID_TO_CORRUPTION,
)

from src.training.metrics import (
    compute_binary_metrics,
)


# ============================================================
# GENERAL HELPERS
# ============================================================


def sigmoid_numpy(
    values,
):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    result = np.empty_like(
        values
    )

    positive = (
        values >= 0
    )

    result[
        positive
    ] = (
        1.0
        / (
            1.0
            + np.exp(
                -values[
                    positive
                ]
            )
        )
    )

    negative = (
        ~positive
    )

    exp_values = np.exp(
        values[
            negative
        ]
    )

    result[
        negative
    ] = (
        exp_values
        / (
            1.0
            + exp_values
        )
    )

    return result


def resolve_image_path(
    value: str,
) -> Path:

    path = Path(
        value
    )

    if path.is_absolute():

        return path

    return (
        Path.cwd()
        / path
    )


# ============================================================
# ATTENTION
# ============================================================


def normalized_attention_entropy(
    attention: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:

    attention = (
        attention
        .clamp_min(
            1e-8
        )
    )

    entropy = -(
        attention
        * torch.log(
            attention
        )
        * mask.float()
    ).sum(
        dim=1
    )

    counts = (
        mask
        .sum(
            dim=1
        )
        .float()
    )

    maximum_entropy = (
        torch.log(
            counts.clamp_min(
                1.0
            )
        )
    )

    output = torch.zeros_like(
        entropy
    )

    multiple_tiles = (
        counts > 1
    )

    output[
        multiple_tiles
    ] = (
        entropy[
            multiple_tiles
        ]
        / maximum_entropy[
            multiple_tiles
        ]
    )

    return output


# ============================================================
# MODEL OUTPUT COLLECTION
# ============================================================


@torch.no_grad()
def collect_diagnostic_outputs(
    model,
    dataset,
    collator,
    device,
    batch_size: int,
    num_workers: int,
    condition_key: str,
    condition_name: str,
    temperature: float = 1.0,
) -> pd.DataFrame:

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

    records = []

    for batch in tqdm(
        dataloader,
        desc=condition_name,
    ):

        pixel_values = (
            batch[
                "pixel_values"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        forensic_tiles = (
            batch[
                "forensic_tiles"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        tile_mask = (
            batch[
                "tile_mask"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        output = (
            model.forward_with_details(

                pixel_values=
                    pixel_values,

                forensic_tiles=
                    forensic_tiles,

                tile_mask=
                    tile_mask,
            )
        )

        logits = (
            output[
                "logits"
            ]
        )

        base_logits = (
            output[
                "base_logits"
            ]
        )

        raw_probability = (
            torch.sigmoid(
                logits
            )
        )

        calibrated_probability = (
            torch.sigmoid(
                logits
                / temperature
            )
        )

        calibrated_base_probability = (
            torch.sigmoid(
                base_logits
                / temperature
            )
        )

        semantic_probability = (
            torch.sigmoid(
                output[
                    "semantic_logits"
                ]
            )
        )

        forensic_probability = (
            torch.sigmoid(
                output[
                    "forensic_logits"
                ]
            )
        )

        corruption_ids = (
            output[
                "corruption_type_probabilities"
            ]
            .argmax(
                dim=-1
            )
        )

        attention = (
            output[
                "attention"
            ]
        )

        entropy = (
            normalized_attention_entropy(
                attention=
                    attention,

                mask=
                    tile_mask,
            )
        )

        masked_attention = (
            attention
            .masked_fill(
                ~tile_mask,
                0.0,
            )
        )

        max_attention = (
            masked_attention.max(
                dim=1
            ).values
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

        logits_np = (
            logits
            .cpu()
            .numpy()
        )

        base_logits_np = (
            base_logits
            .cpu()
            .numpy()
        )

        raw_probability_np = (
            raw_probability
            .cpu()
            .numpy()
        )

        probability_np = (
            calibrated_probability
            .cpu()
            .numpy()
        )

        base_probability_np = (
            calibrated_base_probability
            .cpu()
            .numpy()
        )

        semantic_np = (
            semantic_probability
            .cpu()
            .numpy()
        )

        forensic_np = (
            forensic_probability
            .cpu()
            .numpy()
        )

        semantic_weight_np = (
            output[
                "semantic_weight"
            ]
            .cpu()
            .numpy()
        )

        forensic_weight_np = (
            output[
                "forensic_weight"
            ]
            .cpu()
            .numpy()
        )

        severity_np = (
            output[
                "corruption_severity"
            ]
            .cpu()
            .numpy()
        )

        corruption_ids_np = (
            corruption_ids
            .cpu()
            .numpy()
        )

        entropy_np = (
            entropy
            .cpu()
            .numpy()
        )

        max_attention_np = (
            max_attention
            .cpu()
            .numpy()
        )

        for index in range(
            len(
                labels
            )
        ):

            probability = float(
                probability_np[
                    index
                ]
            )

            base_probability = float(
                base_probability_np[
                    index
                ]
            )

            label = int(
                labels[
                    index
                ]
            )

            prediction = int(
                probability
                >= 0.5
            )

            base_prediction = int(
                base_probability
                >= 0.5
            )

            final_correct = (
                prediction
                == label
            )

            base_correct = (
                base_prediction
                == label
            )

            semantic_value = float(
                semantic_np[
                    index
                ]
            )

            forensic_value = float(
                forensic_np[
                    index
                ]
            )

            records.append(
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

                    "dataset":
                        batch.get(
                            "dataset",
                            [
                                "unknown"
                            ]
                            * len(
                                labels
                            ),
                        )[
                            index
                        ],

                    "source":
                        batch.get(
                            "source",
                            [
                                "unknown"
                            ]
                            * len(
                                labels
                            ),
                        )[
                            index
                        ],

                    "generator":
                        batch.get(
                            "generator",
                            [
                                "unknown"
                            ]
                            * len(
                                labels
                            ),
                        )[
                            index
                        ],

                    "label":
                        label,

                    "condition_key":
                        condition_key,

                    "condition_name":
                        condition_name,

                    "logit":
                        float(
                            logits_np[
                                index
                            ]
                        ),

                    "base_logit":
                        float(
                            base_logits_np[
                                index
                            ]
                        ),

                    "raw_pred":
                        float(
                            raw_probability_np[
                                index
                            ]
                        ),

                    "pred":
                        probability,

                    "base_pred":
                        base_probability,

                    "prediction":
                        prediction,

                    "base_prediction":
                        base_prediction,

                    "correct":
                        bool(
                            final_correct
                        ),

                    "base_correct":
                        bool(
                            base_correct
                        ),

                    "gate_helped":
                        bool(
                            final_correct
                            and not base_correct
                        ),

                    "gate_hurt":
                        bool(
                            base_correct
                            and not final_correct
                        ),

                    "confidence":
                        max(
                            probability,
                            1.0
                            - probability,
                        ),

                    "semantic_pred":
                        semantic_value,

                    "forensic_pred":
                        forensic_value,

                    "branch_disagreement":
                        abs(
                            semantic_value
                            - forensic_value
                        ),

                    "semantic_weight":
                        float(
                            semantic_weight_np[
                                index
                            ]
                        ),

                    "forensic_weight":
                        float(
                            forensic_weight_np[
                                index
                            ]
                        ),

                    "predicted_corruption":
                        ID_TO_CORRUPTION.get(
                            int(
                                corruption_ids_np[
                                    index
                                ]
                            ),
                            "unknown",
                        ),

                    "predicted_severity":
                        float(
                            severity_np[
                                index
                            ]
                        ),

                    "attention_entropy":
                        float(
                            entropy_np[
                                index
                            ]
                        ),

                    "max_tile_attention":
                        float(
                            max_attention_np[
                                index
                            ]
                        ),
                }
            )

    return pd.DataFrame(
        records
    )


# ============================================================
# RELIABILITY
# ============================================================


def attach_reliability(
    dataframe: pd.DataFrame,
    reliability_hashes,
    reliability_probabilities,
) -> pd.DataFrame:

    mapping = {
        content_hash:
            float(
                probability
            )

        for (
            content_hash,
            probability,
        ) in zip(
            reliability_hashes,
            reliability_probabilities,
        )
    }

    dataframe = (
        dataframe.copy()
    )

    dataframe[
        "reliability"
    ] = (
        dataframe[
            "content_hash"
        ]
        .map(
            mapping
        )
    )

    return dataframe


# ============================================================
# CONDITION METRICS
# ============================================================


def build_condition_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for (
        condition_key,
        condition,
    ) in dataframe.groupby(
        "condition_key"
    ):

        labels = (
            condition[
                "label"
            ]
            .to_numpy()
        )

        probabilities = (
            condition[
                "pred"
            ]
            .to_numpy()
        )

        metrics = (
            compute_binary_metrics(

                labels=
                    labels,

                probabilities=
                    probabilities,

                threshold=
                    0.5,
            )
        )

        row = {
            "condition_key":
                condition_key,

            "condition_name":
                condition[
                    "condition_name"
                ].iloc[
                    0
                ],

            **metrics,

            "mean_confidence":
                float(
                    condition[
                        "confidence"
                    ].mean()
                ),

            "mean_reliability":
                (
                    float(
                        condition[
                            "reliability"
                        ].mean()
                    )

                    if (
                        "reliability"
                        in condition.columns
                        and condition[
                            "reliability"
                        ].notna().any()
                    )

                    else np.nan
                ),

            "mean_branch_disagreement":
                float(
                    condition[
                        "branch_disagreement"
                    ].mean()
                ),

            "mean_semantic_weight":
                float(
                    condition[
                        "semantic_weight"
                    ].mean()
                ),

            "mean_forensic_weight":
                float(
                    condition[
                        "forensic_weight"
                    ].mean()
                ),

            "gate_helped":
                int(
                    condition[
                        "gate_helped"
                    ].sum()
                ),

            "gate_hurt":
                int(
                    condition[
                        "gate_hurt"
                    ].sum()
                ),
        }

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# IMAGE-LEVEL STABILITY
# ============================================================


def build_image_stability(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    clean = (
        dataframe[
            dataframe[
                "condition_key"
            ]
            == "clean"
        ]
        .copy()
        .set_index(
            "content_hash"
        )
    )

    rows = []

    for (
        content_hash,
        group,
    ) in dataframe.groupby(
        "content_hash"
    ):

        if (
            content_hash
            not in clean.index
        ):

            continue

        clean_row = clean.loc[
            content_hash
        ]

        clean_probability = float(
            clean_row[
                "pred"
            ]
        )

        clean_prediction = int(
            clean_row[
                "prediction"
            ]
        )

        clean_correct = bool(
            clean_row[
                "correct"
            ]
        )

        transformed = group[
            group[
                "condition_key"
            ]
            != "clean"
        ]

        if transformed.empty:

            continue

        probabilities = (
            transformed[
                "pred"
            ]
            .to_numpy()
        )

        predictions = (
            transformed[
                "prediction"
            ]
            .to_numpy()
        )

        shifts = np.abs(
            probabilities
            - clean_probability
        )

        flip_mask = (
            predictions
            != clean_prediction
        )

        incorrect_mask = (
            ~transformed[
                "correct"
            ].to_numpy(
                dtype=bool
            )
        )

        worst_shift_index = int(
            np.argmax(
                shifts
            )
        )

        worst_shift_row = (
            transformed.iloc[
                worst_shift_index
            ]
        )

        rows.append(
            {
                "content_hash":
                    content_hash,

                "image_path":
                    clean_row[
                        "image_path"
                    ],

                "dataset":
                    clean_row[
                        "dataset"
                    ],

                "generator":
                    clean_row[
                        "generator"
                    ],

                "label":
                    int(
                        clean_row[
                            "label"
                        ]
                    ),

                "clean_pred":
                    clean_probability,

                "clean_correct":
                    clean_correct,

                "min_pred":
                    float(
                        probabilities.min()
                    ),

                "max_pred":
                    float(
                        probabilities.max()
                    ),

                "prediction_range":
                    float(
                        probabilities.max()
                        - probabilities.min()
                    ),

                "max_clean_shift":
                    float(
                        shifts.max()
                    ),

                "flip_count":
                    int(
                        flip_mask.sum()
                    ),

                "flip_rate":
                    float(
                        flip_mask.mean()
                    ),

                "any_flip":
                    bool(
                        flip_mask.any()
                    ),

                "incorrect_condition_count":
                    int(
                        incorrect_mask.sum()
                    ),

                "clean_correct_but_failed":
                    bool(
                        clean_correct
                        and incorrect_mask.any()
                    ),

                "worst_shift_condition":
                    worst_shift_row[
                        "condition_name"
                    ],

                "worst_shift_pred":
                    float(
                        worst_shift_row[
                            "pred"
                        ]
                    ),

                "reliability":
                    (
                        float(
                            clean_row[
                                "reliability"
                            ]
                        )

                        if (
                            "reliability"
                            in clean_row.index
                            and pd.notna(
                                clean_row[
                                    "reliability"
                                ]
                            )
                        )

                        else np.nan
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# FAILURE CATEGORIES
# ============================================================


def select_error_categories(
    predictions: pd.DataFrame,
    stability: pd.DataFrame,
    confident_threshold: float,
    disagreement_threshold: float,
    low_reliability_threshold: float,
) -> dict[
    str,
    pd.DataFrame
]:

    clean = (
        predictions[
            predictions[
                "condition_key"
            ]
            == "clean"
        ]
        .copy()
    )

    false_positive = (
        clean[
            (
                clean[
                    "label"
                ]
                == 0
            )
            & (
                clean[
                    "prediction"
                ]
                == 1
            )
        ]
        .sort_values(
            "pred",
            ascending=False,
        )
    )

    false_negative = (
        clean[
            (
                clean[
                    "label"
                ]
                == 1
            )
            & (
                clean[
                    "prediction"
                ]
                == 0
            )
        ]
        .sort_values(
            "pred",
            ascending=True,
        )
    )

    confident_fp = (
        false_positive[
            false_positive[
                "pred"
            ]
            >= confident_threshold
        ]
    )

    confident_fn = (
        false_negative[
            false_negative[
                "pred"
            ]
            <= (
                1.0
                - confident_threshold
            )
        ]
    )

    disagreement = (
        clean[
            clean[
                "branch_disagreement"
            ]
            >= disagreement_threshold
        ]
        .sort_values(
            "branch_disagreement",
            ascending=False,
        )
    )

    gate_helped = (
        predictions[
            predictions[
                "gate_helped"
            ]
        ]
        .sort_values(
            "confidence",
            ascending=False,
        )
    )

    gate_hurt = (
        predictions[
            predictions[
                "gate_hurt"
            ]
        ]
        .sort_values(
            "confidence",
            ascending=False,
        )
    )

    if (
        "reliability"
        in clean.columns
    ):

        low_reliability = (
            clean[
                clean[
                    "reliability"
                ]
                < low_reliability_threshold
            ]
            .sort_values(
                "reliability",
                ascending=True,
            )
        )

    else:

        low_reliability = (
            clean.iloc[
                0:0
            ]
        )

    transformation_flip_hashes = set(
        stability.loc[
            stability[
                "clean_correct_but_failed"
            ],
            "content_hash",
        ]
    )

    transformation_failures = (
        predictions[
            (
                predictions[
                    "content_hash"
                ]
                .isin(
                    transformation_flip_hashes
                )
            )
            & (
                predictions[
                    "condition_key"
                ]
                != "clean"
            )
            & (
                ~predictions[
                    "correct"
                ]
            )
        ]
        .copy()
    )

    if not transformation_failures.empty:

        transformation_failures[
            "wrong_confidence"
        ] = np.where(

            transformation_failures[
                "label"
            ]
            == 1,

            1.0
            - transformation_failures[
                "pred"
            ],

            transformation_failures[
                "pred"
            ],
        )

        transformation_failures = (
            transformation_failures
            .sort_values(
                "wrong_confidence",
                ascending=False,
            )
        )

    return {
        "clean_false_positive":
            false_positive,

        "clean_false_negative":
            false_negative,

        "confident_false_positive":
            confident_fp,

        "confident_false_negative":
            confident_fn,

        "branch_disagreement":
            disagreement,

        "gate_helped":
            gate_helped,

        "gate_hurt":
            gate_hurt,

        "low_reliability":
            low_reliability,

        "transformation_failure":
            transformation_failures,
    }


# ============================================================
# DATASET / GENERATOR ANALYSIS
# ============================================================


def build_group_summary(
    dataframe: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:

    clean = (
        dataframe[
            dataframe[
                "condition_key"
            ]
            == "clean"
        ]
    )

    rows = []

    for (
        group_name,
        group,
    ) in clean.groupby(
        group_column
    ):

        if len(
            group
        ) < 2:

            continue

        labels = (
            group[
                "label"
            ]
            .to_numpy()
        )

        probabilities = (
            group[
                "pred"
            ]
            .to_numpy()
        )

        # AUROC requires both classes.
        if len(
            np.unique(
                labels
            )
        ) >= 2:

            metrics = (
                compute_binary_metrics(
                    labels=
                        labels,

                    probabilities=
                        probabilities,

                    threshold=
                        0.5,
                )
            )

            auroc = (
                metrics[
                    "auroc"
                ]
            )

        else:

            predictions = (
                probabilities
                >= 0.5
            ).astype(
                np.int64
            )

            auroc = np.nan

            metrics = {
                "accuracy":
                    float(
                        (
                            predictions
                            == labels
                        ).mean()
                    )
            }

        rows.append(
            {
                group_column:
                    group_name,

                "count":
                    len(
                        group
                    ),

                "accuracy":
                    float(
                        metrics[
                            "accuracy"
                        ]
                    ),

                "auroc":
                    auroc,

                "mean_confidence":
                    float(
                        group[
                            "confidence"
                        ].mean()
                    ),

                "mean_reliability":
                    (
                        float(
                            group[
                                "reliability"
                            ].mean()
                        )

                        if (
                            "reliability"
                            in group.columns
                            and group[
                                "reliability"
                            ].notna().any()
                        )

                        else np.nan
                    ),

                "mean_branch_disagreement":
                    float(
                        group[
                            "branch_disagreement"
                        ].mean()
                    ),
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "accuracy",
            ascending=True,
        )
        if rows
        else pd.DataFrame()
    )


# ============================================================
# CONTACT SHEETS
# ============================================================


def create_contact_sheet(
    dataframe: pd.DataFrame,
    output_path: str | Path,
    title: str,
    top_k: int,
    columns: int = 4,
    cell_width: int = 300,
    cell_height: int = 330,
):

    if dataframe.empty:

        return

    dataframe = (
        dataframe
        .head(
            top_k
        )
        .copy()
    )

    rows = math.ceil(
        len(
            dataframe
        )
        / columns
    )

    title_height = 50

    sheet = Image.new(
        "RGB",

        (
            columns
            * cell_width,

            title_height
            + rows
            * cell_height,
        ),

        "white",
    )

    draw = ImageDraw.Draw(
        sheet
    )

    draw.text(
        (
            10,
            15,
        ),

        title,

        fill="black",
    )

    for position, (
        _,
        row,
    ) in enumerate(
        dataframe.iterrows()
    ):

        column = (
            position
            % columns
        )

        grid_row = (
            position
            // columns
        )

        x = (
            column
            * cell_width
        )

        y = (
            title_height
            + grid_row
            * cell_height
        )

        image_path = resolve_image_path(
            str(
                row[
                    "image_path"
                ]
            )
        )

        try:

            with Image.open(
                image_path
            ) as image:

                image = (
                    image
                    .convert(
                        "RGB"
                    )
                )

                image.thumbnail(
                    (
                        cell_width
                        - 20,

                        220,
                    )
                )

                image_x = (
                    x
                    + (
                        cell_width
                        - image.width
                    )
                    // 2
                )

                image_y = (
                    y
                    + 5
                )

                sheet.paste(
                    image,
                    (
                        image_x,
                        image_y,
                    ),
                )

        except Exception:

            draw.text(
                (
                    x + 10,
                    y + 20,
                ),

                "IMAGE LOAD FAILED",

                fill="black",
            )

        probability = float(
            row.get(
                "pred",
                0.0,
            )
        )

        base_probability = float(
            row.get(
                "base_pred",
                0.0,
            )
        )

        label = int(
            row.get(
                "label",
                -1,
            )
        )

        condition = str(
            row.get(
                "condition_name",
                "Clean",
            )
        )

        reliability = (
            row.get(
                "reliability",
                np.nan,
            )
        )

        semantic = float(
            row.get(
                "semantic_pred",
                0.0,
            )
        )

        forensic = float(
            row.get(
                "forensic_pred",
                0.0,
            )
        )

        text_y = (
            y + 235
        )

        text_lines = [
            (
                f"Label: {label}  "
                f"Final: {probability:.3f}"
            ),

            (
                f"Base: {base_probability:.3f}  "
                f"S/F: {semantic:.2f}/{forensic:.2f}"
            ),

            (
                f"{condition[:34]}"
            ),
        ]

        if pd.notna(
            reliability
        ):

            text_lines.append(
                f"Reliability: "
                f"{float(reliability):.3f}"
            )

        for offset, line in enumerate(
            text_lines
        ):

            draw.text(
                (
                    x + 8,
                    text_y
                    + offset
                    * 18,
                ),

                line,

                fill="black",
            )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sheet.save(
        output_path
    )


# ============================================================
# COPY RAW EXAMPLES
# ============================================================


def copy_example_images(
    dataframe: pd.DataFrame,
    output_directory: str | Path,
    top_k: int,
):

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for index, (
        _,
        row,
    ) in enumerate(
        dataframe.head(
            top_k
        ).iterrows(),
        start=1,
    ):

        source = resolve_image_path(
            str(
                row[
                    "image_path"
                ]
            )
        )

        if not source.exists():

            continue

        destination = (
            output_directory
            / (
                f"{index:03d}_"
                f"{source.name}"
            )
        )

        try:

            shutil.copy2(
                source,
                destination,
            )

        except Exception:

            pass


# ============================================================
# REPORT
# ============================================================


def build_report(
    predictions: pd.DataFrame,
    stability: pd.DataFrame,
    categories: dict[
        str,
        pd.DataFrame
    ],
) -> dict:

    clean = predictions[
        predictions[
            "condition_key"
        ]
        == "clean"
    ]

    transformed = predictions[
        predictions[
            "condition_key"
        ]
        != "clean"
    ]

    return {
        "num_images":
            int(
                clean[
                    "content_hash"
                ]
                .nunique()
            ),

        "num_condition_rows":
            int(
                len(
                    predictions
                )
            ),

        "clean_errors":
            int(
                (
                    ~clean[
                        "correct"
                    ]
                ).sum()
            ),

        "clean_false_positives":
            int(
                len(
                    categories[
                        "clean_false_positive"
                    ]
                )
            ),

        "clean_false_negatives":
            int(
                len(
                    categories[
                        "clean_false_negative"
                    ]
                )
            ),

        "confident_false_positives":
            int(
                len(
                    categories[
                        "confident_false_positive"
                    ]
                )
            ),

        "confident_false_negatives":
            int(
                len(
                    categories[
                        "confident_false_negative"
                    ]
                )
            ),

        "transformation_failures":
            int(
                stability[
                    "clean_correct_but_failed"
                ].sum()
            ),

        "images_with_prediction_flip":
            int(
                stability[
                    "any_flip"
                ].sum()
            ),

        "gate_helped_rows":
            int(
                predictions[
                    "gate_helped"
                ].sum()
            ),

        "gate_hurt_rows":
            int(
                predictions[
                    "gate_hurt"
                ].sum()
            ),

        "net_gate_corrections":
            int(
                predictions[
                    "gate_helped"
                ].sum()
                - predictions[
                    "gate_hurt"
                ].sum()
            ),

        "mean_clean_branch_disagreement":
            float(
                clean[
                    "branch_disagreement"
                ].mean()
            ),

        "mean_transformed_branch_disagreement":
            float(
                transformed[
                    "branch_disagreement"
                ].mean()
            ),

        "mean_clean_reliability":
            (
                float(
                    clean[
                        "reliability"
                    ].mean()
                )

                if (
                    "reliability"
                    in clean.columns
                    and clean[
                        "reliability"
                    ].notna().any()
                )

                else None
            ),

        "mean_max_clean_shift":
            (
                float(
                    stability[
                        "max_clean_shift"
                    ].mean()
                )

                if not stability.empty
                else None
            ),
    }