from __future__ import annotations

import math

import numpy as np


RELIABILITY_FEATURE_NAMES = [
    "confidence",
    "mean_probe_confidence",
    "probe_std",
    "probe_max_shift",
    "probe_flip_rate",
    "branch_disagreement",
    "gate_entropy",
    "predicted_severity",
    "attention_entropy",
]


# ============================================================
# BASIC HELPERS
# ============================================================


def sigmoid_numpy(
    values,
):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    positive = (
        values >= 0
    )

    negative = (
        ~positive
    )

    result = np.empty_like(
        values,
        dtype=np.float64,
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


def normalized_binary_entropy(
    probability_a,
    probability_b,
):

    probabilities = np.stack(
        [
            probability_a,
            probability_b,
        ],
        axis=-1,
    )

    probabilities = np.clip(
        probabilities,
        1e-8,
        1.0,
    )

    entropy = -(
        probabilities
        * np.log(
            probabilities
        )
    ).sum(
        axis=-1
    )

    return (
        entropy
        / math.log(
            2.0
        )
    )


# ============================================================
# COLUMN COMPATIBILITY
# ============================================================


def get_column(
    dataframe,
    primary_name: str,
    aliases: list[str] | None = None,
):

    aliases = (
        aliases
        or []
    )

    candidates = [
        primary_name,
        *aliases,
    ]

    for column in candidates:

        if column in dataframe.columns:

            return dataframe[
                column
            ]

    raise KeyError(
        "None of the expected columns "
        f"exist in the dataframe.\n"
        f"Expected one of: {candidates}\n"
        f"Available columns: "
        f"{list(dataframe.columns)}"
    )


def get_numpy_column(
    dataframe,
    primary_name: str,
    aliases: list[str] | None = None,
    dtype=np.float64,
):

    return (
        get_column(
            dataframe=
                dataframe,

            primary_name=
                primary_name,

            aliases=
                aliases,
        )
        .to_numpy(
            dtype=dtype
        )
    )


# ============================================================
# RELIABILITY FEATURE BUILDING
# ============================================================


def build_reliability_features(
    condition_frames: dict,
    temperature: float,
):

    if "clean" not in (
        condition_frames
    ):

        raise ValueError(
            "Reliability features "
            "require a clean condition."
        )

    # ========================================================
    # CLEAN REFERENCE
    # ========================================================

    clean = (
        condition_frames[
            "clean"
        ]
        .copy()
        .set_index(
            "content_hash"
        )
        .sort_index()
    )

    if clean.index.has_duplicates:

        duplicate_count = int(
            clean.index
            .duplicated()
            .sum()
        )

        raise RuntimeError(
            "Duplicate content hashes "
            "were found inside the clean "
            "reliability condition.\n"
            f"Duplicates: {duplicate_count}"
        )

    hashes = (
        clean.index
        .tolist()
    )

    labels = (
        get_numpy_column(
            dataframe=
                clean,

            primary_name=
                "label",

            dtype=
                np.int64,
        )
    )

    clean_logits = (
        get_numpy_column(
            dataframe=
                clean,

            primary_name=
                "logit",
        )
    )

    # Temperature calibrated probability.
    clean_probability = (
        sigmoid_numpy(
            clean_logits
            / temperature
        )
    )

    # ========================================================
    # PROBE PROBABILITIES
    # ========================================================

    condition_probabilities = []

    condition_names = []

    for (
        condition_name,
        dataframe,
    ) in condition_frames.items():

        aligned = (
            dataframe
            .copy()
            .set_index(
                "content_hash"
            )
            .reindex(
                hashes
            )
        )

        logits_column = get_column(
            dataframe=
                aligned,

            primary_name=
                "logit",
        )

        if logits_column.isna().any():

            missing_count = int(
                logits_column
                .isna()
                .sum()
            )

            raise RuntimeError(
                f"Condition "
                f"'{condition_name}' "
                f"is missing "
                f"{missing_count} "
                f"samples relative to "
                f"the clean condition."
            )

        logits = (
            logits_column
            .to_numpy(
                dtype=np.float64
            )
        )

        probabilities = (
            sigmoid_numpy(
                logits
                / temperature
            )
        )

        condition_probabilities.append(
            probabilities
        )

        condition_names.append(
            condition_name
        )

    probability_matrix = (
        np.stack(
            condition_probabilities,
            axis=1,
        )
    )

    # ========================================================
    # PREDICTIONS
    # ========================================================

    clean_predictions = (
        clean_probability
        >= 0.5
    ).astype(
        np.int64
    )

    probe_predictions = (
        probability_matrix
        >= 0.5
    ).astype(
        np.int64
    )

    # ========================================================
    # CONFIDENCE
    # ========================================================

    confidence = np.maximum(
        clean_probability,
        1.0
        - clean_probability,
    )

    probe_confidence = np.maximum(
        probability_matrix,
        1.0
        - probability_matrix,
    )

    mean_probe_confidence = (
        probe_confidence.mean(
            axis=1
        )
    )

    # ========================================================
    # TRANSFORMATION STABILITY
    # ========================================================

    probe_std = (
        probability_matrix.std(
            axis=1
        )
    )

    probe_max_shift = (
        np.abs(
            probability_matrix
            - clean_probability[
                :,
                None
            ]
        )
        .max(
            axis=1
        )
    )

    probe_flip_rate = (
        probe_predictions
        != clean_predictions[
            :,
            None
        ]
    ).mean(
        axis=1
    )

    # ========================================================
    # BRANCH DISAGREEMENT
    #
    # Step 12 schema:
    #   semantic_probability
    #   forensic_probability
    #
    # Step 13 schema:
    #   semantic_pred
    #   forensic_pred
    #
    # Support both.
    # ========================================================

    semantic_probability = (
        get_numpy_column(

            dataframe=
                clean,

            primary_name=
                "semantic_probability",

            aliases=[
                "semantic_pred",
            ],
        )
    )

    forensic_probability = (
        get_numpy_column(

            dataframe=
                clean,

            primary_name=
                "forensic_probability",

            aliases=[
                "forensic_pred",
            ],
        )
    )

    branch_disagreement = (
        np.abs(
            semantic_probability
            - forensic_probability
        )
    )

    # ========================================================
    # RELIABILITY GATE ENTROPY
    # ========================================================

    semantic_weight = (
        get_numpy_column(

            dataframe=
                clean,

            primary_name=
                "semantic_weight",
        )
    )

    forensic_weight = (
        get_numpy_column(

            dataframe=
                clean,

            primary_name=
                "forensic_weight",
        )
    )

    gate_entropy = (
        normalized_binary_entropy(
            semantic_weight,
            forensic_weight,
        )
    )

    # ========================================================
    # CORRUPTION SEVERITY
    # ========================================================

    predicted_severity = (
        get_numpy_column(

            dataframe=
                clean,

            primary_name=
                "predicted_severity",

            aliases=[
                "corruption_severity",
            ],
        )
    )

    # ========================================================
    # TILE ATTENTION ENTROPY
    # ========================================================

    attention_entropy = (
        get_numpy_column(

            dataframe=
                clean,

            primary_name=
                "attention_entropy",
        )
    )

    # ========================================================
    # FEATURE MATRIX
    # ========================================================

    features = np.stack(
        [
            confidence,
            mean_probe_confidence,
            probe_std,
            probe_max_shift,
            probe_flip_rate,
            branch_disagreement,
            gate_entropy,
            predicted_severity,
            attention_entropy,
        ],

        axis=1,
    )

    if not np.isfinite(
        features
    ).all():

        bad_rows = np.where(
            ~np.isfinite(
                features
            ).all(
                axis=1
            )
        )[0]

        raise RuntimeError(
            "Non-finite values found "
            "while building reliability "
            "features.\n"
            f"Bad rows: "
            f"{bad_rows[:20].tolist()}"
        )

    # ========================================================
    # TARGET
    #
    # A sample counts as robustly correct only if the
    # prediction is correct under EVERY reliability probe.
    # ========================================================

    robust_correctness = (
        probe_predictions
        == labels[
            :,
            None
        ]
    ).all(
        axis=1
    ).astype(
        np.int64
    )

    return {
        "content_hashes":
            hashes,

        "labels":
            labels,

        "clean_probability":
            clean_probability,

        "features":
            features,

        "feature_names":
            RELIABILITY_FEATURE_NAMES,

        "robust_correctness":
            robust_correctness,

        "condition_names":
            condition_names,
    }


# ============================================================
# RELIABILITY CALIBRATOR
# ============================================================


class ReliabilityCalibrator:

    def __init__(
        self,
        feature_names=None,
        mean=None,
        scale=None,
        coefficients=None,
        intercept=None,
        constant_probability=None,
    ):

        self.feature_names = (
            feature_names
            or RELIABILITY_FEATURE_NAMES
        )

        self.mean = (
            None
            if mean is None
            else np.asarray(
                mean,
                dtype=np.float64,
            )
        )

        self.scale = (
            None
            if scale is None
            else np.asarray(
                scale,
                dtype=np.float64,
            )
        )

        self.coefficients = (
            None
            if coefficients is None
            else np.asarray(
                coefficients,
                dtype=np.float64,
            )
        )

        self.intercept = (
            intercept
        )

        self.constant_probability = (
            constant_probability
        )


    # ========================================================
    # FIT
    # ========================================================

    @classmethod
    def fit(
        cls,
        features,
        targets,
        feature_names=None,
    ):

        from sklearn.preprocessing import (
            StandardScaler,
        )

        from sklearn.linear_model import (
            LogisticRegression,
        )

        features = np.asarray(
            features,
            dtype=np.float64,
        )

        targets = np.asarray(
            targets,
            dtype=np.int64,
        )

        if features.ndim != 2:

            raise ValueError(
                "Reliability features "
                "must have shape [N, D]."
            )

        if len(
            features
        ) != len(
            targets
        ):

            raise ValueError(
                "Reliability features "
                "and targets must contain "
                "the same number of rows."
            )

        unique = np.unique(
            targets
        )

        # Very strong detectors can occasionally have
        # every reliability target equal to one.
        if len(
            unique
        ) < 2:

            probability = float(
                targets.mean()
            )

            print(
                "\n[WARNING] "
                "Reliability target "
                "contains only one class."
            )

            print(
                f"Using constant reliability: "
                f"{probability:.4f}"
            )

            return cls(

                feature_names=
                    feature_names,

                constant_probability=
                    probability,
            )

        scaler = StandardScaler()

        normalized = (
            scaler.fit_transform(
                features
            )
        )

        classifier = (
            LogisticRegression(

                class_weight=
                    "balanced",

                max_iter=
                    2000,

                random_state=
                    42,
            )
        )

        classifier.fit(
            normalized,
            targets,
        )

        return cls(

            feature_names=
                feature_names,

            mean=
                scaler.mean_,

            scale=
                scaler.scale_,

            coefficients=
                classifier.coef_[
                    0
                ],

            intercept=
                float(
                    classifier.intercept_[
                        0
                    ]
                ),
        )


    # ========================================================
    # PREDICT
    # ========================================================

    def predict_proba(
        self,
        features,
    ):

        features = np.asarray(
            features,
            dtype=np.float64,
        )

        if features.ndim == 1:

            features = (
                features[
                    None,
                    :
                ]
            )

        if (
            self.constant_probability
            is not None
        ):

            return np.full(
                features.shape[
                    0
                ],

                self.constant_probability,

                dtype=np.float64,
            )

        if (
            self.mean is None
            or self.scale is None
            or self.coefficients is None
            or self.intercept is None
        ):

            raise RuntimeError(
                "ReliabilityCalibrator "
                "has not been fitted."
            )

        scale = np.where(
            self.scale
            == 0,

            1.0,

            self.scale,
        )

        normalized = (
            features
            - self.mean
        ) / scale

        logits = (
            normalized
            @ self.coefficients
            + self.intercept
        )

        return sigmoid_numpy(
            logits
        )


    # ========================================================
    # SERIALIZATION
    # ========================================================

    def to_dict(
        self,
    ) -> dict:

        return {
            "feature_names":
                list(
                    self.feature_names
                ),

            "mean":
                (
                    None

                    if self.mean
                    is None

                    else self.mean.tolist()
                ),

            "scale":
                (
                    None

                    if self.scale
                    is None

                    else self.scale.tolist()
                ),

            "coefficients":
                (
                    None

                    if self.coefficients
                    is None

                    else self.coefficients.tolist()
                ),

            "intercept":
                self.intercept,

            "constant_probability":
                self.constant_probability,
        }


    @classmethod
    def from_dict(
        cls,
        payload: dict,
    ):

        return cls(

            feature_names=
                payload[
                    "feature_names"
                ],

            mean=
                payload.get(
                    "mean"
                ),

            scale=
                payload.get(
                    "scale"
                ),

            coefficients=
                payload.get(
                    "coefficients"
                ),

            intercept=
                payload.get(
                    "intercept"
                ),

            constant_probability=
                payload.get(
                    "constant_probability"
                ),
        )