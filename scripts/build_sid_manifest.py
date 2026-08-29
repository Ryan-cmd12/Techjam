from __future__ import annotations

import argparse

from src.data.adapters.sid_set import (
    build_sid_manifests,
)

from src.utils.config import (
    load_config,
)

from src.utils.seed import (
    seed_everything,
)


def optional_int(
    value,
):

    if value is None:

        return None

    if isinstance(
        value,
        str,
    ):

        if (
            value.lower()
            == "none"
        ):

            return None

    return int(
        value
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Materialize SID_Set images "
            "without re-encoding and build "
            "binary AIGC manifests."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/base.yaml",
    )

    parser.add_argument(
        "--source",
        choices=[
            "auto",
            "hub",
            "local",
        ],
        default="auto",
    )

    parser.add_argument(
        "--splits",
        default=(
            "train,validation"
        ),
        help=(
            "Comma-separated: "
            "train,validation"
        ),
    )

    parser.add_argument(
        "--train-max-per-class",
        type=optional_int,
        default=None,
    )

    parser.add_argument(
        "--val-max-per-class",
        type=optional_int,
        default=None,
    )

    parser.add_argument(
        "--include-tampered",
        action="store_true",
    )

    args = parser.parse_args()

    config = load_config(
        args.config
    )

    seed = int(
        config[
            "project"
        ][
            "seed"
        ]
    )

    seed_everything(
        seed
    )

    sid_config = (
        config[
            "datasets"
        ][
            "sid_set"
        ]
    )

    if args.source == "hub":

        local_root = None

    elif args.source == "local":

        local_root = (
            sid_config[
                "local_root"
            ]
        )

    else:

        # Auto mode:
        #
        # Adapter uses local parquet shards if they
        # exist, otherwise falls back to HF Hub.
        local_root = (
            sid_config[
                "local_root"
            ]
        )

    requested_splits = {
        value.strip()
        for value in (
            args.splits.split(
                ","
            )
        )
        if value.strip()
    }

    invalid_splits = (
        requested_splits
        - {
            "train",
            "validation",
        }
    )

    if invalid_splits:

        raise ValueError(
            f"Invalid SID splits: "
            f"{sorted(invalid_splits)}"
        )

    config_train_max = (
        sid_config.get(
            "train_max_per_class"
        )
    )

    config_val_max = (
        sid_config.get(
            "validation_max_per_class"
        )
    )

    train_max = (
        args.train_max_per_class

        if args.train_max_per_class
        is not None

        else config_train_max
    )

    val_max = (
        args.val_max_per_class

        if args.val_max_per_class
        is not None

        else config_val_max
    )

    include_tampered = (
        args.include_tampered
        or bool(
            sid_config.get(
                "include_tampered_auxiliary",
                False,
            )
        )
    )

    print(
        "\n========================================"
    )

    print(
        "SID_SET IMPORT"
    )

    print(
        "========================================"
    )

    print(
        f"\nRequested splits: "
        f"{sorted(requested_splits)}"
    )

    print(
        f"Train max/class: "
        f"{train_max}"
    )

    print(
        f"Validation max/class: "
        f"{val_max}"
    )

    print(
        f"Tampered auxiliary: "
        f"{include_tampered}"
    )

    build_sid_manifests(
        repo_id=
            sid_config[
                "repo_id"
            ],

        local_root=
            local_root,

        materialized_root=
            sid_config[
                "materialized_root"
            ],

        output_directory=
            config[
                "paths"
            ][
                "manifests"
            ],

        train_split=
            sid_config[
                "train_split"
            ],

        validation_split=
            sid_config[
                "validation_split"
            ],

        streaming=
            bool(
                sid_config.get(
                    "streaming",
                    True,
                )
            ),

        shuffle_buffer_size=
            int(
                sid_config.get(
                    "shuffle_buffer_size",
                    10000,
                )
            ),

        train_max_per_class=
            train_max,

        validation_max_per_class=
            val_max,

        include_tampered_auxiliary=
            include_tampered,

        seed=
            seed,

        requested_splits=
            requested_splits,
    )


if __name__ == "__main__":
    main()