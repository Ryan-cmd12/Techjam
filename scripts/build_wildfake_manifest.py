from __future__ import annotations

import argparse

from src.data.adapters.wildfake import (
    build_wildfake_manifest,
)

from src.utils.config import (
    load_config,
)


def optional_integer(
    value: str,
):

    if (
        str(
            value
        ).lower()
        in {
            "none",
            "all",
            "full",
        }
    ):

        return None

    return int(
        value
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/base.yaml",
    )

    parser.add_argument(
        "--root",
        default=None,
    )

    parser.add_argument(
        "--split",
        choices=[
            "train",
            "test",
        ],
        default="test",
    )

    parser.add_argument(
        "--split-file",
        default=None,
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    parser.add_argument(
        "--max-real",
        type=optional_integer,
        default=None,
    )

    parser.add_argument(
        "--max-fake",
        type=optional_integer,
        default=None,
    )

    parser.add_argument(
        "--max-per-generator",
        type=optional_integer,
        default=None,
    )

    parser.add_argument(
        "--full",
        action="store_true",
    )

    args = parser.parse_args()

    config = load_config(
        args.config
    )

    cfg = config.get(
        "wildfake",
        {},
    )

    root = (
        args.root

        or cfg.get(
            "root",
            "data/raw/wildfake",
        )
    )

    split_cfg = cfg.get(
        args.split,
        {},
    )

    if args.full:

        max_real = None
        max_fake = None
        max_per_generator = None

    else:

        max_real = (
            args.max_real

            if args.max_real
            is not None

            else split_cfg.get(
                "max_real",
                5000,
            )
        )

        max_fake = (
            args.max_fake

            if args.max_fake
            is not None

            else split_cfg.get(
                "max_fake",
                5000,
            )
        )

        max_per_generator = (
            args.max_per_generator

            if args.max_per_generator
            is not None

            else split_cfg.get(
                "max_per_generator",
                1000,
            )
        )

    output = (
        args.output

        or (
            "data/manifests/"
            f"wildfake_{args.split}.csv"
        )
    )

    seed = int(
        cfg.get(
            "seed",
            config[
                "project"
            ][
                "seed"
            ],
        )
    )

    build_wildfake_manifest(

        root=
            root,

        split=
            args.split,

        output_path=
            output,

        split_file=
            args.split_file,

        max_real=
            max_real,

        max_fake=
            max_fake,

        max_per_generator=
            max_per_generator,

        seed=
            seed,

        benchmark_root=
            cfg.get(
                "benchmark_root",
                (
                    "data/benchmark/"
                    "hackathon_wildfake"
                ),
            ),

        exclude_benchmark_sources=
            True,
    )


if __name__ == "__main__":

    main()