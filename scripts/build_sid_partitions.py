import argparse

from src.data.sid_partition import (
    build_sid_model_partitions,
)

from src.utils.config import (
    load_config,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/base.yaml",
    )

    parser.add_argument(
        "--dev-fraction",
        type=float,
        default=0.10,
    )

    args = parser.parse_args()

    config = load_config(
        args.config
    )

    build_sid_model_partitions(
        sid_train_path=(
            "data/manifests/"
            "sid_train.csv"
        ),

        sid_official_validation_path=(
            "data/manifests/"
            "sid_val.csv"
        ),

        output_directory=(
            "data/manifests"
        ),

        dev_fraction=
            args.dev_fraction,

        seed=int(
            config[
                "project"
            ][
                "seed"
            ]
        ),
    )


if __name__ == "__main__":
    main()