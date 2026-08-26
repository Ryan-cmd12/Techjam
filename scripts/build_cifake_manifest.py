import argparse

from src.data.adapters.cifake import (
    build_cifake_manifests,
)

from src.utils.config import (
    load_config,
)

from src.utils.seed import (
    seed_everything,
)


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build CIFAKE training, validation "
            "and test manifests."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
    )

    args = parser.parse_args()

    config = (
        load_config(
            args.config
        )
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

    cifake_config = (
        config[
            "datasets"
        ][
            "cifake"
        ]
    )

    output_directory = (
        config[
            "paths"
        ][
            "manifests"
        ]
    )

    build_cifake_manifests(
        dataset_root=
            cifake_config[
                "root"
            ],

        output_directory=
            output_directory,

        validation_fraction=
            float(
                cifake_config[
                    "validation_fraction"
                ]
            ),

        seed=
            seed,

        train_dir_name=
            cifake_config[
                "train_dir"
            ],

        test_dir_name=
            cifake_config[
                "test_dir"
            ],

        real_dir_name=
            cifake_config[
                "real_dir"
            ],

        fake_dir_name=
            cifake_config[
                "fake_dir"
            ],
    )


if __name__ == "__main__":
    main()