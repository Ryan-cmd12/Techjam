from src.data.unified_manifest import (
    build_unified_manifests,
)


def main():

    build_unified_manifests(
        output_directory=(
            "data/manifests"
        ),

        train_paths=[
            (
                "data/manifests/"
                "cifake_train.csv"
            ),

            (
                "data/manifests/"
                "sid_model_train.csv"
            ),
        ],

        val_paths=[
            (
                "data/manifests/"
                "cifake_val.csv"
            ),

            (
                "data/manifests/"
                "sid_model_val.csv"
            ),
        ],

        test_paths=[
            (
                "data/manifests/"
                "cifake_test.csv"
            ),

            (
                "data/manifests/"
                "sid_test.csv"
            ),
        ],
    )


if __name__ == "__main__":
    main()