from torch.utils.data import (
    DataLoader,
)

from src.data.dataset import (
    AIGCImageDataset,
)

from src.data.native_tile_collate import (
    NativeTileCLIPBatchCollator,
)

from src.utils.config import (
    load_config,
)


def test_manifest(
    manifest_path: str,
):

    config = load_config(
        "configs/base.yaml"
    )

    model_name = (
        config[
            "model"
        ][
            "clip_model"
        ]
    )

    tile_config = (
        config[
            "native_tiles"
        ]
    )

    dataset = (
        AIGCImageDataset(
            manifest_path=
                manifest_path,

            return_metadata=
                True,
        )
    )

    collator = (
        NativeTileCLIPBatchCollator(
            model_name=
                model_name,

            tile_size=int(
                tile_config[
                    "tile_size"
                ]
            ),

            max_tiles=int(
                tile_config[
                    "max_tiles"
                ]
            ),

            feature_map_size=int(
                tile_config[
                    "feature_map_size"
                ]
            ),

            sampling_mode="grid",

            seed=int(
                config[
                    "project"
                ][
                    "seed"
                ]
            ),
        )
    )

    dataloader = DataLoader(
        dataset,

        batch_size=4,

        shuffle=False,

        num_workers=0,

        collate_fn=
            collator,
    )

    batch = next(
        iter(
            dataloader
        )
    )

    print(
        "\n========================================"
    )

    print(
        manifest_path
    )

    print(
        "========================================"
    )

    print(
        "\nCLIP:"
    )

    print(
        batch[
            "pixel_values"
        ].shape
    )

    print(
        "\nForensic tiles:"
    )

    print(
        batch[
            "forensic_tiles"
        ].shape
    )

    print(
        "\nTile masks:"
    )

    print(
        batch[
            "tile_mask"
        ]
    )

    print(
        "\nTile boxes:"
    )

    print(
        batch[
            "tile_boxes"
        ]
    )

    print(
        "\nDatasets:"
    )

    print(
        batch[
            "dataset"
        ]
    )


def main():

    test_manifest(
        "data/manifests/"
        "cifake_train.csv"
    )

    test_manifest(
        "data/manifests/"
        "sid_model_train.csv"
    )

    print(
        "\n========================================"
    )

    print(
        "NATIVE TILE PIPELINE WORKING"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()