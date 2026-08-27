import torch

from torch.utils.data import (
    DataLoader,
)

from src.data.collate import (
    CLIPBatchCollator,
)

from src.data.dataset import (
    AIGCImageDataset,
)

from src.models.baseline_detector import (
    BaselineAIGCDetector,
)

from src.utils.config import (
    load_config,
)

from src.utils.device import (
    get_device,
)


def main():

    config = load_config(
        "configs/base.yaml"
    )

    model_config = (
        config[
            "model"
        ]
    )

    clip_model_name = (
        model_config[
            "clip_model"
        ]
    )

    device = (
        get_device()
    )

    print(
        f"\nDevice: {device}"
    )

    dataset = (
        AIGCImageDataset(
            manifest_path=(
                "data/manifests/"
                "cifake_train.csv"
            ),
            return_metadata=True,
        )
    )

    collator = (
        CLIPBatchCollator(
            clip_model_name
        )
    )

    dataloader = (
        DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            collate_fn=
                collator,
        )
    )

    batch = next(
        iter(
            dataloader
        )
    )

    print(
        "\nProcessed pixels:"
    )

    print(
        batch[
            "pixel_values"
        ].shape
    )

    model = (
        BaselineAIGCDetector(
            clip_model_name=
                clip_model_name,

            hidden_dim=
                int(
                    model_config[
                        "classifier"
                    ][
                        "hidden_dim"
                    ]
                ),

            dropout=
                float(
                    model_config[
                        "classifier"
                    ][
                        "dropout"
                    ]
                ),

            freeze_backbone=
                True,

            normalize_embeddings=
                True,
        )
    )

    model = model.to(
        device
    )

    pixel_values = (
        batch[
            "pixel_values"
        ].to(
            device
        )
    )

    model.eval()

    with torch.no_grad():

        features = (
            model.extract_features(
                pixel_values
            )
        )

        logits = (
            model.classify_features(
                features
            )
        )

        probabilities = (
            torch.sigmoid(
                logits
            )
        )

    print(
        "\nCLIP features:"
    )

    print(
        features.shape
    )

    print(
        "\nLogits:"
    )

    print(
        logits
    )

    print(
        "\nAI probabilities:"
    )

    print(
        probabilities
    )

    print(
        "\nLabels:"
    )

    print(
        batch[
            "labels"
        ]
    )

    print(
        "\n=============================="
    )

    print(
        "BASELINE MODEL WORKING"
    )

    print(
        "=============================="
    )


if __name__ == "__main__":
    main()