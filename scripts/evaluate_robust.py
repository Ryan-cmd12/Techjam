import argparse

from src.data.collate import (
    CLIPBatchCollator,
)

from src.data.dataset import (
    AIGCImageDataset,
)

from src.evaluation.robustness import (
    build_corruption_specs,
    run_robustness_benchmark,
)

from src.models.robust_detector import (
    RobustAIGCDetector,
)

from src.training.robust_checkpoint import (
    load_robust_checkpoint,
)

from src.utils.config import (
    load_config,
)

from src.utils.device import (
    get_device,
    print_device_info,
)

from src.utils.seed import (
    seed_everything,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
    )

    parser.add_argument(
        "--manifest",
        type=str,
        default=(
            "data/manifests/"
            "cifake_test.csv"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "checkpoints/"
            "robust_best.pt"
        ),
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

    device = (
        get_device()
    )

    print_device_info(
        device
    )

    model_config = (
        config[
            "model"
        ]
    )

    robust_config = (
        config[
            "robust_training"
        ]
    )

    training_config = (
        config[
            "training"
        ]
    )

    evaluation_config = (
        config[
            "evaluation"
        ]
    )

    clip_model_name = (
        model_config[
            "clip_model"
        ]
    )

    dataset = (
        AIGCImageDataset(
            manifest_path=
                args.manifest,

            return_metadata=
                True,
        )
    )

    collator = (
        CLIPBatchCollator(
            model_name=
                clip_model_name
        )
    )

    model = (
        RobustAIGCDetector(
            clip_model_name=
                clip_model_name,

            classifier_hidden_dim=
                int(
                    model_config[
                        "classifier"
                    ][
                        "hidden_dim"
                    ]
                ),

            classifier_dropout=
                float(
                    model_config[
                        "classifier"
                    ][
                        "dropout"
                    ]
                ),

            adapter_bottleneck_dim=
                int(
                    robust_config[
                        "adapter"
                    ][
                        "bottleneck_dim"
                    ]
                ),

            adapter_dropout=
                float(
                    robust_config[
                        "adapter"
                    ][
                        "dropout"
                    ]
                ),

            freeze_backbone=
                True,

            normalize_embeddings=
                bool(
                    model_config[
                        "normalize_embeddings"
                    ]
                ),
        )
    )

    model = (
        model.to(
            device
        )
    )

    checkpoint = (
        load_robust_checkpoint(
            path=
                args.checkpoint,

            model=
                model,

            device=
                device,
        )
    )

    print(
        f"\nLoaded robust checkpoint "
        f"epoch: "
        f"{checkpoint['epoch']}"
    )

    specs = (
        build_corruption_specs(
            config[
                "robustness"
            ][
                "conditions"
            ]
        )
    )

    output_directory = (
        config[
            "paths"
        ][
            "outputs"
        ]
        + "/evaluation/robust_model"
    )

    run_robustness_benchmark(
        model=
            model,

        base_dataset=
            dataset,

        collator=
            collator,

        specs=
            specs,

        device=
            device,

        batch_size=
            int(
                training_config[
                    "batch_size"
                ]
            ),

        num_workers=
            int(
                training_config[
                    "num_workers"
                ]
            ),

        threshold=
            float(
                evaluation_config[
                    "threshold"
                ]
            ),

        seed=
            seed,

        output_directory=
            output_directory,

        use_amp=
            bool(
                evaluation_config[
                    "use_amp"
                ]
            ),
        output_prefix="robust",
    )


if __name__ == "__main__":
    main()