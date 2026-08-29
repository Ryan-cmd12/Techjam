from src.data.dataset import (
    AIGCImageDataset,
)

from src.data.multisignal_collate import (
    MultiSignalCLIPBatchCollator,
)

from src.evaluation.forensic_robustness import (
    run_forensic_robustness,
)

from src.evaluation.robustness import (
    build_corruption_specs,
)

from src.models.forensic_fusion_detector import (
    ForensicFusionDetector,
)

from src.training.forensic_checkpoint import (
    load_forensic_checkpoint,
)

from src.utils.config import (
    load_config,
)

from src.utils.device import (
    get_device,
    print_device_info,
)


def main():

    config = load_config(
        "configs/base.yaml"
    )

    device = get_device()

    print_device_info(
        device
    )

    model_cfg = config[
        "model"
    ]

    robust_cfg = config[
        "robust_training"
    ]

    forensic_cfg = config[
        "forensic_fusion"
    ]

    forensic_model_cfg = (
        forensic_cfg[
            "forensic"
        ]
    )

    fusion_cfg = forensic_cfg[
        "fusion"
    ]

    model = ForensicFusionDetector(
        clip_model_name=
            model_cfg[
                "clip_model"
            ],

        semantic_hidden_dim=int(
            model_cfg[
                "classifier"
            ][
                "hidden_dim"
            ]
        ),

        semantic_dropout=float(
            model_cfg[
                "classifier"
            ][
                "dropout"
            ]
        ),

        adapter_bottleneck_dim=int(
            robust_cfg[
                "adapter"
            ][
                "bottleneck_dim"
            ]
        ),

        adapter_dropout=float(
            robust_cfg[
                "adapter"
            ][
                "dropout"
            ]
        ),

        forensic_embedding_dim=int(
            forensic_model_cfg[
                "embedding_dim"
            ]
        ),

        forensic_base_channels=int(
            forensic_model_cfg[
                "base_channels"
            ]
        ),

        forensic_dropout=float(
            forensic_model_cfg[
                "dropout"
            ]
        ),

        fusion_projection_dim=int(
            fusion_cfg[
                "projection_dim"
            ]
        ),

        fusion_hidden_dim=int(
            fusion_cfg[
                "hidden_dim"
            ]
        ),

        fusion_dropout=float(
            fusion_cfg[
                "dropout"
            ]
        ),
    ).to(
        device
    )

    checkpoint = load_forensic_checkpoint(
        "checkpoints/"
        "forensic_fusion_best.pt",

        model,

        device,
    )

    print(
        f"\nLoaded forensic fusion epoch: "
        f"{checkpoint['epoch']}"
    )

    dataset = AIGCImageDataset(
        "data/manifests/"
        "cifake_test.csv",

        return_metadata=True,
    )

    collator = (
        MultiSignalCLIPBatchCollator(
            model_cfg[
                "clip_model"
            ]
        )
    )

    specs = build_corruption_specs(
        config[
            "robustness"
        ][
            "conditions"
        ]
    )

    run_forensic_robustness(
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

        batch_size=int(
            config[
                "training"
            ][
                "batch_size"
            ]
        ),

        num_workers=int(
            config[
                "training"
            ][
                "num_workers"
            ]
        ),

        threshold=float(
            config[
                "evaluation"
            ][
                "threshold"
            ]
        ),

        seed=int(
            config[
                "project"
            ][
                "seed"
            ]
        ),

        output_directory=(
            "outputs/evaluation/"
            "forensic_fusion"
        ),
    )


if __name__ == "__main__":
    main()