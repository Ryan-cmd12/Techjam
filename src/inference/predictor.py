from __future__ import annotations

import json

from pathlib import Path

import torch

from torch.utils.data import (
    DataLoader,
)

from tqdm import tqdm

from src.data.native_tile_collate import (
    NativeTileCLIPBatchCollator,
)

from src.inference.dataset import (
    InferenceImageDataset,
)

from src.models.transformation_aware_detector import (
    TransformationAwareDetector,
)

from src.training.corruption_targets import (
    ID_TO_CORRUPTION,
)

from src.training.transformation_aware_checkpoint import (
    load_transformation_aware_checkpoint,
)


def build_transformation_aware_model(
    config: dict,
) -> TransformationAwareDetector:

    model_cfg = (
        config[
            "model"
        ]
    )

    robust_cfg = (
        config[
            "robust_training"
        ]
    )

    tile_cfg = (
        config[
            "native_tiles"
        ]
    )

    corruption_cfg = (
        config[
            "corruption_estimator"
        ]
    )

    adaptive_cfg = (
        config[
            "transformation_aware"
        ]
    )

    return TransformationAwareDetector(

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
            tile_cfg[
                "encoder"
            ][
                "embedding_dim"
            ]
        ),

        forensic_base_channels=int(
            tile_cfg[
                "encoder"
            ][
                "base_channels"
            ]
        ),

        forensic_dropout=float(
            tile_cfg[
                "encoder"
            ][
                "dropout"
            ]
        ),

        attention_hidden_dim=int(
            tile_cfg[
                "attention"
            ][
                "hidden_dim"
            ]
        ),

        attention_dropout=float(
            tile_cfg[
                "attention"
            ][
                "dropout"
            ]
        ),

        fusion_projection_dim=int(
            tile_cfg[
                "fusion"
            ][
                "projection_dim"
            ]
        ),

        fusion_hidden_dim=int(
            tile_cfg[
                "fusion"
            ][
                "hidden_dim"
            ]
        ),

        fusion_dropout=float(
            tile_cfg[
                "fusion"
            ][
                "dropout"
            ]
        ),

        corruption_hidden_dim=int(
            corruption_cfg[
                "hidden_dim"
            ]
        ),

        corruption_embedding_dim=int(
            corruption_cfg[
                "embedding_dim"
            ]
        ),

        num_corruption_types=int(
            corruption_cfg[
                "num_types"
            ]
        ),

        corruption_dropout=float(
            corruption_cfg[
                "dropout"
            ]
        ),

        gate_hidden_dim=int(
            adaptive_cfg[
                "gate"
            ][
                "hidden_dim"
            ]
        ),

        gate_dropout=float(
            adaptive_cfg[
                "gate"
            ][
                "dropout"
            ]
        ),

        initial_residual_scale=float(
            adaptive_cfg[
                "gate"
            ][
                "initial_residual_scale"
            ]
        ),
    )


class AIGCInferenceEngine:

    def __init__(
        self,
        config: dict,
        checkpoint_path: str | Path,
        device: torch.device,
        calibration_path: str | Path | None = None,
    ):

        self.config = config

        self.device = (
            device
        )

        self.checkpoint_path = Path(
            checkpoint_path
        )

        self.model = (
            build_transformation_aware_model(
                config
            )
            .to(
                device
            )
        )

        checkpoint = (
            load_transformation_aware_checkpoint(

                path=
                    self.checkpoint_path,

                model=
                    self.model,

                device=
                    device,
            )
        )

        print(
            f"\nLoaded detector epoch: "
            f"{checkpoint['epoch']}"
        )

        self.model.eval()

        # ==================================================
        # CALIBRATION
        # ==================================================

        self.temperature = 1.0

        if calibration_path is not None:

            calibration_path = Path(
                calibration_path
            )

            if not calibration_path.exists():

                raise FileNotFoundError(
                    "Calibration file not found: "
                    f"{calibration_path}"
                )

            with calibration_path.open(
                "r",
                encoding="utf-8",
            ) as file:

                calibration = json.load(
                    file
                )

            self.temperature = float(
                calibration[
                    "temperature"
                ]
            )

            print(
                f"Calibration temperature: "
                f"{self.temperature:.4f}"
            )

        # ==================================================
        # COLLATOR
        # ==================================================

        tile_cfg = (
            config[
                "native_tiles"
            ]
        )

        self.collator = (
            NativeTileCLIPBatchCollator(

                model_name=
                    config[
                        "model"
                    ][
                        "clip_model"
                    ],

                tile_size=int(
                    tile_cfg[
                        "tile_size"
                    ]
                ),

                max_tiles=int(
                    tile_cfg[
                        "max_tiles"
                    ]
                ),

                feature_map_size=int(
                    tile_cfg[
                        "feature_map_size"
                    ]
                ),

                sampling_mode=
                    tile_cfg[
                        "evaluation_sampling"
                    ],

                seed=int(
                    config[
                        "project"
                    ][
                        "seed"
                    ]
                ),
            )
        )


    @torch.no_grad()
    def predict_directory(
        self,
        input_path: str | Path,
        batch_size: int | None = None,
        num_workers: int | None = None,
    ) -> tuple[
        list[dict],
        list[dict],
    ]:

        dataset = (
            InferenceImageDataset(
                input_path=
                    input_path
            )
        )

        if batch_size is None:

            batch_size = int(
                self.config[
                    "transformation_aware"
                ][
                    "training"
                ][
                    "batch_size"
                ]
            )

        if num_workers is None:

            num_workers = int(
                self.config[
                    "training"
                ][
                    "num_workers"
                ]
            )

        dataloader = (
            DataLoader(

                dataset,

                batch_size=
                    batch_size,

                shuffle=
                    False,

                num_workers=
                    num_workers,

                pin_memory=(
                    self.device.type
                    == "cuda"
                ),

                persistent_workers=(
                    num_workers > 0
                ),

                collate_fn=
                    self.collator,
            )
        )

        predictions = []

        diagnostics = []

        for batch in tqdm(
            dataloader,
            desc="Inference",
        ):

            pixel_values = (
                batch[
                    "pixel_values"
                ]
                .to(
                    self.device,
                    non_blocking=True,
                )
            )

            forensic_tiles = (
                batch[
                    "forensic_tiles"
                ]
                .to(
                    self.device,
                    non_blocking=True,
                )
            )

            tile_mask = (
                batch[
                    "tile_mask"
                ]
                .to(
                    self.device,
                    non_blocking=True,
                )
            )

            output = (
                self.model
                .forward_with_details(

                    pixel_values=
                        pixel_values,

                    forensic_tiles=
                        forensic_tiles,

                    tile_mask=
                        tile_mask,
                )
            )

            raw_probability = (
                torch.sigmoid(
                    output[
                        "logits"
                    ]
                )
            )

            calibrated_probability = (
                torch.sigmoid(
                    output[
                        "logits"
                    ]
                    / self.temperature
                )
            )

            semantic_probability = (
                torch.sigmoid(
                    output[
                        "semantic_logits"
                    ]
                )
            )

            forensic_probability = (
                torch.sigmoid(
                    output[
                        "forensic_logits"
                    ]
                )
            )

            corruption_ids = (
                output[
                    "corruption_type_probabilities"
                ]
                .argmax(
                    dim=-1
                )
            )

            attention = (
                output[
                    "attention"
                ]
            )

            masked_attention = (
                attention
                .masked_fill(
                    ~tile_mask,
                    0.0,
                )
            )

            max_attention = (
                masked_attention
                .max(
                    dim=1
                )
                .values
            )

            raw_probability = (
                raw_probability
                .cpu()
                .numpy()
            )

            calibrated_probability = (
                calibrated_probability
                .cpu()
                .numpy()
            )

            semantic_probability = (
                semantic_probability
                .cpu()
                .numpy()
            )

            forensic_probability = (
                forensic_probability
                .cpu()
                .numpy()
            )

            semantic_weight = (
                output[
                    "semantic_weight"
                ]
                .cpu()
                .numpy()
            )

            forensic_weight = (
                output[
                    "forensic_weight"
                ]
                .cpu()
                .numpy()
            )

            severity = (
                output[
                    "corruption_severity"
                ]
                .cpu()
                .numpy()
            )

            corruption_ids = (
                corruption_ids
                .cpu()
                .numpy()
            )

            max_attention = (
                max_attention
                .cpu()
                .numpy()
            )

            attention_cpu = (
                attention
                .cpu()
                .numpy()
            )

            mask_cpu = (
                tile_mask
                .cpu()
                .numpy()
            )

            boxes_cpu = (
                batch[
                    "tile_boxes"
                ]
                .cpu()
                .numpy()
            )

            for index in range(
                len(
                    batch[
                        "image_path"
                    ]
                )
            ):

                image_path = (
                    batch[
                        "image_path"
                    ][
                        index
                    ]
                )

                pred = float(
                    calibrated_probability[
                        index
                    ]
                )

                # ==================================================
                # REQUIRED COMPETITION FORMAT
                # ==================================================

                predictions.append(
                    {
                        "image_path":
                            image_path,

                        "pred":
                            pred,
                    }
                )

                # ==================================================
                # OPTIONAL FULL DIAGNOSTICS
                # ==================================================

                valid_attention = []

                valid_boxes = []

                for (
                    weight,
                    box,
                    valid,
                ) in zip(
                    attention_cpu[
                        index
                    ],

                    boxes_cpu[
                        index
                    ],

                    mask_cpu[
                        index
                    ],
                ):

                    if not valid:
                        continue

                    valid_attention.append(
                        float(
                            weight
                        )
                    )

                    valid_boxes.append(
                        [
                            float(
                                value
                            )

                            for value
                            in box
                        ]
                    )

                diagnostics.append(
                    {
                        "image_path":
                            image_path,

                        "pred":
                            pred,

                        "raw_pred":
                            float(
                                raw_probability[
                                    index
                                ]
                            ),

                        "semantic_pred":
                            float(
                                semantic_probability[
                                    index
                                ]
                            ),

                        "forensic_pred":
                            float(
                                forensic_probability[
                                    index
                                ]
                            ),

                        "semantic_weight":
                            float(
                                semantic_weight[
                                    index
                                ]
                            ),

                        "forensic_weight":
                            float(
                                forensic_weight[
                                    index
                                ]
                            ),

                        "predicted_corruption":
                            ID_TO_CORRUPTION.get(
                                int(
                                    corruption_ids[
                                        index
                                    ]
                                ),
                                "unknown",
                            ),

                        "predicted_severity":
                            float(
                                severity[
                                    index
                                ]
                            ),

                        "max_tile_attention":
                            float(
                                max_attention[
                                    index
                                ]
                            ),

                        "tile_attention":
                            valid_attention,

                        "tile_boxes":
                            valid_boxes,
                    }
                )

        return (
            predictions,
            diagnostics,
        )