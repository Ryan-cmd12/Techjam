from __future__ import annotations

import torch

from torch import nn

from src.models.corruption_estimator import (
    CorruptionEstimator,
)

from src.models.native_tile_fusion_detector import (
    NativeTileFusionDetector,
)

from src.models.reliability_gate import (
    ReliabilityGate,
)


class TransformationAwareDetector(
    NativeTileFusionDetector
):

    def __init__(
        self,

        clip_model_name: str,

        semantic_hidden_dim: int = 512,
        semantic_dropout: float = 0.20,

        adapter_bottleneck_dim: int = 256,
        adapter_dropout: float = 0.10,

        forensic_embedding_dim: int = 256,
        forensic_base_channels: int = 32,
        forensic_dropout: float = 0.10,

        attention_hidden_dim: int = 128,
        attention_dropout: float = 0.10,

        fusion_projection_dim: int = 256,
        fusion_hidden_dim: int = 256,
        fusion_dropout: float = 0.20,

        corruption_hidden_dim: int = 256,
        corruption_embedding_dim: int = 128,
        num_corruption_types: int = 7,
        corruption_dropout: float = 0.10,

        gate_hidden_dim: int = 256,
        gate_dropout: float = 0.10,

        initial_residual_scale: float = 0.05,
    ):

        super().__init__(

            clip_model_name=
                clip_model_name,

            semantic_hidden_dim=
                semantic_hidden_dim,

            semantic_dropout=
                semantic_dropout,

            adapter_bottleneck_dim=
                adapter_bottleneck_dim,

            adapter_dropout=
                adapter_dropout,

            forensic_embedding_dim=
                forensic_embedding_dim,

            forensic_base_channels=
                forensic_base_channels,

            forensic_dropout=
                forensic_dropout,

            attention_hidden_dim=
                attention_hidden_dim,

            attention_dropout=
                attention_dropout,

            fusion_projection_dim=
                fusion_projection_dim,

            fusion_hidden_dim=
                fusion_hidden_dim,

            fusion_dropout=
                fusion_dropout,
        )

        semantic_dim = (
            self.backbone.feature_dim
        )

        # ==================================================
        # CORRUPTION MODEL
        # ==================================================

        self.corruption_estimator = (
            CorruptionEstimator(

                semantic_dim=
                    semantic_dim,

                forensic_dim=
                    forensic_embedding_dim,

                hidden_dim=
                    corruption_hidden_dim,

                embedding_dim=
                    corruption_embedding_dim,

                num_types=
                    num_corruption_types,

                dropout=
                    corruption_dropout,
            )
        )

        # ==================================================
        # RELIABILITY GATE
        # ==================================================

        self.reliability_gate = (
            ReliabilityGate(

                semantic_dim=
                    semantic_dim,

                forensic_dim=
                    forensic_embedding_dim,

                corruption_embedding_dim=
                    corruption_embedding_dim,

                num_corruption_types=
                    num_corruption_types,

                hidden_dim=
                    gate_hidden_dim,

                dropout=
                    gate_dropout,
            )
        )

        # Existing semantic projection already produces
        # fusion_projection_dim.

        self.forensic_projection = (
            nn.Sequential(

                nn.LayerNorm(
                    forensic_embedding_dim
                ),

                nn.Linear(
                    forensic_embedding_dim,
                    fusion_projection_dim,
                ),

                nn.GELU(),
            )
        )

        adaptive_input_dim = (
            fusion_projection_dim
            * 2
            + corruption_embedding_dim
        )

        self.adaptive_head = (
            nn.Sequential(

                nn.LayerNorm(
                    adaptive_input_dim
                ),

                nn.Linear(
                    adaptive_input_dim,
                    fusion_hidden_dim,
                ),

                nn.GELU(),

                nn.Dropout(
                    fusion_dropout
                ),

                nn.Linear(
                    fusion_hidden_dim,
                    1,
                ),
            )
        )

        self.adaptive_scale = (
            nn.Parameter(
                torch.tensor(
                    float(
                        initial_residual_scale
                    ),
                    dtype=torch.float32,
                )
            )
        )


    # ==================================================
    # CORRUPTION
    # ==================================================

    def estimate_corruption(
        self,

        semantic_features:
            torch.Tensor,

        forensic_features:
            torch.Tensor,

    ) -> dict[
        str,
        torch.Tensor
    ]:

        corruption = (
            self.corruption_estimator(

                semantic_features=
                    semantic_features,

                forensic_features=
                    forensic_features,
            )
        )

        type_probabilities = (
            torch.softmax(

                corruption[
                    "type_logits"
                ],

                dim=-1,
            )
        )

        corruption[
            "type_probabilities"
        ] = (
            type_probabilities
        )

        return corruption


    # ==================================================
    # FUSION FROM FEATURES
    # ==================================================

    def fuse_features(
        self,

        semantic_features:
            torch.Tensor,

        forensic_features:
            torch.Tensor,

    ) -> dict[
        str,
        torch.Tensor
    ]:

        # --------------------------------------------------
        # Original/native-tile fusion path
        # --------------------------------------------------

        projected_semantic = (
            self.semantic_projection(
                semantic_features
            )
        )

        base_fused_features = (
            torch.cat(
                [
                    projected_semantic,
                    forensic_features,
                ],
                dim=-1,
            )
        )

        base_logits = (
            self.fusion_head(
                base_fused_features
            )
            .squeeze(
                -1
            )
        )

        # --------------------------------------------------
        # Corruption representation
        # --------------------------------------------------

        corruption = (
            self.estimate_corruption(

                semantic_features=
                    semantic_features,

                forensic_features=
                    forensic_features,
            )
        )

        # --------------------------------------------------
        # Reliability gating
        # --------------------------------------------------

        gate = (
            self.reliability_gate(

                semantic_features=
                    semantic_features,

                forensic_features=
                    forensic_features,

                corruption_embedding=
                    corruption[
                        "embedding"
                    ],

                corruption_type_probabilities=
                    corruption[
                        "type_probabilities"
                    ],

                corruption_severity=
                    corruption[
                        "severity"
                    ],
            )
        )

        semantic_weight = (
            gate[
                "semantic_weight"
            ].unsqueeze(
                -1
            )
        )

        forensic_weight = (
            gate[
                "forensic_weight"
            ].unsqueeze(
                -1
            )
        )

        projected_forensic = (
            self.forensic_projection(
                forensic_features
            )
        )

        weighted_semantic = (
            projected_semantic
            * semantic_weight
        )

        weighted_forensic = (
            projected_forensic
            * forensic_weight
        )

        adaptive_features = (
            torch.cat(
                [
                    weighted_semantic,
                    weighted_forensic,
                    corruption[
                        "embedding"
                    ],
                ],
                dim=-1,
            )
        )

        adaptive_delta = (
            self.adaptive_head(
                adaptive_features
            )
            .squeeze(
                -1
            )
        )

        # --------------------------------------------------
        # Residual adaptive correction
        # --------------------------------------------------

        logits = (
            base_logits

            + self.adaptive_scale
            * adaptive_delta
        )

        return {
            "base_logits":
                base_logits,

            "adaptive_delta":
                adaptive_delta,

            "logits":
                logits,

            "base_fused_features":
                base_fused_features,

            "adaptive_features":
                adaptive_features,

            "corruption_embedding":
                corruption[
                    "embedding"
                ],

            "corruption_type_logits":
                corruption[
                    "type_logits"
                ],

            "corruption_type_probabilities":
                corruption[
                    "type_probabilities"
                ],

            "corruption_severity":
                corruption[
                    "severity"
                ],

            "gate_weights":
                gate[
                    "weights"
                ],

            "semantic_weight":
                gate[
                    "semantic_weight"
                ],

            "forensic_weight":
                gate[
                    "forensic_weight"
                ],
        }


    # ==================================================
    # SINGLE VIEW
    # ==================================================

    def forward_with_details(
        self,

        pixel_values:
            torch.Tensor,

        forensic_tiles:
            torch.Tensor,

        tile_mask:
            torch.Tensor,

    ) -> dict[
        str,
        torch.Tensor
    ]:

        semantic_features = (
            self.encode_semantic(
                pixel_values
            )
        )

        (
            forensic_features,
            tile_embeddings,
            attention,
        ) = (
            self.encode_forensic(

                forensic_tiles=
                    forensic_tiles,

                tile_mask=
                    tile_mask,
            )
        )

        semantic_logits = (
            self.semantic_head(
                semantic_features
            )
            .squeeze(
                -1
            )
        )

        forensic_logits = (
            self.forensic_head(
                forensic_features
            )
            .squeeze(
                -1
            )
        )

        fusion = (
            self.fuse_features(

                semantic_features=
                    semantic_features,

                forensic_features=
                    forensic_features,
            )
        )

        return {
            "semantic_features":
                semantic_features,

            "forensic_features":
                forensic_features,

            "tile_embeddings":
                tile_embeddings,

            "attention":
                attention,

            "semantic_logits":
                semantic_logits,

            "forensic_logits":
                forensic_logits,

            **fusion,
        }


    def forward(
        self,

        pixel_values:
            torch.Tensor,

        forensic_tiles:
            torch.Tensor,

        tile_mask:
            torch.Tensor,

    ) -> torch.Tensor:

        return (
            self.forward_with_details(

                pixel_values=
                    pixel_values,

                forensic_tiles=
                    forensic_tiles,

                tile_mask=
                    tile_mask,
            )[
                "logits"
            ]
        )


    # ==================================================
    # PAIRED VIEWS
    # ==================================================

    def forward_pair(
        self,

        clean_pixel_values:
            torch.Tensor,

        corrupted_pixel_values:
            torch.Tensor,

        clean_forensic_tiles:
            torch.Tensor,

        corrupted_forensic_tiles:
            torch.Tensor,

        tile_mask:
            torch.Tensor,

    ) -> dict[
        str,
        torch.Tensor
    ]:

        batch_size = (
            clean_pixel_values.shape[
                0
            ]
        )

        # One CLIP pass.
        combined_pixels = (
            torch.cat(
                [
                    clean_pixel_values,
                    corrupted_pixel_values,
                ],
                dim=0,
            )
        )

        combined_semantic = (
            self.encode_semantic(
                combined_pixels
            )
        )

        clean_semantic = (
            combined_semantic[
                :batch_size
            ]
        )

        corrupted_semantic = (
            combined_semantic[
                batch_size:
            ]
        )

        # One tile encoder pass.
        combined_tiles = (
            torch.cat(
                [
                    clean_forensic_tiles,
                    corrupted_forensic_tiles,
                ],
                dim=0,
            )
        )

        combined_mask = (
            torch.cat(
                [
                    tile_mask,
                    tile_mask,
                ],
                dim=0,
            )
        )

        (
            combined_forensic,
            _,
            combined_attention,
        ) = (
            self.encode_forensic(

                forensic_tiles=
                    combined_tiles,

                tile_mask=
                    combined_mask,
            )
        )

        clean_forensic = (
            combined_forensic[
                :batch_size
            ]
        )

        corrupted_forensic = (
            combined_forensic[
                batch_size:
            ]
        )

        clean_attention = (
            combined_attention[
                :batch_size
            ]
        )

        corrupted_attention = (
            combined_attention[
                batch_size:
            ]
        )

        clean_fusion = (
            self.fuse_features(

                semantic_features=
                    clean_semantic,

                forensic_features=
                    clean_forensic,
            )
        )

        corrupted_fusion = (
            self.fuse_features(

                semantic_features=
                    corrupted_semantic,

                forensic_features=
                    corrupted_forensic,
            )
        )

        clean_semantic_logits = (
            self.semantic_head(
                clean_semantic
            )
            .squeeze(-1)
        )

        corrupted_semantic_logits = (
            self.semantic_head(
                corrupted_semantic
            )
            .squeeze(-1)
        )

        clean_forensic_logits = (
            self.forensic_head(
                clean_forensic
            )
            .squeeze(-1)
        )

        corrupted_forensic_logits = (
            self.forensic_head(
                corrupted_forensic
            )
            .squeeze(-1)
        )

        return {
            "clean_semantic_features":
                clean_semantic,

            "corrupted_semantic_features":
                corrupted_semantic,

            "clean_forensic_features":
                clean_forensic,

            "corrupted_forensic_features":
                corrupted_forensic,

            "clean_fused_features":
                clean_fusion[
                    "adaptive_features"
                ],

            "corrupted_fused_features":
                corrupted_fusion[
                    "adaptive_features"
                ],

            "clean_attention":
                clean_attention,

            "corrupted_attention":
                corrupted_attention,

            "clean_semantic_logits":
                clean_semantic_logits,

            "corrupted_semantic_logits":
                corrupted_semantic_logits,

            "clean_forensic_logits":
                clean_forensic_logits,

            "corrupted_forensic_logits":
                corrupted_forensic_logits,

            "clean_logits":
                clean_fusion[
                    "logits"
                ],

            "corrupted_logits":
                corrupted_fusion[
                    "logits"
                ],

            "clean_base_logits":
                clean_fusion[
                    "base_logits"
                ],

            "corrupted_base_logits":
                corrupted_fusion[
                    "base_logits"
                ],

            "clean_gate_weights":
                clean_fusion[
                    "gate_weights"
                ],

            "corrupted_gate_weights":
                corrupted_fusion[
                    "gate_weights"
                ],

            "clean_semantic_weight":
                clean_fusion[
                    "semantic_weight"
                ],

            "clean_forensic_weight":
                clean_fusion[
                    "forensic_weight"
                ],

            "corrupted_semantic_weight":
                corrupted_fusion[
                    "semantic_weight"
                ],

            "corrupted_forensic_weight":
                corrupted_fusion[
                    "forensic_weight"
                ],

            "clean_corruption_type_logits":
                clean_fusion[
                    "corruption_type_logits"
                ],

            "corrupted_corruption_type_logits":
                corrupted_fusion[
                    "corruption_type_logits"
                ],

            "clean_corruption_severity":
                clean_fusion[
                    "corruption_severity"
                ],

            "corrupted_corruption_severity":
                corrupted_fusion[
                    "corruption_severity"
                ],
        }


    # ==================================================
    # PARAMETER GROUPS
    # ==================================================

    def gate_parameters(
        self,
    ):

        yield from (
            self.reliability_gate
            .parameters()
        )


    def adaptive_parameters(
        self,
    ):

        yield from (
            self.forensic_projection
            .parameters()
        )

        yield from (
            self.adaptive_head
            .parameters()
        )

        yield (
            self.adaptive_scale
        )