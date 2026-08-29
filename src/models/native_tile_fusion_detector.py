from __future__ import annotations

import torch

from torch import nn

from src.models.clip_backbone import (
    CLIPImageBackbone,
)

from src.models.native_tile_encoder import (
    NativeTileForensicEncoder,
)

from src.models.residual_adapter import (
    ResidualFeatureAdapter,
)

from src.models.tile_attention import (
    TileAttentionPooler,
)


class NativeTileFusionDetector(
    nn.Module
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
    ):

        super().__init__()

        self.clip_model_name = (
            clip_model_name
        )

        # ==================================================
        # GLOBAL SEMANTIC BRANCH
        # ==================================================

        self.backbone = (
            CLIPImageBackbone(
                model_name=
                    clip_model_name,

                freeze=
                    True,

                normalize_embeddings=
                    True,
            )
        )

        semantic_dim = (
            self.backbone.feature_dim
        )

        self.semantic_adapter = (
            ResidualFeatureAdapter(
                feature_dim=
                    semantic_dim,

                bottleneck_dim=
                    adapter_bottleneck_dim,

                dropout=
                    adapter_dropout,
            )
        )

        self.semantic_head = (
            nn.Sequential(

                nn.LayerNorm(
                    semantic_dim
                ),

                nn.Linear(
                    semantic_dim,
                    semantic_hidden_dim,
                ),

                nn.GELU(),

                nn.Dropout(
                    semantic_dropout
                ),

                nn.Linear(
                    semantic_hidden_dim,
                    1,
                ),
            )
        )

        # ==================================================
        # NATIVE FORENSIC TILES
        # ==================================================

        self.tile_encoder = (
            NativeTileForensicEncoder(
                embedding_dim=
                    forensic_embedding_dim,

                base_channels=
                    forensic_base_channels,

                dropout=
                    forensic_dropout,
            )
        )

        self.tile_attention = (
            TileAttentionPooler(
                embedding_dim=
                    forensic_embedding_dim,

                hidden_dim=
                    attention_hidden_dim,

                dropout=
                    attention_dropout,
            )
        )

        self.forensic_head = (
            nn.Sequential(

                nn.LayerNorm(
                    forensic_embedding_dim
                ),

                nn.Linear(
                    forensic_embedding_dim,
                    128,
                ),

                nn.GELU(),

                nn.Dropout(
                    forensic_dropout
                ),

                nn.Linear(
                    128,
                    1,
                ),
            )
        )

        # ==================================================
        # FUSION
        # ==================================================

        self.semantic_projection = (
            nn.Sequential(

                nn.LayerNorm(
                    semantic_dim
                ),

                nn.Linear(
                    semantic_dim,
                    fusion_projection_dim,
                ),

                nn.GELU(),
            )
        )

        fusion_input_dim = (
            fusion_projection_dim
            + forensic_embedding_dim
        )

        self.fusion_head = (
            nn.Sequential(

                nn.LayerNorm(
                    fusion_input_dim
                ),

                nn.Linear(
                    fusion_input_dim,
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


    # ==================================================
    # ENCODERS
    # ==================================================

    def encode_semantic(
        self,
        pixel_values:
            torch.Tensor,
    ) -> torch.Tensor:

        raw_features = (
            self.backbone(
                pixel_values
            )
        )

        return (
            self.semantic_adapter(
                raw_features
            )
        )


    def encode_forensic(
        self,
        forensic_tiles:
            torch.Tensor,

        tile_mask:
            torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:

        tile_embeddings = (
            self.tile_encoder(
                forensic_tiles
            )
        )

        (
            pooled_forensic,
            attention,
        ) = (
            self.tile_attention(
                tile_embeddings=
                    tile_embeddings,

                tile_mask=
                    tile_mask,
            )
        )

        return (
            pooled_forensic,
            tile_embeddings,
            attention,
        )


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

        projected_semantic = (
            self.semantic_projection(
                semantic_features
            )
        )

        fused_features = torch.cat(
            [
                projected_semantic,
                forensic_features,
            ],
            dim=-1,
        )

        logits = (
            self.fusion_head(
                fused_features
            )
            .squeeze(
                -1
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

            "fused_features":
                fused_features,

            "semantic_logits":
                semantic_logits,

            "forensic_logits":
                forensic_logits,

            "logits":
                logits,
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
            clean_pixel_values
            .shape[
                0
            ]
        )

        # --------------------------------------------------
        # Semantic branch — one CLIP pass
        # --------------------------------------------------

        combined_pixels = torch.cat(
            [
                clean_pixel_values,
                corrupted_pixel_values,
            ],
            dim=0,
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

        # --------------------------------------------------
        # Forensic branch — combine tile batches
        # --------------------------------------------------

        combined_forensic_tiles = (
            torch.cat(
                [
                    clean_forensic_tiles,
                    corrupted_forensic_tiles,
                ],
                dim=0,
            )
        )

        combined_tile_mask = (
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
            combined_tile_embeddings,
            combined_attention,
        ) = (
            self.encode_forensic(
                forensic_tiles=
                    combined_forensic_tiles,

                tile_mask=
                    combined_tile_mask,
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

        # --------------------------------------------------
        # Heads
        # --------------------------------------------------

        clean_semantic_logits = (
            self.semantic_head(
                clean_semantic
            )
            .squeeze(
                -1
            )
        )

        corrupted_semantic_logits = (
            self.semantic_head(
                corrupted_semantic
            )
            .squeeze(
                -1
            )
        )

        clean_forensic_logits = (
            self.forensic_head(
                clean_forensic
            )
            .squeeze(
                -1
            )
        )

        corrupted_forensic_logits = (
            self.forensic_head(
                corrupted_forensic
            )
            .squeeze(
                -1
            )
        )

        clean_projected_semantic = (
            self.semantic_projection(
                clean_semantic
            )
        )

        corrupted_projected_semantic = (
            self.semantic_projection(
                corrupted_semantic
            )
        )

        clean_fused = torch.cat(
            [
                clean_projected_semantic,
                clean_forensic,
            ],
            dim=-1,
        )

        corrupted_fused = torch.cat(
            [
                corrupted_projected_semantic,
                corrupted_forensic,
            ],
            dim=-1,
        )

        clean_logits = (
            self.fusion_head(
                clean_fused
            )
            .squeeze(
                -1
            )
        )

        corrupted_logits = (
            self.fusion_head(
                corrupted_fused
            )
            .squeeze(
                -1
            )
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
                clean_fused,

            "corrupted_fused_features":
                corrupted_fused,

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
                clean_logits,

            "corrupted_logits":
                corrupted_logits,
        }


    # ==================================================
    # OPTIMIZER GROUPS
    # ==================================================

    def semantic_parameters(
        self,
    ):

        yield from (
            self.semantic_adapter
            .parameters()
        )

        yield from (
            self.semantic_head
            .parameters()
        )


    def forensic_parameters(
        self,
    ):

        yield from (
            self.tile_encoder
            .parameters()
        )

        yield from (
            self.forensic_head
            .parameters()
        )


    def attention_parameters(
        self,
    ):

        yield from (
            self.tile_attention
            .parameters()
        )


    def fusion_parameters(
        self,
    ):

        yield from (
            self.semantic_projection
            .parameters()
        )

        yield from (
            self.fusion_head
            .parameters()
        )


    def count_parameters(
        self,
    ) -> dict[
        str,
        int
    ]:

        total = sum(
            parameter.numel()
            for parameter
            in self.parameters()
        )

        trainable = sum(
            parameter.numel()
            for parameter
            in self.parameters()
            if parameter.requires_grad
        )

        return {
            "total":
                total,

            "trainable":
                trainable,

            "frozen":
                total
                - trainable,
        }