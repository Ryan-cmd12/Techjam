from __future__ import annotations

import torch

from torch import nn

from src.models.clip_backbone import (
    CLIPImageBackbone,
)

from src.models.forensic_encoder import (
    ForensicEncoder,
)

from src.models.residual_adapter import (
    ResidualFeatureAdapter,
)


class ForensicFusionDetector(nn.Module):

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
        fusion_projection_dim: int = 256,
        fusion_hidden_dim: int = 256,
        fusion_dropout: float = 0.20,
    ):
        super().__init__()

        self.clip_model_name = (
            clip_model_name
        )

        self.backbone = CLIPImageBackbone(
            model_name=
                clip_model_name,

            freeze=True,

            normalize_embeddings=True,
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

        # Same architecture as the previous robust
        # classifier, allowing exact warm start.
        self.semantic_head = nn.Sequential(
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

        self.forensic_encoder = (
            ForensicEncoder(
                embedding_dim=
                    forensic_embedding_dim,

                base_channels=
                    forensic_base_channels,

                dropout=
                    forensic_dropout,
            )
        )

        self.forensic_head = nn.Sequential(
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

        self.semantic_projection = nn.Sequential(
            nn.LayerNorm(
                semantic_dim
            ),

            nn.Linear(
                semantic_dim,
                fusion_projection_dim,
            ),

            nn.GELU(),
        )

        fusion_input_dim = (
            fusion_projection_dim
            + forensic_embedding_dim
        )

        self.fusion_head = nn.Sequential(
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

    def encode(
        self,
        pixel_values: torch.Tensor,
        forensic_images: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        raw_semantic = self.backbone(
            pixel_values
        )

        semantic = self.semantic_adapter(
            raw_semantic
        )

        forensic = self.forensic_encoder(
            forensic_images
        )

        projected_semantic = (
            self.semantic_projection(
                semantic
            )
        )

        fused = torch.cat(
            [
                projected_semantic,
                forensic,
            ],
            dim=-1,
        )

        return {
            "semantic_features":
                semantic,

            "forensic_features":
                forensic,

            "fused_features":
                fused,
        }

    def forward_with_details(
        self,
        pixel_values: torch.Tensor,
        forensic_images: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        features = self.encode(
            pixel_values=
                pixel_values,

            forensic_images=
                forensic_images,
        )

        semantic_logits = (
            self.semantic_head(
                features[
                    "semantic_features"
                ]
            )
            .squeeze(-1)
        )

        forensic_logits = (
            self.forensic_head(
                features[
                    "forensic_features"
                ]
            )
            .squeeze(-1)
        )

        final_logits = (
            self.fusion_head(
                features[
                    "fused_features"
                ]
            )
            .squeeze(-1)
        )

        return {
            **features,

            "semantic_logits":
                semantic_logits,

            "forensic_logits":
                forensic_logits,

            "logits":
                final_logits,
        }

    def forward(
        self,
        pixel_values: torch.Tensor,
        forensic_images: torch.Tensor,
    ) -> torch.Tensor:

        return self.forward_with_details(
            pixel_values,
            forensic_images,
        )["logits"]

    def forward_pair(
        self,
        clean_pixel_values: torch.Tensor,
        corrupted_pixel_values: torch.Tensor,
        clean_forensic_images: torch.Tensor,
        corrupted_forensic_images: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        batch_size = clean_pixel_values.shape[0]

        combined_pixels = torch.cat(
            [
                clean_pixel_values,
                corrupted_pixel_values,
            ],
            dim=0,
        )

        combined_forensic = torch.cat(
            [
                clean_forensic_images,
                corrupted_forensic_images,
            ],
            dim=0,
        )

        output = self.forward_with_details(
            pixel_values=
                combined_pixels,

            forensic_images=
                combined_forensic,
        )

        result = {}

        for key, values in output.items():

            result[
                f"clean_{key}"
            ] = values[
                :batch_size
            ]

            result[
                f"corrupted_{key}"
            ] = values[
                batch_size:
            ]

        return result

    def semantic_parameters(
        self,
    ):
        modules = [
            self.semantic_adapter,
            self.semantic_head,
        ]

        for module in modules:
            yield from module.parameters()

    def new_branch_parameters(
        self,
    ):
        modules = [
            self.forensic_encoder,
            self.forensic_head,
            self.semantic_projection,
            self.fusion_head,
        ]

        for module in modules:
            yield from module.parameters()

    def count_parameters(
        self,
    ) -> dict[str, int]:

        total = sum(
            p.numel()
            for p in self.parameters()
        )

        trainable = sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

        return {
            "total":
                total,

            "trainable":
                trainable,

            "frozen":
                total - trainable,
        }