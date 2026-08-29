from __future__ import annotations

import torch

from torch import nn

from src.models.forensic_features import (
    ForensicFeatureExtractor,
)


class ConvForensicBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),

            nn.GroupNorm(
                num_groups=min(
                    8,
                    out_channels,
                ),
                num_channels=out_channels,
            ),

            nn.GELU(),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.GroupNorm(
                num_groups=min(
                    8,
                    out_channels,
                ),
                num_channels=out_channels,
            ),

            nn.GELU(),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.block(x)


class ForensicEncoder(nn.Module):

    def __init__(
        self,
        embedding_dim: int = 256,
        base_channels: int = 32,
        dropout: float = 0.10,
    ):
        super().__init__()

        self.feature_extractor = (
            ForensicFeatureExtractor()
        )

        self.encoder = nn.Sequential(
            ConvForensicBlock(
                8,
                base_channels,
            ),

            ConvForensicBlock(
                base_channels,
                base_channels * 2,
                stride=2,
            ),

            ConvForensicBlock(
                base_channels * 2,
                base_channels * 4,
                stride=2,
            ),

            ConvForensicBlock(
                base_channels * 4,
                base_channels * 8,
                stride=2,
            ),

            nn.AdaptiveAvgPool2d(
                1
            ),
        )

        final_channels = (
            base_channels
            * 8
        )

        self.projection = nn.Sequential(
            nn.Flatten(),

            nn.LayerNorm(
                final_channels
            ),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                final_channels,
                embedding_dim,
            ),

            nn.GELU(),

            nn.LayerNorm(
                embedding_dim
            ),
        )

    def forward(
        self,
        images: torch.Tensor,
        return_maps: bool = False,
    ):

        maps = self.feature_extractor(
            images
        )

        encoded = self.encoder(
            maps["combined"]
        )

        embedding = self.projection(
            encoded
        )

        if return_maps:
            return (
                embedding,
                maps,
            )

        return embedding