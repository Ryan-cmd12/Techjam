from __future__ import annotations

import torch

from torch import nn

from src.models.forensic_encoder import (
    ConvForensicBlock,
)


class NativeTileForensicEncoder(
    nn.Module
):

    def __init__(
        self,
        embedding_dim: int = 256,
        base_channels: int = 32,
        dropout: float = 0.10,
    ):

        super().__init__()

        self.embedding_dim = (
            embedding_dim
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
        forensic_tiles:
            torch.Tensor,
    ) -> torch.Tensor:
        """
        forensic_tiles:

            [B, T, 8, H, W]

        Returns:

            [B, T, embedding_dim]
        """

        batch_size = (
            forensic_tiles.shape[
                0
            ]
        )

        tile_count = (
            forensic_tiles.shape[
                1
            ]
        )

        channels = (
            forensic_tiles.shape[
                2
            ]
        )

        height = (
            forensic_tiles.shape[
                3
            ]
        )

        width = (
            forensic_tiles.shape[
                4
            ]
        )

        flattened_tiles = (
            forensic_tiles.reshape(
                batch_size
                * tile_count,

                channels,
                height,
                width,
            )
        )

        encoded = (
            self.encoder(
                flattened_tiles
            )
        )

        embeddings = (
            self.projection(
                encoded
            )
        )

        embeddings = (
            embeddings.reshape(
                batch_size,
                tile_count,
                self.embedding_dim,
            )
        )

        return embeddings