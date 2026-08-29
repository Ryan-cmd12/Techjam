from __future__ import annotations

import torch

from torch import nn


class TileAttentionPooler(
    nn.Module
):

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.10,
    ):

        super().__init__()

        self.scorer = nn.Sequential(

            nn.LayerNorm(
                embedding_dim
            ),

            nn.Linear(
                embedding_dim,
                hidden_dim,
            ),

            nn.Tanh(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                1,
            ),
        )


    def forward(
        self,
        tile_embeddings:
            torch.Tensor,

        tile_mask:
            torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        tile_embeddings:
            [B, T, D]

        tile_mask:
            [B, T]

        Returns:

            pooled:
                [B, D]

            attention:
                [B, T]
        """

        scores = (
            self.scorer(
                tile_embeddings
            )
            .squeeze(
                -1
            )
        )

        tile_mask = (
            tile_mask.bool()
        )

        # Padded tiles must get zero attention.
        scores = scores.masked_fill(
            ~tile_mask,
            float(
                "-inf"
            ),
        )

        attention = (
            torch.softmax(
                scores,
                dim=1,
            )
        )

        attention = torch.where(
            tile_mask,
            attention,
            torch.zeros_like(
                attention
            ),
        )

        # Numerical protection.
        denominator = (
            attention.sum(
                dim=1,
                keepdim=True,
            )
            .clamp_min(
                1e-8
            )
        )

        attention = (
            attention
            / denominator
        )

        pooled = torch.sum(
            tile_embeddings
            * attention.unsqueeze(
                -1
            ),

            dim=1,
        )

        return (
            pooled,
            attention,
        )