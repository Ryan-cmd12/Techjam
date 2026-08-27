from __future__ import annotations

import torch

import torch.nn.functional as F

from torch import nn


class ResidualFeatureAdapter(
    nn.Module
):

    def __init__(
        self,
        feature_dim: int,
        bottleneck_dim: int = 256,
        dropout: float = 0.10,
    ):

        super().__init__()

        self.feature_dim = (
            feature_dim
        )

        self.bottleneck_dim = (
            bottleneck_dim
        )

        self.adapter = (
            nn.Sequential(
                nn.LayerNorm(
                    feature_dim
                ),

                nn.Linear(
                    feature_dim,
                    bottleneck_dim,
                ),

                nn.GELU(),

                nn.Dropout(
                    dropout
                ),

                nn.Linear(
                    bottleneck_dim,
                    feature_dim,
                ),
            )
        )

        # Start with a small residual contribution.
        #
        # This lets training begin close to the original
        # CLIP representation rather than immediately
        # destroying it.
        self.residual_scale = (
            nn.Parameter(
                torch.tensor(
                    0.10,
                    dtype=torch.float32,
                )
            )
        )


    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:

        residual = (
            self.adapter(
                features
            )
        )

        adapted = (
            features
            + self.residual_scale
            * residual
        )

        adapted = (
            F.normalize(
                adapted,
                p=2,
                dim=-1,
            )
        )

        return adapted