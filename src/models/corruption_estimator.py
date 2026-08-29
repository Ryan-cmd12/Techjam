from __future__ import annotations

import torch

from torch import nn


class CorruptionEstimator(
    nn.Module
):

    def __init__(
        self,
        semantic_dim: int,
        forensic_dim: int,
        hidden_dim: int = 256,
        embedding_dim: int = 128,
        num_types: int = 7,
        dropout: float = 0.10,
    ):

        super().__init__()

        input_dim = (
            semantic_dim
            + forensic_dim
        )

        self.encoder = (
            nn.Sequential(

                nn.LayerNorm(
                    input_dim
                ),

                nn.Linear(
                    input_dim,
                    hidden_dim,
                ),

                nn.GELU(),

                nn.Dropout(
                    dropout
                ),

                nn.Linear(
                    hidden_dim,
                    embedding_dim,
                ),

                nn.GELU(),

                nn.LayerNorm(
                    embedding_dim
                ),
            )
        )

        self.type_head = (
            nn.Linear(
                embedding_dim,
                num_types,
            )
        )

        self.severity_head = (
            nn.Sequential(

                nn.Linear(
                    embedding_dim,
                    1,
                ),

                nn.Sigmoid(),
            )
        )


    def forward(
        self,
        semantic_features:
            torch.Tensor,

        forensic_features:
            torch.Tensor,
    ) -> dict[
        str,
        torch.Tensor
    ]:

        combined = torch.cat(
            [
                semantic_features,
                forensic_features,
            ],
            dim=-1,
        )

        embedding = (
            self.encoder(
                combined
            )
        )

        type_logits = (
            self.type_head(
                embedding
            )
        )

        severity = (
            self.severity_head(
                embedding
            )
            .squeeze(
                -1
            )
        )

        return {
            "embedding":
                embedding,

            "type_logits":
                type_logits,

            "severity":
                severity,
        }