from __future__ import annotations

import torch

from torch import nn


class ReliabilityGate(
    nn.Module
):

    def __init__(
        self,
        semantic_dim: int,
        forensic_dim: int,
        corruption_embedding_dim: int,
        num_corruption_types: int,
        hidden_dim: int = 256,
        dropout: float = 0.10,
    ):

        super().__init__()

        input_dim = (
            semantic_dim
            + forensic_dim
            + corruption_embedding_dim
            + num_corruption_types
            + 1
        )

        self.network = (
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
                    hidden_dim // 2,
                ),

                nn.GELU(),

                nn.Linear(
                    hidden_dim // 2,
                    2,
                ),
            )
        )


    def forward(
        self,

        semantic_features:
            torch.Tensor,

        forensic_features:
            torch.Tensor,

        corruption_embedding:
            torch.Tensor,

        corruption_type_probabilities:
            torch.Tensor,

        corruption_severity:
            torch.Tensor,

    ) -> dict[
        str,
        torch.Tensor
    ]:

        severity = (
            corruption_severity
            .unsqueeze(
                -1
            )
        )

        gate_input = (
            torch.cat(
                [
                    semantic_features,
                    forensic_features,
                    corruption_embedding,
                    corruption_type_probabilities,
                    severity,
                ],

                dim=-1,
            )
        )

        gate_logits = (
            self.network(
                gate_input
            )
        )

        weights = (
            torch.softmax(
                gate_logits,
                dim=-1,
            )
        )

        return {
            "gate_logits":
                gate_logits,

            "weights":
                weights,

            "semantic_weight":
                weights[
                    :,
                    0
                ],

            "forensic_weight":
                weights[
                    :,
                    1
                ],
        }