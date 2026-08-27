from __future__ import annotations

import torch

from torch import nn

from src.models.clip_backbone import (
    CLIPImageBackbone,
)


class BaselineAIGCDetector(
    nn.Module
):

    def __init__(
        self,
        clip_model_name: str,
        hidden_dim: int = 512,
        dropout: float = 0.20,
        freeze_backbone: bool = True,
        normalize_embeddings: bool = True,
    ):

        super().__init__()

        self.clip_model_name = (
            clip_model_name
        )

        self.backbone = (
            CLIPImageBackbone(
                model_name=
                    clip_model_name,

                freeze=
                    freeze_backbone,

                normalize_embeddings=
                    normalize_embeddings,
            )
        )

        feature_dim = (
            self.backbone.feature_dim
        )

        print(
            f"CLIP feature dimension: "
            f"{feature_dim}"
        )

        self.classifier = (
            nn.Sequential(
                nn.LayerNorm(
                    feature_dim
                ),

                nn.Linear(
                    feature_dim,
                    hidden_dim,
                ),

                nn.GELU(),

                nn.Dropout(
                    dropout
                ),

                nn.Linear(
                    hidden_dim,
                    1,
                ),
            )
        )


    def extract_features(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:

        return self.backbone(
            pixel_values
        )


    def classify_features(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:

        logits = (
            self.classifier(
                features
            )
        )

        logits = (
            logits.squeeze(
                -1
            )
        )

        return logits


    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:

        features = (
            self.extract_features(
                pixel_values
            )
        )

        logits = (
            self.classify_features(
                features
            )
        )

        return logits


    def predict_proba(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:

        logits = self(
            pixel_values
        )

        probabilities = (
            torch.sigmoid(
                logits
            )
        )

        return probabilities


    def get_trainable_parameters(
        self,
    ):

        return (
            parameter
            for parameter
            in self.parameters()
            if parameter.requires_grad
        )


    def count_parameters(
        self,
    ) -> dict[str, int]:

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

        frozen = (
            total
            - trainable
        )

        return {
            "total":
                total,

            "trainable":
                trainable,

            "frozen":
                frozen,
        }