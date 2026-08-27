from __future__ import annotations

import torch

from torch import nn

from src.models.clip_backbone import (
    CLIPImageBackbone,
)

from src.models.residual_adapter import (
    ResidualFeatureAdapter,
)


class RobustAIGCDetector(
    nn.Module
):

    def __init__(
        self,
        clip_model_name: str,
        classifier_hidden_dim: int = 512,
        classifier_dropout: float = 0.20,
        adapter_bottleneck_dim: int = 256,
        adapter_dropout: float = 0.10,
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
            f"Robust model CLIP feature dimension: "
            f"{feature_dim}"
        )

        self.adapter = (
            ResidualFeatureAdapter(
                feature_dim=
                    feature_dim,

                bottleneck_dim=
                    adapter_bottleneck_dim,

                dropout=
                    adapter_dropout,
            )
        )

        self.classifier = (
            nn.Sequential(
                nn.LayerNorm(
                    feature_dim
                ),

                nn.Linear(
                    feature_dim,
                    classifier_hidden_dim,
                ),

                nn.GELU(),

                nn.Dropout(
                    classifier_dropout
                ),

                nn.Linear(
                    classifier_hidden_dim,
                    1,
                ),
            )
        )


    def extract_raw_features(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:

        return (
            self.backbone(
                pixel_values
            )
        )


    def adapt_features(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:

        return (
            self.adapter(
                features
            )
        )


    def classify_features(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:

        logits = (
            self.classifier(
                features
            )
            .squeeze(
                -1
            )
        )

        return logits


    def forward_with_features(
        self,
        pixel_values: torch.Tensor,
    ) -> dict[
        str,
        torch.Tensor
    ]:

        raw_features = (
            self.extract_raw_features(
                pixel_values
            )
        )

        adapted_features = (
            self.adapt_features(
                raw_features
            )
        )

        logits = (
            self.classify_features(
                adapted_features
            )
        )

        return {
            "raw_features":
                raw_features,

            "features":
                adapted_features,

            "logits":
                logits,
        }


    def forward_pair(
        self,
        clean_pixel_values:
            torch.Tensor,

        corrupted_pixel_values:
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

        # One backbone pass instead of two.
        combined_pixels = (
            torch.cat(
                [
                    clean_pixel_values,
                    corrupted_pixel_values,
                ],
                dim=0,
            )
        )

        combined_raw = (
            self.extract_raw_features(
                combined_pixels
            )
        )

        combined_features = (
            self.adapt_features(
                combined_raw
            )
        )

        combined_logits = (
            self.classify_features(
                combined_features
            )
        )

        clean_features = (
            combined_features[
                :batch_size
            ]
        )

        corrupted_features = (
            combined_features[
                batch_size:
            ]
        )

        clean_logits = (
            combined_logits[
                :batch_size
            ]
        )

        corrupted_logits = (
            combined_logits[
                batch_size:
            ]
        )

        return {
            "clean_features":
                clean_features,

            "corrupted_features":
                corrupted_features,

            "clean_logits":
                clean_logits,

            "corrupted_logits":
                corrupted_logits,
        }


    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:

        output = (
            self.forward_with_features(
                pixel_values
            )
        )

        return (
            output[
                "logits"
            ]
        )


    def predict_proba(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:

        return (
            torch.sigmoid(
                self(
                    pixel_values
                )
            )
        )


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