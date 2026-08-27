from __future__ import annotations

import torch

import torch.nn.functional as F

from torch import nn

from transformers import (
    CLIPVisionModelWithProjection,
)


class CLIPImageBackbone(
    nn.Module
):

    def __init__(
        self,
        model_name: str,
        freeze: bool = True,
        normalize_embeddings: bool = True,
    ):

        super().__init__()

        self.model_name = (
            model_name
        )

        self.freeze = (
            freeze
        )

        self.normalize_embeddings = (
            normalize_embeddings
        )

        print(
            f"Loading CLIP vision model: "
            f"{model_name}"
        )

        self.model = (
            CLIPVisionModelWithProjection
            .from_pretrained(
                model_name
            )
        )

        self.feature_dim = (
            self.model
            .visual_projection
            .out_features
        )

        if self.freeze:

            self.freeze_backbone()


    def freeze_backbone(
        self,
    ) -> None:

        for parameter in (
            self.model.parameters()
        ):

            parameter.requires_grad = (
                False
            )

        self.model.eval()


    def unfreeze_backbone(
        self,
    ) -> None:

        for parameter in (
            self.model.parameters()
        ):

            parameter.requires_grad = (
                True
            )

        self.freeze = False


    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:

        if self.freeze:

            with torch.no_grad():

                outputs = self.model(
                    pixel_values=
                        pixel_values,
                )

        else:

            outputs = self.model(
                pixel_values=
                    pixel_values,
            )

        embeddings = (
            outputs.image_embeds
        )

        if self.normalize_embeddings:

            embeddings = (
                F.normalize(
                    embeddings,
                    p=2,
                    dim=-1,
                )
            )

        return embeddings


    def train(
        self,
        mode: bool = True,
    ):

        super().train(
            mode
        )

        # If the CLIP encoder is frozen, always keep it
        # in evaluation mode even while the classifier
        # itself is training.
        if self.freeze:

            self.model.eval()

        return self