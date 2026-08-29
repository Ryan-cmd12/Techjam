from __future__ import annotations

import torch

from torch import nn


class TemperatureScaler(
    nn.Module
):

    def __init__(
        self,
        initial_temperature: float = 1.0,
    ):

        super().__init__()

        self.log_temperature = (
            nn.Parameter(
                torch.tensor(
                    float(
                        initial_temperature
                    )
                ).log()
            )
        )


    @property
    def temperature(
        self,
    ) -> torch.Tensor:

        return torch.exp(
            self.log_temperature
        )


    def forward(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:

        temperature = (
            self.temperature
            .clamp(
                0.05,
                20.0,
            )
        )

        return (
            logits
            / temperature
        )


    def probabilities(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:

        return torch.sigmoid(
            self(
                logits
            )
        )


    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        max_iterations: int = 100,
    ) -> float:

        logits = (
            logits
            .detach()
            .float()
            .cpu()
        )

        labels = (
            labels
            .detach()
            .float()
            .cpu()
        )

        self.cpu()

        criterion = (
            nn.BCEWithLogitsLoss()
        )

        optimizer = (
            torch.optim.LBFGS(

                [
                    self.log_temperature
                ],

                lr=0.1,

                max_iter=
                    max_iterations,

                line_search_fn=
                    "strong_wolfe",
            )
        )


        def closure():

            optimizer.zero_grad()

            scaled_logits = self(
                logits
            )

            loss = criterion(
                scaled_logits,
                labels,
            )

            loss.backward()

            return loss


        optimizer.step(
            closure
        )

        with torch.no_grad():

            self.log_temperature.clamp_(
                min=torch.log(
                    torch.tensor(
                        0.05
                    )
                ),

                max=torch.log(
                    torch.tensor(
                        20.0
                    )
                ),
            )

        return float(
            self.temperature.item()
        )