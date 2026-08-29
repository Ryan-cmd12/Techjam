from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from torch import nn


class NativeForensicFeatureExtractor(
    nn.Module
):

    def __init__(
        self,
        gaussian_sigma: float = 1.0,
        dct_block_size: int = 8,
    ):

        super().__init__()

        self.dct_block_size = (
            dct_block_size
        )

        gaussian_kernel = (
            self._build_gaussian_kernel(
                size=5,
                sigma=gaussian_sigma,
            )
        )

        gaussian_kernel = (
            gaussian_kernel
            .view(
                1,
                1,
                5,
                5,
            )
            .repeat(
                3,
                1,
                1,
                1,
            )
        )

        self.register_buffer(
            "gaussian_kernel",
            gaussian_kernel,
            persistent=False,
        )

        dct_matrix = (
            self._build_dct_matrix(
                dct_block_size
            )
        )

        self.register_buffer(
            "dct_matrix",
            dct_matrix,
            persistent=False,
        )

    @staticmethod
    def _build_gaussian_kernel(
        size: int,
        sigma: float,
    ) -> torch.Tensor:

        coordinates = torch.arange(
            size,
            dtype=torch.float32,
        )

        coordinates -= (
            size - 1
        ) / 2.0

        kernel_1d = torch.exp(
            -(
                coordinates ** 2
            )
            / (
                2.0
                * sigma ** 2
            )
        )

        kernel_1d /= kernel_1d.sum()

        return torch.outer(
            kernel_1d,
            kernel_1d,
        )

    @staticmethod
    def _build_dct_matrix(
        size: int,
    ) -> torch.Tensor:

        n = torch.arange(
            size,
            dtype=torch.float32,
        )

        k = torch.arange(
            size,
            dtype=torch.float32,
        ).unsqueeze(1)

        matrix = torch.cos(
            (
                math.pi
                / size
            )
            * (
                n + 0.5
            )
            * k
        )

        matrix[0] *= (
            math.sqrt(
                1.0
                / size
            )
        )

        if size > 1:

            matrix[1:] *= (
                math.sqrt(
                    2.0
                    / size
                )
            )

        return matrix

    @staticmethod
    def _standardize(
        values: torch.Tensor,
    ) -> torch.Tensor:

        mean = values.mean(
            dim=(-2, -1),
            keepdim=True,
        )

        std = values.std(
            dim=(-2, -1),
            keepdim=True,
            unbiased=False,
        )

        return (
            (
                values
                - mean
            )
            / (
                std
                + 1e-6
            )
        ).clamp(
            -5.0,
            5.0,
        )

    @staticmethod
    def rgb_to_gray(
        image: torch.Tensor,
    ) -> torch.Tensor:

        weights = torch.tensor(
            [
                0.299,
                0.587,
                0.114,
            ],
            device=image.device,
            dtype=image.dtype,
        ).view(
            1,
            3,
            1,
            1,
        )

        return (
            image
            * weights
        ).sum(
            dim=1,
            keepdim=True,
        )

    def high_pass_residual(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:

        kernel = (
            self.gaussian_kernel
            .to(
                device=image.device,
                dtype=image.dtype,
            )
        )

        blurred = F.conv2d(
            image,
            kernel,
            padding=2,
            groups=3,
        )

        residual = (
            image
            - blurred
        )

        return (
            residual
            * 4.0
        ).clamp(
            -1.0,
            1.0,
        )

    def fft_map(
        self,
        gray: torch.Tensor,
    ) -> torch.Tensor:

        spectrum = torch.fft.fft2(
            gray.squeeze(1),
            norm="ortho",
        )

        spectrum = torch.fft.fftshift(
            spectrum,
            dim=(-2, -1),
        )

        magnitude = (
            torch.log1p(
                torch.abs(
                    spectrum
                )
            )
            .unsqueeze(1)
        )

        return self._standardize(
            magnitude
        )

    def block_dct_map(
        self,
        gray: torch.Tensor,
    ) -> torch.Tensor:

        block_size = (
            self.dct_block_size
        )

        original_height = (
            gray.shape[-2]
        )

        original_width = (
            gray.shape[-1]
        )

        pad_height = (
            block_size
            - (
                original_height
                % block_size
            )
        ) % block_size

        pad_width = (
            block_size
            - (
                original_width
                % block_size
            )
        ) % block_size

        if (
            pad_height
            or pad_width
        ):

            gray = F.pad(
                gray,
                (
                    0,
                    pad_width,
                    0,
                    pad_height,
                ),
                mode="replicate",
            )

        batch_size = (
            gray.shape[0]
        )

        padded_height = (
            gray.shape[-2]
        )

        padded_width = (
            gray.shape[-1]
        )

        blocks = F.unfold(
            gray,

            kernel_size=
                block_size,

            stride=
                block_size,
        )

        number_of_blocks = (
            blocks.shape[-1]
        )

        blocks = (
            blocks
            .transpose(
                1,
                2,
            )
            .reshape(
                batch_size,
                number_of_blocks,
                block_size,
                block_size,
            )
        )

        dct_matrix = (
            self.dct_matrix
            .to(
                device=
                    gray.device,

                dtype=
                    gray.dtype,
            )
        )

        transformed = (
            torch.matmul(
                dct_matrix,
                blocks,
            )
        )

        transformed = (
            torch.matmul(
                transformed,
                dct_matrix.transpose(
                    0,
                    1,
                ),
            )
        )

        transformed = torch.log1p(
            torch.abs(
                transformed
            )
        )

        transformed = (
            transformed.clone()
        )

        transformed[
            :,
            :,
            0,
            0,
        ] = 0.0

        transformed = (
            transformed
            .reshape(
                batch_size,
                number_of_blocks,
                block_size
                * block_size,
            )
            .transpose(
                1,
                2,
            )
        )

        reconstructed = F.fold(
            transformed,

            output_size=(
                padded_height,
                padded_width,
            ),

            kernel_size=
                block_size,

            stride=
                block_size,
        )

        reconstructed = (
            reconstructed[
                :,
                :,
                :original_height,
                :original_width,
            ]
        )

        return self._standardize(
            reconstructed
        )

    def haar_details(
        self,
        gray: torch.Tensor,
    ) -> torch.Tensor:

        original_height = (
            gray.shape[-2]
        )

        original_width = (
            gray.shape[-1]
        )

        pad_bottom = (
            original_height
            % 2
        )

        pad_right = (
            original_width
            % 2
        )

        if (
            pad_bottom
            or pad_right
        ):

            gray = F.pad(
                gray,
                (
                    0,
                    pad_right,
                    0,
                    pad_bottom,
                ),
                mode="replicate",
            )

        a = gray[
            :,
            :,
            0::2,
            0::2,
        ]

        b = gray[
            :,
            :,
            0::2,
            1::2,
        ]

        c = gray[
            :,
            :,
            1::2,
            0::2,
        ]

        d = gray[
            :,
            :,
            1::2,
            1::2,
        ]

        horizontal = (
            a
            + b
            - c
            - d
        ) / 2.0

        vertical = (
            a
            - b
            + c
            - d
        ) / 2.0

        diagonal = (
            a
            - b
            - c
            + d
        ) / 2.0

        details = torch.cat(
            [
                horizontal,
                vertical,
                diagonal,
            ],
            dim=1,
        )

        details = F.interpolate(
            details,

            size=(
                original_height,
                original_width,
            ),

            mode="bilinear",

            align_corners=False,
        )

        return self._standardize(
            details
        )

    def forward(
        self,
        images: torch.Tensor,
    ) -> dict[
        str,
        torch.Tensor
    ]:

        images = (
            images
            .float()
            .clamp(
                0.0,
                1.0,
            )
        )

        gray = self.rgb_to_gray(
            images
        )

        residual = (
            self.high_pass_residual(
                images
            )
        )

        fft = self.fft_map(
            gray
        )

        dct = self.block_dct_map(
            gray
        )

        wavelet = self.haar_details(
            gray
        )

        combined = torch.cat(
            [
                residual,
                fft,
                dct,
                wavelet,
            ],
            dim=1,
        )

        return {
            "residual":
                residual,

            "fft":
                fft,

            "dct":
                dct,

            "wavelet":
                wavelet,

            "combined":
                combined,
        }