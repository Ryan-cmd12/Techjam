from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class ForensicFeatureExtractor(nn.Module):

    def __init__(
        self,
        gaussian_sigma: float = 1.0,
    ):
        super().__init__()

        kernel = self._build_gaussian_kernel(
            size=5,
            sigma=gaussian_sigma,
        )

        kernel = kernel.view(
            1,
            1,
            5,
            5,
        )

        kernel = kernel.repeat(
            3,
            1,
            1,
            1,
        )

        self.register_buffer(
            "gaussian_kernel",
            kernel,
            persistent=False,
        )

        self._dct_cache = {}

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

        kernel_2d = torch.outer(
            kernel_1d,
            kernel_1d,
        )

        return kernel_2d

    @staticmethod
    def _standardize_map(
        values: torch.Tensor,
    ) -> torch.Tensor:

        mean = values.mean(
            dim=(-2, -1),
            keepdim=True,
        )

        std = values.std(
            dim=(-2, -1),
            keepdim=True,
        )

        values = (
            values - mean
        ) / (
            std + 1e-6
        )

        return values.clamp(
            -5.0,
            5.0,
        )

    @staticmethod
    def rgb_to_gray(
        images: torch.Tensor,
    ) -> torch.Tensor:

        weights = torch.tensor(
            [
                0.299,
                0.587,
                0.114,
            ],
            device=images.device,
            dtype=images.dtype,
        ).view(
            1,
            3,
            1,
            1,
        )

        return (
            images
            * weights
        ).sum(
            dim=1,
            keepdim=True,
        )

    def high_pass_residual(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:

        blurred = F.conv2d(
            images,
            self.gaussian_kernel.to(
                dtype=images.dtype
            ),
            padding=2,
            groups=3,
        )

        residual = (
            images - blurred
        )

        # Make subtle residuals easier for the CNN
        # to represent while retaining sign.
        residual = (
            residual * 4.0
        ).clamp(
            -1.0,
            1.0,
        )

        return residual

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

        magnitude = torch.log1p(
            torch.abs(
                spectrum
            )
        ).unsqueeze(1)

        return self._standardize_map(
            magnitude
        )

    def _dct_matrix(
        self,
        size: int,
        device,
        dtype,
    ) -> torch.Tensor:

        key = (
            size,
            str(device),
            str(dtype),
        )

        if key in self._dct_cache:
            return self._dct_cache[key]

        n = torch.arange(
            size,
            device=device,
            dtype=dtype,
        )

        k = torch.arange(
            size,
            device=device,
            dtype=dtype,
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

        matrix[0] *= math.sqrt(
            1.0 / size
        )

        if size > 1:
            matrix[1:] *= math.sqrt(
                2.0 / size
            )

        self._dct_cache[key] = matrix

        return matrix

    def dct_map(
        self,
        gray: torch.Tensor,
    ) -> torch.Tensor:

        gray_2d = gray.squeeze(1)

        height = gray_2d.shape[-2]
        width = gray_2d.shape[-1]

        dct_h = self._dct_matrix(
            height,
            gray_2d.device,
            gray_2d.dtype,
        )

        dct_w = self._dct_matrix(
            width,
            gray_2d.device,
            gray_2d.dtype,
        )

        transformed = torch.matmul(
            dct_h,
            gray_2d,
        )

        transformed = torch.matmul(
            transformed,
            dct_w.transpose(
                0,
                1,
            ),
        )

        transformed = torch.log1p(
            torch.abs(
                transformed
            )
        )

        # Suppress DC coefficient so overall brightness
        # does not dominate the visualization/features.
        transformed = transformed.clone()

        transformed[
            :,
            0,
            0,
        ] = 0.0

        transformed = transformed.unsqueeze(
            1
        )

        return self._standardize_map(
            transformed
        )

    def haar_wavelet_details(
        self,
        gray: torch.Tensor,
    ) -> torch.Tensor:

        original_height = gray.shape[-2]
        original_width = gray.shape[-1]

        pad_bottom = (
            original_height % 2
        )

        pad_right = (
            original_width % 2
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
            a + b - c - d
        ) / 2.0

        vertical = (
            a - b + c - d
        ) / 2.0

        diagonal = (
            a - b - c + d
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

        details = self._standardize_map(
            details
        )

        return details

    def forward(
        self,
        images: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        # FFT/DCT are safest in FP32.
        images = images.float().clamp(
            0.0,
            1.0,
        )

        gray = self.rgb_to_gray(
            images
        )

        residual = self.high_pass_residual(
            images
        )

        fft = self.fft_map(
            gray
        )

        dct = self.dct_map(
            gray
        )

        wavelet = self.haar_wavelet_details(
            gray
        )

        combined = torch.cat(
            [
                residual,   # 3
                fft,        # 1
                dct,        # 1
                wavelet,    # 3
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