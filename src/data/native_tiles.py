from __future__ import annotations

import hashlib
import math
import random

from dataclasses import dataclass

import numpy as np

from PIL import Image


@dataclass(
    frozen=True
)
class TileBox:

    left: int
    top: int
    right: int
    bottom: int

    def as_tuple(
        self,
    ) -> tuple[
        int,
        int,
        int,
        int,
    ]:

        return (
            self.left,
            self.top,
            self.right,
            self.bottom,
        )

    def normalized(
        self,
        image_width: int,
        image_height: int,
    ) -> tuple[
        float,
        float,
        float,
        float,
    ]:

        return (
            self.left
            / image_width,

            self.top
            / image_height,

            self.right
            / image_width,

            self.bottom
            / image_height,
        )


class NativeTileSampler:

    def __init__(
        self,
        tile_size: int = 256,
        max_tiles: int = 6,
        mode: str = "grid",
        seed: int = 42,
    ):

        if tile_size <= 0:

            raise ValueError(
                "tile_size must be > 0"
            )

        if max_tiles <= 0:

            raise ValueError(
                "max_tiles must be > 0"
            )

        if mode not in {
            "grid",
            "random",
        }:

            raise ValueError(
                "mode must be "
                "'grid' or 'random'"
            )

        self.tile_size = (
            tile_size
        )

        self.max_tiles = (
            max_tiles
        )

        self.mode = (
            mode
        )

        self.seed = (
            seed
        )

    def _build_seed(
        self,
        sample_key: str,
        sampling_token: int,
    ) -> int:

        value = (
            f"{self.seed}|"
            f"{sample_key}|"
            f"{sampling_token}"
        )

        digest = hashlib.sha256(
            value.encode(
                "utf-8"
            )
        ).hexdigest()

        return int(
            digest[:8],
            16,
        )

    def _tile_dimensions(
        self,
        image: Image.Image,
    ) -> tuple[
        int,
        int,
    ]:

        width, height = (
            image.size
        )

        tile_width = min(
            self.tile_size,
            width,
        )

        tile_height = min(
            self.tile_size,
            height,
        )

        return (
            tile_width,
            tile_height,
        )

    def _grid_coordinates(
        self,
        image: Image.Image,
    ) -> list[
        TileBox
    ]:

        width, height = (
            image.size
        )

        (
            tile_width,
            tile_height,
        ) = self._tile_dimensions(
            image
        )

        max_left = max(
            0,
            width
            - tile_width,
        )

        max_top = max(
            0,
            height
            - tile_height,
        )

        if (
            max_left == 0
            and max_top == 0
        ):

            return [
                TileBox(
                    0,
                    0,
                    width,
                    height,
                )
            ]

        aspect_ratio = (
            width
            / max(
                height,
                1,
            )
        )

        approximate_columns = (
            math.sqrt(
                self.max_tiles
                * aspect_ratio
            )
        )

        columns = max(
            1,
            int(
                round(
                    approximate_columns
                )
            ),
        )

        columns = min(
            columns,
            self.max_tiles,
        )

        rows = max(
            1,
            math.ceil(
                self.max_tiles
                / columns
            ),
        )

        x_positions = (
            np.linspace(
                0,
                max_left,
                columns,
            )
            .round()
            .astype(
                int
            )
            .tolist()
        )

        y_positions = (
            np.linspace(
                0,
                max_top,
                rows,
            )
            .round()
            .astype(
                int
            )
            .tolist()
        )

        coordinates = []

        for top in y_positions:

            for left in x_positions:

                coordinates.append(
                    TileBox(
                        left=
                            left,

                        top=
                            top,

                        right=
                            left
                            + tile_width,

                        bottom=
                            top
                            + tile_height,
                    )
                )

        # Remove duplicates caused by linspace rounding.
        coordinates = list(
            dict.fromkeys(
                coordinates
            )
        )

        if (
            len(
                coordinates
            )
            <= self.max_tiles
        ):

            return coordinates

        indices = (
            np.linspace(
                0,
                len(
                    coordinates
                )
                - 1,

                self.max_tiles,
            )
            .round()
            .astype(
                int
            )
        )

        return [
            coordinates[
                index
            ]
            for index
            in indices
        ]

    def _random_coordinates(
        self,
        image: Image.Image,
        sample_key: str,
        sampling_token: int,
    ) -> list[
        TileBox
    ]:

        width, height = (
            image.size
        )

        (
            tile_width,
            tile_height,
        ) = self._tile_dimensions(
            image
        )

        max_left = max(
            0,
            width
            - tile_width,
        )

        max_top = max(
            0,
            height
            - tile_height,
        )

        if (
            max_left == 0
            and max_top == 0
        ):

            return [
                TileBox(
                    0,
                    0,
                    width,
                    height,
                )
            ]

        rng = random.Random(
            self._build_seed(
                sample_key=
                    sample_key,

                sampling_token=
                    sampling_token,
            )
        )

        coordinates = []

        # Always include center tile.
        center_left = (
            max_left
            // 2
        )

        center_top = (
            max_top
            // 2
        )

        coordinates.append(
            TileBox(
                center_left,
                center_top,
                center_left
                + tile_width,
                center_top
                + tile_height,
            )
        )

        seen = set(
            coordinates
        )

        attempts = 0

        maximum_attempts = (
            self.max_tiles
            * 20
        )

        while (
            len(
                coordinates
            )
            < self.max_tiles
            and attempts
            < maximum_attempts
        ):

            attempts += 1

            left = (
                rng.randint(
                    0,
                    max_left,
                )
                if max_left > 0
                else 0
            )

            top = (
                rng.randint(
                    0,
                    max_top,
                )
                if max_top > 0
                else 0
            )

            box = TileBox(
                left,
                top,
                left
                + tile_width,
                top
                + tile_height,
            )

            if box in seen:
                continue

            seen.add(
                box
            )

            coordinates.append(
                box
            )

        return coordinates

    def sample(
        self,
        image: Image.Image,
        sample_key: str,
        sampling_token: int = 0,
    ) -> list[
        TileBox
    ]:

        if self.mode == "grid":

            return (
                self._grid_coordinates(
                    image
                )
            )

        return (
            self._random_coordinates(
                image=
                    image,

                sample_key=
                    sample_key,

                sampling_token=
                    sampling_token,
            )
        )


def crop_tiles(
    image: Image.Image,
    boxes: list[
        TileBox
    ],
) -> list[
    Image.Image
]:

    return [
        image.crop(
            box.as_tuple()
        )
        for box in boxes
    ]