from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset

from src.data.image_utils import (
    load_rgb_image,
)


class AIGCImageDataset(
    Dataset
):

    def __init__(
        self,
        manifest_path: str | Path,
        transform: Callable | None = None,
        return_metadata: bool = True,
    ):

        self.manifest_path = Path(
            manifest_path
        )

        if not self.manifest_path.exists():

            raise FileNotFoundError(
                f"Manifest does not exist: "
                f"{self.manifest_path}"
            )

        self.dataframe = (
            pd.read_csv(
                self.manifest_path
            )
        )

        required_columns = {
            "image_path",
            "label",
            "class_name",
            "dataset",
            "source",
            "generator",
            "original_split",
            "split",
            "content_hash",
        }

        missing_columns = (
            required_columns
            - set(
                self.dataframe.columns
            )
        )

        if missing_columns:

            raise ValueError(
                "Manifest is missing "
                f"required columns: "
                f"{sorted(missing_columns)}"
            )

        self.transform = (
            transform
        )

        self.return_metadata = (
            return_metadata
        )

    def __len__(
        self,
    ) -> int:

        return len(
            self.dataframe
        )

    def _resolve_image_path(
        self,
        image_path: str,
    ) -> Path:

        path = Path(
            image_path
        )

        if path.is_absolute():
            return path

        return (
            Path.cwd()
            / path
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:

        row = (
            self.dataframe
            .iloc[
                index
            ]
        )

        image_path = (
            self._resolve_image_path(
                str(
                    row[
                        "image_path"
                    ]
                )
            )
        )

        image: Image.Image = (
            load_rgb_image(
                image_path
            )
        )

        if self.transform is not None:

            image = (
                self.transform(
                    image
                )
            )

        sample: dict[str, Any] = {
            "image":
                image,

            "label":
                torch.tensor(
                    float(
                        row[
                            "label"
                        ]
                    ),
                    dtype=
                        torch.float32,
                ),
        }

        if self.return_metadata:

            sample.update(
                {
                    "image_path":
                        str(
                            image_path
                        ),

                    "class_name":
                        str(
                            row[
                                "class_name"
                            ]
                        ),

                    "dataset":
                        str(
                            row[
                                "dataset"
                            ]
                        ),

                    "source":
                        str(
                            row[
                                "source"
                            ]
                        ),

                    "generator":
                        str(
                            row[
                                "generator"
                            ]
                        ),

                    "original_split":
                        str(
                            row[
                                "original_split"
                            ]
                        ),

                    "split":
                        str(
                            row[
                                "split"
                            ]
                        ),

                    "content_hash":
                        str(
                            row[
                                "content_hash"
                            ]
                        ),
                }
            )

        return sample