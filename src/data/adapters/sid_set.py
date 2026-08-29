from __future__ import annotations

import hashlib
import io
import re
import shutil

from pathlib import Path
from typing import Iterable

import pandas as pd

from PIL import Image

from datasets import (
    Image as HFImage,
    load_dataset,
)


DATASET_NAME = "sid_set"


SID_LABELS = {
    0: {
        "label": 0,
        "class_name": "real",
        "scope": "real",
        "source": "openimages_v7",
        "generator": "none",
    },

    1: {
        "label": 1,
        "class_name": "fake",
        "scope": "full_synthetic",
        "source": "sid_set",
        "generator": "flux",
    },

    2: {
        "label": 1,
        "class_name": "fake",
        "scope": "tampered",
        "source": "sid_set",
        "generator": "latent_diffusion_tampering",
    },
}


FORMAT_TO_EXTENSION = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
    "BMP": ".bmp",
    "TIFF": ".tif",
}


def normalize_manifest_path(
    path: Path,
) -> str:

    path = path.resolve()

    project_root = (
        Path.cwd()
        .resolve()
    )

    try:

        return (
            path
            .relative_to(
                project_root
            )
            .as_posix()
        )

    except ValueError:

        return path.as_posix()


def safe_filename(
    value: str,
) -> str:

    value = str(
        value
    ).strip()

    value = re.sub(
        r"[^a-zA-Z0-9_.-]+",
        "_",
        value,
    )

    value = value.strip(
        "._"
    )

    if not value:

        value = "sid_image"

    return value


def sha256_bytes(
    raw_bytes: bytes,
) -> str:

    return (
        hashlib.sha256(
            raw_bytes
        )
        .hexdigest()
    )


def inspect_image_bytes(
    raw_bytes: bytes,
) -> tuple[
    int,
    int,
    str,
]:

    with Image.open(
        io.BytesIO(
            raw_bytes
        )
    ) as image:

        width, height = (
            image.size
        )

        image_format = (
            image.format
            or "PNG"
        ).upper()

        image.verify()

    return (
        width,
        height,
        image_format,
    )


def infer_extension(
    raw_bytes: bytes,
    path_hint: str | None = None,
) -> str:

    if path_hint:

        suffix = (
            Path(
                path_hint
            )
            .suffix
            .lower()
        )

        if suffix in {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".bmp",
            ".tif",
            ".tiff",
        }:

            if suffix == ".jpeg":
                return ".jpg"

            if suffix == ".tiff":
                return ".tif"

            return suffix

    _, _, image_format = (
        inspect_image_bytes(
            raw_bytes
        )
    )

    return (
        FORMAT_TO_EXTENSION.get(
            image_format,
            ".bin",
        )
    )


def extract_image_bytes(
    image_payload,
) -> tuple[
    bytes,
    str | None,
]:
    """
    Hugging Face Image(decode=False) generally returns:

        {
            "bytes": ...,
            "path": ...
        }

    Depending on how the dataset is stored, either bytes
    or path may be populated.

    We always retrieve the ORIGINAL encoded bytes.

    We do not decode and re-encode the image.
    """

    if isinstance(
        image_payload,
        bytes,
    ):

        return (
            image_payload,
            None,
        )

    if not isinstance(
        image_payload,
        dict,
    ):

        raise TypeError(
            "Unexpected SID image payload "
            f"type: {type(image_payload)}"
        )

    raw_bytes = (
        image_payload.get(
            "bytes"
        )
    )

    path_hint = (
        image_payload.get(
            "path"
        )
    )

    if raw_bytes is not None:

        return (
            raw_bytes,
            path_hint,
        )

    if path_hint:

        source_path = Path(
            path_hint
        )

        if source_path.exists():

            return (
                source_path.read_bytes(),
                path_hint,
            )

    raise RuntimeError(
        "SID image contains neither "
        "accessible bytes nor an accessible path."
    )


def find_local_parquet_files(
    local_root: str | Path,
    split: str,
) -> list[str]:

    local_root = Path(
        local_root
    )

    data_directory = (
        local_root
        / "data"
    )

    if not data_directory.exists():

        raise FileNotFoundError(
            "SID local data directory "
            f"does not exist: "
            f"{data_directory}"
        )

    files = sorted(
        data_directory.glob(
            f"{split}-*.parquet"
        )
    )

    if not files:

        raise FileNotFoundError(
            f"No SID {split} parquet "
            f"shards found under "
            f"{data_directory}"
        )

    return [
        str(
            path
        )
        for path in files
    ]


def load_sid_split(
    split: str,
    repo_id: str = "saberzl/SID_Set",
    local_root: str | Path | None = None,
    streaming: bool = True,
):
    """
    Load SID from either:

    1. the Hugging Face Hub, or
    2. a locally downloaded HF snapshot.

    The image column is explicitly made non-decoding so
    original image bytes can be preserved.
    """

    if (
        local_root is not None
        and Path(
            local_root
        ).exists()
        and (
            Path(
                local_root
            )
            / "data"
        ).exists()
    ):

        files = (
            find_local_parquet_files(
                local_root=
                    local_root,

                split=
                    split,
            )
        )

        print(
            f"\nLoading SID {split} "
            f"from local parquet shards."
        )

        print(
            f"Shards: {len(files)}"
        )

        dataset = load_dataset(
            "parquet",

            data_files={
                split:
                    files,
            },

            split=
                split,

            streaming=
                streaming,
        )

    else:

        print(
            f"\nLoading SID {split} "
            f"from Hugging Face:"
        )

        print(
            repo_id
        )

        dataset = load_dataset(
            repo_id,

            split=
                split,

            streaming=
                streaming,
        )

    # Avoid image decoding/re-encoding.
    dataset = dataset.cast_column(
        "image",
        HFImage(
            decode=False
        ),
    )

    return dataset


def prepare_iteration_order(
    dataset,
    seed: int,
    streaming: bool,
    shuffle_buffer_size: int,
):
    """
    Apply deterministic shuffling.

    For streaming datasets Hugging Face uses a finite
    shuffle buffer, which gives us good approximate
    random sampling without materializing the entire
    100+ GB dataset first.
    """

    if streaming:

        return dataset.shuffle(
            seed=
                seed,

            buffer_size=
                shuffle_buffer_size,
        )

    return dataset.shuffle(
        seed=
            seed
    )


def quota_complete(
    counts: dict[int, int],
    maximum_per_class: int | None,
    labels: set[int],
) -> bool:

    if maximum_per_class is None:

        return False

    return all(
        counts.get(
            label,
            0,
        )
        >= maximum_per_class

        for label in labels
    )


def write_original_image(
    raw_bytes: bytes,
    destination: Path,
) -> None:

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists():

        existing_hash = (
            hashlib.sha256(
                destination.read_bytes()
            )
            .hexdigest()
        )

        incoming_hash = (
            sha256_bytes(
                raw_bytes
            )
        )

        if (
            existing_hash
            == incoming_hash
        ):

            return

    temporary_path = (
        destination.parent
        / (
            destination.name
            + ".tmp"
        )
    )

    with temporary_path.open(
        "wb"
    ) as file:

        file.write(
            raw_bytes
        )

    shutil.move(
        str(
            temporary_path
        ),
        str(
            destination
        ),
    )


def materialize_sid_split(
    dataset,
    split: str,
    materialized_root: str | Path,
    maximum_per_class: int | None,
    include_tampered_auxiliary: bool = False,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    materialized_root = Path(
        materialized_root
    )

    primary_labels = {
        0,
        1,
    }

    counts = {
        0: 0,
        1: 0,
        2: 0,
    }

    primary_records = []

    tampered_records = []

    processed_rows = 0
    failed_rows = 0

    print(
        "\n========================================"
    )

    print(
        f"MATERIALIZING SID {split.upper()}"
    )

    print(
        "========================================"
    )

    for row in dataset:

        processed_rows += 1

        try:

            original_label = int(
                row[
                    "label"
                ]
            )

        except Exception:

            failed_rows += 1
            continue

        if (
            original_label
            not in SID_LABELS
        ):

            failed_rows += 1
            continue

        # ------------------------------------------
        # Main binary task only uses labels 0 and 1.
        # ------------------------------------------

        if original_label == 2:

            if not include_tampered_auxiliary:
                continue

        elif (
            maximum_per_class
            is not None
            and counts[
                original_label
            ]
            >= maximum_per_class
        ):

            if quota_complete(
                counts=
                    counts,

                maximum_per_class=
                    maximum_per_class,

                labels=
                    primary_labels,
            ):

                break

            continue

        try:

            raw_bytes, path_hint = (
                extract_image_bytes(
                    row[
                        "image"
                    ]
                )
            )

            width, height, _ = (
                inspect_image_bytes(
                    raw_bytes
                )
            )

            extension = infer_extension(
                raw_bytes=
                    raw_bytes,

                path_hint=
                    path_hint,
            )

        except Exception as error:

            failed_rows += 1

            print(
                "\n[WARNING] "
                f"Failed SID row "
                f"{processed_rows}: "
                f"{error}"
            )

            continue

        img_id = safe_filename(
            row.get(
                "img_id",
                f"sid_{processed_rows:09d}",
            )
        )

        metadata = (
            SID_LABELS[
                original_label
            ]
        )

        if original_label == 0:

            output_class = (
                "real"
            )

        elif original_label == 1:

            output_class = (
                "full_synthetic"
            )

        else:

            output_class = (
                "tampered"
            )

        destination = (
            materialized_root
            / split
            / output_class
            / (
                img_id
                + extension
            )
        )

        write_original_image(
            raw_bytes=
                raw_bytes,

            destination=
                destination,
        )

        content_hash = (
            sha256_bytes(
                raw_bytes
            )
        )

        # Prefer official dimensions but confirm against
        # the actual encoded image.
        official_width = row.get(
            "width"
        )

        official_height = row.get(
            "height"
        )

        record = {
            "image_path":
                normalize_manifest_path(
                    destination
                ),

            "label":
                metadata[
                    "label"
                ],

            "class_name":
                metadata[
                    "class_name"
                ],

            "dataset":
                DATASET_NAME,

            "source":
                metadata[
                    "source"
                ],

            "generator":
                metadata[
                    "generator"
                ],

            "scope":
                metadata[
                    "scope"
                ],

            "original_label":
                original_label,

            "img_id":
                img_id,

            "original_split":
                split,

            "split":
                (
                    "val"
                    if split
                    == "validation"
                    else split
                ),

            "width":
                width,

            "height":
                height,

            "official_width":
                (
                    int(
                        official_width
                    )
                    if official_width
                    is not None
                    else width
                ),

            "official_height":
                (
                    int(
                        official_height
                    )
                    if official_height
                    is not None
                    else height
                ),

            "content_hash":
                content_hash,
        }

        if original_label == 2:

            tampered_records.append(
                record
            )

        else:

            primary_records.append(
                record
            )

            counts[
                original_label
            ] += 1

        if (
            len(
                primary_records
            )
            % 1000
            == 0
            and primary_records
        ):

            print(
                f"\rPrimary images saved: "
                f"{len(primary_records):,} "
                f"| Real: "
                f"{counts[0]:,} "
                f"| Synthetic: "
                f"{counts[1]:,}",
                end="",
                flush=True,
            )

        if quota_complete(
            counts=
                counts,

            maximum_per_class=
                maximum_per_class,

            labels=
                primary_labels,
        ):

            break

    print()

    primary_dataframe = (
        pd.DataFrame(
            primary_records
        )
    )

    tampered_dataframe = (
        pd.DataFrame(
            tampered_records
        )
    )

    print(
        f"\nProcessed source rows: "
        f"{processed_rows:,}"
    )

    print(
        f"Primary real:          "
        f"{counts[0]:,}"
    )

    print(
        f"Primary synthetic:     "
        f"{counts[1]:,}"
    )

    print(
        f"Aux tampered:          "
        f"{len(tampered_records):,}"
    )

    print(
        f"Failed rows:           "
        f"{failed_rows:,}"
    )

    return (
        primary_dataframe,
        tampered_dataframe,
    )


def check_duplicates(
    dataframe: pd.DataFrame,
    name: str,
) -> None:

    if dataframe.empty:

        return

    duplicated = (
        dataframe[
            "content_hash"
        ]
        .duplicated(
            keep=False
        )
    )

    duplicate_count = int(
        duplicated.sum()
    )

    if duplicate_count == 0:

        print(
            f"\nNo exact duplicates "
            f"inside {name}."
        )

        return

    print(
        f"\n[WARNING] "
        f"{duplicate_count:,} rows "
        f"inside {name} participate "
        f"in duplicate groups."
    )


def check_cross_split_leakage(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    output_directory: Path,
) -> None:

    train_hashes = set(
        train_dataframe[
            "content_hash"
        ]
    )

    val_hashes = set(
        validation_dataframe[
            "content_hash"
        ]
    )

    overlaps = (
        train_hashes
        & val_hashes
    )

    if not overlaps:

        print(
            "\nNo exact SID train/"
            "validation leakage detected."
        )

        return

    print(
        "\n[WARNING] "
        f"SID train/validation share "
        f"{len(overlaps):,} exact image hashes."
    )

    combined = pd.concat(
        [
            train_dataframe,
            validation_dataframe,
        ],
        ignore_index=True,
    )

    leakage = combined[
        combined[
            "content_hash"
        ].isin(
            overlaps
        )
    ]

    leakage.to_csv(
        output_directory
        / "sid_cross_split_duplicates.csv",

        index=False,
    )


def build_sid_manifests(
    repo_id: str,
    local_root: str | Path | None,
    materialized_root: str | Path,
    output_directory: str | Path,
    train_split: str = "train",
    validation_split: str = "validation",
    streaming: bool = True,
    shuffle_buffer_size: int = 10000,
    train_max_per_class: int | None = 10000,
    validation_max_per_class: int | None = 5000,
    include_tampered_auxiliary: bool = False,
    seed: int = 42,
    requested_splits: set[str] | None = None,
) -> None:

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if requested_splits is None:

        requested_splits = {
            "train",
            "validation",
        }

    dataframes = {}

    auxiliary_frames = {}

    split_configs = [
        (
            "train",
            train_split,
            train_max_per_class,
            seed,
        ),

        (
            "validation",
            validation_split,
            validation_max_per_class,
            seed + 1,
        ),
    ]

    for (
        canonical_name,
        source_split,
        maximum_per_class,
        split_seed,
    ) in split_configs:

        if (
            canonical_name
            not in requested_splits
        ):

            continue

        dataset = load_sid_split(
            split=
                source_split,

            repo_id=
                repo_id,

            local_root=
                local_root,

            streaming=
                streaming,
        )

        dataset = prepare_iteration_order(
            dataset=
                dataset,

            seed=
                split_seed,

            streaming=
                streaming,

            shuffle_buffer_size=
                shuffle_buffer_size,
        )

        (
            primary_dataframe,
            tampered_dataframe,
        ) = materialize_sid_split(
            dataset=
                dataset,

            split=
                source_split,

            materialized_root=
                materialized_root,

            maximum_per_class=
                maximum_per_class,

            include_tampered_auxiliary=
                include_tampered_auxiliary,
        )

        dataframes[
            canonical_name
        ] = primary_dataframe

        auxiliary_frames[
            canonical_name
        ] = tampered_dataframe

        output_name = (
            "sid_train.csv"
            if canonical_name
            == "train"

            else "sid_val.csv"
        )

        primary_dataframe.to_csv(
            output_directory
            / output_name,

            index=False,
        )

        if (
            include_tampered_auxiliary
            and not tampered_dataframe.empty
        ):

            tampered_dataframe.to_csv(
                output_directory
                / (
                    f"sid_tampered_"
                    f"{canonical_name}.csv"
                ),

                index=False,
            )

        check_duplicates(
            dataframe=
                primary_dataframe,

            name=
                f"SID {canonical_name}",
        )

    if (
        "train" in dataframes
        and "validation" in dataframes
    ):

        check_cross_split_leakage(
            train_dataframe=
                dataframes[
                    "train"
                ],

            validation_dataframe=
                dataframes[
                    "validation"
                ],

            output_directory=
                output_directory,
        )

        combined = pd.concat(
            [
                dataframes[
                    "train"
                ],

                dataframes[
                    "validation"
                ],
            ],

            ignore_index=True,
        )

        combined.to_csv(
            output_directory
            / "sid_all.csv",

            index=False,
        )

    print(
        "\n========================================"
    )

    print(
        "SID MANIFEST BUILD COMPLETE"
    )

    print(
        "========================================"
    )