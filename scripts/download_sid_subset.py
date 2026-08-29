from __future__ import annotations

import argparse
import hashlib
import io
import re

from pathlib import Path

import pandas as pd

from PIL import Image

from datasets import (
    Image as HFImage,
    load_dataset,
)

from tqdm import tqdm


# ============================================================
# CONSTANTS
# ============================================================

REPO_ID = (
    "saberzl/SID_Set"
)


LABEL_INFO = {

    # SID:
    #
    # 0 = real
    # 1 = full synthetic
    # 2 = tampered

    0: {
        "binary_label": 0,
        "class_name": "real",
        "folder": "real",
        "scope": "real",
        "source": "openimages_v7",
        "generator": "none",
    },

    1: {
        "binary_label": 1,
        "class_name": "fake",
        "folder": "full_synthetic",
        "scope": "full_synthetic",
        "source": "sid_set",
        "generator": "flux",
    },
}


FORMAT_EXTENSION = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
    "BMP": ".bmp",
    "TIFF": ".tif",
}


# ============================================================
# PATH HELPERS
# ============================================================

def normalize_manifest_path(
    path: Path,
) -> str:

    path = path.resolve()

    project_root = (
        Path.cwd()
        .resolve()
    )

    try:

        relative = path.relative_to(
            project_root
        )

        return (
            relative.as_posix()
        )

    except ValueError:

        return (
            path.as_posix()
        )


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

        return "sid_image"

    return value


# ============================================================
# HASHING
# ============================================================

def sha256_bytes(
    raw_bytes: bytes,
) -> str:

    return (
        hashlib.sha256(
            raw_bytes
        )
        .hexdigest()
    )


# ============================================================
# IMAGE BYTE HANDLING
# ============================================================

def extract_original_bytes(
    image_payload,
) -> bytes:
    """
    SID is stored as Parquet.

    When image decoding is disabled, Hugging Face returns
    something similar to:

        {
            "bytes": b"...",
            "path": "..."
        }

    We save those original bytes directly.

    We DO NOT do:

        PIL.Image.open(...)
        image.save(...)

    because re-encoding could alter forensic evidence.
    """

    if isinstance(
        image_payload,
        bytes,
    ):

        return image_payload

    if not isinstance(
        image_payload,
        dict,
    ):

        raise TypeError(
            "Unexpected image payload "
            f"type: {type(image_payload)}"
        )

    raw_bytes = (
        image_payload.get(
            "bytes"
        )
    )

    if raw_bytes is not None:

        return raw_bytes

    path_value = (
        image_payload.get(
            "path"
        )
    )

    if path_value:

        path = Path(
            path_value
        )

        if path.exists():

            return (
                path.read_bytes()
            )

    raise RuntimeError(
        "Image payload contains neither "
        "usable bytes nor a local path."
    )


def inspect_image(
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


def extension_from_format(
    image_format: str,
) -> str:

    return (
        FORMAT_EXTENSION.get(
            image_format.upper(),
            ".img",
        )
    )


# ============================================================
# FILE WRITING
# ============================================================

def write_bytes_atomic(
    raw_bytes: bytes,
    destination: Path,
) -> None:

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Already downloaded.
    if destination.exists():

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

    temporary_path.replace(
        destination
    )


# ============================================================
# HUGGING FACE STREAM
# ============================================================

def load_label_stream(
    split: str,
    label: int,
    seed: int,
    shuffle_buffer_size: int,
):
    """
    Load only one SID label.

    Using filters with streaming=True is particularly useful
    because SID uses Parquet, allowing us to avoid materializing
    the complete 140 GB dataset.
    """

    print(
        "\n----------------------------------------"
    )

    print(
        f"Opening SID stream"
    )

    print(
        f"Split: {split}"
    )

    print(
        f"Label: {label}"
    )

    print(
        "----------------------------------------"
    )

    dataset = load_dataset(
        REPO_ID,

        split=
            split,

        streaming=
            True,

        filters=[
            (
                "label",
                "=",
                label,
            )
        ],
    )

    # --------------------------------------------------------
    # Disable image decoding.
    #
    # This lets us access the underlying encoded image bytes.
    # --------------------------------------------------------

    try:

        dataset = dataset.cast_column(
            "image",
            HFImage(
                decode=False
            ),
        )

    except Exception:

        # Newer datasets versions also expose decode(False)
        # for streaming IterableDataset.
        dataset = dataset.decode(
            False
        )

    # We don't need masks at all for the primary
    # real-vs-full-synthetic detector.
    try:

        dataset = dataset.select_columns(
            [
                "img_id",
                "image",
                "width",
                "height",
                "label",
            ]
        )

    except Exception:

        # select_columns support can differ between
        # datasets library versions.
        pass

    # Approximate deterministic streaming shuffle.
    dataset = dataset.shuffle(
        seed=
            seed,

        buffer_size=
            shuffle_buffer_size,
    )

    return dataset


# ============================================================
# DOWNLOAD ONE LABEL
# ============================================================

def download_label_subset(
    split: str,
    label: int,
    count: int,
    output_root: Path,
    seed: int,
    shuffle_buffer_size: int,
) -> pd.DataFrame:

    info = (
        LABEL_INFO[
            label
        ]
    )

    dataset = load_label_stream(
        split=
            split,

        label=
            label,

        seed=
            seed,

        shuffle_buffer_size=
            shuffle_buffer_size,
    )

    output_directory = (
        output_root
        / split
        / info[
            "folder"
        ]
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    iterator = iter(
        dataset
    )

    progress = tqdm(
        total=count,

        desc=(
            f"{split} "
            f"{info['folder']}"
        ),
    )

    source_rows_seen = 0

    while (
        len(
            records
        )
        < count
    ):

        try:

            row = next(
                iterator
            )

        except StopIteration:

            print(
                "\n[WARNING] SID stream ended "
                "before the requested number "
                "of images was collected."
            )

            break

        source_rows_seen += 1

        try:

            original_label = int(
                row[
                    "label"
                ]
            )

            if (
                original_label
                != label
            ):

                continue

            raw_bytes = (
                extract_original_bytes(
                    row[
                        "image"
                    ]
                )
            )

            (
                actual_width,
                actual_height,
                image_format,
            ) = inspect_image(
                raw_bytes
            )

            content_hash = (
                sha256_bytes(
                    raw_bytes
                )
            )

            img_id = safe_filename(
                row.get(
                    "img_id",
                    (
                        f"sid_"
                        f"{split}_"
                        f"{label}_"
                        f"{source_rows_seen}"
                    ),
                )
            )

            extension = (
                extension_from_format(
                    image_format
                )
            )

            # Hash suffix prevents accidental filename
            # collisions while keeping img_id readable.
            filename = (
                f"{img_id}_"
                f"{content_hash[:12]}"
                f"{extension}"
            )

            destination = (
                output_directory
                / filename
            )

            write_bytes_atomic(
                raw_bytes=
                    raw_bytes,

                destination=
                    destination,
            )

            official_width = (
                row.get(
                    "width"
                )
            )

            official_height = (
                row.get(
                    "height"
                )
            )

            manifest_split = (
                "val"
                if split
                == "validation"

                else "train"
            )

            records.append(
                {
                    "image_path":
                        normalize_manifest_path(
                            destination
                        ),

                    "label":
                        info[
                            "binary_label"
                        ],

                    "class_name":
                        info[
                            "class_name"
                        ],

                    "dataset":
                        "sid_set",

                    "source":
                        info[
                            "source"
                        ],

                    "generator":
                        info[
                            "generator"
                        ],

                    "scope":
                        info[
                            "scope"
                        ],

                    "original_label":
                        original_label,

                    "img_id":
                        img_id,

                    "original_split":
                        split,

                    "split":
                        manifest_split,

                    "width":
                        actual_width,

                    "height":
                        actual_height,

                    "official_width":
                        (
                            int(
                                official_width
                            )
                            if official_width
                            is not None
                            else actual_width
                        ),

                    "official_height":
                        (
                            int(
                                official_height
                            )
                            if official_height
                            is not None
                            else actual_height
                        ),

                    "image_format":
                        image_format,

                    "content_hash":
                        content_hash,
                }
            )

            progress.update(
                1
            )

        except Exception as error:

            print(
                "\n[WARNING] "
                f"Skipping SID row: "
                f"{error}"
            )

            continue

    progress.close()

    dataframe = pd.DataFrame(
        records
    )

    print(
        f"\nCollected "
        f"{len(dataframe):,}/"
        f"{count:,} images."
    )

    return dataframe


# ============================================================
# SPLIT DOWNLOAD
# ============================================================

def download_split(
    split: str,
    per_class: int,
    output_root: Path,
    seed: int,
    shuffle_buffer_size: int,
) -> pd.DataFrame:

    print(
        "\n"
        "========================================"
    )

    print(
        f"SID {split.upper()}"
    )

    print(
        "========================================"
    )

    real_dataframe = (
        download_label_subset(
            split=
                split,

            label=
                0,

            count=
                per_class,

            output_root=
                output_root,

            seed=
                seed,

            shuffle_buffer_size=
                shuffle_buffer_size,
        )
    )

    fake_dataframe = (
        download_label_subset(
            split=
                split,

            label=
                1,

            count=
                per_class,

            output_root=
                output_root,

            seed=
                seed + 100,

            shuffle_buffer_size=
                shuffle_buffer_size,
        )
    )

    dataframe = pd.concat(
        [
            real_dataframe,
            fake_dataframe,
        ],

        ignore_index=True,
    )

    # Shuffle resulting manifest so classes aren't
    # stored sequentially.
    dataframe = (
        dataframe.sample(
            frac=1.0,

            random_state=
                seed,
        )
        .reset_index(
            drop=True
        )
    )

    return dataframe


# ============================================================
# DUPLICATE CHECKS
# ============================================================

def check_internal_duplicates(
    dataframe: pd.DataFrame,
    split_name: str,
) -> None:

    duplicated = (
        dataframe[
            "content_hash"
        ]
        .duplicated(
            keep=False
        )
    )

    count = int(
        duplicated.sum()
    )

    if count == 0:

        print(
            f"\nNo exact duplicates "
            f"inside SID {split_name}."
        )

    else:

        print(
            f"\n[WARNING] "
            f"{count:,} rows inside "
            f"SID {split_name} belong "
            f"to duplicate-image groups."
        )


def check_cross_split_duplicates(
    train_dataframe: pd.DataFrame,
    val_dataframe: pd.DataFrame,
) -> None:

    train_hashes = set(
        train_dataframe[
            "content_hash"
        ]
    )

    val_hashes = set(
        val_dataframe[
            "content_hash"
        ]
    )

    overlap = (
        train_hashes
        & val_hashes
    )

    print(
        f"\nSID train ↔ validation "
        f"exact overlap: "
        f"{len(overlap):,}"
    )

    if overlap:

        raise RuntimeError(
            "SID train/validation leakage "
            "was detected."
        )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    dataframe: pd.DataFrame,
    title: str,
) -> None:

    print(
        "\n"
        "========================================"
    )

    print(
        title
    )

    print(
        "========================================"
    )

    print(
        f"\nImages: "
        f"{len(dataframe):,}"
    )

    print(
        "\nClasses:"
    )

    print(
        dataframe[
            "class_name"
        ]
        .value_counts()
    )

    print(
        "\nGenerators:"
    )

    print(
        dataframe[
            "generator"
        ]
        .value_counts()
    )

    print(
        "\nResolution:"
    )

    print(
        dataframe[
            [
                "width",
                "height",
            ]
        ].describe()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Stream a balanced subset of SID_Set "
            "from Hugging Face without downloading "
            "the full ~140 GB dataset."
        )
    )

    parser.add_argument(
        "--output-root",

        type=str,

        default=(
            "data/processed/"
            "sid_set"
        ),
    )

    parser.add_argument(
        "--manifest-dir",

        type=str,

        default=(
            "data/manifests"
        ),
    )

    parser.add_argument(
        "--train-per-class",

        type=int,

        default=8000,
    )

    parser.add_argument(
        "--val-per-class",

        type=int,

        default=2000,
    )

    parser.add_argument(
        "--splits",

        type=str,

        default="train,validation",

        help=(
            "train, validation, "
            "or train,validation"
        ),
    )

    parser.add_argument(
        "--seed",

        type=int,

        default=42,
    )

    parser.add_argument(
        "--shuffle-buffer",

        type=int,

        default=10000,
    )

    args = parser.parse_args()

    output_root = Path(
        args.output_root
    )

    manifest_directory = Path(
        args.manifest_dir
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    requested_splits = {
        split.strip()
        for split
        in args.splits.split(
            ","
        )
        if split.strip()
    }

    allowed_splits = {
        "train",
        "validation",
    }

    invalid = (
        requested_splits
        - allowed_splits
    )

    if invalid:

        raise ValueError(
            "Invalid SID splits: "
            f"{sorted(invalid)}"
        )

    print(
        "\n"
        "========================================"
    )

    print(
        "SID_SET SUBSET DOWNLOAD"
    )

    print(
        "========================================"
    )

    print(
        f"\nRepository:"
        f"\n{REPO_ID}"
    )

    print(
        f"\nOutput:"
        f"\n{output_root.resolve()}"
    )

    print(
        f"\nTrain per class: "
        f"{args.train_per_class:,}"
    )

    print(
        f"Validation per class: "
        f"{args.val_per_class:,}"
    )

    print(
        "\nTampered label 2:"
        "\nEXCLUDED"
    )

    print(
        "\nOriginal image encoding:"
        "\nPRESERVED"
    )

    print(
        "\nFull 140 GB dataset:"
        "\nNOT DOWNLOADED"
    )

    train_dataframe = None
    val_dataframe = None

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    if (
        "train"
        in requested_splits
    ):

        train_dataframe = (
            download_split(
                split=
                    "train",

                per_class=
                    args.train_per_class,

                output_root=
                    output_root,

                seed=
                    args.seed,

                shuffle_buffer_size=
                    args.shuffle_buffer,
            )
        )

        check_internal_duplicates(
            dataframe=
                train_dataframe,

            split_name=
                "train",
        )

        train_path = (
            manifest_directory
            / "sid_train.csv"
        )

        train_dataframe.to_csv(
            train_path,
            index=False,
        )

        print_summary(
            train_dataframe,
            "SID TRAIN SUMMARY",
        )

        print(
            f"\nSaved manifest:"
            f"\n{train_path.resolve()}"
        )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if (
        "validation"
        in requested_splits
    ):

        val_dataframe = (
            download_split(
                split=
                    "validation",

                per_class=
                    args.val_per_class,

                output_root=
                    output_root,

                seed=
                    args.seed + 1000,

                shuffle_buffer_size=
                    args.shuffle_buffer,
            )
        )

        check_internal_duplicates(
            dataframe=
                val_dataframe,

            split_name=
                "validation",
        )

        val_path = (
            manifest_directory
            / "sid_val.csv"
        )

        val_dataframe.to_csv(
            val_path,
            index=False,
        )

        print_summary(
            val_dataframe,
            "SID VALIDATION SUMMARY",
        )

        print(
            f"\nSaved manifest:"
            f"\n{val_path.resolve()}"
        )

    # --------------------------------------------------------
    # COMBINED
    # --------------------------------------------------------

    if (
        train_dataframe
        is not None

        and val_dataframe
        is not None
    ):

        check_cross_split_duplicates(
            train_dataframe=
                train_dataframe,

            val_dataframe=
                val_dataframe,
        )

        combined = pd.concat(
            [
                train_dataframe,
                val_dataframe,
            ],

            ignore_index=True,
        )

        all_path = (
            manifest_directory
            / "sid_all.csv"
        )

        combined.to_csv(
            all_path,
            index=False,
        )

        print(
            f"\nSaved combined manifest:"
            f"\n{all_path.resolve()}"
        )

    print(
        "\n"
        "========================================"
    )

    print(
        "SID SUBSET DOWNLOAD COMPLETE"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()