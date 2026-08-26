from __future__ import annotations

import hashlib
import random

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.image_utils import (
    DEFAULT_IMAGE_EXTENSIONS,
    is_image_file,
    validate_image,
)


@dataclass
class ManifestEntry:
    image_path: str
    label: int
    class_name: str
    source: str
    file_id: str


def build_file_id(
    path: Path,
) -> str:
    """
    Generate a stable identifier from the relative path.

    We deliberately do not hash the image bytes here because
    that would substantially slow down manifest generation
    for large datasets.
    """

    normalized = str(path).replace("\\", "/")

    return hashlib.sha1(
        normalized.encode("utf-8")
    ).hexdigest()


def infer_source(
    image_path: Path,
    class_root: Path,
) -> str:
    """
    Infer dataset/generator source from directory structure.

    Example:

        data/raw/fake/stable_diffusion/img001.png

    returns:

        stable_diffusion

    If the image sits directly inside fake/, the source will
    be 'unknown'.
    """

    try:
        relative = image_path.relative_to(class_root)

    except ValueError:
        return "unknown"

    parts = relative.parts

    if len(parts) <= 1:
        return "unknown"

    return parts[0]


def scan_class_directory(
    directory: str | Path,
    label: int,
    class_name: str,
    extensions: set[str] | None = None,
    verify_images: bool = True,
) -> list[ManifestEntry]:

    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {directory}"
        )

    valid_extensions = (
        extensions
        if extensions is not None
        else DEFAULT_IMAGE_EXTENSIONS
    )

    entries: list[ManifestEntry] = []

    all_paths = sorted(
        path
        for path in directory.rglob("*")
        if is_image_file(
            path,
            extensions=valid_extensions,
        )
    )

    print(
        f"Scanning {class_name}: "
        f"{len(all_paths)} candidate files"
    )

    skipped = 0

    for path in all_paths:

        if verify_images and not validate_image(path):
            skipped += 1
            print(
                f"[WARNING] Skipping unreadable image: {path}"
            )
            continue

        source = infer_source(
            image_path=path,
            class_root=directory,
        )

        entries.append(
            ManifestEntry(
                image_path=str(path.resolve()),
                label=label,
                class_name=class_name,
                source=source,
                file_id=build_file_id(path.resolve()),
            )
        )

    if skipped:
        print(
            f"Skipped {skipped} corrupted/unreadable "
            f"{class_name} images."
        )

    return entries


def entries_to_dataframe(
    entries: list[ManifestEntry],
) -> pd.DataFrame:

    rows = [
        {
            "image_path": entry.image_path,
            "label": entry.label,
            "class_name": entry.class_name,
            "source": entry.source,
            "file_id": entry.file_id,
        }
        for entry in entries
    ]

    return pd.DataFrame(rows)


def stratified_split(
    dataframe: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Split while preserving label/source composition as much
    as possible.

    We stratify manually over (label, source), which matters
    because fake-generator imbalance can otherwise leak into
    train/validation composition.
    """

    ratio_sum = (
        train_ratio
        + val_ratio
        + test_ratio
    )

    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(
            "train_ratio + val_ratio + test_ratio "
            "must equal 1.0"
        )

    rng = random.Random(seed)

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    grouped = dataframe.groupby(
        ["label", "source"],
        dropna=False,
    )

    for _, group in grouped:

        indices = group.index.tolist()

        rng.shuffle(indices)

        n = len(indices)

        n_train = int(
            round(n * train_ratio)
        )

        n_val = int(
            round(n * val_ratio)
        )

        # Avoid rounding producing too many samples.
        if n_train + n_val > n:
            overflow = (
                n_train + n_val - n
            )

            n_val = max(
                0,
                n_val - overflow,
            )

        train_end = n_train
        val_end = n_train + n_val

        train_indices.extend(
            indices[:train_end]
        )

        val_indices.extend(
            indices[train_end:val_end]
        )

        test_indices.extend(
            indices[val_end:]
        )

    train_df = (
        dataframe
        .loc[train_indices]
        .sample(
            frac=1,
            random_state=seed,
        )
        .reset_index(drop=True)
    )

    val_df = (
        dataframe
        .loc[val_indices]
        .sample(
            frac=1,
            random_state=seed + 1,
        )
        .reset_index(drop=True)
    )

    test_df = (
        dataframe
        .loc[test_indices]
        .sample(
            frac=1,
            random_state=seed + 2,
        )
        .reset_index(drop=True)
    )

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    return (
        train_df,
        val_df,
        test_df,
    )


def build_manifest(
    real_dir: str | Path,
    fake_dir: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    verify_images: bool = True,
) -> None:

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    real_entries = scan_class_directory(
        directory=real_dir,
        label=0,
        class_name="real",
        verify_images=verify_images,
    )

    fake_entries = scan_class_directory(
        directory=fake_dir,
        label=1,
        class_name="fake",
        verify_images=verify_images,
    )

    entries = (
        real_entries
        + fake_entries
    )

    if not entries:
        raise RuntimeError(
            "No images were found."
        )

    dataframe = entries_to_dataframe(
        entries
    )

    duplicate_mask = dataframe[
        "file_id"
    ].duplicated(
        keep=False
    )

    if duplicate_mask.any():

        duplicates = dataframe[
            duplicate_mask
        ]

        print(
            "\n[WARNING] Duplicate file IDs detected:"
        )

        print(
            duplicates[
                [
                    "image_path",
                    "file_id",
                ]
            ]
        )

    (
        train_df,
        val_df,
        test_df,
    ) = stratified_split(
        dataframe=dataframe,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    full_df = pd.concat(
        [
            train_df,
            val_df,
            test_df,
        ],
        ignore_index=True,
    )

    train_path = (
        output_dir
        / "train.csv"
    )

    val_path = (
        output_dir
        / "val.csv"
    )

    test_path = (
        output_dir
        / "test.csv"
    )

    full_path = (
        output_dir
        / "all.csv"
    )

    train_df.to_csv(
        train_path,
        index=False,
    )

    val_df.to_csv(
        val_path,
        index=False,
    )

    test_df.to_csv(
        test_path,
        index=False,
    )

    full_df.to_csv(
        full_path,
        index=False,
    )

    print(
        "\n=============================="
    )

    print(
        "MANIFEST BUILD COMPLETE"
    )

    print(
        "=============================="
    )

    print(
        f"\nTotal: {len(full_df)}"
    )

    print(
        f"Train: {len(train_df)}"
    )

    print(
        f"Val:   {len(val_df)}"
    )

    print(
        f"Test:  {len(test_df)}"
    )

    print(
        "\nClass distribution:"
    )

    print(
        full_df[
            "class_name"
        ]
        .value_counts()
    )

    print(
        "\nSource distribution:"
    )

    print(
        full_df[
            [
                "class_name",
                "source",
            ]
        ]
        .value_counts()
    )

    print(
        f"\nSaved manifests to:"
        f"\n{output_dir.resolve()}"
    )