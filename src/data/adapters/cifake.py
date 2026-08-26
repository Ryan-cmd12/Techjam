
from __future__ import annotations

import hashlib

from pathlib import Path

import pandas as pd

from sklearn.model_selection import (
    train_test_split,
)

from src.data.image_utils import (
    DEFAULT_IMAGE_EXTENSIONS,
    get_image_size,
    is_image_file,
    validate_image,
)


DATASET_NAME = "cifake"


def find_directory_case_insensitive(
    parent: Path,
    target_name: str,
) -> Path:
    """
    Find a child directory without depending on capitalization.

    This allows:

        REAL
        Real
        real

    to all work.
    """

    expected = (
        parent
        / target_name
    )

    if expected.exists():
        return expected

    target_lower = (
        target_name.lower()
    )

    for child in parent.iterdir():

        if (
            child.is_dir()
            and child.name.lower()
            == target_lower
        ):

            return child

    raise FileNotFoundError(
        f"Could not find directory "
        f"'{target_name}' inside "
        f"'{parent}'."
    )


def calculate_sha256(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """
    Calculate an SHA256 hash of an image.

    We use content hashes so duplicate images can be detected
    even if they have different filenames.
    """

    hasher = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:

        while True:

            chunk = file.read(
                chunk_size
            )

            if not chunk:
                break

            hasher.update(
                chunk
            )

    return hasher.hexdigest()


def normalize_manifest_path(
    path: Path,
) -> str:
    """
    Prefer project-relative paths so manifests do not contain
    machine-specific paths such as:

        C:\\Users\\...

    If the image is outside the project directory, fall back
    to an absolute path.
    """

    path = path.resolve()

    project_root = (
        Path.cwd()
        .resolve()
    )

    try:

        relative_path = (
            path.relative_to(
                project_root
            )
        )

        return (
            relative_path
            .as_posix()
        )

    except ValueError:

        return (
            path.as_posix()
        )


def scan_class(
    directory: Path,
    label: int,
    class_name: str,
    original_split: str,
    extensions: set[str] | None = None,
) -> pd.DataFrame:
    """
    Scan one CIFAKE class directory.

    Example:

        train/REAL
        train/FAKE
    """

    valid_extensions = (
        extensions
        if extensions is not None
        else DEFAULT_IMAGE_EXTENSIONS
    )

    image_paths = sorted(
        path
        for path in directory.rglob("*")
        if is_image_file(
            path,
            extensions=valid_extensions,
        )
    )

    print(
        f"\nScanning "
        f"{original_split}/{class_name}"
    )

    print(
        f"Found "
        f"{len(image_paths)} "
        f"candidate images."
    )

    records = []

    invalid_count = 0

    for index, path in enumerate(
        image_paths,
        start=1,
    ):

        if not validate_image(
            path
        ):

            invalid_count += 1

            print(
                f"[WARNING] "
                f"Unreadable image: "
                f"{path}"
            )

            continue

        try:

            width, height = (
                get_image_size(
                    path
                )
            )

        except Exception as error:

            invalid_count += 1

            print(
                f"[WARNING] "
                f"Could not inspect "
                f"{path}: "
                f"{error}"
            )

            continue

        content_hash = (
            calculate_sha256(
                path
            )
        )

        records.append(
            {
                "image_path":
                    normalize_manifest_path(
                        path
                    ),

                "label":
                    label,

                "class_name":
                    class_name,

                "dataset":
                    DATASET_NAME,

                "source":
                    "cifake",

                # We intentionally don't invent generator
                # metadata that isn't present in the folder.
                "generator":
                    "unknown",

                "original_split":
                    original_split,

                "width":
                    width,

                "height":
                    height,

                "content_hash":
                    content_hash,
            }
        )

        if (
            index % 10000
            == 0
        ):

            print(
                f"Processed "
                f"{index}/"
                f"{len(image_paths)}"
            )

    dataframe = pd.DataFrame(
        records
    )

    print(
        f"Valid images: "
        f"{len(dataframe)}"
    )

    print(
        f"Invalid images: "
        f"{invalid_count}"
    )

    return dataframe


def scan_original_split(
    split_directory: Path,
    original_split: str,
    real_dir_name: str = "REAL",
    fake_dir_name: str = "FAKE",
) -> pd.DataFrame:

    real_directory = (
        find_directory_case_insensitive(
            parent=split_directory,
            target_name=real_dir_name,
        )
    )

    fake_directory = (
        find_directory_case_insensitive(
            parent=split_directory,
            target_name=fake_dir_name,
        )
    )

    real_dataframe = (
        scan_class(
            directory=real_directory,
            label=0,
            class_name="real",
            original_split=original_split,
        )
    )

    fake_dataframe = (
        scan_class(
            directory=fake_directory,
            label=1,
            class_name="fake",
            original_split=original_split,
        )
    )

    dataframe = pd.concat(
        [
            real_dataframe,
            fake_dataframe,
        ],
        ignore_index=True,
    )

    return dataframe


def check_internal_duplicates(
    dataframe: pd.DataFrame,
    name: str,
) -> pd.DataFrame:

    duplicate_mask = (
        dataframe[
            "content_hash"
        ]
        .duplicated(
            keep=False
        )
    )

    duplicates = dataframe[
        duplicate_mask
    ].copy()

    if len(duplicates) == 0:

        print(
            f"\nNo duplicate images "
            f"detected inside {name}."
        )

        return duplicates

    print(
        f"\n[WARNING] "
        f"{len(duplicates)} rows "
        f"in {name} are involved "
        f"in duplicate-image groups."
    )

    return duplicates


def check_cross_split_leakage(
    train_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Detect images that occur in both CIFAKE's original
    training and test sets.
    """

    train_hashes = set(
        train_dataframe[
            "content_hash"
        ].tolist()
    )

    test_hashes = set(
        test_dataframe[
            "content_hash"
        ].tolist()
    )

    overlapping_hashes = (
        train_hashes
        & test_hashes
    )

    if not overlapping_hashes:

        print(
            "\nNo exact image leakage detected "
            "between original train and test sets."
        )

        return pd.DataFrame()

    print(
        "\n[WARNING]"
    )

    print(
        f"Found "
        f"{len(overlapping_hashes)} "
        f"image hashes shared between "
        f"original train and test."
    )

    combined = pd.concat(
        [
            train_dataframe,
            test_dataframe,
        ],
        ignore_index=True,
    )

    leakage_dataframe = (
        combined[
            combined[
                "content_hash"
            ]
            .isin(
                overlapping_hashes
            )
        ]
        .copy()
    )

    return leakage_dataframe


def create_train_validation_split(
    original_train_dataframe: pd.DataFrame,
    validation_fraction: float = 0.10,
    seed: int = 42,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    if not (
        0.0
        < validation_fraction
        < 1.0
    ):

        raise ValueError(
            "validation_fraction must "
            "be between 0 and 1."
        )

    train_dataframe, validation_dataframe = (
        train_test_split(
            original_train_dataframe,

            test_size=
                validation_fraction,

            random_state=
                seed,

            stratify=
                original_train_dataframe[
                    "label"
                ],
        )
    )

    train_dataframe = (
        train_dataframe
        .copy()
        .reset_index(
            drop=True
        )
    )

    validation_dataframe = (
        validation_dataframe
        .copy()
        .reset_index(
            drop=True
        )
    )

    train_dataframe[
        "split"
    ] = "train"

    validation_dataframe[
        "split"
    ] = "val"

    return (
        train_dataframe,
        validation_dataframe,
    )


def print_split_summary(
    dataframe: pd.DataFrame,
    name: str,
) -> None:

    print(
        "\n=============================="
    )

    print(
        name.upper()
    )

    print(
        "=============================="
    )

    print(
        f"Images: "
        f"{len(dataframe)}"
    )

    print(
        "\nClass counts:"
    )

    print(
        dataframe[
            "class_name"
        ]
        .value_counts()
    )

    print(
        "\nClass proportions:"
    )

    proportions = (
        dataframe[
            "class_name"
        ]
        .value_counts(
            normalize=True
        )
    )

    print(
        proportions
    )


def build_cifake_manifests(
    dataset_root: str | Path,
    output_directory: str | Path,
    validation_fraction: float = 0.10,
    seed: int = 42,
    train_dir_name: str = "train",
    test_dir_name: str = "test",
    real_dir_name: str = "REAL",
    fake_dir_name: str = "FAKE",
) -> None:

    dataset_root = Path(
        dataset_root
    )

    output_directory = Path(
        output_directory
    )

    if not dataset_root.exists():

        raise FileNotFoundError(
            f"CIFAKE root does not exist: "
            f"{dataset_root}"
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_directory = (
        find_directory_case_insensitive(
            parent=dataset_root,
            target_name=train_dir_name,
        )
    )

    test_directory = (
        find_directory_case_insensitive(
            parent=dataset_root,
            target_name=test_dir_name,
        )
    )

    print(
        "\n=============================="
    )

    print(
        "BUILDING CIFAKE MANIFEST"
    )

    print(
        "=============================="
    )

    print(
        f"\nRoot: "
        f"{dataset_root.resolve()}"
    )

    print(
        f"\nOriginal training folder:"
        f"\n{train_directory}"
    )

    print(
        f"\nOriginal test folder:"
        f"\n{test_directory}"
    )

    original_train_dataframe = (
        scan_original_split(
            split_directory=
                train_directory,

            original_split=
                "train",

            real_dir_name=
                real_dir_name,

            fake_dir_name=
                fake_dir_name,
        )
    )

    original_test_dataframe = (
        scan_original_split(
            split_directory=
                test_directory,

            original_split=
                "test",

            real_dir_name=
                real_dir_name,

            fake_dir_name=
                fake_dir_name,
        )
    )

    check_internal_duplicates(
        dataframe=
            original_train_dataframe,

        name=
            "original training set",
    )

    check_internal_duplicates(
        dataframe=
            original_test_dataframe,

        name=
            "original test set",
    )

    leakage_dataframe = (
        check_cross_split_leakage(
            train_dataframe=
                original_train_dataframe,

            test_dataframe=
                original_test_dataframe,
        )
    )

    if len(
        leakage_dataframe
    ) > 0:

        leakage_path = (
            output_directory
            / "cifake_cross_split_duplicates.csv"
        )

        leakage_dataframe.to_csv(
            leakage_path,
            index=False,
        )

        print(
            f"\nDuplicate report saved:"
            f"\n{leakage_path}"
        )

    (
        train_dataframe,
        validation_dataframe,
    ) = (
        create_train_validation_split(
            original_train_dataframe=
                original_train_dataframe,

            validation_fraction=
                validation_fraction,

            seed=
                seed,
        )
    )

    test_dataframe = (
        original_test_dataframe
        .copy()
        .reset_index(
            drop=True
        )
    )

    test_dataframe[
        "split"
    ] = "test"

    full_dataframe = pd.concat(
        [
            train_dataframe,
            validation_dataframe,
            test_dataframe,
        ],
        ignore_index=True,
    )

    train_path = (
        output_directory
        / "cifake_train.csv"
    )

    validation_path = (
        output_directory
        / "cifake_val.csv"
    )

    test_path = (
        output_directory
        / "cifake_test.csv"
    )

    all_path = (
        output_directory
        / "cifake_all.csv"
    )

    train_dataframe.to_csv(
        train_path,
        index=False,
    )

    validation_dataframe.to_csv(
        validation_path,
        index=False,
    )

    test_dataframe.to_csv(
        test_path,
        index=False,
    )

    full_dataframe.to_csv(
        all_path,
        index=False,
    )

    print_split_summary(
        train_dataframe,
        "CIFAKE Training",
    )

    print_split_summary(
        validation_dataframe,
        "CIFAKE Validation",
    )

    print_split_summary(
        test_dataframe,
        "CIFAKE Test",
    )

    print(
        "\n=============================="
    )

    print(
        "CIFAKE MANIFEST COMPLETE"
    )

    print(
        "=============================="
    )

    print(
        f"\nTrain:"
        f"\n{train_path}"
    )

    print(
        f"\nValidation:"
        f"\n{validation_path}"
    )

    print(
        f"\nTest:"
        f"\n{test_path}"
    )

    print(
        f"\nCombined:"
        f"\n{all_path}"
    )