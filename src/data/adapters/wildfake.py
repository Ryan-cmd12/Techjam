from __future__ import annotations

import csv
import hashlib
import json
import random
import re

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from PIL import Image


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}


PATH_KEYS = [
    "path",
    "image_path",
    "file_path",
    "filepath",
    "file",
    "filename",
    "file_name",
    "img_path",
    "image",
]


@dataclass(
    frozen=True
)
class WildFakeCandidate:

    image_path: Path
    relative_path: str

    label: int
    class_name: str

    generation_family: str
    subcategory: str
    generator: str
    source: str

    original_split: str


# ============================================================
# GENERIC HELPERS
# ============================================================


def normalize_path_string(
    value: str,
) -> str:

    value = str(
        value
    ).strip()

    value = value.replace(
        "\\",
        "/",
    )

    while value.startswith(
        "./"
    ):

        value = value[
            2:
        ]

    return value


def safe_string(
    value: Any,
) -> str:

    if value is None:

        return ""

    return str(
        value
    ).strip()


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:

        while True:

            chunk = file.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def inspect_image(
    path: Path,
) -> tuple[
    int,
    int,
]:

    with Image.open(
        path
    ) as image:

        width, height = (
            image.size
        )

    return (
        int(
            width
        ),
        int(
            height
        ),
    )


# ============================================================
# WILDFAKE STRUCTURE
# ============================================================


def find_images_directory(
    root: Path,
) -> Path:

    direct = (
        root
        / "Images"
    )

    if direct.exists():

        return direct

    for child in (
        root.iterdir()
    ):

        if (
            child.is_dir()
            and child.name.lower()
            == "images"
        ):

            return child

    raise FileNotFoundError(
        "Could not find WildFake "
        "'Images/' directory under:\n"
        f"{root}"
    )


def find_split_file(
    root: Path,
    split: str,
) -> Path:

    split = (
        split.lower()
        .strip()
    )

    candidates = [
        root
        / f"{split}.jsonl",

        root
        / f"{split}.json",

        root
        / f"{split}.csv",

        root
        / f"{split}.txt",

        root
        / f"{split}_list.jsonl",

        root
        / f"{split}_list.json",

        root
        / f"{split}_list.txt",
    ]

    for candidate in candidates:

        if candidate.exists():

            return candidate

    # Case-insensitive fallback.
    for path in root.iterdir():

        if not path.is_file():

            continue

        stem = (
            path.stem.lower()
        )

        if (
            split in stem
            and path.suffix.lower()
            in {
                ".jsonl",
                ".json",
                ".csv",
                ".txt",
            }
        ):

            return path

    raise FileNotFoundError(
        f"Could not locate official "
        f"WildFake split annotation "
        f"for split '{split}' under:\n"
        f"{root}"
    )


# ============================================================
# SPLIT FILE PARSING
# ============================================================


def extract_path_from_object(
    value: Any,
) -> str | None:

    if isinstance(
        value,
        str,
    ):

        value = value.strip()

        if value:

            return value

        return None

    if isinstance(
        value,
        dict,
    ):

        for key in PATH_KEYS:

            if key not in value:
                continue

            path = extract_path_from_object(
                value[
                    key
                ]
            )

            if path:

                return path

        # Occasionally a JSON entry has one meaningful value.
        string_values = [
            item

            for item
            in value.values()

            if isinstance(
                item,
                str,
            )
        ]

        image_like = [
            item

            for item
            in string_values

            if Path(
                item
            ).suffix.lower()
            in IMAGE_EXTENSIONS
        ]

        if len(
            image_like
        ) == 1:

            return image_like[
                0
            ]

    return None


def recursively_extract_json_paths(
    payload: Any,
) -> list[str]:

    paths = []

    if isinstance(
        payload,
        list,
    ):

        for item in payload:

            direct = (
                extract_path_from_object(
                    item
                )
            )

            if direct:

                paths.append(
                    direct
                )

            elif isinstance(
                item,
                (
                    dict,
                    list,
                ),
            ):

                paths.extend(
                    recursively_extract_json_paths(
                        item
                    )
                )

    elif isinstance(
        payload,
        dict,
    ):

        direct = (
            extract_path_from_object(
                payload
            )
        )

        if direct:

            paths.append(
                direct
            )

        else:

            for value in (
                payload.values()
            ):

                if isinstance(
                    value,
                    (
                        dict,
                        list,
                    ),
                ):

                    paths.extend(
                        recursively_extract_json_paths(
                            value
                        )
                    )

    return paths


def read_jsonl_paths(
    path: Path,
) -> list[str]:

    paths = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:

                payload = json.loads(
                    line
                )

            except json.JSONDecodeError:

                # Also support plain path-per-line
                # files accidentally named jsonl.
                payload = line

            image_path = (
                extract_path_from_object(
                    payload
                )
            )

            if image_path is None:

                raise RuntimeError(
                    "Could not extract an "
                    "image path from:\n"
                    f"{path}\n"
                    f"line {line_number}:\n"
                    f"{line[:300]}"
                )

            paths.append(
                image_path
            )

    return paths


def read_json_paths(
    path: Path,
) -> list[str]:

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        payload = json.load(
            file
        )

    paths = (
        recursively_extract_json_paths(
            payload
        )
    )

    if not paths:

        raise RuntimeError(
            f"No image paths could be "
            f"extracted from:\n{path}"
        )

    return paths


def read_csv_paths(
    path: Path,
) -> list[str]:

    dataframe = pd.read_csv(
        path
    )

    column = None

    for candidate in PATH_KEYS:

        if candidate in (
            dataframe.columns
        ):

            column = candidate
            break

    if column is None:

        # Try any column that is mostly image paths.
        for candidate in (
            dataframe.columns
        ):

            values = (
                dataframe[
                    candidate
                ]
                .astype(
                    str
                )
            )

            image_fraction = float(
                values.map(
                    lambda value:
                        Path(
                            value
                        ).suffix.lower()
                        in IMAGE_EXTENSIONS
                ).mean()
            )

            if image_fraction > 0.5:

                column = candidate
                break

    if column is None:

        raise RuntimeError(
            "Could not determine path "
            f"column in:\n{path}"
        )

    return (
        dataframe[
            column
        ]
        .astype(
            str
        )
        .tolist()
    )


def read_text_paths(
    path: Path,
) -> list[str]:

    paths = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            paths.append(
                line
            )

    return paths


def read_split_paths(
    split_file: Path,
) -> list[str]:

    suffix = (
        split_file
        .suffix
        .lower()
    )

    if suffix == ".jsonl":

        return read_jsonl_paths(
            split_file
        )

    if suffix == ".json":

        return read_json_paths(
            split_file
        )

    if suffix == ".csv":

        return read_csv_paths(
            split_file
        )

    if suffix == ".txt":

        return read_text_paths(
            split_file
        )

    raise ValueError(
        f"Unsupported split annotation: "
        f"{split_file}"
    )


# ============================================================
# PATH RESOLUTION
# ============================================================


def strip_images_prefix(
    value: str,
) -> str:

    value = normalize_path_string(
        value
    )

    pieces = value.split(
        "/"
    )

    if (
        pieces
        and pieces[
            0
        ].lower()
        == "images"
    ):

        pieces = pieces[
            1:
        ]

    return "/".join(
        pieces
    )


def resolve_annotation_path(
    root: Path,
    images_root: Path,
    annotation_path: str,
) -> tuple[
    Path,
    str,
]:

    normalized = (
        normalize_path_string(
            annotation_path
        )
    )

    raw = Path(
        normalized
    )

    if raw.is_absolute():

        resolved = raw

        try:

            relative = (
                resolved
                .relative_to(
                    images_root
                )
            )

        except ValueError:

            relative = Path(
                resolved.name
            )

        return (
            resolved,
            normalize_path_string(
                str(
                    relative
                )
            ),
        )

    relative_string = (
        strip_images_prefix(
            normalized
        )
    )

    relative = Path(
        relative_string
    )

    candidates = [
        images_root
        / relative,

        root
        / Path(
            normalized
        ),

        root
        / relative,
    ]

    for candidate in candidates:

        if candidate.exists():

            return (
                candidate,
                relative_string,
            )

    # Return expected path so caller can report it.
    return (
        images_root
        / relative,
        relative_string,
    )


# ============================================================
# HIERARCHY METADATA
# ============================================================


def slug_tokens(
    value: str,
) -> list[str]:

    return [
        token

        for token
        in re.split(
            r"[^a-z0-9]+",
            value.lower(),
        )

        if token
    ]


def parse_hierarchy(
    relative_path: str,
) -> dict[str, Any]:

    relative_path = (
        normalize_path_string(
            relative_path
        )
    )

    parts = [
        part

        for part
        in relative_path.split(
            "/"
        )

        if part
    ]

    if not parts:

        raise ValueError(
            "Empty WildFake relative path."
        )

    family = parts[
        0
    ]

    family_lower = (
        family.lower()
    )

    is_real = (
        family_lower
        == "real"
        or family_lower.startswith(
            "real_"
        )
    )

    if is_real:

        label = 0
        class_name = "real"

        subcategory = (
            parts[
                1
            ]
            if len(
                parts
            ) >= 3
            else "real"
        )

        generator = "none"

        source = (
            parts[
                1
            ]
            if len(
                parts
            ) >= 3
            else "wildfake_real"
        )

    else:

        label = 1
        class_name = "fake"

        if len(
            parts
        ) >= 4:

            # Example:
            #
            # GAN_based/
            # Typical/
            # styleGAN/
            # image.png
            subcategory = (
                parts[
                    1
                ]
            )

            generator = (
                parts[
                    2
                ]
            )

        elif len(
            parts
        ) == 3:

            # Example:
            #
            # GAN_based/
            # styleGAN/
            # image.png
            subcategory = (
                parts[
                    1
                ]
            )

            generator = (
                parts[
                    1
                ]
            )

        else:

            subcategory = (
                "unknown"
            )

            generator = (
                "unknown"
            )

        source = (
            generator
        )

    return {
        "label":
            label,

        "class_name":
            class_name,

        "generation_family":
            family,

        "subcategory":
            subcategory,

        "generator":
            generator,

        "source":
            source,
    }


# ============================================================
# HACKATHON BENCHMARK PROTECTION
# ============================================================


def looks_like_hackathon_source(
    relative_path: str,
) -> bool:
    """
    Source-level protection for the benchmark sources
    described by the challenge:

      - COCO val2017 real
      - DALL-E Advanced fake

    This intentionally requires compound token matches
    so we don't blacklist unrelated DALL-E/COCO data.
    """

    tokens = set(
        slug_tokens(
            relative_path
        )
    )

    normalized = (
        relative_path
        .lower()
        .replace(
            "\\",
            "/",
        )
    )

    coco_like = (
        "coco"
        in tokens
        and (
            "val2017"
            in normalized
            or (
                "val"
                in tokens
                and "2017"
                in tokens
            )
        )
    )

    dalle_like = (
        (
            "dalle"
            in tokens
            or "dall"
            in tokens
            or "dall-e"
            in normalized
            or "dall_e"
            in normalized
        )
        and "advanced"
        in tokens
    )

    return (
        coco_like
        or dalle_like
    )


def build_benchmark_hashes(
    benchmark_root: Path | None,
) -> set[str]:

    if (
        benchmark_root is None
        or not benchmark_root.exists()
    ):

        print(
            "\nBenchmark directory not found; "
            "exact benchmark hash protection "
            "is unavailable."
        )

        print(
            "Source-level exclusion is still "
            "enabled."
        )

        return set()

    image_paths = [
        path

        for path
        in benchmark_root.rglob(
            "*"
        )

        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ]

    print(
        f"\nHashing local benchmark images: "
        f"{len(image_paths):,}"
    )

    hashes = set()

    for path in image_paths:

        try:

            hashes.add(
                sha256_file(
                    path
                )
            )

        except Exception as exception:

            print(
                f"[WARNING] Could not hash "
                f"{path}: {exception}"
            )

    print(
        f"Benchmark hashes loaded: "
        f"{len(hashes):,}"
    )

    return hashes


# ============================================================
# CANDIDATES
# ============================================================


def build_candidates(
    root: Path,
    split: str,
    split_file: Path | None = None,
    exclude_benchmark_sources: bool = True,
) -> tuple[
    list[WildFakeCandidate],
    dict[str, int],
]:

    images_root = (
        find_images_directory(
            root
        )
    )

    if split_file is None:

        split_file = (
            find_split_file(
                root=
                    root,

                split=
                    split,
            )
        )

    print(
        f"\nWildFake root:"
        f"\n{root}"
    )

    print(
        f"\nImages:"
        f"\n{images_root}"
    )

    print(
        f"\nOfficial {split} annotation:"
        f"\n{split_file}"
    )

    annotated_paths = (
        read_split_paths(
            split_file
        )
    )

    print(
        f"\nAnnotated entries: "
        f"{len(annotated_paths):,}"
    )

    candidates = []

    stats = {
        "annotation_rows":
            len(
                annotated_paths
            ),

        "missing":
            0,

        "duplicate_paths":
            0,

        "benchmark_source_excluded":
            0,
    }

    seen_paths = set()

    for annotation_path in (
        annotated_paths
    ):

        (
            image_path,
            relative_path,
        ) = (
            resolve_annotation_path(
                root=
                    root,

                images_root=
                    images_root,

                annotation_path=
                    annotation_path,
            )
        )

        key = str(
            image_path.resolve()
            if image_path.exists()
            else image_path
        )

        if key in seen_paths:

            stats[
                "duplicate_paths"
            ] += 1

            continue

        seen_paths.add(
            key
        )

        if not image_path.exists():

            stats[
                "missing"
            ] += 1

            continue

        if (
            image_path.suffix.lower()
            not in IMAGE_EXTENSIONS
        ):

            continue

        if (
            exclude_benchmark_sources
            and looks_like_hackathon_source(
                relative_path
            )
        ):

            stats[
                "benchmark_source_excluded"
            ] += 1

            continue

        hierarchy = (
            parse_hierarchy(
                relative_path
            )
        )

        candidates.append(
            WildFakeCandidate(

                image_path=
                    image_path,

                relative_path=
                    relative_path,

                label=int(
                    hierarchy[
                        "label"
                    ]
                ),

                class_name=str(
                    hierarchy[
                        "class_name"
                    ]
                ),

                generation_family=str(
                    hierarchy[
                        "generation_family"
                    ]
                ),

                subcategory=str(
                    hierarchy[
                        "subcategory"
                    ]
                ),

                generator=str(
                    hierarchy[
                        "generator"
                    ]
                ),

                source=str(
                    hierarchy[
                        "source"
                    ]
                ),

                original_split=
                    split,
            )
        )

    return (
        candidates,
        stats,
    )


# ============================================================
# BALANCED GENERATOR SAMPLING
# ============================================================


def deterministic_shuffle(
    values: list,
    seed: int,
) -> list:

    values = list(
        values
    )

    rng = random.Random(
        seed
    )

    rng.shuffle(
        values
    )

    return values


def sample_fake_balanced(
    candidates: list[
        WildFakeCandidate
    ],
    max_total: int | None,
    max_per_generator: int | None,
    seed: int,
) -> list[
    WildFakeCandidate
]:

    groups = defaultdict(
        list
    )

    for candidate in candidates:

        groups[
            candidate.generator
        ].append(
            candidate
        )

    generators = sorted(
        groups.keys()
    )

    for index, generator in enumerate(
        generators
    ):

        values = (
            deterministic_shuffle(
                groups[
                    generator
                ],

                seed=
                    seed
                    + index
                    * 7919,
            )
        )

        if (
            max_per_generator
            is not None
        ):

            values = values[
                :max_per_generator
            ]

        groups[
            generator
        ] = values

    if max_total is None:

        selected = []

        for generator in generators:

            selected.extend(
                groups[
                    generator
                ]
            )

        return selected

    selected = []

    positions = {
        generator:
            0

        for generator
        in generators
    }

    # Round-robin gives smaller generators a fair chance.
    while len(
        selected
    ) < max_total:

        added = False

        for generator in generators:

            position = positions[
                generator
            ]

            values = groups[
                generator
            ]

            if position >= len(
                values
            ):

                continue

            selected.append(
                values[
                    position
                ]
            )

            positions[
                generator
            ] += 1

            added = True

            if len(
                selected
            ) >= max_total:

                break

        if not added:

            break

    return selected


def sample_candidates(
    candidates: list[
        WildFakeCandidate
    ],
    max_real: int | None,
    max_fake: int | None,
    max_per_generator: int | None,
    seed: int,
) -> list[
    WildFakeCandidate
]:

    real = [
        candidate

        for candidate
        in candidates

        if candidate.label == 0
    ]

    fake = [
        candidate

        for candidate
        in candidates

        if candidate.label == 1
    ]

    real = deterministic_shuffle(
        real,
        seed=seed,
    )

    if max_real is not None:

        real = real[
            :max_real
        ]

    fake = (
        sample_fake_balanced(
            candidates=
                fake,

            max_total=
                max_fake,

            max_per_generator=
                max_per_generator,

            seed=
                seed + 100000,
        )
    )

    selected = (
        real
        + fake
    )

    return deterministic_shuffle(
        selected,
        seed=
            seed + 200000,
    )


# ============================================================
# MATERIALIZE MANIFEST
# ============================================================


def candidates_to_manifest(
    candidates: list[
        WildFakeCandidate
    ],
    split: str,
    benchmark_hashes: set[str],
) -> tuple[
    pd.DataFrame,
    dict[str, int],
]:

    rows = []

    stats = {
        "invalid_images":
            0,

        "benchmark_hash_excluded":
            0,

        "same_hash_duplicates":
            0,
    }

    seen_hash_labels = {}

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):

        if (
            index == 1
            or index % 1000
            == 0
        ):

            print(
                f"Processing selected images: "
                f"{index:,}/"
                f"{len(candidates):,}"
            )

        try:

            content_hash = (
                sha256_file(
                    candidate.image_path
                )
            )

            width, height = (
                inspect_image(
                    candidate.image_path
                )
            )

        except Exception as exception:

            stats[
                "invalid_images"
            ] += 1

            print(
                f"[WARNING] Invalid image: "
                f"{candidate.image_path}\n"
                f"{exception}"
            )

            continue

        if (
            content_hash
            in benchmark_hashes
        ):

            stats[
                "benchmark_hash_excluded"
            ] += 1

            continue

        if (
            content_hash
            in seen_hash_labels
        ):

            previous_label = (
                seen_hash_labels[
                    content_hash
                ]
            )

            if (
                previous_label
                != candidate.label
            ):

                raise RuntimeError(
                    "WildFake exact duplicate has "
                    "conflicting labels.\n"
                    f"Hash: {content_hash}\n"
                    f"Previous label: "
                    f"{previous_label}\n"
                    f"Current label: "
                    f"{candidate.label}\n"
                    f"Path: "
                    f"{candidate.image_path}"
                )

            stats[
                "same_hash_duplicates"
            ] += 1

            continue

        seen_hash_labels[
            content_hash
        ] = (
            candidate.label
        )

        rows.append(
            {
                "image_path":
                    str(
                        candidate.image_path
                    ),

                "label":
                    candidate.label,

                "class_name":
                    candidate.class_name,

                "dataset":
                    "wildfake",

                "source":
                    candidate.source,

                "generator":
                    candidate.generator,

                "generation_family":
                    candidate.generation_family,

                "subcategory":
                    candidate.subcategory,

                "wildfake_relative_path":
                    candidate.relative_path,

                "original_split":
                    candidate.original_split,

                "width":
                    width,

                "height":
                    height,

                "content_hash":
                    content_hash,

                "split":
                    split,
            }
        )

    return (
        pd.DataFrame(
            rows
        ),
        stats,
    )


# ============================================================
# PUBLIC BUILDER
# ============================================================


def build_wildfake_manifest(
    root: str | Path,
    split: str,
    output_path: str | Path,
    split_file: str | Path | None = None,
    max_real: int | None = 5000,
    max_fake: int | None = 5000,
    max_per_generator: int | None = 1000,
    seed: int = 42,
    benchmark_root: str | Path | None = None,
    exclude_benchmark_sources: bool = True,
) -> pd.DataFrame:

    root = Path(
        root
    )

    output_path = Path(
        output_path
    )

    if split_file is not None:

        split_file = Path(
            split_file
        )

    benchmark_path = (
        Path(
            benchmark_root
        )
        if benchmark_root
        else None
    )

    benchmark_hashes = (
        build_benchmark_hashes(
            benchmark_path
        )
    )

    (
        candidates,
        candidate_stats,
    ) = (
        build_candidates(
            root=
                root,

            split=
                split,

            split_file=
                split_file,

            exclude_benchmark_sources=
                exclude_benchmark_sources,
        )
    )

    print(
        "\n========================================"
    )

    print(
        "WILDFAKE CANDIDATES"
    )

    print(
        "========================================"
    )

    print(
        f"\nAvailable after filtering: "
        f"{len(candidates):,}"
    )

    print(
        f"Missing annotation paths: "
        f"{candidate_stats['missing']:,}"
    )

    print(
        f"Duplicate annotation paths: "
        f"{candidate_stats['duplicate_paths']:,}"
    )

    print(
        f"Benchmark-source excluded: "
        f"{candidate_stats['benchmark_source_excluded']:,}"
    )

    selected = (
        sample_candidates(
            candidates=
                candidates,

            max_real=
                max_real,

            max_fake=
                max_fake,

            max_per_generator=
                max_per_generator,

            seed=
                seed,
        )
    )

    print(
        f"\nSelected before hashing: "
        f"{len(selected):,}"
    )

    (
        dataframe,
        materialization_stats,
    ) = (
        candidates_to_manifest(
            candidates=
                selected,

            split=
                split,

            benchmark_hashes=
                benchmark_hashes,
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    print(
        "\n========================================"
    )

    print(
        "WILDFAKE MANIFEST COMPLETE"
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
        ].value_counts()
    )

    print(
        "\nGeneration families:"
    )

    print(
        dataframe[
            "generation_family"
        ].value_counts()
    )

    print(
        "\nFake generators:"
    )

    fake = dataframe[
        dataframe[
            "label"
        ] == 1
    ]

    print(
        fake[
            "generator"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nImage dimensions:"
    )

    print(
        dataframe[
            [
                "width",
                "height",
            ]
        ].describe()
    )

    print(
        "\nRemoved after selection:"
    )

    print(
        materialization_stats
    )

    print(
        "\nSaved:"
    )

    print(
        output_path.resolve()
    )

    return dataframe