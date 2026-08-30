from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from tqdm import tqdm


# ============================================================
# CONSTANTS
# ============================================================

DATASET_ID = "hy2628982280/WildFake"

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
    "filepath",
    "file_path",
    "file",
    "filename",
    "file_name",
    "img_path",
    "image",
]

# WildFake's actual official overall split paths.
OFFICIAL_TEST_CANDIDATES = [
    "split_train_test/csv_file/total_split/test_metadata.csv",
    "split_train_test/csv_file_detail/total_split/test_metadata.csv",
    # Generic fallbacks, in case the repository layout changes later.
    "test_metadata.csv",
    "test.csv",
    "test.jsonl",
    "test.json",
    "splits/test.jsonl",
    "splits/test.json",
    "metadata/test.jsonl",
    "metadata/test.json",
    "test.txt",
]

# Archive layout visible in the official WildFake repository.
SINGLE_ARCHIVE_DIFFUSION_ARCHITECTURES = {
    "adm": "ADM",
    "dalle": "DALLE",
    "ddim": "DDIM",
    "ddpm": "DDPM",
    "imagen": "Imagen",
    "vqdm": "VQDM",
}

MIDJOURNEY_PART_COUNTS = {
    "advanced": 7,
    "typical": 4,
}

ORIGINAL_SD_PART_COUNTS = {
    "advanced": 7,
    "typical": 3,
}

EMPTY_METADATA_VALUES = {
    "",
    "none",
    "null",
    "nan",
    "n/a",
    "na",
    "-",
}

# Small-download defaults for this project.  These choices intentionally
# restrict the run to four repository ZIP files:
#   Images/Diffusion_based/DDIM.zip
#   Images/Real/afhq.zip
#   Images/Real/church.zip
#   Images/Real/ffhq.zip
DEFAULT_FAKE_ARCHITECTURES = ["DDIM"]
DEFAULT_REAL_SOURCES = ["afhq", "church", "ffhq"]


# ============================================================
# BASIC HELPERS
# ============================================================


def normalize_path(value: str) -> str:
    value = str(value).strip().replace("\\", "/")

    while value.startswith("./"):
        value = value[2:]

    while "//" in value:
        value = value.replace("//", "/")

    return value.lstrip("/")


def strip_images_prefix(path: str) -> str:
    path = normalize_path(path)

    if path.lower().startswith("images/"):
        return path[len("Images/") :]

    return path


def remote_image_path(annotation_path: str) -> str:
    annotation_path = normalize_path(annotation_path)

    if annotation_path.lower().startswith("images/"):
        return annotation_path

    return f"Images/{annotation_path}"


def tokenize(value: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", value.lower())
        if token
    }


def clean_metadata_value(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip()

    if text.lower() in EMPTY_METADATA_VALUES:
        return ""

    return text


def parse_optional_bool(value: Any) -> bool | None:
    text = clean_metadata_value(value).lower()

    if not text:
        return None

    if text in {"1", "1.0", "true", "yes", "y", "fake"}:
        return True

    if text in {"0", "0.0", "false", "no", "n", "real"}:
        return False

    return None


def get_case_insensitive(row: dict[str, Any], key: str) -> Any:
    target = key.lower().strip()

    for current_key, value in row.items():
        if str(current_key).lower().strip() == target:
            return value

    return None


# ============================================================
# SOURCE PROTECTION
# ============================================================


def source_search_text(record: dict[str, Any]) -> str:
    values = [
        record.get("image_path", ""),
        record.get("annotation_path", ""),
        record.get("remote_path", ""),
        record.get("metadata_generator", ""),
        record.get("metadata_architecture", ""),
        record.get("metadata_weight", ""),
        record.get("metadata_category", ""),
    ]

    return " ".join(
        clean_metadata_value(value)
        for value in values
        if clean_metadata_value(value)
    )


def is_forbidden_benchmark_source(
    record_or_path: dict[str, Any] | str,
) -> tuple[bool, str | None]:
    """
    Hard source-level exclusion before any image/archive extraction.

    Excludes:
      1. COCO val2017
      2. DALL-E Advanced
    """

    if isinstance(record_or_path, dict):
        text = source_search_text(record_or_path)
        is_advanced_metadata = record_or_path.get("metadata_is_advanced") is True
    else:
        text = str(record_or_path)
        is_advanced_metadata = False

    normalized = normalize_path(text).lower()
    tokens = tokenize(normalized)

    # COCO val2017
    coco_val2017 = (
        "coco" in tokens
        and (
            "val2017" in normalized
            or ("val" in tokens and "2017" in tokens)
        )
    )

    if coco_val2017:
        return True, "COCO val2017"

    # DALL-E Advanced
    dalle_present = (
        "dalle" in tokens
        or ("dall" in tokens and "e" in tokens)
        or "dall-e" in normalized
        or "dall_e" in normalized
        or "dalle" in normalized
    )

    advanced_present = (
        "advanced" in tokens
        or "advanced" in normalized
        or is_advanced_metadata
    )

    if dalle_present and advanced_present:
        return True, "DALL-E Advanced"

    return False, None


# ============================================================
# MODELSCOPE
# ============================================================


def create_hub_api():
    try:
        from modelscope_hub import HubApi
    except ImportError as exception:
        raise RuntimeError(
            "\nmodelscope-hub is not installed.\n\n"
            "Install it with:\n\n"
            "pip install -U modelscope modelscope-hub\n"
        ) from exception

    return HubApi()


def download_repo_file(
    api,
    remote_path: str,
    local_root: Path,
    force: bool = False,
) -> Path:
    """Download one repository file from the ModelScope dataset."""

    result = api.download_file(
        DATASET_ID,
        "dataset",
        remote_path,
        local_dir=str(local_root),
        force=force,
    )

    return Path(result)


# ============================================================
# ANNOTATION READING
# ============================================================


def extract_path(value: Any) -> str | None:
    if isinstance(value, str):
        value = value.strip()
        return value if value else None

    if isinstance(value, dict):
        normalized = {
            str(key).lower().strip(): item
            for key, item in value.items()
        }

        for key in PATH_KEYS:
            if key not in normalized:
                continue

            result = extract_path(normalized[key])
            if result:
                return result

    return None


def recursively_extract_paths(value: Any) -> list[str]:
    paths: list[str] = []

    if isinstance(value, list):
        for item in value:
            direct = extract_path(item)

            if direct:
                paths.append(direct)
            elif isinstance(item, (list, dict)):
                paths.extend(recursively_extract_paths(item))

    elif isinstance(value, dict):
        direct = extract_path(value)

        if direct:
            paths.append(direct)
        else:
            for item in value.values():
                if isinstance(item, (list, dict)):
                    paths.extend(recursively_extract_paths(item))

    return paths


def make_path_only_record(path: str) -> dict[str, Any]:
    return {
        "image_path": normalize_path(path),
        "metadata_generator": "",
        "metadata_architecture": "",
        "metadata_weight": "",
        "metadata_category": "",
        "metadata_is_advanced": None,
        "metadata_is_fake": None,
        "metadata_num": "",
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                payload = line

            result = extract_path(payload)

            if result is None:
                raise RuntimeError(
                    f"Could not extract image path from {path}, "
                    f"line {line_number}."
                )

            records.append(make_path_only_record(result))

    return records


def read_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    paths = recursively_extract_paths(payload)

    if not paths:
        raise RuntimeError(f"No paths found in {path}")

    return [make_path_only_record(value) for value in paths]


def find_column(
    columns: list[str],
    candidates: list[str],
) -> str | None:
    normalized_columns = {
        str(column).lower().strip(): column
        for column in columns
    }

    for candidate in candidates:
        key = candidate.lower().strip()
        if key in normalized_columns:
            return normalized_columns[key]

    return None


def read_csv_file(path: Path) -> list[dict[str, Any]]:
    """
    Read WildFake CSV metadata and KEEP the hierarchy columns.

    This fixes the old bug where CSV rows were reduced to paths and the
    script later guessed the generator from directory positions.
    """

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        raise RuntimeError(f"CSV is empty: {path}")

    columns = list(rows[0].keys())

    path_column = find_column(
        columns,
        [
            "image_path",
            "imagepath",
            "path",
            "filepath",
            "file_path",
            "filename",
            "file",
        ],
    )

    if path_column is None:
        raise RuntimeError(
            "\nCould not identify image path column in "
            f"{path}.\nColumns: {columns}"
        )

    records: list[dict[str, Any]] = []

    for row in rows:
        image_path = clean_metadata_value(row.get(path_column))

        if not image_path:
            continue

        records.append(
            {
                "image_path": normalize_path(image_path),
                "metadata_generator": clean_metadata_value(
                    get_case_insensitive(row, "Generator")
                ),
                "metadata_architecture": clean_metadata_value(
                    get_case_insensitive(row, "Architecture")
                ),
                "metadata_weight": clean_metadata_value(
                    get_case_insensitive(row, "Weight")
                ),
                "metadata_category": clean_metadata_value(
                    get_case_insensitive(row, "Category")
                ),
                "metadata_is_advanced": parse_optional_bool(
                    get_case_insensitive(row, "IsAdvanced")
                ),
                "metadata_is_fake": parse_optional_bool(
                    get_case_insensitive(row, "IsFake")
                ),
                "metadata_num": clean_metadata_value(
                    get_case_insensitive(row, "Num")
                ),
            }
        )

    return records


def read_text(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [
            make_path_only_record(line.strip())
            for line in file
            if line.strip()
        ]


def read_annotation(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        return read_jsonl(path)

    if suffix == ".json":
        return read_json(path)

    if suffix == ".csv":
        return read_csv_file(path)

    if suffix == ".txt":
        return read_text(path)

    raise ValueError(f"Unsupported annotation format: {path}")


# ============================================================
# DOWNLOAD OFFICIAL TEST ANNOTATION
# ============================================================


def download_test_annotation(
    api,
    local_root: Path,
    explicit_remote_path: str | None,
) -> Path:
    if explicit_remote_path:
        print("\nDownloading requested split annotation:")
        print(explicit_remote_path)

        return download_repo_file(
            api=api,
            remote_path=explicit_remote_path,
            local_root=local_root,
        )

    print("\nSearching for official WildFake test annotation...")

    errors: list[tuple[str, str]] = []

    for remote_path in OFFICIAL_TEST_CANDIDATES:
        try:
            local = download_repo_file(
                api=api,
                remote_path=remote_path,
                local_root=local_root,
            )

            print(f"\nFound official split:\n{remote_path}")
            return local

        except Exception as exception:
            errors.append((remote_path, str(exception)))

    raise RuntimeError(
        "\nCould not automatically locate WildFake's official test annotation.\n\n"
        "Rerun with:\n\n"
        "  --split-file <remote/path/to/test_metadata.csv>\n\n"
        "Tried:\n"
        + "\n".join(f"  {remote_path}" for remote_path, _ in errors)
    )


# ============================================================
# WILDFAKE HIERARCHY
# ============================================================


def meaningful_metadata_values(record: dict[str, Any]) -> list[str]:
    values = [
        clean_metadata_value(record.get("metadata_generator")),
        clean_metadata_value(record.get("metadata_architecture")),
        clean_metadata_value(record.get("metadata_weight")),
        clean_metadata_value(record.get("metadata_category")),
    ]

    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        if not value:
            continue

        key = value.lower()
        if key in seen:
            continue

        seen.add(key)
        result.append(value)

    return result


def fallback_generator_from_path(path: str) -> str:
    relative = strip_images_prefix(path)
    parts = [part for part in relative.split("/") if part]

    if len(parts) <= 2:
        return "unknown"

    # Start from the directory directly above the filename.
    for value in reversed(parts[:-1]):
        if value.lower() not in {
            "img",
            "imgs",
            "image",
            "images",
            "advanced",
            "typical",
        }:
            return value

    return parts[-2]


def make_generator_key(record: dict[str, Any], is_real: bool) -> str:
    if is_real:
        return "none"

    values = meaningful_metadata_values(record)

    if values:
        # Keep the full hierarchy instead of pretending one ambiguous CSV
        # column is always the exact generator model.
        return "/".join(values)

    return fallback_generator_from_path(record["image_path"])


def parse_wildfake_record(record: dict[str, Any]) -> dict[str, Any]:
    annotation_path = normalize_path(record["image_path"])
    relative = strip_images_prefix(annotation_path)
    parts = [part for part in relative.split("/") if part]

    if not parts:
        raise ValueError(f"Invalid path: {annotation_path}")

    path_family = parts[0]
    path_is_real = path_family.lower() == "real" or path_family.lower().startswith("real_")

    metadata_is_fake = record.get("metadata_is_fake")

    if metadata_is_fake is None:
        is_real = path_is_real
    else:
        is_real = not bool(metadata_is_fake)

    family = "Real" if is_real else path_family

    if is_real:
        subcategory = parts[1] if len(parts) > 1 else "real"
    else:
        subcategory = (
            clean_metadata_value(record.get("metadata_architecture"))
            or (parts[1] if len(parts) > 1 else "unknown")
        )

    return {
        "label": 0 if is_real else 1,
        "class_name": "real" if is_real else "fake",
        "family": family,
        "subcategory": subcategory,
        "generator": make_generator_key(record, is_real=is_real),
    }


# ============================================================
# FILTER SPLIT
# ============================================================


def build_candidates(
    annotation_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    for record in annotation_records:
        path = normalize_path(record["image_path"])
        remote_path = remote_image_path(path)

        if remote_path in seen_paths:
            continue

        seen_paths.add(remote_path)

        working = {
            **record,
            "annotation_path": strip_images_prefix(path),
            "remote_path": remote_path,
        }

        hierarchy = parse_wildfake_record(working)
        working.update(hierarchy)

        forbidden, reason = is_forbidden_benchmark_source(working)

        if forbidden:
            excluded.append(
                {
                    "path": remote_path,
                    "reason": reason,
                    "metadata": {
                        "generator": working.get("metadata_generator", ""),
                        "architecture": working.get("metadata_architecture", ""),
                        "weight": working.get("metadata_weight", ""),
                        "category": working.get("metadata_category", ""),
                    },
                }
            )
            continue

        candidates.append(working)

    return candidates, excluded


# ============================================================
# SOURCE RESTRICTION
# ============================================================


def candidate_real_source(candidate: dict[str, Any]) -> str:
    relative = strip_images_prefix(candidate["annotation_path"])
    parts = [part for part in relative.split("/") if part]

    if len(parts) >= 2 and parts[0].lower() == "real":
        return parts[1]

    return ""


def candidate_fake_architecture(candidate: dict[str, Any]) -> str:
    architecture = clean_metadata_value(candidate.get("metadata_architecture"))

    if architecture:
        return architecture

    relative = strip_images_prefix(candidate["annotation_path"])
    parts = [part for part in relative.split("/") if part]

    if len(parts) >= 2 and parts[0].lower() == "diffusion_based":
        return parts[1]

    return ""


def restrict_candidates(
    candidates: list[dict[str, Any]],
    real_sources: list[str],
    fake_architectures: list[str],
) -> list[dict[str, Any]]:
    real_allowed = {value.lower() for value in real_sources}
    fake_allowed = {value.lower() for value in fake_architectures}

    restricted: list[dict[str, Any]] = []

    for candidate in candidates:
        if candidate["label"] == 0:
            source = candidate_real_source(candidate).lower()

            if source in real_allowed:
                candidate = {**candidate, "real_source": source}
                restricted.append(candidate)

            continue

        architecture = candidate_fake_architecture(candidate).lower()

        if architecture in fake_allowed:
            candidate = {**candidate, "fake_architecture": architecture}
            restricted.append(candidate)

    return restricted


def sample_real_balanced_by_source(
    candidates: list[dict[str, Any]],
    count: int,
    sources: list[str],
    seed: int,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for candidate in candidates:
        if candidate["label"] != 0:
            continue

        source = candidate_real_source(candidate).lower()

        if source:
            groups[source].append(candidate)

    requested_sources = [source.lower() for source in sources]
    requested_sources = [source for source in requested_sources if groups.get(source)]

    if not requested_sources:
        return []

    for index, source in enumerate(requested_sources):
        local_rng = random.Random(seed + index * 104729)
        local_rng.shuffle(groups[source])

    selected: list[dict[str, Any]] = []
    positions = {source: 0 for source in requested_sources}

    # Round-robin gives an approximately even number of real images from
    # each selected source while never pulling from any unlisted ZIP.
    while len(selected) < count:
        added = False

        for source in requested_sources:
            position = positions[source]
            values = groups[source]

            if position >= len(values):
                continue

            selected.append(values[position])
            positions[source] += 1
            added = True

            if len(selected) >= count:
                break

        if not added:
            break

    return selected


def sample_fake_from_restricted_architectures(
    candidates: list[dict[str, Any]],
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    fake = [candidate for candidate in candidates if candidate["label"] == 1]

    rng = random.Random(seed)
    rng.shuffle(fake)

    return fake[: min(count, len(fake))]


def build_restricted_selection(
    candidates: list[dict[str, Any]],
    real_count: int,
    fake_count: int,
    real_sources: list[str],
    fake_architectures: list[str],
    seed: int,
) -> list[dict[str, Any]]:
    restricted = restrict_candidates(
        candidates=candidates,
        real_sources=real_sources,
        fake_architectures=fake_architectures,
    )

    real = sample_real_balanced_by_source(
        candidates=restricted,
        count=real_count,
        sources=real_sources,
        seed=seed,
    )

    fake = sample_fake_from_restricted_architectures(
        candidates=restricted,
        count=fake_count,
        seed=seed + 10000,
    )

    selected = real + fake

    rng = random.Random(seed + 20000)
    rng.shuffle(selected)

    return selected


# ============================================================
# BALANCED SAMPLING
# ============================================================


def sample_real(
    candidates: list[dict[str, Any]],
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    real = [candidate for candidate in candidates if candidate["label"] == 0]

    rng = random.Random(seed)
    rng.shuffle(real)

    return real[: min(count, len(real))]


def sample_fake_balanced(
    candidates: list[dict[str, Any]],
    total_count: int,
    max_per_generator: int,
    seed: int,
) -> list[dict[str, Any]]:
    fake = [candidate for candidate in candidates if candidate["label"] == 1]

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for candidate in fake:
        key = (candidate["family"], candidate["generator"])
        groups[key].append(candidate)

    group_keys = sorted(groups.keys())
    rng = random.Random(seed)

    # Deterministically randomize within each generator key.
    for index, key in enumerate(group_keys):
        local_rng = random.Random(seed + index * 104729)
        local_rng.shuffle(groups[key])
        groups[key] = groups[key][:max_per_generator]

    rng.shuffle(group_keys)

    positions = {key: 0 for key in group_keys}
    selected: list[dict[str, Any]] = []

    # Round-robin across generator keys for diversity.
    while len(selected) < total_count:
        added = False

        for key in group_keys:
            position = positions[key]
            values = groups[key]

            if position >= len(values):
                continue

            selected.append(values[position])
            positions[key] += 1
            added = True

            if len(selected) >= total_count:
                break

        if not added:
            break

    return selected


def build_selection(
    candidates: list[dict[str, Any]],
    real_count: int,
    fake_count: int,
    max_per_generator: int,
    seed: int,
) -> list[dict[str, Any]]:
    real = sample_real(
        candidates=candidates,
        count=real_count,
        seed=seed,
    )

    fake = sample_fake_balanced(
        candidates=candidates,
        total_count=fake_count,
        max_per_generator=max_per_generator,
        seed=seed + 10000,
    )

    selected = real + fake

    rng = random.Random(seed + 20000)
    rng.shuffle(selected)

    return selected


# ============================================================
# HASH PROTECTION
# ============================================================


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def load_benchmark_hashes(benchmark_root: Path) -> set[str]:
    if not benchmark_root.exists():
        return set()

    image_paths = [
        path
        for path in benchmark_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not image_paths:
        return set()

    print(
        f"\nHashing {len(image_paths):,} local benchmark images "
        "for an additional leakage check..."
    )

    hashes: set[str] = set()

    for path in tqdm(image_paths, desc="Benchmark SHA256"):
        try:
            hashes.add(sha256_file(path))
        except Exception as exception:
            print(f"\n[WARNING] Could not hash {path}: {exception}")

    return hashes


# ============================================================
# PATH -> ARCHIVE MAPPING
# ============================================================


def archive_candidates_for_path(annotation_path: str) -> list[str]:
    """
    Return the repository ZIP file(s) that can contain one annotation path.

    For multipart Midjourney/originalSD folders, the metadata does not tell
    us which part_N.zip contains a particular image, so we return all parts
    in order. The extraction stage stops as soon as every selected image in
    that archive group has been found.
    """

    relative = strip_images_prefix(annotation_path)
    parts = [part for part in PurePosixPath(relative).parts if part]

    if not parts:
        raise ValueError(f"Invalid WildFake path: {annotation_path}")

    family = parts[0].lower()

    # Real/<source>/...
    if family == "real":
        if len(parts) < 2:
            raise ValueError(f"Invalid real-image path: {annotation_path}")

        source = parts[1]
        return [f"Images/Real/{source}.zip"]

    # All GAN images are in one archive.
    if family == "gan_based":
        return ["Images/GAN_based.zip"]

    # All Other-based images are in one archive.
    if family == "other_based":
        return ["Images/Other_based.zip"]

    if family != "diffusion_based":
        raise ValueError(
            f"Unknown WildFake top-level family in path: {annotation_path}"
        )

    if len(parts) < 2:
        raise ValueError(f"Invalid diffusion path: {annotation_path}")

    architecture_lower = parts[1].lower()

    # ADM / DALLE / DDIM / DDPM / Imagen / VQDM
    if architecture_lower in SINGLE_ARCHIVE_DIFFUSION_ARCHITECTURES:
        canonical = SINGLE_ARCHIVE_DIFFUSION_ARCHITECTURES[architecture_lower]
        return [f"Images/Diffusion_based/{canonical}.zip"]

    # Midjourney/<Advanced|Typical>/...
    if architecture_lower == "midjourney":
        if len(parts) < 3:
            raise ValueError(f"Invalid Midjourney path: {annotation_path}")

        category = parts[2]
        count = MIDJOURNEY_PART_COUNTS.get(category.lower())

        if count is None:
            raise ValueError(
                f"Unknown Midjourney category '{category}' in {annotation_path}"
            )

        canonical_category = "Advanced" if category.lower() == "advanced" else "Typical"

        return [
            f"Images/Diffusion_based/Midjourney/{canonical_category}/part_{index}.zip"
            for index in range(1, count + 1)
        ]

    # SD/<weight>/...
    if architecture_lower == "sd":
        if len(parts) < 3:
            raise ValueError(f"Invalid Stable Diffusion path: {annotation_path}")

        weight = parts[2]
        weight_lower = weight.lower()

        if weight_lower == "originalsd":
            if len(parts) < 4:
                raise ValueError(f"Invalid originalSD path: {annotation_path}")

            category = parts[3]
            count = ORIGINAL_SD_PART_COUNTS.get(category.lower())

            if count is None:
                raise ValueError(
                    f"Unknown originalSD category '{category}' in {annotation_path}"
                )

            canonical_category = "Advanced" if category.lower() == "advanced" else "Typical"

            return [
                f"Images/Diffusion_based/SD/originalSD/{canonical_category}/part_{index}.zip"
                for index in range(1, count + 1)
            ]

        if weight_lower == "personalizedsd":
            return ["Images/Diffusion_based/SD/personalizedSD.zip"]

        if weight_lower == "sdwithadaptor":
            return ["Images/Diffusion_based/SD/SDwithAdaptor.zip"]

        raise ValueError(
            f"Unknown Stable Diffusion weight '{weight}' in {annotation_path}"
        )

    raise ValueError(
        f"Could not determine archive for WildFake path: {annotation_path}"
    )


def print_archive_plan(selection: list[dict[str, Any]]) -> None:
    archives: set[str] = set()

    for item in selection:
        for archive in archive_candidates_for_path(item["annotation_path"]):
            archives.add(archive)

    print("\nRepository ZIP files required by this selection:")

    for archive in sorted(archives):
        print(f"  {archive}")

    print(f"\nTotal ZIP files: {len(archives):,}")


# ============================================================
# ZIP MEMBER MATCHING
# ============================================================


def normalized_zip_member_name(name: str) -> str:
    return normalize_path(name)


def trailing_match_score(target: str, member: str) -> int:
    target_parts = [part.lower() for part in normalize_path(target).split("/") if part]
    member_parts = [part.lower() for part in normalize_path(member).split("/") if part]

    score = 0

    while (
        score < len(target_parts)
        and score < len(member_parts)
        and target_parts[-1 - score] == member_parts[-1 - score]
    ):
        score += 1

    return score


def build_member_basename_index(
    archive: zipfile.ZipFile,
) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)

    for info in archive.infolist():
        if info.is_dir():
            continue

        name = normalized_zip_member_name(info.filename)
        basename = PurePosixPath(name).name.lower()

        if basename:
            index[basename].append(info.filename)

    return index


def find_member_for_target(
    target: str,
    basename_index: dict[str, list[str]],
) -> str | None:
    target = strip_images_prefix(target)
    basename = PurePosixPath(target).name.lower()
    candidates = basename_index.get(basename, [])

    if not candidates:
        return None

    scored = [
        (trailing_match_score(target, candidate), candidate)
        for candidate in candidates
    ]

    best_score = max(score for score, _ in scored)
    best = [candidate for score, candidate in scored if score == best_score]

    # Prefer at least filename + one parent directory. If there is only one
    # basename match in the whole archive, basename-only is still safe enough.
    if best_score >= 2 and len(best) == 1:
        return best[0]

    if len(candidates) == 1:
        return candidates[0]

    # Multiple equally plausible members -> do not silently extract the wrong file.
    return None


# ============================================================
# ARCHIVE-AWARE DOWNLOAD + SELECTIVE EXTRACTION
# ============================================================


def destination_for_item(item: dict[str, Any], output_root: Path) -> Path:
    return output_root / "Images" / strip_images_prefix(item["annotation_path"])


def finalize_extracted_image(
    item: dict[str, Any],
    output_path: Path,
    benchmark_hashes: set[str],
    archive_path: str,
    archive_member: str,
) -> dict[str, Any]:
    content_hash: str | None = None

    if benchmark_hashes:
        try:
            content_hash = sha256_file(output_path)
        except Exception:
            content_hash = None

    if content_hash is not None and content_hash in benchmark_hashes:
        try:
            output_path.unlink()
        except OSError:
            pass

        return {
            **item,
            "status": "benchmark_hash_removed",
            "reason": "Exact SHA256 matches local benchmark",
            "archive_path": archive_path,
            "archive_member": archive_member,
            "content_hash": content_hash,
        }

    return {
        **item,
        "status": "downloaded",
        "local_path": str(output_path),
        "archive_path": archive_path,
        "archive_member": archive_member,
        "content_hash": content_hash,
    }


def validate_existing_image(
    item: dict[str, Any],
    output_root: Path,
    benchmark_hashes: set[str],
) -> dict[str, Any] | None:
    output_path = destination_for_item(item, output_root)

    if not output_path.exists() or not output_path.is_file():
        return None

    if benchmark_hashes:
        try:
            content_hash = sha256_file(output_path)
        except Exception:
            content_hash = None

        if content_hash is not None and content_hash in benchmark_hashes:
            try:
                output_path.unlink()
            except OSError:
                pass

            return {
                **item,
                "status": "benchmark_hash_removed",
                "reason": "Existing file exactly matches local benchmark SHA256",
                "local_path": str(output_path),
                "content_hash": content_hash,
            }

    return {
        **item,
        "status": "reused",
        "local_path": str(output_path),
    }


def safely_delete_archive(path: Path, output_root: Path) -> None:
    try:
        resolved_path = path.resolve()
        resolved_root = output_root.resolve()

        # Only delete files that were downloaded inside our requested output tree.
        if resolved_root in resolved_path.parents and resolved_path.is_file():
            resolved_path.unlink()
    except OSError:
        pass


def process_archive_group(
    api,
    items: list[dict[str, Any]],
    archive_candidates: tuple[str, ...],
    output_root: Path,
    benchmark_hashes: set[str],
    keep_archives: bool,
    force_archive_download: bool,
    progress: tqdm,
) -> dict[str, dict[str, Any]]:
    pending = {
        item["annotation_path"]: item
        for item in items
    }

    results: dict[str, dict[str, Any]] = {}
    archive_errors: list[str] = []

    for remote_archive in archive_candidates:
        if not pending:
            break

        print(
            f"\nArchive: {remote_archive}\n"
            f"  still needed from this group: {len(pending):,}"
        )

        try:
            local_archive = download_repo_file(
                api=api,
                remote_path=remote_archive,
                local_root=output_root,
                force=force_archive_download,
            )
        except Exception as exception:
            archive_errors.append(f"{remote_archive}: {exception}")
            print(f"  [WARNING] archive download failed: {exception}")
            continue

        try:
            with zipfile.ZipFile(local_archive, "r") as archive:
                basename_index = build_member_basename_index(archive)

                found_this_archive: list[str] = []

                for annotation_path, item in list(pending.items()):
                    member = find_member_for_target(
                        annotation_path,
                        basename_index,
                    )

                    if member is None:
                        continue

                    output_path = destination_for_item(item, output_root)
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    try:
                        with archive.open(member, "r") as source, output_path.open("wb") as destination:
                            shutil.copyfileobj(source, destination)

                        result = finalize_extracted_image(
                            item=item,
                            output_path=output_path,
                            benchmark_hashes=benchmark_hashes,
                            archive_path=remote_archive,
                            archive_member=member,
                        )

                    except Exception as exception:
                        try:
                            output_path.unlink()
                        except OSError:
                            pass

                        result = {
                            **item,
                            "status": "failed",
                            "reason": (
                                f"Found member in {remote_archive} but extraction failed: "
                                f"{exception}"
                            ),
                            "archive_path": remote_archive,
                            "archive_member": member,
                        }

                    results[annotation_path] = result
                    found_this_archive.append(annotation_path)
                    progress.update(1)

                for annotation_path in found_this_archive:
                    pending.pop(annotation_path, None)

                print(
                    f"  extracted selected images: {len(found_this_archive):,}\n"
                    f"  remaining for this group: {len(pending):,}"
                )

        except zipfile.BadZipFile as exception:
            archive_errors.append(
                f"{remote_archive}: downloaded file is not a valid ZIP ({exception})"
            )
            print(f"  [WARNING] invalid ZIP: {exception}")

        except Exception as exception:
            archive_errors.append(f"{remote_archive}: ZIP processing failed: {exception}")
            print(f"  [WARNING] ZIP processing failed: {exception}")

        finally:
            if not keep_archives:
                safely_delete_archive(local_archive, output_root)

    if pending:
        tried = ", ".join(archive_candidates)
        error_text = " | ".join(archive_errors)

        for annotation_path, item in pending.items():
            reason = f"Image was not found in expected archive(s): {tried}"

            if error_text:
                reason += f". Archive errors: {error_text}"

            results[annotation_path] = {
                **item,
                "status": "failed",
                "reason": reason,
            }
            progress.update(1)

    return results


def download_selection(
    selection: list[dict[str, Any]],
    output_root: Path,
    benchmark_hashes: set[str],
    workers: int,
    keep_archives: bool,
    force_archive_download: bool,
) -> list[dict[str, Any]]:
    print("\n========================================")
    print("DOWNLOADING WILDFAKE SUBSET")
    print("========================================")
    print(f"\nImages requested: {len(selection):,}")
    print(
        "Download mode: archive-aware selective extraction\n"
        "ModelScope stores WildFake images inside ZIP files, so individual "
        "image paths are not requested directly."
    )

    if workers != 1:
        print(
            f"\nNote: --workers={workers} is accepted for CLI compatibility, "
            "but archive downloads are intentionally processed sequentially "
            "to avoid downloading several very large ZIPs at once."
        )

    api = create_hub_api()

    results_by_path: dict[str, dict[str, Any]] = {}
    pending_items: list[dict[str, Any]] = []

    progress = tqdm(total=len(selection), desc="WildFake images", unit="image")

    # Reuse already-extracted images first.
    for item in selection:
        forbidden, reason = is_forbidden_benchmark_source(item)

        if forbidden:
            results_by_path[item["annotation_path"]] = {
                **item,
                "status": "excluded",
                "reason": reason,
            }
            progress.update(1)
            continue

        existing = validate_existing_image(
            item=item,
            output_root=output_root,
            benchmark_hashes=benchmark_hashes,
        )

        if existing is not None:
            results_by_path[item["annotation_path"]] = existing
            progress.update(1)
            continue

        pending_items.append(item)

    # Group by the archive search set. This means, for example, every
    # Midjourney/Advanced selected image shares one 7-part search group.
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)

    for item in pending_items:
        try:
            archives = tuple(archive_candidates_for_path(item["annotation_path"]))
        except Exception as exception:
            results_by_path[item["annotation_path"]] = {
                **item,
                "status": "failed",
                "reason": f"Archive mapping failed: {exception}",
            }
            progress.update(1)
            continue

        groups[archives].append(item)

    print(f"\nArchive groups required: {len(groups):,}")

    for archives, items in groups.items():
        group_results = process_archive_group(
            api=api,
            items=items,
            archive_candidates=archives,
            output_root=output_root,
            benchmark_hashes=benchmark_hashes,
            keep_archives=keep_archives,
            force_archive_download=force_archive_download,
            progress=progress,
        )

        results_by_path.update(group_results)

    progress.close()

    # Preserve the same order as selected_test_subset.jsonl.
    results: list[dict[str, Any]] = []

    for item in selection:
        result = results_by_path.get(item["annotation_path"])

        if result is None:
            result = {
                **item,
                "status": "failed",
                "reason": "Internal error: item produced no download result.",
            }

        results.append(result)

    return results


# ============================================================
# REPORTING
# ============================================================


def print_selection_summary(
    candidates: list[dict[str, Any]],
    selection: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
) -> None:
    real_available = sum(candidate["label"] == 0 for candidate in candidates)
    fake_available = sum(candidate["label"] == 1 for candidate in candidates)

    selected_real = sum(item["label"] == 0 for item in selection)
    selected_fake = sum(item["label"] == 1 for item in selection)

    generators: dict[str, int] = defaultdict(int)
    families: dict[str, int] = defaultdict(int)

    for item in selection:
        if item["label"] != 1:
            continue

        generators[item["generator"]] += 1
        families[item["family"]] += 1

    print("\n========================================")
    print("WILDFAKE SELECTION")
    print("========================================")

    print(f"\nAvailable real: {real_available:,}")
    print(f"Available fake: {fake_available:,}")

    print(f"\nSelected real: {selected_real:,}")
    print(f"Selected fake: {selected_fake:,}")
    print(f"Total selected: {len(selection):,}")

    print("\nForbidden benchmark paths removed BEFORE download:")
    print(f"{len(excluded):,}")

    if excluded:
        reason_counts: dict[str, int] = defaultdict(int)

        for item in excluded:
            reason_counts[item["reason"]] += 1

        for reason, count in sorted(reason_counts.items()):
            print(f"  {reason}: {count:,}")

    print("\nFake families:")
    for family, count in sorted(families.items()):
        print(f"  {family:25s} {count:,}")

    print("\nFake generator keys (from official CSV hierarchy):")
    for generator, count in sorted(
        generators.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        print(f"  {generator:55s} {count:,}")


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for item in rows:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Selectively download a balanced WildFake official-test subset. "
            "The downloader understands WildFake's ZIP archive layout and "
            "extracts only the selected images from each downloaded archive."
        )
    )

    parser.add_argument(
        "--output",
        default="data/raw/wildfake",
    )

    parser.add_argument(
        "--split-file",
        default=None,
        help=(
            "Optional remote path to the official WildFake test annotation. "
            "Default auto-detection now includes the official "
            "split_train_test/csv_file/total_split/test_metadata.csv path."
        ),
    )

    parser.add_argument(
        "--real",
        type=int,
        default=5000,
        help="Number of real test images to select.",
    )

    parser.add_argument(
        "--fake",
        type=int,
        default=5000,
        help="Number of fake test images to select.",
    )

    parser.add_argument(
        "--max-per-generator",
        type=int,
        default=500,
        help=(
            "Legacy option retained for compatibility. In restricted-source "
            "mode the requested fake architecture is sampled directly, so "
            "this cap is not used."
        ),
    )

    parser.add_argument(
        "--fake-architectures",
        nargs="+",
        default=DEFAULT_FAKE_ARCHITECTURES,
        help=(
            "Fake architectures to allow. Default: DDIM only. "
            "Using DDIM only means the fake side requires exactly "
            "Images/Diffusion_based/DDIM.zip."
        ),
    )

    parser.add_argument(
        "--real-sources",
        nargs="+",
        default=DEFAULT_REAL_SOURCES,
        help=(
            "Real WildFake sources to allow. Default: afhq church ffhq. "
            "The real sample is balanced across these sources."
        ),
    )

    parser.add_argument(
        "--reset-images",
        action="store_true",
        help=(
            "Delete the existing output Images directory before extraction. "
            "Use this when changing the selected subset so old images do not "
            "inflate the manifest count."
        ),
    )

    parser.add_argument("--seed", type=int, default=42)

    # Retained so old commands do not break. Archive downloads are sequential.
    parser.add_argument("--workers", type=int, default=8)

    parser.add_argument(
        "--benchmark-root",
        default="data/benchmark/hackathon_wildfake",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Select and print files without downloading image archives.",
    )

    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help=(
            "Keep downloaded WildFake ZIP files after extracting the selected "
            "images. Default behavior deletes each ZIP after processing to "
            "save disk space."
        ),
    )

    parser.add_argument(
        "--force-archive-download",
        action="store_true",
        help="Force ModelScope to redownload ZIP archives even if cached locally.",
    )

    args = parser.parse_args()

    if args.real < 0 or args.fake < 0:
        raise ValueError("--real and --fake must be >= 0")

    if args.max_per_generator <= 0:
        raise ValueError("--max-per-generator must be > 0")

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.reset_images:
        images_root = output_root / "Images"

        if images_root.exists():
            print(f"\nRemoving previous extracted images: {images_root}")
            shutil.rmtree(images_root)

    print("\n========================================")
    print("SELECTIVE WILDFAKE DOWNLOAD")
    print("========================================")

    print(f"\nDataset:\n{DATASET_ID}")

    print("\nHARD EXCLUSIONS:")
    print("  COCO val2017")
    print("  DALL-E Advanced")

    print("\nSOURCE RESTRICTION:")
    print("  Fake architectures: " + ", ".join(args.fake_architectures))
    print("  Real sources: " + ", ".join(args.real_sources))

    # --------------------------------------------------------
    # Official test split
    # --------------------------------------------------------

    api = create_hub_api()

    annotation_path = download_test_annotation(
        api=api,
        local_root=output_root,
        explicit_remote_path=args.split_file,
    )

    annotation_records = read_annotation(annotation_path)

    print(f"\nOfficial test entries: {len(annotation_records):,}")

    # --------------------------------------------------------
    # Filter before any image archive download
    # --------------------------------------------------------

    candidates, excluded = build_candidates(annotation_records)

    # --------------------------------------------------------
    # Restricted selection
    # --------------------------------------------------------

    selection = build_restricted_selection(
        candidates=candidates,
        real_count=args.real,
        fake_count=args.fake,
        real_sources=args.real_sources,
        fake_architectures=args.fake_architectures,
        seed=args.seed,
    )

    selected_real = sum(item["label"] == 0 for item in selection)
    selected_fake = sum(item["label"] == 1 for item in selection)

    if selected_real < args.real:
        raise RuntimeError(
            f"Only {selected_real:,} eligible real images were found in "
            f"the requested real sources {args.real_sources}; requested "
            f"{args.real:,}."
        )

    if selected_fake < args.fake:
        raise RuntimeError(
            f"Only {selected_fake:,} eligible fake images were found in "
            f"the requested fake architectures {args.fake_architectures}; "
            f"requested {args.fake:,}."
        )

    print_selection_summary(
        candidates=candidates,
        selection=selection,
        excluded=excluded,
    )

    print_archive_plan(selection)

    # --------------------------------------------------------
    # Save selection before large archive downloads
    # --------------------------------------------------------

    selection_path = output_root / "selected_test_subset.jsonl"
    save_jsonl(selection_path, selection)

    excluded_path = output_root / "excluded_benchmark_sources.jsonl"
    save_jsonl(excluded_path, excluded)

    print("\nSelection saved:")
    print(selection_path)

    print("\nExcluded source list:")
    print(excluded_path)

    if args.dry_run:
        print("\nDRY RUN — no image archives downloaded.")
        return

    # --------------------------------------------------------
    # Optional exact benchmark hash protection
    # --------------------------------------------------------

    benchmark_hashes = load_benchmark_hashes(Path(args.benchmark_root))

    if benchmark_hashes:
        print(f"\nExact benchmark protection: {len(benchmark_hashes):,} hashes")
    else:
        print("\nNo local benchmark hashes found.")
        print("Source-level benchmark protection is still active.")

    # --------------------------------------------------------
    # Archive-aware download
    # --------------------------------------------------------

    results = download_selection(
        selection=selection,
        output_root=output_root,
        benchmark_hashes=benchmark_hashes,
        workers=args.workers,
        keep_archives=args.keep_archives,
        force_archive_download=args.force_archive_download,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    result_path = output_root / "download_results.jsonl"
    save_jsonl(result_path, results)

    status_counts: dict[str, int] = defaultdict(int)

    for item in results:
        status_counts[item["status"]] += 1

    print("\n========================================")
    print("DOWNLOAD COMPLETE")
    print("========================================")

    for status, count in sorted(status_counts.items()):
        print(f"{status:30s} {count:,}")

    print("\nExpected dataset location:")
    print(output_root / "Images")

    print("\nDownload report:")
    print(result_path)

    if status_counts.get("failed", 0) > 0:
        print("\nSome selected images were not found in the expected archive(s).")
        print(
            "Check the failed rows in download_results.jsonl. "
            "Re-running is safe; extracted images are reused."
        )


if __name__ == "__main__":
    main()