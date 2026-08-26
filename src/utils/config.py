from pathlib import Path
from typing import Any

import yaml


def load_config(
    path: str | Path,
) -> dict[str, Any]:
    """
    Load a YAML configuration file.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Config file does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(
            f"Config file is empty: {path}"
        )

    return config