from pathlib import Path

from PIL import Image


DEFAULT_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}


def is_image_file(
    path: str | Path,
    extensions: set[str] | None = None,
) -> bool:

    path = Path(path)

    valid_extensions = (
        extensions
        if extensions is not None
        else DEFAULT_IMAGE_EXTENSIONS
    )

    return (
        path.is_file()
        and path.suffix.lower()
        in valid_extensions
    )


def validate_image(
    path: str | Path,
) -> bool:

    path = Path(path)

    try:

        with Image.open(path) as image:
            image.verify()

        return True

    except Exception:

        return False


def load_rgb_image(
    path: str | Path,
) -> Image.Image:

    path = Path(path)

    with Image.open(path) as image:

        image = image.convert(
            "RGB"
        )

        return image.copy()


def get_image_size(
    path: str | Path,
) -> tuple[int, int]:

    path = Path(path)

    with Image.open(path) as image:

        width, height = image.size

    return (
        width,
        height,
    )