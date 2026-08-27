import torch


def get_device() -> torch.device:
    """
    Select the best available PyTorch device.

    Priority:
        CUDA
        MPS
        CPU
    """

    if torch.cuda.is_available():

        device = torch.device(
            "cuda"
        )

    elif (
        hasattr(
            torch.backends,
            "mps",
        )
        and torch.backends.mps.is_available()
    ):

        device = torch.device(
            "mps"
        )

    else:

        device = torch.device(
            "cpu"
        )

    return device


def print_device_info(
    device: torch.device,
) -> None:

    print(
        "\n=============================="
    )

    print(
        "DEVICE"
    )

    print(
        "=============================="
    )

    print(
        f"\nUsing: {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        properties = (
            torch.cuda.get_device_properties(
                0
            )
        )

        memory_gb = (
            properties.total_memory
            / 1024**3
        )

        print(
            f"VRAM: "
            f"{memory_gb:.2f} GB"
        )