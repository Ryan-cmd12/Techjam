import os
import random

import numpy as np
import torch


def seed_everything(
    seed: int = 42,
) -> None:
    """
    Seed common random number generators so experiments
    are as reproducible as reasonably possible.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic algorithms can hurt performance and some
    # operations do not support them, so we keep benchmark mode
    # disabled while avoiding forcing full determinism.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True