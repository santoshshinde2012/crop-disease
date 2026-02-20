"""
Seed management for full reproducibility.

Must be called once at the very beginning of the notebook
before any data loading or model creation.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility.

    Sets seeds for: Python random, NumPy, PyTorch CPU, PyTorch CUDA,
    and configures CuDNN for deterministic behavior.

    Args:
        seed: Integer seed value. Default: 42
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Deterministic CuDNN behavior (slightly slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
