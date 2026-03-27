# src/utils/seed.py
import os
import random
import numpy as np
import torch
from src.dino.config import CFG  # Import the configuration class from the project configuration module


def seed_everything(seed: int = CFG.SEED) -> None:
    """
    Set random seed across Python, NumPy, and PyTorch for reproducibility.

    Ensures consistent behavior across runs, including multi-GPU settings
    and hash-based operations.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)