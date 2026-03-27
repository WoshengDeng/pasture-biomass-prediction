# src/siglip/data/dataset.py
"""
Dataset classes for SigLIP-based feature extraction.
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


class SiglipDataset(Dataset):
    """
    Dataset for SigLIP feature extraction.

    Returns image path and metadata for each sample. Actual image loading
    is deferred to the feature extraction pipeline to handle variable patch
    sizes and batching complexities.

    Args:
        df (pd.DataFrame): DataFrame with at least an 'image_path' column.
        image_root (Path): Root directory where images are stored.
    """

    def __init__(self, df: pd.DataFrame, image_root: Path):
        self.df = df.reset_index(drop=True)
        self.image_root = Path(image_root)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[str, Dict[str, Any]]:
        """
        Get sample by index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (image_path, metadata) where:
                - image_path (str): Full path to the image file.
                - metadata (dict): Original row data as dictionary.
        """
        row = self.df.iloc[idx]
        # Ensure image_path is relative to root
        img_path = self.image_root / row["image_path"]
        metadata = row.to_dict()
        return str(img_path), metadata