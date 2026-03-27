# src/siglip/data/transforms.py
"""
Image transformation utilities for SigLIP feature extraction.

Includes functions for splitting images into overlapping patches.
"""

from typing import List

import numpy as np


def split_image(image: np.ndarray, patch_size: int = 520, overlap: int = 16) -> List[np.ndarray]:
    """
    Split a large image into overlapping patches for SigLIP inference.

    Handles edge cases where the image size is not a multiple of patch size.

    Args:
        image (np.ndarray): Input image array (H, W, C).
        patch_size (int): Size of each patch (square).
        overlap (int): Overlap between patches.

    Returns:
        List[np.ndarray]: List of image patches.
    """
    h, w, _ = image.shape
    stride = patch_size - overlap
    patches = []

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # Ensure patch fits within image boundaries
            y2 = min(y + patch_size, h)
            x2 = min(x + patch_size, w)
            y1 = max(0, y2 - patch_size)
            x1 = max(0, x2 - patch_size)

            patches.append(image[y1:y2, x1:x2, :])

    return patches