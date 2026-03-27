# src/siglip/utils/helpers.py
"""
Helper utilities for SigLIP data processing and feature management.
"""

import os
from pathlib import Path
from typing import List, Union

import pandas as pd


def prepare_image_paths(
    df: pd.DataFrame,
    base_path: Union[str, Path],
    subdir: str = None,
    image_col: str = "image_path",
) -> pd.DataFrame:
    """
    Convert relative image paths to absolute paths.

    If subdir is provided, it is inserted between base_path and the base name of the
    original path. Otherwise, the path is simply joined with base_path.

    Args:
        df (pd.DataFrame): DataFrame containing image paths.
        base_path (Union[str, Path]): Root directory for images.
        subdir (str, optional): Subdirectory to insert (e.g., 'train').
        image_col (str): Name of the column containing image paths.

    Returns:
        pd.DataFrame: DataFrame with updated absolute paths.
    """
    base_path = Path(base_path)

    def _to_absolute(p: str) -> str:
        if subdir is not None:
            # Use only basename of original path
            return str(base_path / subdir / os.path.basename(p))
        else:
            return str(base_path / p)

    # Check if paths are already absolute (simple heuristic: starts with '/')
    if not str(df[image_col].iloc[0]).startswith('/'):
        df = df.copy()
        df[image_col] = df[image_col].apply(_to_absolute)
    return df


def add_embeddings_to_df(
    df: pd.DataFrame,
    embeddings: pd.DataFrame,
    prefix: str = "emb",
) -> pd.DataFrame:
    """
    Add embedding columns to a DataFrame.

    Args:
        df (pd.DataFrame): Original DataFrame.
        embeddings (pd.DataFrame): Embeddings with same number of rows as df.
        prefix (str): Prefix for embedding column names.

    Returns:
        pd.DataFrame: DataFrame with embedding columns appended.
    """
    emb_cols = [f"{prefix}{i}" for i in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols)
    return pd.concat([df.reset_index(drop=True), emb_df], axis=1)


def extract_image_id(image_path: str) -> str:
    """
    Extract image ID from file path (without extension).

    Args:
        image_path (str): Full or relative path to an image.

    Returns:
        str: Image ID (filename without extension).
    """
    return os.path.splitext(os.path.basename(image_path))[0]