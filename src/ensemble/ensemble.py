# src/ensemble/ensemble.py
"""
Ensemble module for combining DINO and SigLIP predictions.

Provides functions for weighted ensemble, physical constraint enforcement,
and submission file generation.
"""

import os
from typing import List, Dict

import pandas as pd


def weighted_ensemble(
    dino_df: pd.DataFrame,
    siglip_df: pd.DataFrame,
    weights: Dict[str, float],
    targets: List[str],
    clover_only_dino: bool = True,
) -> pd.DataFrame:
    """
    Perform weighted ensemble of DINO and SigLIP predictions.

    Args:
        dino_df: DataFrame with DINO predictions (must contain 'image_path' and target columns).
        siglip_df: DataFrame with SigLIP predictions (same columns).
        weights: Dictionary with keys 'dino' and 'siglip' for ensemble weights.
        targets: List of target column names to ensemble.
        clover_only_dino: If True, always use DINO predictions for 'Dry_Clover_g'.

    Returns:
        pd.DataFrame: DataFrame with ensembled predictions.
    """
    # Ensure both DataFrames have the same index and order
    if not dino_df['image_path'].equals(siglip_df['image_path']):
        raise ValueError("DINO and SigLIP DataFrames have mismatched image paths")

    final_df = dino_df[['image_path']].copy()

    for target in targets:
        if target == 'Dry_Clover_g' and clover_only_dino:
            final_df[target] = dino_df[target]
        else:
            final_df[target] = (
                dino_df[target] * weights['dino'] + siglip_df[target] * weights['siglip']
            )

    return final_df


def enforce_physical_constraints(
    df: pd.DataFrame,
    targets: List[str],
    recompute_derived: bool = True,
) -> pd.DataFrame:
    """
    Enforce non-negativity and mass balance constraints on predictions.

    Args:
        df: DataFrame with predictions. Must contain columns:
            'Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g'
            if recompute_derived is True.
        targets: List of all target column names to clip.
        recompute_derived: If True, recompute 'GDM_g' and 'Dry_Total_g' after clipping.

    Returns:
        pd.DataFrame: DataFrame with constraints applied.
    """
    df = df.copy()

    # Clip all targets to non‑negative
    for col in targets:
        if col in df.columns:
            df[col] = df[col].clip(lower=0.0)

    # Recompute derived targets to maintain consistency (if columns exist)
    if recompute_derived:
        if 'Dry_Green_g' in df.columns and 'Dry_Clover_g' in df.columns:
            df['GDM_g'] = df['Dry_Green_g'] + df['Dry_Clover_g']
        if 'GDM_g' in df.columns and 'Dry_Dead_g' in df.columns:
            df['Dry_Total_g'] = df['GDM_g'] + df['Dry_Dead_g']

    return df


def create_submission(
    df: pd.DataFrame,
    targets: List[str],
    image_path_col: str = 'image_path',
    sample_id_format: str = "{image_id}__{target}",
) -> pd.DataFrame:
    """
    Create submission DataFrame in the required format.

    The submission must have two columns: 'sample_id' and 'target'.
    Each row corresponds to one target per image.

    Args:
        df: DataFrame with predictions. Must contain `image_path_col` and all columns in `targets`.
        targets: List of target names (as they appear in the competition).
        image_path_col: Column name containing image paths.
        sample_id_format: Format string for sample_id. It should contain placeholders
            `{image_id}` (image filename without extension) and `{target}`.

    Returns:
        pd.DataFrame: Submission DataFrame with columns ['sample_id', 'target'].
    """
    submission_rows = []
    for _, row in df.iterrows():
        image_path = row[image_path_col]
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        for target in targets:
            sample_id = sample_id_format.format(image_id=image_id, target=target)
            submission_rows.append({
                'sample_id': sample_id,
                'target': row[target]
            })
    return pd.DataFrame(submission_rows)