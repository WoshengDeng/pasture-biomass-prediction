# inference/siglip_inference.py
"""
SigLIP inference module for biomass prediction.

Provides functions to compute SigLIP embeddings, train GBDT models,
and generate predictions for test images.
"""

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

# Import SigLIP components
from src.siglip.data.dataset import SiglipDataset
from src.siglip.features.embedding_extractor import (
    compute_siglip_embeddings,
    generate_semantic_features
)
from src.siglip.models.gbdt_models import train_gbdt_cv
from src.siglip.utils.helpers import add_embeddings_to_df, prepare_image_paths

# Import GBDT model classes
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


def run_siglip_inference(
    config: Dict[str, Any],
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run full SigLIP inference pipeline (embedding extraction + GBDT training + prediction).

    Note: This function trains GBDT models on the fly using cross-validation.
    For production, pre-trained models should be loaded instead.

    Args:
        config: Configuration dictionary with SigLIP-related keys:
            - data_path: Root directory for images.
            - split_path: Path to CSV with fold information.
            - siglip_model_path: Path to pretrained SigLIP model.
            - siglip: sub-dictionary containing:
                - patch_size, overlap, embedding_dim
                - target_names, target_max, skip_targets
                - gbdt_models: list of dicts with 'class' and 'params'
            - device (optional): 'cuda' or 'cpu'.
        test_df: DataFrame with column 'image_path' (relative to data_path).

    Returns:
        DataFrame with predictions for each test image, containing columns:
            image_path, and the targets (with Clover set to 0).
    """
    data_path = Path(config['data_path'])
    split_path = Path(config['split_path'])
    siglip_path = config['siglip_model_path']
    siglip_cfg = config['siglip']
    device = config.get('device', 'cuda')

    # Load training split with fold information
    train_split = pd.read_csv(split_path)
    # Remove any existing embedding columns
    cols_keep = [c for c in train_split.columns if not c.startswith('emb')]
    train_split = train_split[cols_keep]

    # Prepare image paths (make absolute)
    train_split = prepare_image_paths(train_split, data_path, subdir='train')
    test_df = prepare_image_paths(test_df, data_path)

    # Compute embeddings for train and test
    print("  Computing train embeddings...")
    train_emb = compute_siglip_embeddings(
        siglip_path, train_split, data_path,
        device=device,
        patch_size=siglip_cfg['patch_size'],
        overlap=siglip_cfg['overlap'],
        embedding_dim=siglip_cfg['embedding_dim']
    )
    print("  Computing test embeddings...")
    test_emb = compute_siglip_embeddings(
        siglip_path, test_df, data_path,
        device=device,
        patch_size=siglip_cfg['patch_size'],
        overlap=siglip_cfg['overlap'],
        embedding_dim=siglip_cfg['embedding_dim']
    )

    # Add embeddings to DataFrames
    emb_cols = [f"emb{i}" for i in range(train_emb.shape[1])]
    train_feat = add_embeddings_to_df(train_split, train_emb, prefix='emb')
    test_feat = add_embeddings_to_df(test_df, test_emb, prefix='emb')

    # Generate semantic features (using all images to avoid recomputing)
    print("  Generating semantic features...")
    all_emb = np.vstack([train_emb, test_emb])
    all_sem = generate_semantic_features(all_emb, siglip_path, device=device)
    sem_train = all_sem[:len(train_split)]
    sem_test = all_sem[len(train_split):]

    # Prepare target names and max values
    target_names = siglip_cfg['target_names']
    target_max = siglip_cfg['target_max']
    skip_targets = siglip_cfg.get('skip_targets', [])

    # Train GBDT models and accumulate predictions
    print("  Training GBDT models...")
    predictions = []
    for model_cfg in siglip_cfg['gbdt_models']:
        model_name = model_cfg['name']
        model_class_name = model_cfg['class']
        params = model_cfg['params']

        # Map class name to actual class
        if model_class_name == 'HistGradientBoostingRegressor':
            model_cls = HistGradientBoostingRegressor
        elif model_class_name == 'GradientBoostingRegressor':
            model_cls = GradientBoostingRegressor
        elif model_class_name == 'CatBoostRegressor':
            model_cls = CatBoostRegressor
        elif model_class_name == 'LGBMRegressor':
            model_cls = LGBMRegressor
        else:
            raise ValueError(f"Unsupported model class: {model_class_name}")

        print(f"    {model_name}...")
        pred = train_gbdt_cv(
            model_cls, params,
            train_feat, test_feat,
            sem_train, sem_test,
            emb_cols,
            target_names=target_names,
            target_max=target_max,
            skip_targets=skip_targets,
            fold_column='fold'
        )
        predictions.append(pred)

    # Average predictions across models
    siglip_pred = np.mean(predictions, axis=0)

    # Build output DataFrame
    siglip_df = test_df.copy()
    # Fill predictions for all targets
    siglip_df[target_names] = siglip_pred
    # Set Clover to 0 (since it's skipped)
    if 'Dry_Clover_g' in skip_targets:
        siglip_df['Dry_Clover_g'] = 0.0
    # Recompute derived targets (GDM and Total) for consistency
    siglip_df['GDM_g'] = siglip_df['Dry_Green_g']
    siglip_df['Dry_Total_g'] = siglip_df['GDM_g'] + siglip_df['Dry_Dead_g']

    return siglip_df