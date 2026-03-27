#!/usr/bin/env python
"""
Script to train SigLIP-based GBDT models for biomass prediction.

This script loads training data, computes SigLIP embeddings and semantic features,
and trains multiple GBDT models using cross-validation. Trained models and feature
engines are saved for later inference.
"""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
import yaml

# SigLIP components
from src.siglip.config import SiglipConfig
from src.siglip.features.embedding_extractor import compute_siglip_embeddings, generate_semantic_features
from src.siglip.features.embedding_engine import SupervisedEmbeddingEngine
from src.siglip.utils.helpers import prepare_image_paths, add_embeddings_to_df

# GBDT model classes
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Mapping from class name to actual class
MODEL_CLASSES = {
    'HistGradientBoostingRegressor': HistGradientBoostingRegressor,
    'GradientBoostingRegressor': GradientBoostingRegressor,
    'CatBoostRegressor': CatBoostRegressor,
    'LGBMRegressor': LGBMRegressor,
}


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def train_and_save_models(
    config: SiglipConfig,
    train_data: pd.DataFrame,
    embeddings: np.ndarray,
    semantic_features: np.ndarray,
    logger: logging.Logger,
) -> None:
    """
    Train GBDT models for each target and save them to disk.

    Args:
        config: SigLIP configuration.
        train_data: DataFrame with columns: image_path, fold, and target columns.
        embeddings: SigLIP embeddings for training images (N, D).
        semantic_features: Semantic features for training images (N, S).
        logger: Logger instance.
    """
    # Prepare data
    target_names = config.target_names
    target_max = np.array([config.target_max[t] for t in target_names])
    skip_targets = config.skip_targets
    fold_column = config.fold_column
    emb_cols = [f"emb{i}" for i in range(embeddings.shape[1])]

    X = embeddings
    y = train_data[target_names].values.astype(np.float32)
    folds = train_data[fold_column].values
    n_folds = config.n_folds

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # For each GBDT model defined in config
    for model_cfg in config.gbdt_models:
        model_name = model_cfg['name']
        model_class_name = model_cfg['class']
        params = model_cfg['params']

        if model_class_name not in MODEL_CLASSES:
            logger.error(f"Unknown model class: {model_class_name}")
            continue

        model_cls = MODEL_CLASSES[model_class_name]
        model_dir = output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Training {model_name}...")

        # For each fold
        for fold_idx in range(n_folds):
            logger.info(f"  Fold {fold_idx}")
            # Split train/validation within this fold
            train_mask = folds != fold_idx
            X_tr = X[train_mask]
            y_tr = y[train_mask]
            sem_tr = semantic_features[train_mask]

            # Normalize targets
            y_tr_norm = y_tr / target_max

            # For each target (skip some if needed)
            for k, target in enumerate(target_names):
                if target in skip_targets:
                    logger.debug(f"    Skipping target {target}")
                    continue

                logger.info(f"    Target: {target}")

                # Feature transformation engine
                eng = SupervisedEmbeddingEngine(
                    pca_n_components=config.pca_n_components,
                    pls_n_components=config.pls_n_components,
                    gmm_n_components=config.gmm_n_components,
                    random_state=config.random_state,
                )
                eng.fit(X_tr, y=y_tr_norm[:, k], X_semantic=sem_tr)

                # Transform features
                X_tr_eng = eng.transform(X_tr, X_semantic=sem_tr)

                # Train GBDT model
                model = model_cls(**params)
                model.fit(X_tr_eng, y_tr_norm[:, k])

                # Save engine and model
                engine_path = model_dir / f"fold{fold_idx}_target_{target}_engine.pkl"
                model_path = model_dir / f"fold{fold_idx}_target_{target}.pkl"

                joblib.dump(eng, engine_path)
                joblib.dump(model, model_path)

                logger.debug(f"      Saved engine to {engine_path}")
                logger.debug(f"      Saved model to {model_path}")

        logger.info(f"Finished training {model_name}.")


def main():
    parser = argparse.ArgumentParser(description="Train SigLIP GBDT models")
    parser.add_argument('--config', type=str, default='configs/siglip_train.yaml',
                        help='Path to configuration YAML file')
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("Loading configuration...")
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Convert to SiglipConfig dataclass (if needed, we can also use dict directly)
    # Here we'll assume config_dict is already structured according to SiglipConfig fields.
    # For simplicity, we'll use the dict directly but will instantiate SiglipConfig for validation.
    config = SiglipConfig(**config_dict)

    logger.info(f"Configuration loaded. Output directory: {config.output_dir}")

    # Set random seeds
    np.random.seed(config.seed)
    # torch seeds not needed as we only use CPU for GBDT

    # Load training split data
    logger.info("Loading training split...")
    train_split = pd.read_csv(config.split_path)
    # Remove any existing embedding columns
    cols_keep = [c for c in train_split.columns if not c.startswith('emb')]
    train_split = train_split[cols_keep]

    # Prepare image paths (make absolute)
    train_split = prepare_image_paths(train_split, config.data_path, subdir='train')

    # Compute SigLIP embeddings
    logger.info("Computing SigLIP embeddings for training images...")
    train_emb = compute_siglip_embeddings(
        model_path=config.siglip_model_path,
        df=train_split,
        img_dir=config.data_path,
        device=config.device,
        patch_size=config.patch_size,
        overlap=config.overlap,
        embedding_dim=config.embedding_dim,
    )

    # Add embeddings to DataFrame (optional, but useful for debugging)
    train_feat = add_embeddings_to_df(train_split, train_emb, prefix='emb')
    emb_cols = [f"emb{i}" for i in range(train_emb.shape[1])]

    # Generate semantic features
    logger.info("Generating semantic features...")
    all_emb = train_emb  # Only training data available
    all_sem = generate_semantic_features(all_emb, config.siglip_model_path, device=config.device)
    sem_train = all_sem[:len(train_split)]

    # Train and save models
    train_and_save_models(config, train_feat, train_emb, sem_train, logger)

    logger.info("Training completed successfully.")


if __name__ == "__main__":
    main()