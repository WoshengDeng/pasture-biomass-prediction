# src/siglip/models/gbdt_models.py
"""
GBDT model training and prediction functions for SigLIP-based features.
"""

from typing import List, Dict, Any, Type

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from src.siglip.features.embedding_engine import SupervisedEmbeddingEngine


def train_gbdt_cv(
    model_cls: Type,
    params: Dict[str, Any],
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    sem_tr: np.ndarray,
    sem_te: np.ndarray,
    emb_cols: List[str],
    target_names: List[str],
    target_max: Dict[str, float],
    skip_targets: List[str] = None,
    fold_column: str = "fold",
) -> np.ndarray:
    """
    Train GBDT models using K-fold cross-validation on precomputed embeddings.

    Args:
        model_cls: GBDT class (e.g., LGBMRegressor, HistGradientBoostingRegressor).
        params (dict): Model hyperparameters.
        train_data (pd.DataFrame): Training data including embeddings, targets, and fold column.
        test_data (pd.DataFrame): Test data embeddings.
        sem_tr (np.ndarray): Semantic features for training data.
        sem_te (np.ndarray): Semantic features for test data.
        emb_cols (List[str]): Columns of embeddings to use as features.
        target_names (List[str]): Ordered list of target column names.
        target_max (Dict[str, float]): Maximum values for each target (for normalization).
        skip_targets (List[str], optional): Target names to skip (not predicted by this model).
        fold_column (str): Column name in train_data indicating fold index.

    Returns:
        np.ndarray: Averaged predictions across folds [n_test_samples, n_targets].
    """
    if skip_targets is None:
        skip_targets = []

    # Convert target max to array in the order of target_names
    target_max_arr = np.array([target_max[t] for t in target_names])
    y_pred_test = np.zeros([len(test_data), len(target_names)])
    n_splits = int(train_data[fold_column].nunique())

    X_train = train_data[emb_cols].values.astype(np.float32)
    X_test = test_data[emb_cols].values.astype(np.float32)
    y_train = train_data[target_names].values.astype(np.float32)

    # Train per fold
    for fold in range(n_splits):
        train_mask = train_data[fold_column] != fold
        X_tr = X_train[train_mask]
        y_tr = y_train[train_mask] / target_max_arr  # Normalize targets
        sem_tr_fold = sem_tr[train_mask]  # Semantic features for current fold

        # Feature transformation using supervised embedding engine
        eng = SupervisedEmbeddingEngine()
        eng.fit(X_tr, y=y_tr, X_semantic=sem_tr_fold)
        x_tr_eng = eng.transform(X_tr, X_semantic=sem_tr_fold)
        x_te_eng = eng.transform(X_test, X_semantic=sem_te)

        # Train separate model for each target
        for k, target in enumerate(target_names):
            if target in skip_targets:
                continue
            model = model_cls(**params)
            model.fit(x_tr_eng, y_tr[:, k])
            # Predict and denormalize
            y_pred_test[:, k] += model.predict(x_te_eng) * target_max_arr[k]

    # Average predictions over folds
    return y_pred_test / n_splits