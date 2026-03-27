# src/siglip/models/model_loader.py
"""
Model loading utilities for trained SigLIP GBDT models.

Provides functions to load saved GBDT models and feature engineering
components for inference.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Union, Any

import joblib
import numpy as np

from src.siglip.features.embedding_engine import SupervisedEmbeddingEngine


def load_gbdt_model(model_path: Union[str, Path]) -> Any:
    """
    Load a saved GBDT model.

    Supports both joblib and pickle formats.

    Args:
        model_path (Union[str, Path]): Path to the saved model file.

    Returns:
        Any: Loaded model object (e.g., LGBMRegressor, CatBoostRegressor, etc.).

    Raises:
        FileNotFoundError: If model file does not exist.
        ValueError: If file format is not supported.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Try joblib first (common for scikit-learn and LightGBM)
    try:
        return joblib.load(model_path)
    except Exception:
        pass

    # Fallback to pickle (for CatBoost, etc.)
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load model from {model_path}: {e}")


def load_embedding_engine(engine_path: Union[str, Path]) -> SupervisedEmbeddingEngine:
    """
    Load a saved SupervisedEmbeddingEngine.

    Args:
        engine_path (Union[str, Path]): Path to the saved engine file.

    Returns:
        SupervisedEmbeddingEngine: Loaded feature engineering engine.

    Raises:
        FileNotFoundError: If engine file does not exist.
    """
    engine_path = Path(engine_path)
    if not engine_path.exists():
        raise FileNotFoundError(f"Engine file not found: {engine_path}")

    try:
        return joblib.load(engine_path)
    except Exception:
        with open(engine_path, 'rb') as f:
            return pickle.load(f)


def load_all_models(
    output_dir: Union[str, Path],
    model_names: List[str],
    target_names: List[str],
    skip_targets: List[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Load all saved GBDT models and corresponding feature engines for each target.

    This function assumes a specific directory structure:
        output_dir/
            {model_name}/
                target_{target_name}.pkl   (model file)
                engine_{target_name}.pkl   (feature engine)

    Args:
        output_dir (Union[str, Path]): Directory where models are saved.
        model_names (List[str]): List of model names (e.g., ['histgb', 'gb', 'catboost', 'lgbm']).
        target_names (List[str]): List of target names.
        skip_targets (List[str], optional): Target names to skip (not used). Defaults to None.

    Returns:
        Dict[str, Dict[str, Any]]: Nested dict of models and engines:
            {model_name: {'models': {target: model}, 'engines': {target: engine}}}
    """
    output_dir = Path(output_dir)
    if skip_targets is None:
        skip_targets = []

    all_models = {}
    for model_name in model_names:
        model_dir = output_dir / model_name
        if not model_dir.exists():
            print(f"Warning: Model directory {model_dir} not found, skipping {model_name}")
            continue

        models = {}
        engines = {}
        for target in target_names:
            if target in skip_targets:
                continue
            model_file = model_dir / f"target_{target}.pkl"
            engine_file = model_dir / f"engine_{target}.pkl"
            if model_file.exists() and engine_file.exists():
                models[target] = load_gbdt_model(model_file)
                engines[target] = load_embedding_engine(engine_file)
            else:
                print(f"Warning: Model or engine for {target} not found in {model_dir}")

        if models:
            all_models[model_name] = {
                'models': models,
                'engines': engines
            }
    return all_models