# inference/dino_inference.py
"""
DINO inference module for biomass prediction.

Provides functions to run inference with trained DINO models, including
loading checkpoints, processing test images, and applying post-processing heuristics.
"""

import gc
import warnings
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from tqdm.auto import tqdm

# Import DINO components
from src.dino.models.biomass_model import BiomassModel
from src.dino.data.dataset import TestDataset, collate_fn
from src.dino.utils.seed import seed_everything

warnings.filterwarnings('ignore')


@torch.no_grad()
def predict_dino(model: torch.nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    """
    Run inference using a DINO model on a DataLoader.

    Args:
        model: Trained DINO model.
        loader: DataLoader providing batches of (left, right) images.
        device: Device to run inference on ("cpu" or "cuda").

    Returns:
        Predictions stacked as [N, 5] array (Green, Dead, Clover, GDM, Total).
    """
    model.eval()
    preds_all = []

    for lefts, rights, _ in tqdm(loader, desc="DINO Inference"):
        lefts = lefts.to(device)
        rights = rights.to(device)

        with autocast():
            pred = model(lefts, rights)

        preds_all.append(pred.cpu().numpy())

    return np.vstack(preds_all)


def run_dino_inference(
    config: Dict[str, Any],
    test_df: pd.DataFrame,
    device: str = "cuda",
) -> pd.DataFrame:
    """
    Run full DINO inference on test images.

    Args:
        config: Configuration dictionary with DINO-related keys:
            - model_name: DINO backbone name.
            - img_size: Image size.
            - batch_size: Batch size for DataLoader.
            - n_folds: Number of cross‑validation folds.
            - models_dir: Path to directory containing fold checkpoints.
            - postprocess: Dict with postprocessing parameters.
        test_df: DataFrame with column 'image_path' (relative to data_path).
        device: Device for inference.

    Returns:
        DataFrame with predictions for each test image, containing columns:
            image_path, Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g.
    """
    data_path = Path(config['data_path'])
    dino_cfg = config['dino']
    postprocess_cfg = dino_cfg.get('postprocess', {})

    # Prepare dataset and loader
    test_dataset = TestDataset(test_df, data_path, img_size=dino_cfg['img_size'])
    test_loader = DataLoader(
        test_dataset,
        batch_size=dino_cfg['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    all_fold_preds = []
    models_dir = Path(dino_cfg['models_dir'])

    # Iterate over folds
    for fold in range(dino_cfg['n_folds']):
        model_path = models_dir / f"fold{fold}_best.pth"
        if not model_path.exists():
            print(f"  Fold {fold} not found, skipping...")
            continue

        print(f"  Loading fold {fold}...")
        model = BiomassModel(dino_cfg['model_name'], pretrained=False).to(device)
        state_dict = torch.load(model_path, map_location=device)

        # Handle DataParallel saved weights
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        fold_preds = predict_dino(model, test_loader, device)
        all_fold_preds.append(fold_preds)

        # Clean up
        del model, state_dict
        gc.collect()
        torch.cuda.empty_cache()

    # Average predictions across folds
    dino_preds = np.mean(all_fold_preds, axis=0)
    print(f"DINO predictions shape: {dino_preds.shape}")

    # Build output DataFrame
    dino_df = test_df.copy()
    dino_df['Dry_Green_g'] = dino_preds[:, 0]
    dino_df['Dry_Dead_g'] = dino_preds[:, 1]
    dino_df['Dry_Clover_g'] = dino_preds[:, 2]

    # Apply post-processing heuristics
    clover_scale = postprocess_cfg.get('clover_scale', 1.0)
    dino_df['Dry_Clover_g'] *= clover_scale

    # Dead adjustment
    dead_adjust = postprocess_cfg.get('dead_adjust', {})
    if dead_adjust:
        high_thresh = dead_adjust.get('threshold_high', 20.0)
        high_factor = dead_adjust.get('factor_high', 1.1)
        low_thresh = dead_adjust.get('threshold_low', 10.0)
        low_factor = dead_adjust.get('factor_low', 0.9)

        for i in range(len(dino_df)):
            if dino_df.loc[i, 'Dry_Dead_g'] > high_thresh:
                dino_df.loc[i, 'Dry_Dead_g'] *= high_factor
            elif dino_df.loc[i, 'Dry_Dead_g'] < low_thresh:
                dino_df.loc[i, 'Dry_Dead_g'] *= low_factor

    # Recompute derived targets
    dino_df['GDM_g'] = dino_df['Dry_Green_g'] + dino_df['Dry_Clover_g']
    dino_df['Dry_Total_g'] = dino_df['GDM_g'] + dino_df['Dry_Dead_g']

    # Clip negative values
    for col in ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']:
        dino_df[col] = dino_df[col].clip(lower=0.0)

    return dino_df