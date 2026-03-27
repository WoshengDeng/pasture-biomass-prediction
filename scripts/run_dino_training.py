#!/usr/bin/env python3
"""
CSIRO Pasture Biomass Prediction: DINOv3 ViT-Huge Fine-Tuning
Main training script using modular components.
"""

import os
import sys
import json
import warnings
import numpy as np
import torch
from tqdm.auto import tqdm

# Add project root to path if necessary
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CFG
from src.dino.data.data_loader import load_train_data, load_test_data
from src.dino.utils.seed import seed_everything
from src.dino.train.fold_train import train_fold


def main() -> None:
    warnings.filterwarnings('ignore')

    # ============================================================================
    # Performance tweaks
    # ============================================================================
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    # ============================================================================
    # Initial prints
    # ============================================================================
    print("✓ Imports complete")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    try:
        import timm
        print(f"timm version: {timm.__version__}")
    except ImportError:
        print("timm not available for version check")

    # ============================================================================
    # Ensure necessary directories exist
    # ============================================================================
    os.makedirs(CFG.MODEL_DIR, exist_ok=True)
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

    # ============================================================================
    # Reproducibility
    # ============================================================================
    seed_everything()

    # ============================================================================
    # Print configuration summary
    # ============================================================================
    print(f"\n{'='*60}")
    print("CONFIGURATION - VIT_HUGE_PLUS")
    print(f"{'='*60}")
    print(f"Device: {CFG.DEVICE}")
    print(f"Model: {CFG.MODEL_NAME}")
    print(f"Image Size: {CFG.IMG_SIZE}")
    print(f"Batch Size: {CFG.BATCH_SIZE}")
    print(f"Epochs: {CFG.EPOCHS}")
    print(f"Folds: {CFG.N_FOLDS}")

    # ============================================================================
    # Load data
    # ============================================================================
    print(f"\n{'='*60}")
    print("STEP 1: Loading Data")
    print(f"{'='*60}")

    train_df = load_train_data()
    test_df = load_test_data()

    # ============================================================================
    # Train folds
    # ============================================================================
    print(f"\n{'='*60}")
    print("STEP 2: Training DINO HUGE Models")
    print(f"{'='*60}")

    fold_scores = []
    fold_pbar = tqdm(CFG.FOLDS_TO_TRAIN, desc='Training Folds')

    for fold in fold_pbar:
        fold_pbar.set_description(f'Training Fold {fold}')
        score = train_fold(fold, train_df)
        fold_scores.append(score)

        fold_pbar.set_postfix({
            'current_r2': f'{score:.4f}',
            'mean_r2': f'{np.mean(fold_scores):.4f}'
        })

    # ============================================================================
    # Training summary
    # ============================================================================
    print(f"\n{'='*60}")
    print("DINO HUGE TRAINING COMPLETE!")
    print(f"{'='*60}")

    print(f"Fold scores: {fold_scores}")
    print(f"Mean CV R²: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")

    # Save training summary
    summary = {
        'model': CFG.MODEL_NAME,
        'folds': CFG.N_FOLDS,
        'epochs': CFG.EPOCHS,
        'batch_size': CFG.BATCH_SIZE,
        'fold_scores': fold_scores,
        'mean_cv': float(np.mean(fold_scores)),
        'std_cv': float(np.std(fold_scores))
    }

    with open(f'{CFG.OUTPUT_DIR}/training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")

    # Output artifact locations
    print(f"\nModels saved to: {CFG.MODEL_DIR}/")
    for fold in CFG.FOLDS_TO_TRAIN:
        print(f"  - fold{fold}_best.pth")


if __name__ == "__main__":
    main()