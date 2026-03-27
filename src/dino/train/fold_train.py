# src/train/fold_train.py
import gc
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dino.config import CFG
from src.dino.data.dataset import BiomassDataset
from src.dino.data.transforms import get_train_transforms, get_val_transforms
from src.dino.models.biomass_model import BiomassModel
from src.dino.utils.optim import build_optimizer, build_scheduler
from src.dino.train.engine import train_one_epoch, validate


def train_fold(fold, train_df):
    """
    Train a single cross-validation fold.

    Workflow:
    - Split data into train/validation based on fold index
    - Build datasets and dataloaders
    - Initialize model, optimizer, and scheduler
    - Run training loop with validation
    - Apply early stopping and save best checkpoint

    Args:
        fold (int): Fold index
        train_df (pd.DataFrame): Full training dataframe with fold assignments

    Returns:
        float: Best validation weighted R² for this fold
    """
    print(f"\n{'=' * 60}")
    print(f"TRAINING FOLD {fold}")
    print(f"{'=' * 60}")

    # Split data for current fold
    train_data = train_df[train_df['fold'] != fold].reset_index(drop=True)
    val_data = train_df[train_df['fold'] == fold].reset_index(drop=True)

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Dataset and dataloaders
    train_dataset = BiomassDataset(train_data, CFG.TRAIN_IMAGE_DIR, get_train_transforms())
    val_dataset = BiomassDataset(val_data, CFG.TRAIN_IMAGE_DIR, get_val_transforms())

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.BATCH_SIZE,
        shuffle=True,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
        drop_last=True  # Ensure consistent batch statistics
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True
    )

    # Model and optimization setup
    model = BiomassModel(CFG.MODEL_NAME, pretrained=True).to(CFG.DEVICE)
    optimizer = build_optimizer(model)

    total_steps = len(train_loader) * CFG.EPOCHS
    scheduler = build_scheduler(optimizer, total_steps)

    # Tracking best performance
    best_r2 = -float('inf')
    best_epoch = 0
    epochs_without_improvement = 0

    epoch_pbar = tqdm(range(CFG.EPOCHS), desc=f'Fold {fold} Epochs')
    for epoch in epoch_pbar:
        print(f"\nEpoch {epoch + 1}/{CFG.EPOCHS}")

        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, CFG.DEVICE)
        val_r2, per_r2 = validate(model, val_loader, CFG.DEVICE)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val R²: {val_r2:.4f}")
        print(
            f"Per-target: Green={per_r2[0]:.3f}, Dead={per_r2[1]:.3f}, Clover={per_r2[2]:.3f}, GDM={per_r2[3]:.3f}, Total={per_r2[4]:.3f}")

        epoch_pbar.set_postfix({
            'loss': f'{train_loss:.4f}',
            'val_r2': f'{val_r2:.4f}',
            'best_r2': f'{best_r2:.4f}'
        })

        # Checkpoint best model based on validation metric
        if val_r2 > best_r2:
            best_r2 = val_r2
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            save_path = f"{CFG.MODEL_DIR}/fold{fold}_best.pth"
            torch.save(model.state_dict(), save_path)

            print(f"✓ Saved best model (R²={best_r2:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s)")

            # Early stopping based on validation plateau
            if epochs_without_improvement >= CFG.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    print(f"\nFold {fold} Best: R²={best_r2:.4f} at epoch {best_epoch}")

    # Explicit memory cleanup (important for large models / CV loops)
    del model, optimizer, scheduler, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()

    return best_r2