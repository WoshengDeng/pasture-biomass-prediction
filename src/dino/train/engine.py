# src/train/engine.py
import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm
import numpy as np
from src.dino.config import CFG
from src.dino.utils.metrics import biomass_loss, weighted_r2_score

# AMP gradient scaling for stable mixed precision training
scaler = GradScaler()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    device: torch.device
) -> float:
    """
    Run one training epoch.

    Includes:
    - Mixed precision training (AMP)
    - Gradient clipping for stability
    - Step-based LR scheduling

    Args:
        model (nn.Module): Model to train
        loader (DataLoader): Training data loader
        optimizer (Optimizer): Optimizer
        scheduler (LRScheduler): Learning rate scheduler
        device (torch.device): Training device

    Returns:
        float: Average training loss
    """
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc='Training')
    for i, (left, right, labels) in enumerate(pbar):
        left = left.to(device)
        right = right.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward + loss under mixed precision
        with autocast():
            preds = model(left, right)
            loss = biomass_loss(preds, labels)

        # Backprop with gradient scaling
        scaler.scale(loss).backward()

        # Stabilize training before optimizer step
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.CLIP_GRAD_NORM)

        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{total_loss / (i + 1):.4f}'})

    return total_loss / len(loader)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Tuple[float, np.ndarray]:
    """
    Run validation loop and compute evaluation metrics.

    - Disables gradient computation for efficiency
    - Uses AMP for faster inference
    - Aggregates predictions across the full dataset
    - Computes weighted R² as the primary metric

    Args:
        model (nn.Module): Trained model
        loader (DataLoader): Validation data loader
        device (torch.device): Inference device

    Returns:
        Tuple[float, np.ndarray]:
            - Weighted R² score
            - Per-target R² scores
    """
    model.eval()
    all_preds = []
    all_labels = []

    for left, right, labels in tqdm(loader, desc='Validating'):
        left = left.to(device)
        right = right.to(device)

        # Inference under mixed precision
        with autocast():
            preds = model(left, right)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())

    # Concatenate batch-level outputs
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Compute evaluation metrics
    weighted_r2, per_target_r2 = weighted_r2_score(all_labels, all_preds)

    return weighted_r2, per_target_r2