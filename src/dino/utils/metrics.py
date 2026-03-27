# src/utils/metrics.py
import numpy as np
import torch.nn as nn


def biomass_loss(preds: Tensor, labels: Tensor) -> Tensor:
    """
    Regression loss for multi-target biomass prediction.

    Uses Smooth L1 (Huber) loss for robustness to outliers,
    which are common in biomass measurements.

    Args:
        preds (Tensor): Model predictions [B, 5]
        labels (Tensor): Ground truth targets [B, 5]

    Returns:
        Tensor: Scalar loss
    """
    huber = nn.SmoothL1Loss(beta=5.0)
    loss = huber(preds, labels)
    return loss


def weighted_r2_score(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Compute weighted R² score across multiple targets.

    - Evaluates per-target R²
    - Applies higher weight to more important targets (e.g., Total biomass)

    Args:
        y_true (np.ndarray): Ground truth [N, 5]
        y_pred (np.ndarray): Predictions [N, 5]

    Returns:
        Tuple[float, np.ndarray]:
            - Weighted R² score
            - Per-target R² scores
    """
    weights = np.array([0.1, 0.1, 0.1, 0.2, 0.5])
    r2_scores = []

    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)

        # Handle degenerate case where variance is zero
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        r2_scores.append(r2)

    r2_scores = np.array(r2_scores)
    weighted = np.sum(r2_scores * weights) / np.sum(weights)

    return weighted, r2_scores