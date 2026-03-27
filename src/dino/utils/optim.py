# src/utils/optim.py
import math
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from src.dino.config import CFG


def build_optimizer(model: nn.Module) -> optim.Optimizer:
    """
    Build optimizer with differential learning rates.

    - Lower LR for pretrained backbone (stable fine-tuning)
    - Higher LR for task-specific layers (faster adaptation)

    Args:
        model (nn.Module): BiomassModel

    Returns:
        torch.optim.Optimizer: Configured AdamW optimizer
    """
    backbone_params = list(model.backbone.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]

    # Parameter grouping for fine-tuning strategy
    return optim.AdamW([
        {'params': backbone_params, 'lr': CFG.LR_BACKBONE},
        {'params': head_params, 'lr': CFG.LR_HEAD}
    ], weight_decay=CFG.WD)


def build_scheduler(optimizer: optim.Optimizer, total_steps: int) -> LambdaLR:
    """
    Build learning rate scheduler with linear warmup + cosine decay.

    - Warmup stabilizes early training, especially for large models
    - Cosine decay provides smooth convergence

    Args:
        optimizer (Optimizer): Optimizer instance
        total_steps (int): Total training steps

    Returns:
        LambdaLR: Step-based learning rate scheduler
    """

    def lr_lambda(step):
        warmup_steps = CFG.WARMUP_EPOCHS * (total_steps // CFG.EPOCHS)

        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)