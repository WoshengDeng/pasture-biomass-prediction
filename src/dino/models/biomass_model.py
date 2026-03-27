# src/models/biomass_model.py
import torch
import torch.nn as nn
import timm
from src.dino.config import CFG
from src.dino.models.local_mamba import LocalMambaBlock


class BiomassModel(nn.Module):
    """
    Multi-branch regression model for pasture biomass estimation.

    Architecture:
    - Backbone: Vision Transformer (DINOv3) extracting token-level features
    - Dual input streams (left/right image halves)
    - Token-level fusion via lightweight local mixing blocks
    - Global pooling to obtain image-level representation
    - Multi-head regression for base components (Green, Dead, Clover)
    - Derived targets (GDM, Total) computed from base predictions

    Args:
        model_name (str): timm backbone identifier
        pretrained (bool): Whether to load pretrained weights
    """

    def __init__(self, model_name: str, pretrained: bool = True) -> None:
        super().__init__()
        self.model_name = model_name

        # Backbone without classification head, returning token embeddings
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
        )
        nf = self.backbone.num_features
        print(f"✓ Backbone: {model_name}, features={nf}")

        # Token-level fusion to model cross-view interactions
        self.fusion = nn.Sequential(
            LocalMambaBlock(nf, kernel_size=5, dropout=CFG.DROPOUT),
            LocalMambaBlock(nf, kernel_size=5, dropout=CFG.DROPOUT)
        )

        # Aggregate token features into a global representation
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Independent regression heads for base components (non-negative outputs)
        self.head_green = nn.Sequential(
            nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(CFG.DROPOUT),
            nn.Linear(nf // 2, 1), nn.Softplus()
        )
        self.head_dead = nn.Sequential(
            nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(CFG.DROPOUT),
            nn.Linear(nf // 2, 1), nn.Softplus()
        )
        self.head_clover = nn.Sequential(
            nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(CFG.DROPOUT),
            nn.Linear(nf // 2, 1), nn.Softplus()
        )

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        """
        Args:
            left (Tensor): Left image batch
            right (Tensor): Right image batch

        Returns:
            Tensor: Shape [B, 5] with targets ordered as:
                    [Green, Dead, Clover, GDM, Total]
        """
        x_l = self.backbone(left)
        x_r = self.backbone(right)

        # Concatenate token sequences from both views
        x_cat = torch.cat([x_l, x_r], dim=1)

        # Fuse local information across tokens
        x_fused = self.fusion(x_cat)

        # Pool token dimension to obtain global feature vector
        x_pool = self.pool(x_fused.transpose(1, 2)).flatten(1)

        # Base component predictions
        green = self.head_green(x_pool)
        dead = self.head_dead(x_pool)
        clover = self.head_clover(x_pool)

        # Derived targets (enforcing physical relationships)
        gdm = green + clover
        total = gdm + dead

        return torch.cat([green, dead, clover, gdm, total], dim=1)