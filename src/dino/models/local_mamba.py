# src/models/local_mamba.py
import torch
import torch.nn as nn


class LocalMambaBlock(nn.Module):
    """
    Lightweight token-mixing block inspired by Mamba-style local modeling.

    Design:
    - Applies LayerNorm + gated modulation
    - Uses depthwise 1D convolution for local token interaction
    - Residual connection for stable training

    Args:
        dim (int): Feature dimension (C)
        kernel_size (int): Convolution kernel size for local context
        dropout (float): Dropout rate
    """

    def __init__(self, dim: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Shape [B, L, C]

        Returns:
            Tensor: Shape [B, L, C]
        """
        shortcut = x

        x = self.norm(x)

        # Gated feature modulation
        g = torch.sigmoid(self.gate(x))
        x = x * g

        # Depthwise convolution over token dimension
        x = x.transpose(1, 2)
        x = self.dwconv(x)
        x = x.transpose(1, 2)

        x = self.proj(x)
        x = self.drop(x)

        return shortcut + x  # Residual connection