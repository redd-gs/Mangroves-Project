"""
Model architectures for mangrove coverage classification.

Implements:
- Classifier: simple fully-connected baseline
- CNNClassifier: convolutional baseline
- ViTClassifier: Vision Transformer (proposed approach from the paper)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Simple Fully-Connected Classifier

class Classifier(nn.Module):
    """Simple fully-connected classifier operating on flattened embeddings."""

    def __init__(self, in_channels: int = 64, out_features: int = 6,
                 hidden_dim: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # (B, C, 1, 1)
            nn.Flatten(),                      # (B, C)
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) -> (B, out_features)"""
        return self.net(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# CNN Classifier

class CNNClassifier(nn.Module):
    """Convolutional baseline for spatial pattern recognition."""

    def __init__(self, in_channels: int = 64, out_features: int = 6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



# Vision Transformer

class PatchEmbedding(nn.Module):
    """Split image into non-overlapping patches and project to embedding dim."""

    def __init__(self, in_channels: int = 64, patch_size: int = 16,
                 embed_dim: int = 256, img_size: int = 244):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, n_patches, embed_dim)
        x = self.proj(x)                       # (B, E, H', W')
        x = x.flatten(2).transpose(1, 2)       # (B, N, E)
        return x


class TransformerEncoderBlock(nn.Module):
    """Single Transformer encoder layer with pre-norm."""

    def __init__(self, embed_dim: int = 256, num_heads: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        # Pre-norm MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class ViTClassifier(nn.Module):
    """
    Vision Transformer for mangrove coverage classification.

    Architecture follows the paper:
    - Input: (B, 64, 244, 244) AlphaEarth embedding tensor
    - Patch embedding with positional encoding
    - L Transformer encoder blocks with multi-head self-attention
    - [CLS] token -> MLP head -> 6-class coverage prediction
    """

    def __init__(self, in_channels: int = 64, out_features: int = 6,
                 img_size: int = 244, patch_size: int = 16,
                 embed_dim: int = 256, depth: int = 6,
                 num_heads: int = 8, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size,
                                          embed_dim, img_size)
        n_patches = self.patch_embed.n_patches

        # Learnable [CLS] token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + n_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, out_features),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 64, H, W)  AlphaEarth embedding tensor
        Returns:
            logits: (B, out_features)
        """
        B = x.shape[0]
        x = self.patch_embed(x)                               # (B, N, E)
        cls_tokens = self.cls_token.expand(B, -1, -1)         # (B, 1, E)
        x = torch.cat([cls_tokens, x], dim=1)                 # (B, 1+N, E)
        x = self.pos_drop(x + self.pos_embed)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0]                                     # (B, E)
        return self.head(cls_out)

    def get_last_attention(self, x: torch.Tensor):
        """Return attention weights from the last transformer block (for Grad-CAM)."""
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

        for blk in self.blocks[:-1]:
            x = blk(x)

        # Last block – extract attention weights
        x_norm = self.blocks[-1].norm1(x)
        _, attn_weights = self.blocks[-1].attn(
            x_norm, x_norm, x_norm, need_weights=True)
        return attn_weights

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
