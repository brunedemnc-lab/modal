"""
DINOv2 + Temporal Attention classifier for video action anticipation (Track B).

Designed for short clips (4 frames): strong per-frame features from DINOv2,
lightweight temporal transformer to reason across frames.

Forward:
    Input:  (B, T, C, H, W)
    DINOv2 ViT-B/14 per frame (shared weights) -> (B, T, 768)
    + learnable temporal positional embedding
    Temporal TransformerEncoder (2 layers) -> (B, T, 768)
    Mean pool + LayerNorm -> (B, 768)
    Linear head -> (B, num_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm


class DINOv2TemporalClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        num_temporal_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_frames: int = 16,
    ) -> None:
        super().__init__()

        self.backbone = timm.create_model(
            "vit_base_patch14_dinov2.lvd142m",
            pretrained=pretrained,
            num_classes=0,  # remove classification head, return CLS token
        )
        feature_dim: int = self.backbone.num_features  # 768 for ViT-B/14

        # Learnable temporal positional embedding (supports up to max_frames)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames, feature_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-norm: more stable for fine-tuning
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_temporal_layers
        )
        self.norm = nn.LayerNorm(feature_dim)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        """
        video_batch: (B, T, C, H, W)
        returns logits: (B, num_classes)
        """
        B, T, C, H, W = video_batch.shape

        # Extract per-frame CLS features with shared DINOv2 backbone
        frames = video_batch.reshape(B * T, C, H, W)
        frame_features = self.backbone(frames)          # (B*T, 768)
        frame_features = frame_features.view(B, T, -1)  # (B, T, 768)

        # Add temporal positional embedding
        frame_features = frame_features + self.pos_embed[:, :T, :]

        # Reason across frames
        temporal_out = self.temporal_transformer(frame_features)  # (B, T, 768)

        # Mean pool over time, normalize, classify
        pooled = self.norm(temporal_out.mean(dim=1))   # (B, 768)
        logits = self.classifier(self.drop(pooled))
        return logits

    def param_groups(self, backbone_lr: float, head_lr: float) -> list:
        """Separate LRs for pretrained backbone vs. new temporal head."""
        backbone_params = [p for n, p in self.named_parameters() if "backbone" in n]
        head_params = [p for n, p in self.named_parameters() if "backbone" not in n]
        return [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ]
