"""
VideoMAE-based classifier for video action anticipation (Track B).

Uses MCG-NJU/videomae-base-finetuned-ssv2 pretrained on Something-Something V2,
which is directly relevant to this task. The classification head is replaced to
match the competition's num_classes.

Forward:
    Input:  (B, T, C, H, W)  — T should be 16 for VideoMAE-base
    VideoMAE ViT backbone -> patch token sequence
    Mean pool over patch tokens -> (B, 768)
    Linear head -> (B, num_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import VideoMAEForVideoClassification, VideoMAEConfig


SSV2_MODEL = "MCG-NJU/videomae-base-finetuned-ssv2"
KINETICS_MODEL = "MCG-NJU/videomae-base"


class VideoMAEClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        model_name: str = SSV2_MODEL,
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        if pretrained:
            self.videomae = VideoMAEForVideoClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            config = VideoMAEConfig(num_labels=num_classes)
            self.videomae = VideoMAEForVideoClassification(config)

        if gradient_checkpointing:
            self.videomae.gradient_checkpointing_enable()

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        """
        video_batch: (B, T, C, H, W) — T=16 matches VideoMAE-base pretraining.
        returns logits: (B, num_classes)
        """
        outputs = self.videomae(pixel_values=video_batch)
        return outputs.logits

    def param_groups(self, backbone_lr: float, head_lr: float) -> list:
        """Return param groups with separate LRs for backbone vs. head."""
        backbone_params = [
            p for n, p in self.named_parameters() if "classifier" not in n
        ]
        head_params = [
            p for n, p in self.named_parameters() if "classifier" in n
        ]
        return [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ]
