
"""Baseline CNN for OceanEmbed."""
import torch
import torch.nn as nn


class OceanCNN(nn.Module):
    def __init__(self, in_channels=5, out_depths=15, hidden=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Conv2d(hidden, out_depths, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)
        out = self.head(features)
        return out


def masked_mse_loss(pred, target, mask):
    mask_expanded = mask.unsqueeze(1).expand_as(pred)
    diff2 = (pred - target) ** 2
    diff2_masked = diff2[mask_expanded]
    return diff2_masked.mean()
