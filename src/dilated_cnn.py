"""Multi-scale dilated CNN building block."""
import torch
from torch import nn

class MultiScaleDilatedConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(channels, channels // 2, 3, padding=1, dilation=1),
            nn.Conv2d(channels, channels // 2, 3, padding=2, dilation=2),
        ])
        self.fuse = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        features = [branch(x) for branch in self.branches]
        return self.fuse(torch.cat(features, dim=1))
