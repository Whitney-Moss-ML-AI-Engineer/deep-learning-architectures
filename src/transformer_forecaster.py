"""Compact Transformer encoder for sequence regression."""
import torch
from torch import nn

class TransformerForecaster(nn.Module):
    def __init__(self, features, d_model=64, heads=4, layers=2):
        super().__init__()
        self.projection = nn.Linear(features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        z = self.encoder(self.projection(x))
        return self.head(z[:, -1, :])
