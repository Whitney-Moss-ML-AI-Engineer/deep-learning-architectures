"""LSTM sequence model for regression/forecasting."""
import torch
from torch import nn

class LSTMForecaster(nn.Module):
    def __init__(self, features, hidden=64, layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(features, hidden, layers, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, output_size)

    def forward(self, x):
        sequence, _ = self.lstm(x)
        return self.head(sequence[:, -1, :])
