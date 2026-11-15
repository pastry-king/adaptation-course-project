"""Univariate LSTM with scalar next-month output."""

from torch import nn


class CreditLSTM(nn.Module):
    def __init__(self, hidden: int, layers: int, dropout: float = 0.0):
        super().__init__()
        if hidden < 1 or layers < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid LSTM architecture")
        self.lstm = nn.LSTM(1, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.output = nn.Linear(hidden, 1)

    def forward(self, x):
        sequence, _ = self.lstm(x)
        return self.output(sequence[:, -1, :])
