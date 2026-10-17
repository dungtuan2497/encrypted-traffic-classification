"""Bidirectional GRU on directed packet-size sequences (simplified FS-Net; review Section 4.2.3).

Input (B, 32, 2): [size * direction / 1500, log1p(IAT ms)]. A 2-layer BiGRU encodes the
sequence; the final hidden states of both directions (top layer) are concatenated and fed
to a linear classifier. Padding is left as zeros (no packing) for simplicity.
"""

from __future__ import annotations

import torch
from torch import nn


class BiGRU(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 2,
        hidden: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            in_channels, hidden, num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)  # h: (num_layers * 2, B, hidden)
        last = torch.cat([h[-2], h[-1]], dim=-1)
        return self.head(last)
