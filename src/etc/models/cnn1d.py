"""1D-CNN on raw session bytes (Wang et al. 2017; review Section 4.2.2).

784 bytes (scaled to [0, 1]) -> Conv1d(1->32, k25) -> ReLU -> MaxPool(3)
-> Conv1d(32->64, k25) -> ReLU -> MaxPool(3) -> FC(1024) -> Dropout(0.5) -> FC(classes).
'same' padding and pool size 3 follow the original end-to-end 1D-CNN.
"""

from __future__ import annotations

import torch
from torch import nn


class CNN1D(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_len: int = 784,
        channels: tuple[int, int] = (32, 64),
        kernel_size: int = 25,
        fc_dim: int = 1024,
        dropout: float = 0.5,
        pool: int = 3,
    ) -> None:
        super().__init__()
        c1, c2 = channels
        self.features = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(pool),
            nn.Conv1d(c1, c2, kernel_size, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(pool),
        )
        out_len = input_len // pool // pool
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c2 * out_len, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, input_len) float in [0, 1] -> logits (B, num_classes)."""
        return self.classifier(self.features(x))
