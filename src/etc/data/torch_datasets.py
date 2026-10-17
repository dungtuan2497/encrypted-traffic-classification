"""Model-specific input tensors built from the shared processed arrays.

  * bytes  -> (N, 1, max_bytes) float32 in [0, 1]          (1D-CNN)
  * seq    -> (N, seq_len, 2) float32, size scaled by 1/1500 (BiGRU)
  * tokens -> see etc.models.transformer (Phase 4)
Inputs are materialised once as tensors (the dataset fits in memory easily).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from etc.data.processed import ProcessedData

MTU = 1500.0


def bytes_input(data: ProcessedData, variant: str, max_bytes: int) -> np.ndarray:
    flat = np.asarray(data.bytes(variant, "bytes"))[:, :max_bytes]
    return (flat.astype(np.float32) / 255.0)[:, None, :]


def seq_input(data: ProcessedData) -> np.ndarray:
    seqs = data.seqs().astype(np.float32).copy()
    seqs[..., 0] /= MTU
    return seqs


def make_loader(
    x: np.ndarray, y: np.ndarray, idx: np.ndarray, batch_size: int, shuffle: bool, seed: int = 0
) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(np.ascontiguousarray(x[idx])), torch.from_numpy(y[idx]).long())
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g if shuffle else None)
