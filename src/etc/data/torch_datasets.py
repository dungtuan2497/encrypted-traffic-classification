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


def tokenize_session(pkt_bytes: np.ndarray, pkt_lens: np.ndarray, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    """[CLS] p1 [SEP] p2 [SEP] ... -> (token ids, packet indices), padded/cut to max_len.

    Packet index: 0 for [CLS] and padding, i+1 for the bytes of packet i and its [SEP].
    """
    from etc.models.transformer import CLS, PAD, SEP

    tokens = np.full(max_len, PAD, dtype=np.int16)
    pidx = np.zeros(max_len, dtype=np.int16)
    tokens[0] = CLS
    pos = 1
    for i, n in enumerate(pkt_lens):
        n = int(n)
        if n == 0 or pos >= max_len:
            break
        take = min(n, max_len - pos)
        tokens[pos : pos + take] = pkt_bytes[i, :take]
        pidx[pos : pos + take] = i + 1
        pos += take
        if pos < max_len:
            tokens[pos] = SEP
            pidx[pos] = i + 1
            pos += 1
    return tokens, pidx


def token_input(data: ProcessedData, variant: str, max_len: int) -> np.ndarray:
    """(N, 2, max_len) int16: token ids and packet indices for the byte Transformer."""
    pkts = np.asarray(data.bytes(variant, "pkt_bytes"))
    lens = np.asarray(data.bytes(variant, "pkt_lens"))
    out = np.empty((len(pkts), 2, max_len), dtype=np.int16)
    for i in range(len(pkts)):
        out[i, 0], out[i, 1] = tokenize_session(pkts[i], lens[i], max_len)
    return out


class LengthBucketSampler(torch.utils.data.Sampler[list[int]]):
    """Batches of similar sequence length (shuffled in chunks) to avoid wasted padding."""

    def __init__(self, lengths: np.ndarray, batch_size: int, seed: int = 0, chunk_batches: int = 50) -> None:
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.chunk = batch_size * chunk_batches
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        perm = self.rng.permutation(len(self.lengths))
        batches = []
        for s in range(0, len(perm), self.chunk):
            chunk = perm[s : s + self.chunk]
            chunk = chunk[np.argsort(self.lengths[chunk], kind="stable")]
            batches += [chunk[i : i + self.batch_size].tolist() for i in range(0, len(chunk), self.batch_size)]
        for j in self.rng.permutation(len(batches)):
            yield batches[j]

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def make_token_loader(
    x: np.ndarray, y: np.ndarray | None, idx: np.ndarray, batch_size: int, shuffle: bool, seed: int = 0
) -> DataLoader:
    """Loader for (N, 2, L) token inputs; length-bucketed when shuffling, sorted otherwise."""
    from etc.models.transformer import PAD

    xs = torch.from_numpy(np.ascontiguousarray(x[idx]).astype(np.int64))
    ys = torch.from_numpy(y[idx]).long() if y is not None else torch.zeros(len(idx), dtype=torch.long)
    ds = TensorDataset(xs, ys)
    lengths = (xs[:, 0] != PAD).sum(1).numpy()
    if shuffle:
        return DataLoader(ds, batch_sampler=LengthBucketSampler(lengths, batch_size, seed))
    order = np.argsort(lengths, kind="stable")
    batches = [order[i : i + batch_size].tolist() for i in range(0, len(order), batch_size)]
    return DataLoader(ds, batch_sampler=batches)


def make_loader(
    x: np.ndarray, y: np.ndarray, idx: np.ndarray, batch_size: int, shuffle: bool, seed: int = 0
) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(np.ascontiguousarray(x[idx])), torch.from_numpy(y[idx]).long())
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g if shuffle else None)
