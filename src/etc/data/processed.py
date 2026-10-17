"""Loaders for the shared processed dataset (sessions, splits, arrays) and label subsampling.

Every model goes through these functions, so all of them see exactly the same rows and
the same train/val/test indices from `splits.json`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from etc.config import Config, resolve_path


@dataclass
class ProcessedData:
    sessions: pd.DataFrame
    splits: dict[str, np.ndarray]
    classes: list[str]
    feature_names: list[str]
    root: Path

    @property
    def y(self) -> np.ndarray:
        return self.sessions["label_idx"].to_numpy()

    def features(self) -> np.ndarray:
        return self.sessions[[f"f_{n}" for n in self.feature_names]].to_numpy(dtype=np.float64)

    def seqs(self) -> np.ndarray:
        return np.load(self.root / "seqs.npy")

    def bytes(self, variant: str, kind: str = "bytes") -> np.ndarray:
        """kind: 'bytes' (flat), 'pkt_bytes' or 'pkt_lens' for mask variant none/ip/ip_port."""
        return np.load(self.root / "bytes" / variant / f"{kind}.npy", mmap_mode="r")


def load_processed(cfg: Config) -> ProcessedData:
    root = resolve_path(cfg["paths"]["processed_dir"])
    if not (root / "splits.json").exists():
        raise FileNotFoundError(f"{root}/splits.json not found - run `make preprocess` first")
    meta = json.loads((root / "meta.json").read_text())
    raw = json.loads((root / "splits.json").read_text())
    splits = {k: np.asarray(raw[k], dtype=np.int64) for k in ("train", "val", "test")}
    return ProcessedData(
        sessions=pd.read_parquet(root / "sessions.parquet"),
        splits=splits,
        classes=meta["classes"],
        feature_names=meta["feature_names"],
        root=root,
    )


def subsample_labels(
    train_idx: np.ndarray, y: np.ndarray, fraction: float, seed: int
) -> np.ndarray:
    """Stratified subset of the labelled training indices (RQ2).

    Keeps round(fraction * n_c) samples per class, with at least one per class so every
    class is still represented at 1% labels.
    """
    if fraction >= 1.0:
        return np.sort(train_idx)
    rng = np.random.default_rng(seed)
    out = []
    for c in np.unique(y[train_idx]):
        idx = train_idx[y[train_idx] == c]
        k = max(1, int(round(len(idx) * fraction)))
        out.append(rng.choice(idx, size=k, replace=False))
    return np.sort(np.concatenate(out))
