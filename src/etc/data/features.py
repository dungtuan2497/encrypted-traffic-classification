"""Hand-crafted flow statistics and packet-size sequences (review Sections 3.2, 4.2.3).

(a) Statistical features for traditional ML (Random Forest / XGBoost): counts, bytes,
    duration, packet-size and inter-arrival-time statistics, per direction and overall.
    IP addresses and ports are deliberately excluded.
(b) Directed packet-size sequence (size x direction) plus log-scaled inter-arrival time,
    used by the GRU (a simplified FS-Net input).
All values are computed over the first `max_packets_per_session` kept packets.
"""

from __future__ import annotations

import numpy as np

SCOPES = ("all", "fwd", "bwd")
SIZE_STATS = ("min", "max", "mean", "std", "p25", "p50", "p75")
IAT_STATS = ("min", "max", "mean", "std")


def feature_names() -> list[str]:
    names: list[str] = []
    for s in SCOPES:
        names += [f"{s}_n_pkts", f"{s}_n_bytes", f"{s}_duration"]
        names += [f"{s}_size_{k}" for k in SIZE_STATS]
        names += [f"{s}_iat_{k}" for k in IAT_STATS]
    names += ["ratio_pkts_fwd", "ratio_bytes_fwd", "is_tcp"]
    return names


FEATURE_NAMES = feature_names()


def _size_stats(x: np.ndarray) -> list[float]:
    if x.size == 0:
        return [0.0] * len(SIZE_STATS)
    p25, p50, p75 = np.percentile(x, [25, 50, 75])
    return [x.min(), x.max(), x.mean(), x.std(), p25, p50, p75]


def _iat_stats(ts: np.ndarray) -> list[float]:
    if ts.size < 2:
        return [0.0] * len(IAT_STATS)
    iat = np.diff(ts)
    return [iat.min(), iat.max(), iat.mean(), iat.std()]


def statistical_features(
    ts: np.ndarray, lens: np.ndarray, dirs: np.ndarray, proto: int
) -> np.ndarray:
    """Feature vector aligned with FEATURE_NAMES (float64)."""
    ts = np.asarray(ts, dtype=np.float64)
    lens = np.asarray(lens, dtype=np.float64)
    dirs = np.asarray(dirs)
    feats: list[float] = []
    for scope in SCOPES:
        sel = slice(None) if scope == "all" else (dirs > 0 if scope == "fwd" else dirs < 0)
        t, x = ts[sel], lens[sel]
        duration = float(t.max() - t.min()) if t.size else 0.0
        feats += [float(x.size), float(x.sum()), duration]
        feats += _size_stats(x)
        feats += _iat_stats(t)
    n_fwd, n_all = float((dirs > 0).sum()), float(dirs.size)
    b_fwd, b_all = float(lens[dirs > 0].sum()), float(lens.sum())
    feats += [
        n_fwd / n_all if n_all else 0.0,
        b_fwd / b_all if b_all else 0.0,
        1.0 if proto == 6 else 0.0,
    ]
    return np.asarray(feats, dtype=np.float64)


def size_sequence(
    ts: np.ndarray, lens: np.ndarray, dirs: np.ndarray, seq_len: int
) -> np.ndarray:
    """(seq_len, 2) float32: [size * direction, log1p(inter-arrival in ms)], zero-padded."""
    out = np.zeros((seq_len, 2), dtype=np.float32)
    n = min(seq_len, len(lens))
    if n == 0:
        return out
    ts = np.asarray(ts[:n], dtype=np.float64)
    out[:n, 0] = np.asarray(lens[:n], dtype=np.float32) * np.asarray(dirs[:n], dtype=np.float32)
    iat_ms = np.diff(ts, prepend=ts[0]) * 1e3
    out[:n, 1] = np.log1p(np.clip(iat_ms, 0, None)).astype(np.float32)
    return out
