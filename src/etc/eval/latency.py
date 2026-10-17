"""Inference latency / throughput (the "real-time" criterion of the review, Section 2.4).

Measured on CPU after warm-up: mean milliseconds per sample at batch size 1, and samples
per second at a large batch (default 256). Works for any `predict(batch) -> labels`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np


def measure_latency(
    predict: Callable[[Any], Any],
    x: np.ndarray,
    warmup: int = 20,
    iters: int = 200,
    batch: int = 256,
) -> dict[str, float]:
    n = len(x)
    rng = np.random.default_rng(0)
    for i in range(warmup):
        predict(x[i % n : i % n + 1])

    idx = rng.integers(0, n, size=iters)
    t0 = time.perf_counter()
    for i in idx:
        predict(x[i : i + 1])
    per_sample_ms = (time.perf_counter() - t0) / iters * 1e3

    big = x[np.arange(batch) % n]
    predict(big)  # warm-up for the large batch
    reps = 5
    t0 = time.perf_counter()
    for _ in range(reps):
        predict(big)
    throughput = batch * reps / (time.perf_counter() - t0)
    return {
        "latency_ms_batch1": per_sample_ms,
        "throughput_per_s": throughput,
        "throughput_batch": batch,
    }
