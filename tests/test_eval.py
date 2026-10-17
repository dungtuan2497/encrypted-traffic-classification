"""Tests for label subsampling (RQ2), metrics and latency helpers."""

import numpy as np

from etc.data.processed import subsample_labels
from etc.eval.latency import measure_latency
from etc.eval.metrics import classification_metrics, run_name


def test_subsample_labels_stratified_and_deterministic() -> None:
    y = np.array([0] * 900 + [1] * 90 + [2] * 10)
    train = np.arange(1000)
    sub = subsample_labels(train, y, 0.1, seed=1)
    assert np.bincount(y[sub]).tolist() == [90, 9, 1]
    assert np.array_equal(sub, subsample_labels(train, y, 0.1, seed=1))
    tiny = subsample_labels(train, y, 0.01, seed=1)
    assert np.bincount(y[tiny]).tolist() == [9, 1, 1]  # every class keeps >= 1 sample
    assert np.array_equal(subsample_labels(train, y, 1.0, seed=1), train)


def test_classification_metrics() -> None:
    m = classification_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), ["a", "b"])
    assert m["accuracy"] == 0.75
    assert m["confusion_matrix"] == [[1, 1], [0, 2]]
    assert set(m["per_class_f1"]) == {"a", "b"}
    assert run_name("rf", 42, 0.05) == "rf_lf0.05_seed42"


def test_measure_latency() -> None:
    out = measure_latency(lambda x: x.sum(axis=1), np.ones((10, 4)), warmup=2, iters=5, batch=8)
    assert out["latency_ms_batch1"] > 0 and out["throughput_per_s"] > 0
