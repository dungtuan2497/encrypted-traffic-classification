"""Smoke tests for Phase 0 infrastructure: config loading and seeding."""

import random

import numpy as np
import pytest

from etc.config import apply_overrides, deep_merge, load_config
from etc.utils import get_device, set_seed, timer


def test_deep_merge_overrides_nested_keys() -> None:
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    merged = deep_merge(base, {"a": {"b": 10}})
    assert merged == {"a": {"b": 10, "c": 2}, "d": 3}
    assert base["a"]["b"] == 1  # base untouched


def test_apply_overrides_parses_yaml_scalars() -> None:
    cfg = apply_overrides({"x": {"y": 1}}, ["x.y=0.5", "x.z=true", "w=null"])
    assert cfg == {"x": {"y": 0.5, "z": True}, "w": None}
    with pytest.raises(ValueError):
        apply_overrides({}, ["no_equals_sign"])


def test_load_config_merges_model_config() -> None:
    cfg = load_config("cnn", overrides=["preprocess.max_sessions_per_class=200"])
    assert cfg["seed"] == 42
    assert cfg["model"]["name"] == "cnn1d"
    assert cfg["preprocess"]["max_sessions_per_class"] == 200
    assert abs(sum(cfg["split"].values()) - 1.0) < 1e-9


def test_set_seed_is_reproducible() -> None:
    set_seed(123)
    a = (random.random(), np.random.rand())
    set_seed(123)
    b = (random.random(), np.random.rand())
    assert a == b


def test_get_device_and_timer() -> None:
    assert get_device().type in {"cuda", "mps", "cpu"}
    with timer() as t:
        sum(range(1000))
    assert t["seconds"] >= 0
