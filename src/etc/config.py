"""YAML configuration loading with layered merging and CLI overrides.

All paths and hyperparameters live in `configs/*.yaml`; code never hard-codes them.
A stage loads `default.yaml`, merges a model config on top, then applies
`key.sub=value` overrides coming from the command line.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

Config = dict[str, Any]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def deep_merge(base: Config, override: Config) -> Config:
    """Recursively merge `override` into a copy of `base` (override wins)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_overrides(cfg: Config, overrides: list[str]) -> Config:
    """Apply dotted `a.b.c=value` overrides; values are parsed as YAML scalars."""
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got: {item!r}")
        dotted, raw = item.split("=", 1)
        keys = dotted.strip().split(".")
        node = out
        for key in keys[:-1]:
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                raise ValueError(f"Cannot override {dotted!r}: {key!r} is not a mapping")
        node[keys[-1]] = yaml.safe_load(raw)
    return out


def load_yaml(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(*names: str, overrides: list[str] | None = None) -> Config:
    """Load `default.yaml` plus the given config files (name or path), then overrides."""
    cfg = load_yaml(CONFIG_DIR / "default.yaml")
    for name in names:
        path = Path(name)
        if not path.exists():
            path = CONFIG_DIR / (name if name.endswith(".yaml") else f"{name}.yaml")
        cfg = deep_merge(cfg, load_yaml(path))
    return apply_overrides(cfg, overrides or [])


def load_dataset_config(cfg: Config) -> Config:
    """Load the dataset definition (pcap glob -> label) referenced by `cfg['dataset']`."""
    path = Path(cfg["dataset"])
    if not path.is_absolute():
        path = CONFIG_DIR / path
    return load_yaml(path)


def resolve_path(path: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
