"""Evaluation metrics (review Section 2.4) and result persistence.

Macro-F1 is the primary metric because classes are imbalanced; accuracy, macro
precision/recall, per-class F1 and the confusion matrix are reported alongside.
Each run is saved as one JSON file in `results/` (model, config, seed, metrics, timing,
git commit) - the report only ever reads numbers from these files.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from etc.config import Config, resolve_path
from etc.utils import git_commit_hash


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict[str, Any]:
    labels = np.arange(len(classes))
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f1),
        "per_class_f1": {c: float(v) for c, v in zip(classes, per_class)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def plot_confusion_matrix(cm: list[list[int]], classes: list[str], title: str, path: Path) -> None:
    """Row-normalised confusion matrix heatmap (recall per true class on the diagonal)."""
    m = np.asarray(cm, dtype=np.float64)
    norm = m / np.clip(m.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if norm[i, j] >= 0.01:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_name(model: str, seed: int, label_fraction: float = 1.0, tag: str | None = None) -> str:
    parts = [model]
    if tag:
        parts.append(tag)
    parts += [f"lf{label_fraction:g}", f"seed{seed}"]
    return "_".join(parts)


def save_result(name: str, cfg: Config, payload: dict[str, Any]) -> Path:
    """Write results/<name>.json with config, git commit and timestamp."""
    out_dir = resolve_path(cfg["paths"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "run": name,
        **payload,
        "config": cfg,
        "git_commit": git_commit_hash(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(record, indent=2, default=_json_default))
    return path


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serialisable: {type(o)}")
