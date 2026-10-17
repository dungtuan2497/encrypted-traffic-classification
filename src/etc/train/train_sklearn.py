"""Train/evaluate tree-ensemble baselines on statistical features (Phase 2, RQ1/RQ2).

Hyperparameters are chosen by grid search on the validation split (models are fitted on
the training split only); the selected model is then evaluated once on the test split.
Feature importances illustrate the interpretability advantage of traditional ML that
the review points out.
"""

from __future__ import annotations

import itertools
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight

from etc.config import Config, resolve_path
from etc.data.processed import load_processed, subsample_labels
from etc.eval.latency import measure_latency
from etc.eval.metrics import classification_metrics, macro_f1, plot_confusion_matrix, run_name, save_result
from etc.models.rf import build_rf, build_xgb, tree_node_count


def grid_points(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = sorted(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def plot_feature_importance(names: list[str], importances: np.ndarray, top_k: int, title: str, path) -> list[dict]:
    order = np.argsort(importances)[::-1][:top_k]
    top = [{"feature": names[i], "importance": float(importances[i])} for i in order]
    fig, ax = plt.subplots(figsize=(7, 0.35 * top_k + 1))
    ax.barh([t["feature"] for t in top][::-1], [t["importance"] for t in top][::-1], color="#4C72B0")
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return top


def train_sklearn(cfg: Config, model_name: str, seed: int, label_fraction: float, log: Any) -> dict[str, Any]:
    data = load_processed(cfg)
    X, y = data.features(), data.y
    tr = subsample_labels(data.splits["train"], y, label_fraction, seed)
    va, te = data.splits["val"], data.splits["test"]
    log.info(f"{model_name}: train={len(tr):,} (label_fraction={label_fraction:g}) val={len(va):,} test={len(te):,}")

    if model_name == "rf":
        base = {k: v for k, v in cfg["model"].items() if k not in ("grid", "name")}
        grid = cfg["model"]["grid"]
        build = lambda p: build_rf({**base, **p}, seed)  # noqa: E731
        fit_kwargs = lambda: {}  # noqa: E731
    elif model_name == "xgb":
        base = {k: v for k, v in cfg["xgb"].items() if k != "grid"}
        grid = cfg["xgb"]["grid"]
        build = lambda p: build_xgb({**base, **p}, seed, len(data.classes))  # noqa: E731
        fit_kwargs = lambda: {"sample_weight": compute_sample_weight("balanced", y[tr])}  # noqa: E731
    else:
        raise ValueError(model_name)

    best: dict[str, Any] = {"val_macro_f1": -1.0}
    search = []
    grid_start = time.perf_counter()
    for params in grid_points(grid):
        model = build(params)
        t0 = time.perf_counter()
        model.fit(X[tr], y[tr], **fit_kwargs())
        fit_s = time.perf_counter() - t0
        f1 = macro_f1(y[va], model.predict(X[va]))
        search.append({"params": params, "val_macro_f1": f1, "fit_seconds": fit_s})
        log.info(f"  {params} -> val macro-F1 {f1:.4f} ({fit_s:.1f}s)")
        if f1 > best["val_macro_f1"]:
            best = {"val_macro_f1": f1, "params": params, "model": model, "fit_seconds": fit_s}
    grid_s = time.perf_counter() - grid_start

    model = best["model"]
    test_metrics = classification_metrics(y[te], model.predict(X[te]), data.classes)
    log.info(f"best {best['params']} -> test macro-F1 {test_metrics['macro_f1']:.4f}, acc {test_metrics['accuracy']:.4f}")

    # Latency on a single CPU thread (same protocol for every model).
    model.set_params(n_jobs=1)
    ecfg = cfg["eval"]
    latency = measure_latency(model.predict, X[te], ecfg["latency_warmup"], ecfg["latency_iters"], ecfg["throughput_batch"])

    name = run_name(model_name, seed, label_fraction)
    fig_dir = resolve_path(cfg["paths"]["reports_dir"]) / "figures"
    plot_confusion_matrix(test_metrics["confusion_matrix"], data.classes, f"{model_name.upper()} (test)", fig_dir / f"cm_{name}.png")
    top = plot_feature_importance(
        data.feature_names, model.feature_importances_, int(cfg.get("top_k_features", 15)),
        f"{model_name.upper()} top features", fig_dir / f"fi_{name}.png",
    )

    payload = {
        "model": model_name,
        "generation": "machine_learning",
        "input": "statistical_features",
        "seed": seed,
        "label_fraction": label_fraction,
        "mask_variant": None,
        "n_train": int(len(tr)),
        "best_params": best["params"],
        "val_macro_f1": best["val_macro_f1"],
        "grid_search": search,
        "test": test_metrics,
        "n_params": tree_node_count(model),
        "n_params_kind": "tree_nodes",
        "train_seconds": best["fit_seconds"],
        "grid_search_seconds": grid_s,
        "latency": latency,
        "feature_importance_top": top,
        "classes": data.classes,
    }
    path = save_result(name, cfg, payload)
    log.info(f"saved {path}")
    return payload
