"""Shared supervised training loop for all torch models (Phase 3, reused by fine-tuning).

AdamW + cosine learning-rate schedule (optional linear warm-up), class-weighted
cross-entropy, early stopping on validation macro-F1 (patience), best-checkpoint saving,
per-epoch logging. `finalize_run` evaluates on test, measures CPU latency and writes the
results JSON + figures, so every torch model is reported identically.
"""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from etc.config import Config, resolve_path
from etc.data.bytes_repr import variant_name
from etc.data.processed import ProcessedData, load_processed, subsample_labels
from etc.data.torch_datasets import bytes_input, make_loader, seq_input
from etc.eval.latency import measure_latency
from etc.eval.metrics import classification_metrics, macro_f1, plot_confusion_matrix, run_name, save_result
from etc.utils import get_device


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    """'balanced' weights: n / (k * n_c); classes absent from y get weight 0."""
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    w = np.where(counts > 0, len(y) / (num_classes * np.maximum(counts, 1)), 0.0)
    return torch.tensor(w, dtype=torch.float32)


def cosine_schedule(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int = 0) -> torch.optim.lr_scheduler.LambdaLR:
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    ys, preds, total, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if criterion is not None:
            total += float(criterion(logits, y)) * len(y)
            n += len(y)
        ys.append(y.cpu().numpy())
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(preds), total / max(n, 1)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    patience: int,
    device: torch.device,
    criterion: nn.Module,
    log: Any,
    ckpt_path: Path | None = None,
    warmup_ratio: float = 0.0,
    grad_clip: float | None = 1.0,
) -> dict[str, Any]:
    """Train with early stopping on val macro-F1; restores the best weights at the end."""
    total_steps = epochs * len(train_loader)
    scheduler = cosine_schedule(optimizer, total_steps, int(warmup_ratio * total_steps))
    history: list[dict[str, float]] = []
    best_f1, best_epoch, best_state, bad = -1.0, -1, None, 0
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        t0, run_loss, n = time.perf_counter(), 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            run_loss += float(loss.detach()) * len(y)
            n += len(y)
        y_val, p_val, val_loss = evaluate(model, val_loader, device, criterion)
        val_f1 = macro_f1(y_val, p_val)
        rec = {
            "epoch": epoch,
            "train_loss": run_loss / max(n, 1),
            "val_loss": val_loss,
            "val_macro_f1": val_f1,
            "val_accuracy": float((y_val == p_val).mean()),
            "seconds": time.perf_counter() - t0,
        }
        history.append(rec)
        improved = val_f1 > best_f1
        if improved:
            best_f1, best_epoch, bad = val_f1, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            if ckpt_path is not None:
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, ckpt_path)
        else:
            bad += 1
        log.info(
            f"epoch {epoch:2d} | train {rec['train_loss']:.4f} | val {val_loss:.4f} "
            f"| val F1 {val_f1:.4f}{' *' if improved else ''} | {rec['seconds']:.1f}s"
        )
        if bad >= patience:
            log.info(f"early stopping (no val F1 improvement for {patience} epochs)")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "train_seconds": time.perf_counter() - start,
    }


def plot_history(history: list[dict[str, float]], title: str, path: Path) -> None:
    ep = [h["epoch"] for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    a1.plot(ep, [h["train_loss"] for h in history], label="train")
    a1.plot(ep, [h["val_loss"] for h in history], label="val")
    a1.set_xlabel("epoch")
    a1.set_ylabel("loss")
    a1.legend()
    a2.plot(ep, [h["val_macro_f1"] for h in history], color="#C44E52")
    a2.set_xlabel("epoch")
    a2.set_ylabel("val macro-F1")
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def torch_latency(model: nn.Module, x: np.ndarray, ecfg: Config) -> dict[str, float]:
    """CPU, single-thread latency (same protocol as the sklearn models)."""
    cpu_model = copy.deepcopy(model).cpu().eval()
    prev = torch.get_num_threads()
    torch.set_num_threads(1)

    @torch.no_grad()
    def predict(batch: np.ndarray) -> np.ndarray:
        return cpu_model(torch.from_numpy(np.ascontiguousarray(batch))).argmax(1).numpy()

    try:
        return measure_latency(predict, x, ecfg["latency_warmup"], ecfg["latency_iters"], ecfg["throughput_batch"])
    finally:
        torch.set_num_threads(prev)


def finalize_run(
    cfg: Config,
    data: ProcessedData,
    model: nn.Module,
    test_loader: DataLoader,
    x_test: np.ndarray,
    device: torch.device,
    fit_out: dict[str, Any],
    name: str,
    extra: dict[str, Any],
    log: Any,
    results_dir: str | None = None,
) -> dict[str, Any]:
    y_true, y_pred, _ = evaluate(model, test_loader, device)
    test_metrics = classification_metrics(y_true, y_pred, data.classes)
    log.info(f"{name}: test macro-F1 {test_metrics['macro_f1']:.4f}, acc {test_metrics['accuracy']:.4f}")
    latency = torch_latency(model, x_test, cfg["eval"])
    log.info(f"latency {latency['latency_ms_batch1']:.2f} ms/sample, throughput {latency['throughput_per_s']:.0f}/s")

    fig_dir = resolve_path(cfg["paths"]["reports_dir"]) / "figures"
    if results_dir is None:
        plot_history(fit_out["history"], name, fig_dir / f"curve_{name}.png")
        plot_confusion_matrix(test_metrics["confusion_matrix"], data.classes, f"{name} (test)", fig_dir / f"cm_{name}.png")
    payload = {
        **extra,
        "test": test_metrics,
        "n_params": count_params(model),
        "n_params_kind": "parameters",
        "train_seconds": fit_out["train_seconds"],
        "best_epoch": fit_out["best_epoch"],
        "val_macro_f1": fit_out["best_val_macro_f1"],
        "history": fit_out["history"],
        "latency": latency,
        "device": str(device),
        "classes": data.classes,
    }
    run_cfg = cfg if results_dir is None else {**cfg, "paths": {**cfg["paths"], "results_dir": results_dir}}
    path = save_result(name, run_cfg, payload)
    log.info(f"saved {path}")
    return payload


def effective_schedule(tcfg: Config, batches_per_epoch: int) -> tuple[int, int]:
    """(epochs, patience) guaranteeing at least `min_steps` optimizer steps.

    With few labels (RQ2) an epoch is only a handful of steps, so the epoch budget and
    the early-stopping patience are stretched by the same factor. Applied identically to
    every torch model so the label-efficiency comparison stays fair.
    """
    epochs, patience = int(tcfg["epochs"]), int(tcfg["patience"])
    min_steps = int(tcfg.get("min_steps", 0))
    if min_steps and epochs * batches_per_epoch < min_steps:
        factor = math.ceil(min_steps / batches_per_epoch) / epochs
        return math.ceil(epochs * factor), math.ceil(patience * factor)
    return epochs, patience


def limit_indices(idx: np.ndarray, limit: int | None, seed: int) -> np.ndarray:
    """Random subset for quick smoke runs (--limit)."""
    if limit is None or len(idx) <= limit:
        return idx
    return np.sort(np.random.default_rng(seed).choice(idx, size=limit, replace=False))


MODEL_INPUTS: dict[str, str] = {"cnn1d": "bytes", "gru": "seq"}


def build_model(cfg: Config, num_classes: int) -> nn.Module:
    m = cfg["model"]
    if m["name"] == "cnn1d":
        from etc.models.cnn1d import CNN1D

        return CNN1D(num_classes, m["input_len"], tuple(m["channels"]), m["kernel_size"], m["fc_dim"], m["dropout"])
    if m["name"] == "gru":
        from etc.models.gru import BiGRU

        return BiGRU(num_classes, m["in_channels"], m["hidden"], m["num_layers"], m["dropout"])
    raise ValueError(f"Unknown model {m['name']}")


def run_supervised(cfg: Config, seed: int, label_fraction: float, log: Any, limit: int | None = None) -> dict[str, Any]:
    """Train + evaluate one supervised torch model (CNN or GRU) and save its results."""
    data = load_processed(cfg)
    y = data.y
    mname = cfg["model"]["name"]
    pcfg, tcfg = cfg["preprocess"], cfg["train"]
    variant = variant_name(pcfg["mask_ip"], pcfg["mask_port"])
    if MODEL_INPUTS[mname] == "bytes":
        x = bytes_input(data, variant, int(cfg["model"]["input_len"]))
    else:
        x, variant = seq_input(data), None

    tr = subsample_labels(data.splits["train"], y, label_fraction, seed)
    tr = limit_indices(tr, limit, seed)
    va = limit_indices(data.splits["val"], limit, seed)
    te = limit_indices(data.splits["test"], limit, seed)
    bs = int(tcfg["batch_size"])
    train_loader = make_loader(x, y, tr, bs, shuffle=True, seed=seed)
    val_loader = make_loader(x, y, va, 512, shuffle=False)
    test_loader = make_loader(x, y, te, 512, shuffle=False)

    device = get_device(cfg.get("device"))
    model = build_model(cfg, len(data.classes)).to(device)
    n_params = count_params(model)
    log.info(f"{mname}: {n_params:,} params on {device}; train={len(tr):,} val={len(va):,} test={len(te):,} variant={variant}")

    weights = class_weights(y[tr], len(data.classes)).to(device) if tcfg.get("class_weight", True) else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))
    name = run_name(mname, seed, label_fraction, tag=f"mask-{variant}" if variant else None)
    ckpt = resolve_path(cfg["paths"]["checkpoints_dir"]) / f"{name}.pt"
    epochs, patience = effective_schedule(tcfg, len(train_loader))
    fit_out = fit(
        model, train_loader, val_loader, optimizer,
        epochs=epochs, patience=patience, device=device,
        criterion=criterion, log=log, ckpt_path=ckpt, warmup_ratio=float(tcfg.get("warmup_ratio", 0.0)),
    )
    extra = {
        "model": mname,
        "generation": "deep_learning",
        "input": MODEL_INPUTS[mname],
        "seed": seed,
        "label_fraction": label_fraction,
        "mask_variant": variant,
        "n_train": int(len(tr)),
        "epochs_run_max": epochs,
        "limit": limit,
    }
    smoke_dir = str(Path(cfg["paths"]["results_dir"]) / "_smoke") if limit else None
    return finalize_run(cfg, data, model, test_loader, x[te], device, fit_out, name, extra, log, smoke_dir)


TrainFn = Callable[..., dict[str, Any]]
