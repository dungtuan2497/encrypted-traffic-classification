"""Masked Byte Modeling pre-training (review Section 5.3.1, BERT/ET-BERT masking rule).

Uses ONLY the training split, without labels (pre-training on val/test is exactly the
leakage the review criticises in YaTC). A small slice of the training split (2%) is held
out to monitor MLM loss/accuracy; val/test are never touched.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from etc.config import Config, resolve_path
from etc.data.bytes_repr import variant_name
from etc.data.processed import load_processed
from etc.data.torch_datasets import make_token_loader, token_input
from etc.eval.metrics import save_result
from etc.models.transformer import ByteTransformer, build_transformer, mask_tokens
from etc.train.train_torch import cosine_schedule, count_params, limit_indices
from etc.utils import get_device

MONITOR_FRACTION = 0.02


def pretrain_ckpt_path(cfg: Config, variant: str, seed: int) -> Path:
    return resolve_path(cfg["paths"]["checkpoints_dir"]) / f"transformer_pretrain_mask-{variant}_seed{seed}.pt"


def mlm_step(
    model: ByteTransformer, x: torch.Tensor, pcfg: Config, generator: torch.Generator | None
) -> tuple[torch.Tensor, int, int]:
    """Mask a batch, return (loss, #correct masked predictions, #masked tokens)."""
    x = ByteTransformer.trim(x)
    masked, labels = mask_tokens(
        x[:, 0], float(pcfg["mask_prob"]), float(pcfg["mask_token_ratio"]),
        float(pcfg["random_token_ratio"]), generator,
    )
    inp = torch.stack([masked, x[:, 1]], dim=1)
    logits = model.mlm_logits(inp)
    loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
    sel = labels != -100
    correct = int((logits.argmax(-1)[sel] == labels[sel]).sum())
    return loss, correct, int(sel.sum())


@torch.no_grad()
def mlm_evaluate(model: ByteTransformer, loader, pcfg: Config, device: torch.device) -> tuple[float, float]:
    model.eval()
    g = torch.Generator().manual_seed(1234)  # identical masks every epoch
    tot_loss, tot_c, tot_n = 0.0, 0, 0
    for x, _ in loader:
        loss, c, n = mlm_step(model, x.to(device), pcfg, g)
        tot_loss += float(loss) * n
        tot_c += c
        tot_n += n
    return tot_loss / max(tot_n, 1), tot_c / max(tot_n, 1)


def plot_mlm(history: list[dict[str, float]], path: Path) -> None:
    ep = [h["epoch"] for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    a1.plot(ep, [h["train_loss"] for h in history], label="train")
    a1.plot(ep, [h["monitor_loss"] for h in history], label="held-out (train split)")
    a1.set_xlabel("epoch")
    a1.set_ylabel("MLM loss")
    a1.legend()
    a2.plot(ep, [h["train_acc"] for h in history], label="train")
    a2.plot(ep, [h["monitor_acc"] for h in history], label="held-out (train split)")
    a2.set_xlabel("epoch")
    a2.set_ylabel("masked-byte accuracy")
    a2.legend()
    fig.suptitle("Masked Byte Modeling pre-training")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pretrain(cfg: Config, seed: int, log: Any, limit: int | None = None) -> dict[str, Any]:
    data = load_processed(cfg)
    mcfg, pcfg = cfg["model"], cfg["pretrain"]
    variant = variant_name(cfg["preprocess"]["mask_ip"], cfg["preprocess"]["mask_port"])
    x = token_input(data, variant, int(mcfg["max_len"]))

    train_idx = limit_indices(data.splits["train"], limit, seed)  # train split ONLY
    rng = np.random.default_rng(seed)
    perm = rng.permutation(train_idx)
    n_mon = max(1, int(len(perm) * MONITOR_FRACTION))
    mon_idx, fit_idx = np.sort(perm[:n_mon]), np.sort(perm[n_mon:])
    bs = int(pcfg["batch_size"])
    train_loader = make_token_loader(x, None, fit_idx, bs, shuffle=True, seed=seed)
    mon_loader = make_token_loader(x, None, mon_idx, 64, shuffle=False)

    device = get_device(cfg.get("device"))
    model = build_transformer(mcfg, num_classes=None).to(device)
    n_params = count_params(model)
    log.info(f"pre-train: {n_params:,} params on {device}; {len(fit_idx):,} train sessions (+{n_mon:,} held-out), variant={variant}")

    opt = torch.optim.AdamW(model.parameters(), lr=float(pcfg["lr"]), weight_decay=float(pcfg["weight_decay"]))
    epochs = int(pcfg["epochs"])
    total = epochs * len(train_loader)
    sched = cosine_schedule(opt, total, int(float(pcfg["warmup_ratio"]) * total))
    g = torch.Generator().manual_seed(seed)

    history: list[dict[str, float]] = []
    loss0, acc0 = mlm_evaluate(model, mon_loader, pcfg, device)
    log.info(f"epoch  0 | held-out MLM loss {loss0:.4f} acc {acc0:.4f} (random init; ln(260)={math.log(260):.2f})")
    ckpt = pretrain_ckpt_path(cfg, variant, seed)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        t0, tl, tc, tn = time.perf_counter(), 0.0, 0, 0
        for xb, _ in train_loader:
            loss, c, n = mlm_step(model, xb.to(device), pcfg, g)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tl += float(loss.detach()) * n
            tc += c
            tn += n
        mon_loss, mon_acc = mlm_evaluate(model, mon_loader, pcfg, device)
        rec = {
            "epoch": epoch,
            "train_loss": tl / max(tn, 1),
            "train_acc": tc / max(tn, 1),
            "monitor_loss": mon_loss,
            "monitor_acc": mon_acc,
            "seconds": time.perf_counter() - t0,
        }
        history.append(rec)
        log.info(
            f"epoch {epoch:2d} | train MLM loss {rec['train_loss']:.4f} acc {rec['train_acc']:.4f} "
            f"| held-out loss {mon_loss:.4f} acc {mon_acc:.4f} | {rec['seconds']:.0f}s"
        )
        # Save after every epoch so an interrupted run still leaves a usable encoder.
        torch.save(model.encoder_state_dict(), ckpt)
    train_s = time.perf_counter() - start

    log.info(f"saved encoder -> {ckpt}")

    name = f"pretrain_mask-{variant}_seed{seed}"
    if not limit:
        plot_mlm(history, resolve_path(cfg["paths"]["reports_dir"]) / "figures" / f"{name}.png")
    payload = {
        "model": "transformer_pretrain",
        "stage": "pretrain",
        "seed": seed,
        "mask_variant": variant,
        "n_params": n_params,
        "n_pretrain_sessions": int(len(fit_idx)),
        "n_monitor_sessions": int(n_mon),
        "initial_monitor": {"loss": loss0, "acc": acc0},
        "history": history,
        "train_seconds": train_s,
        "device": str(device),
        "limit": limit,
        "checkpoint": str(ckpt),
    }
    run_cfg = cfg if not limit else {**cfg, "paths": {**cfg["paths"], "results_dir": str(Path(cfg["paths"]["results_dir"]) / "_smoke")}}
    save_result(name, run_cfg, payload)
    return payload
