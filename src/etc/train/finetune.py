"""Fine-tune the byte Transformer for classification (review Section 5.3, RQ1/RQ2/RQ3).

Loads the MLM-pre-trained encoder (or random init with `from_scratch`), adds a
classification head on [CLS] and trains everything with discriminative learning rates
(small for the encoder, larger for the head). `label_fraction` subsamples the labelled
training split (stratified, fixed seed) for the label-efficiency experiment.

Design choice (documented): the from-scratch baseline uses a larger encoder learning
rate (`lr_scratch`) because 5e-5 is a fine-tuning rate; using it from random init
would handicap the baseline unfairly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from etc.config import Config, resolve_path
from etc.data.bytes_repr import variant_name
from etc.data.processed import load_processed, subsample_labels
from etc.data.torch_datasets import make_token_loader, token_input
from etc.eval.metrics import run_name
from etc.models.transformer import build_transformer
from etc.train.pretrain_mlm import pretrain_ckpt_path
from etc.train.train_torch import class_weights, count_params, effective_schedule, finalize_run, fit, limit_indices
from etc.utils import get_device


def finetune(
    cfg: Config,
    seed: int,
    label_fraction: float,
    from_scratch: bool,
    log: Any,
    limit: int | None = None,
    pretrain_seed: int | None = None,
) -> dict[str, Any]:
    data = load_processed(cfg)
    y = data.y
    mcfg, fcfg = cfg["model"], cfg["finetune"]
    variant = variant_name(cfg["preprocess"]["mask_ip"], cfg["preprocess"]["mask_port"])
    x = token_input(data, variant, int(mcfg["max_len"]))

    tr = limit_indices(subsample_labels(data.splits["train"], y, label_fraction, seed), limit, seed)
    va = limit_indices(data.splits["val"], limit, seed)
    te = limit_indices(data.splits["test"], limit, seed)
    bs = int(fcfg["batch_size"])
    train_loader = make_token_loader(x, y, tr, bs, shuffle=True, seed=seed)
    val_loader = make_token_loader(x, y, va, 64, shuffle=False)
    test_loader = make_token_loader(x, y, te, 64, shuffle=False)

    device = get_device(cfg.get("device"))
    model = build_transformer(mcfg, num_classes=len(data.classes))
    ckpt_used = None
    if not from_scratch:
        pseed = int(pretrain_seed if pretrain_seed is not None else cfg["seed"])
        ckpt = pretrain_ckpt_path(cfg, variant, pseed)
        if not ckpt.exists():
            raise FileNotFoundError(f"{ckpt} not found - run `make pretrain` (mask variant {variant}) first")
        missing, unexpected = model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
        assert all(k.startswith("cls_head.") for k in missing) and not unexpected, (missing, unexpected)
        ckpt_used = str(ckpt)
    model.to(device)
    log.info(
        f"fine-tune ({'from scratch' if from_scratch else 'pre-trained'}): {count_params(model):,} params on {device}; "
        f"train={len(tr):,} (label_fraction={label_fraction:g}) val={len(va):,} test={len(te):,} variant={variant}"
    )

    lr_enc = float(fcfg["lr_scratch"] if from_scratch else fcfg["lr_encoder"])
    head = [p for n, p in model.named_parameters() if n.startswith("cls_head.")]
    enc = [p for n, p in model.named_parameters() if not n.startswith("cls_head.")]
    opt = torch.optim.AdamW(
        [{"params": enc, "lr": lr_enc}, {"params": head, "lr": float(fcfg["lr_head"])}],
        weight_decay=float(fcfg["weight_decay"]),
    )
    weights = class_weights(y[tr], len(data.classes)).to(device) if fcfg.get("class_weight", True) else None
    epochs, patience, eval_every = effective_schedule(fcfg, len(train_loader))
    mname = "transformer-scratch" if from_scratch else "transformer-pt"
    name = run_name(mname, seed, label_fraction, tag=f"mask-{variant}")
    fit_out = fit(
        model, train_loader, val_loader, opt,
        epochs=epochs, patience=patience, device=device, eval_every=eval_every,
        criterion=nn.CrossEntropyLoss(weight=weights), log=log,
        ckpt_path=resolve_path(cfg["paths"]["checkpoints_dir"]) / f"{name}.pt",
        warmup_ratio=float(fcfg.get("warmup_ratio", 0.0)),
    )
    extra = {
        "model": mname,
        "generation": "pretraining" if not from_scratch else "deep_learning",
        "input": "byte_tokens",
        "seed": seed,
        "label_fraction": label_fraction,
        "mask_variant": variant,
        "from_scratch": from_scratch,
        "pretrained_checkpoint": ckpt_used,
        "n_train": int(len(tr)),
        "epochs_run_max": epochs,
        "limit": limit,
    }
    smoke_dir = str(Path(cfg["paths"]["results_dir"]) / "_smoke") if limit else None
    return finalize_run(cfg, data, model, test_loader, x[te], device, fit_out, name, extra, log, smoke_dir)
