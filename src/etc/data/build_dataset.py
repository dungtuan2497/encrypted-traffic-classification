"""Assemble the shared dataset: sessions -> 3 representations -> de-dup -> one split.

Every model reads the same `sessions.parquet` rows and the same `splits.json`, which is
what makes the generation comparison (RQ1-RQ3) fair. Leakage rules from the spec:
  * split by session (never by packet);
  * drop duplicate sessions (hash of masked bytes) BEFORE splitting; sessions whose
    duplicates carry different labels are dropped entirely;
  * split once (stratified 80/10/10, fixed seed) and persist the indices.

Outputs in `processed_dir`:
  sessions.parquet             metadata + statistical features (f_*) + label
  seqs.npy                     (N, seq_len, 2) float32 packet-size sequences
  bytes/<variant>/bytes.npy    (N, max_bytes) uint8 flat bytes      (CNN)
  bytes/<variant>/pkt_bytes.npy (N, P, bytes_per_packet) uint8       (Transformer)
  bytes/<variant>/pkt_lens.npy (N, P) valid bytes per packet
  splits.json, meta.json
where <variant> is one of none / ip / ip_port (RQ3 mask ablation).
"""

from __future__ import annotations

import json
import socket
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from etc.config import Config, resolve_path
from etc.data.bytes_repr import MASK_VARIANTS, content_hash, session_bytes
from etc.data.features import FEATURE_NAMES, size_sequence, statistical_features
from etc.data.pcap_reader import extract_class_sessions


def discover_files(ds_cfg: Config) -> dict[str, list[Path]]:
    """Resolve the dataset's glob -> label mapping into label -> sorted pcap files."""
    raw_dir = resolve_path(ds_cfg["raw_dir"])
    out: dict[str, list[Path]] = {}
    for pattern, label in ds_cfg["files"].items():
        out.setdefault(label, []).extend(sorted(raw_dir.glob(pattern)))
    return {k: sorted(set(v)) for k, v in sorted(out.items())}


def process_class(label: str, files: list[Path], pcfg: dict[str, Any], seed: int) -> dict[str, Any]:
    """Extract one class's sessions and build all representations (runs in a worker)."""
    sessions, stats = extract_class_sessions(
        files, label, pcfg, seed, limit=pcfg.get("max_packets_per_file")
    )
    P, B = int(pcfg["max_packets_per_session"]), int(pcfg["bytes_per_packet"])
    max_bytes, seq_len = int(pcfg["max_bytes"]), int(pcfg["seq_len"])

    meta, feats, seqs = [], [], []
    flat = {v: [] for v in MASK_VARIANTS}
    pkts = {v: [] for v in MASK_VARIANTS}
    plens = {v: [] for v in MASK_VARIANTS}
    for fname, s in sessions:
        ts, lens, dirs = np.asarray(s.ts), np.asarray(s.lens), np.asarray(s.dirs)
        init_ip, init_port = s.initiator
        resp = (s.key[2], s.key[3]) if (s.key[0], s.key[1]) == s.initiator else (s.key[0], s.key[1])
        meta.append(
            {
                "label": label,
                "file": fname,
                "proto": "TCP" if s.proto == 6 else "UDP",
                "src_ip": socket.inet_ntoa(init_ip),
                "src_port": init_port,
                "dst_ip": socket.inet_ntoa(resp[0]),
                "dst_port": resp[1],
                "n_packets": len(s.ts),
                "start_ts": float(ts[0]),
            }
        )
        feats.append(statistical_features(ts, lens, dirs, s.proto))
        seqs.append(size_sequence(ts, lens, dirs, seq_len))
        for v, (m_ip, m_port) in MASK_VARIANTS.items():
            f, pa, pl = session_bytes(s.raw, m_ip, m_port, P, B, max_bytes)
            flat[v].append(f)
            pkts[v].append(pa)
            plens[v].append(pl)

    def stack(xs: list[np.ndarray], shape: tuple[int, ...], dtype: Any) -> np.ndarray:
        return np.stack(xs) if xs else np.zeros((0, *shape), dtype=dtype)

    return {
        "label": label,
        "stats": stats,
        "meta": meta,
        "feats": stack(feats, (len(FEATURE_NAMES),), np.float64),
        "seqs": stack(seqs, (seq_len, 2), np.float32),
        "flat": {v: stack(x, (max_bytes,), np.uint8) for v, x in flat.items()},
        "pkts": {v: stack(x, (P, B), np.uint8) for v, x in pkts.items()},
        "plens": {v: stack(x, (P,), np.uint8) for v, x in plens.items()},
    }


def deduplicate(hashes: list[str], labels: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    """Keep the first occurrence of each hash; drop all copies of label-conflicting hashes."""
    labels_by_hash: dict[str, set[str]] = {}
    for h, y in zip(hashes, labels):
        labels_by_hash.setdefault(h, set()).add(y)
    seen: set[str] = set()
    keep = np.zeros(len(hashes), dtype=bool)
    n_dup = n_conflict = 0
    for i, h in enumerate(hashes):
        if len(labels_by_hash[h]) > 1:
            n_conflict += 1
        elif h in seen:
            n_dup += 1
        else:
            seen.add(h)
            keep[i] = True
    return keep, {"duplicates_removed": n_dup, "label_conflicts_removed": n_conflict}


def stratified_split(
    labels: np.ndarray, ratios: dict[str, float], seed: int
) -> dict[str, list[int]]:
    """Per-class shuffle and cut into train/val/test (rounded per class, seeded).

    A hand-rolled split (instead of sklearn's) so tiny classes never raise errors.
    """
    rng = np.random.default_rng(seed)
    out: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for y in sorted(set(labels.tolist())):
        idx = np.flatnonzero(labels == y)
        rng.shuffle(idx)
        n_test = int(round(len(idx) * ratios["test"]))
        n_val = int(round(len(idx) * ratios["val"]))
        out["test"] += idx[:n_test].tolist()
        out["val"] += idx[n_test : n_test + n_val].tolist()
        out["train"] += idx[n_test + n_val :].tolist()
    return {k: sorted(v) for k, v in out.items()}


def build(cfg: Config, ds_cfg: Config, log: Any = None) -> dict[str, Any]:
    """Run the full preprocessing and write all outputs. Returns the summary dict."""
    pcfg, seed = cfg["preprocess"], int(cfg["seed"])
    files = discover_files(ds_cfg)
    missing = [k for k, v in files.items() if not v]
    if missing:
        raise FileNotFoundError(f"No pcap found for classes: {missing} (raw_dir={ds_cfg['raw_dir']})")

    # Largest classes first so the pool finishes sooner.
    order = sorted(files, key=lambda k: -sum(p.stat().st_size for p in files[k]))
    workers = max(1, int(pcfg.get("num_workers", 1)))
    results: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {lab: ex.submit(process_class, lab, files[lab], pcfg, seed) for lab in order}
        for lab in order:
            results[lab] = futs[lab].result()
            if log:
                st = results[lab]["stats"]
                log.info(
                    f"{lab}: {st.get('packets_read', 0):,} pkts read, "
                    f"{st.get('sessions_eligible', 0):,} eligible sessions, "
                    f"{st.get('sessions_sampled', 0):,} sampled"
                )

    labels_sorted = sorted(results)
    parts = [results[k] for k in labels_sorted]
    meta = pd.DataFrame([m for r in parts for m in r["meta"]])
    feats = np.concatenate([r["feats"] for r in parts])
    seqs = np.concatenate([r["seqs"] for r in parts])
    variants = {
        v: {
            "flat": np.concatenate([r["flat"][v] for r in parts]),
            "pkts": np.concatenate([r["pkts"][v] for r in parts]),
            "plens": np.concatenate([r["plens"][v] for r in parts]),
        }
        for v in MASK_VARIANTS
    }

    # De-duplicate on the strictest mask so no variant has cross-split duplicates.
    dedup_v = pcfg.get("dedup_variant", "ip_port")
    hashes = [content_hash(x) for x in variants[dedup_v]["flat"]]
    keep, dup_stats = deduplicate(hashes, meta["label"].tolist())
    meta = meta[keep].reset_index(drop=True)
    feats, seqs = feats[keep], seqs[keep]
    for v in variants:
        variants[v] = {k: a[keep] for k, a in variants[v].items()}

    # Exclude classes too small to train/evaluate on.
    min_cls = int(pcfg.get("min_sessions_per_class", 1))
    sizes = meta["label"].value_counts()
    dropped_classes = sorted(lab for lab in labels_sorted if sizes.get(lab, 0) < min_cls)
    if dropped_classes:
        if log:
            log.warning(f"Excluding classes with < {min_cls} sessions: {dropped_classes}")
        keep_cls = (~meta["label"].isin(dropped_classes)).to_numpy()
        meta = meta[keep_cls].reset_index(drop=True)
        feats, seqs = feats[keep_cls], seqs[keep_cls]
        for v in variants:
            variants[v] = {k: a[keep_cls] for k, a in variants[v].items()}

    classes = sorted(meta["label"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    meta.insert(0, "session_id", np.arange(len(meta)))
    meta["label_idx"] = meta["label"].map(class_to_idx).astype(np.int64)
    groups = {lab: g for g, labs in ds_cfg.get("groups", {}).items() for lab in labs}
    meta["group"] = meta["label"].map(groups).fillna("")
    feat_df = pd.DataFrame(feats, columns=[f"f_{n}" for n in FEATURE_NAMES])
    sessions = pd.concat([meta, feat_df], axis=1)

    splits = stratified_split(meta["label"].to_numpy(), cfg["split"], seed)

    # ---- write outputs ----
    out_dir = resolve_path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions.to_parquet(out_dir / "sessions.parquet", index=False)
    np.save(out_dir / "seqs.npy", seqs)
    for v, arrs in variants.items():
        vdir = out_dir / "bytes" / v
        vdir.mkdir(parents=True, exist_ok=True)
        np.save(vdir / "bytes.npy", arrs["flat"])
        np.save(vdir / "pkt_bytes.npy", arrs["pkts"])
        np.save(vdir / "pkt_lens.npy", arrs["plens"])
    with open(out_dir / "splits.json", "w") as f:
        json.dump({"seed": seed, "ratios": cfg["split"], **splits}, f)

    per_class: dict[str, dict[str, int]] = {}
    split_of = np.empty(len(meta), dtype=object)
    for name, idx in splits.items():
        split_of[idx] = name
    for lab in labels_sorted:
        st = results[lab]["stats"]
        mask = (meta["label"] == lab).to_numpy()
        c = Counter(split_of[mask].tolist())
        per_class[lab] = {
            "sessions_total": int(st.get("sessions_total", 0)),
            "sessions_eligible": int(st.get("sessions_eligible", 0)),
            "sessions_sampled": int(st.get("sessions_sampled", 0)),
            "after_dedup": int(mask.sum()),
            "train": c.get("train", 0),
            "val": c.get("val", 0),
            "test": c.get("test", 0),
        }
    packet_stats: Counter[str] = Counter()
    for r in parts:
        packet_stats.update(
            {k: v for k, v in r["stats"].items() if k.startswith(("packets_", "drop_"))}
        )
    summary = {
        "dataset": ds_cfg.get("name"),
        "classes": classes,
        "feature_names": FEATURE_NAMES,
        "mask_variants": list(MASK_VARIANTS),
        "dedup_variant": dedup_v,
        "n_sessions": int(len(meta)),
        "dropped_classes": dropped_classes,
        "dedup": dup_stats,
        "packet_stats": dict(sorted(packet_stats.items())),
        "per_class": per_class,
        "preprocess": pcfg,
        "seed": seed,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary
