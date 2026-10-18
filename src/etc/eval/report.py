"""Generate reports/results.md (Vietnamese) and figures from results/*.json (Phase 6).

Every number in the report is read from a results file; anything missing is rendered as
"chưa có kết quả" (no result yet). Sections: RQ1 generation comparison (mean ± std over
seeds), RQ2 label-efficiency curve, RQ3 IP/port masking ablation, pre-training curve,
confusion matrix of the best model, limitations.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from etc.config import Config, resolve_path

NO_RESULT = "chưa có kết quả"

# Display order, labels and generation of each model (RQ1 table rows).
MODELS: list[tuple[str, str, str]] = [
    ("rf", "Random Forest", "Học máy"),
    ("xgb", "XGBoost", "Học máy"),
    ("gru", "BiGRU", "Học sâu"),
    ("cnn1d", "1D-CNN", "Học sâu"),
    ("transformer-scratch", "Transformer (học từ đầu)", "Học sâu"),
    ("transformer-pt", "Transformer (pre-train)", "Pre-training"),
]
LABEL = {m: lab for m, lab, _ in MODELS}
RQ2_MODELS = ["rf", "cnn1d", "transformer-scratch", "transformer-pt"]
RQ2_FRACTIONS = [0.01, 0.05, 0.1, 1.0]
RQ3_MODELS = ["cnn1d", "transformer-pt"]
RQ3_VARIANTS = [("none", "Không che"), ("ip", "Che IP (mặc định)"), ("ip_port", "Che IP + port")]

# Reference categorical palette (light mode), slots 1-4 in fixed order; colour follows
# the model, never its rank. Text and axes use neutral ink.
COLORS = {"rf": "#2a78d6", "cnn1d": "#eb6834", "transformer-scratch": "#1baf7a", "transformer-pt": "#eda100"}
MARKERS = {"rf": "o", "cnn1d": "s", "transformer-scratch": "^", "transformer-pt": "D"}
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load_runs(results_dir: Path) -> list[dict[str, Any]]:
    """All committed run records (smoke runs live in a subfolder and are ignored)."""
    runs = []
    for p in sorted(results_dir.glob("*.json")):
        rec = json.loads(p.read_text())
        rec["_file"] = p.name
        runs.append(rec)
    return runs


def is_default_mask(run: dict[str, Any]) -> bool:
    return run.get("mask_variant") in (None, "ip")


def classifier_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in runs if r.get("stage") != "pretrain" and "test" in r]


def mean_std(values: list[float]) -> tuple[float, float, int]:
    a = np.asarray(values, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0, len(a)


def fmt_ms(mean: float, std: float, n: int) -> str:
    return f"{mean:.4f} ± {std:.4f}" if n > 1 else f"{mean:.4f}"


def rq1_rows(runs: list[dict[str, Any]], pretrain_seconds: float | None) -> list[dict[str, Any]]:
    """Mean ± std over seeds of every model at 100% labels and default masking."""
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in classifier_runs(runs):
        if r.get("label_fraction") == 1.0 and is_default_mask(r):
            by_model[r["model"]].append(r)
    rows = []
    for model, label, gen in MODELS:
        rs = sorted(by_model.get(model, []), key=lambda r: r["seed"])
        if not rs:
            continue
        acc = mean_std([r["test"]["accuracy"] for r in rs])
        f1 = mean_std([r["test"]["macro_f1"] for r in rs])
        train_s = float(np.mean([r["train_seconds"] for r in rs]))
        rows.append(
            {
                "model": model,
                "label": label,
                "generation": gen,
                "seeds": [r["seed"] for r in rs],
                "accuracy": acc,
                "macro_f1": f1,
                "n_params": rs[0]["n_params"],
                "n_params_kind": rs[0].get("n_params_kind", "parameters"),
                "train_seconds": train_s,
                "pretrain_seconds": pretrain_seconds if model == "transformer-pt" else None,
                "latency_ms": float(np.mean([r["latency"]["latency_ms_batch1"] for r in rs])),
                "throughput": float(np.mean([r["latency"]["throughput_per_s"] for r in rs])),
            }
        )
    return rows


def rq2_table(runs: list[dict[str, Any]], seed: int = 42) -> dict[str, dict[float, float]]:
    out: dict[str, dict[float, float]] = defaultdict(dict)
    for r in classifier_runs(runs):
        if r["model"] in RQ2_MODELS and r.get("seed") == seed and is_default_mask(r):
            out[r["model"]][float(r["label_fraction"])] = r["test"]["macro_f1"]
    return out


def rq3_table(runs: list[dict[str, Any]], seed: int = 42) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for r in classifier_runs(runs):
        if r["model"] in RQ3_MODELS and r.get("seed") == seed and r.get("label_fraction") == 1.0:
            out[r["model"]][r["mask_variant"]] = {"accuracy": r["test"]["accuracy"], "macro_f1": r["test"]["macro_f1"]}
    return out


def plot_rq2(table: dict[str, dict[float, float]], path: Path) -> bool:
    """Line chart: label fraction (log x) vs macro-F1, one line per model."""
    if not any(table.values()):
        return False
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ends: list[tuple[float, float, str]] = []
    for model in RQ2_MODELS:
        pts = sorted(table.get(model, {}).items())
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=COLORS[model], linewidth=2, marker=MARKERS[model], markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=LABEL[model], zorder=3)
        ends.append((xs[-1], ys[-1], f"{ys[-1]:.3f}"))
    # End-of-line value labels, nudged apart vertically so close values never overlap.
    lo, hi = ax.get_ylim()
    gap = 0.045 * (hi - lo)
    placed: list[float] = []
    for x, y, text in sorted(ends, key=lambda e: e[1]):
        ty = max(y, placed[-1] + gap) if placed else y
        placed.append(ty)
        ax.annotate(text, (x, y), xytext=(x * 1.12, ty), textcoords="data",
                    va="center", fontsize=9, color=INK_MUTED)
    ax.set_xscale("log")
    ax.set_xticks(RQ2_FRACTIONS, ["1%", "5%", "10%", "100%"])
    ax.minorticks_off()
    ax.set_xlim(0.008, 1.9)
    ax.set_xlabel("Tỷ lệ dữ liệu có nhãn dùng để huấn luyện (thang log)", color=INK_MUTED)
    ax.set_ylabel("Macro-F1 trên tập test", color=INK_MUTED)
    ax.set_title("RQ2 — Hiệu quả dữ liệu nhãn (seed 42)", color=INK, loc="left")
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED)
    leg = ax.legend(frameon=False, loc="lower right", fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return True


def fmt_size(row: dict[str, Any]) -> str:
    unit = "nút cây" if row["n_params_kind"] == "tree_nodes" else "tham số"
    return f"{row['n_params']:,} {unit}"


def fmt_time(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    return f"{seconds:.1f} s" if seconds < 120 else f"{seconds / 60:.1f} phút"


def render(cfg: Config, runs: list[dict[str, Any]], meta: dict[str, Any] | None, fig_dir: Path) -> str:
    pretrain = [r for r in runs if r.get("stage") == "pretrain"]
    pt_default = next((r for r in pretrain if r.get("mask_variant") == "ip" and r.get("seed") == 42), None)
    rows = rq1_rows(runs, pt_default["train_seconds"] if pt_default else None)
    rq2 = rq2_table(runs)
    rq3 = rq3_table(runs)
    has_rq2_fig = plot_rq2(rq2, fig_dir / "label_efficiency.png")

    L: list[str] = []
    L.append("# Kết quả thực nghiệm — Phân loại lưu lượng mạng mã hóa")
    L.append("")
    L.append("> File này được sinh tự động bởi `make report` (`src/etc/eval/report.py`) từ các file `results/*.json`. "
             f"Mọi con số đều lấy từ kết quả chạy thật; mục nào chưa chạy ghi \"{NO_RESULT}\". "
             "Quy ước số: dấu chấm là phần thập phân, dấu phẩy ngăn cách hàng nghìn.")
    L.append("")

    # ---- dataset ----
    L.append("## 1. Dữ liệu")
    L.append("")
    if meta:
        n_train = sum(v["train"] for v in meta["per_class"].values())
        n_val = sum(v["val"] for v in meta["per_class"].values())
        n_test = sum(v["test"] for v in meta["per_class"].values())
        L.append(f"- Bộ dữ liệu: **{meta['dataset']}**, {len(meta['classes'])} lớp, "
                 f"{meta['n_sessions']:,} session (train/val/test = {n_train:,} / {n_val:,} / {n_test:,}).")
        if meta.get("dropped_classes"):
            L.append(f"- Lớp bị loại vì quá ít session: {', '.join(meta['dropped_classes'])}.")
        L.append(f"- Session trùng đã loại (hash byte biến thể `{meta['dedup_variant']}`): "
                 f"{meta['dedup']['duplicates_removed']} (+{meta['dedup']['label_conflicts_removed']} trùng khác nhãn).")
        L.append("- Chia theo session, phân tầng 80/10/10, seed 42; mọi mô hình dùng chung `splits.json`.")
    else:
        L.append(f"- Thống kê dữ liệu: {NO_RESULT} (chưa có `data/processed/meta.json`).")
    L.append("")

    # ---- RQ1 ----
    L.append("## 2. RQ1 — So sánh các thế hệ (100% nhãn)")
    L.append("")
    if rows:
        L.append("| Mô hình | Thế hệ | Seed | Accuracy | **Macro-F1** | Kích thước | Thời gian train | Độ trễ (batch 1) | Throughput (batch 256) |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            train = fmt_time(r["train_seconds"])
            if r["pretrain_seconds"]:
                train = f"{train} fine-tune + {fmt_time(r['pretrain_seconds'])} pre-train"
            L.append(
                f"| {r['label']} | {r['generation']} | {', '.join(map(str, r['seeds']))} | {fmt_ms(*r['accuracy'])} | "
                f"**{fmt_ms(*r['macro_f1'])}** | {fmt_size(r)} | {train} | {r['latency_ms']:.2f} ms | {r['throughput']:,.0f} mẫu/s |"
            )
        L.append("")
        L.append("Ghi chú: giá trị là trung bình ± độ lệch chuẩn qua các seed (nếu có hơn một seed). "
                 "Độ trễ và throughput đo trên **1 luồng CPU** sau warm-up; huấn luyện trên GPU Apple M2 (MPS). "
                 "Kích thước của Random Forest là tổng số nút cây. Transformer fine-tune ở mọi seed dùng chung encoder pre-train seed 42.")
    else:
        L.append(f"_{NO_RESULT}_")
    L.append("")

    # ---- RQ2 ----
    L.append("## 3. RQ2 — Hiệu quả dữ liệu nhãn (seed 42)")
    L.append("")
    L.append("| Mô hình | " + " | ".join(f"{int(f * 100)}% nhãn" for f in RQ2_FRACTIONS) + " |")
    L.append("|---|" + "---|" * len(RQ2_FRACTIONS))
    for m in RQ2_MODELS:
        vals = [f"{rq2[m][f]:.4f}" if f in rq2.get(m, {}) else NO_RESULT for f in RQ2_FRACTIONS]
        L.append(f"| {LABEL[m]} | " + " | ".join(vals) + " |")
    L.append("")
    if "transformer-pt" in rq2 and "transformer-scratch" in rq2:
        gains = [f"{int(f * 100)}%: {rq2['transformer-pt'][f] - rq2['transformer-scratch'][f]:+.4f}"
                 for f in RQ2_FRACTIONS if f in rq2["transformer-pt"] and f in rq2["transformer-scratch"]]
        if gains:
            L.append("Lợi ích riêng của pre-train (Transformer pre-train − Transformer học từ đầu, macro-F1): " + "; ".join(gains) + ".")
            L.append("")
    if has_rq2_fig:
        L.append("![RQ2 label efficiency](figures/label_efficiency.png)")
        L.append("")

    # ---- RQ3 ----
    L.append("## 4. RQ3 — Học tắt: che IP / port (100% nhãn, seed 42)")
    L.append("")
    L.append("| Mô hình | " + " | ".join(lab for _, lab in RQ3_VARIANTS) + " |")
    L.append("|---|" + "---|" * len(RQ3_VARIANTS))
    for m in RQ3_MODELS:
        cells = []
        base = rq3.get(m, {}).get("ip")
        for v, _ in RQ3_VARIANTS:
            res = rq3.get(m, {}).get(v)
            if res is None:
                cells.append(NO_RESULT)
            elif base is not None and v != "ip":
                cells.append(f"{res['macro_f1']:.4f} ({res['macro_f1'] - base['macro_f1']:+.4f})")
            else:
                cells.append(f"{res['macro_f1']:.4f}")
        L.append(f"| {LABEL[m]} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("Giá trị là macro-F1 trên tập test; số trong ngoặc là chênh lệch so với cấu hình mặc định (che IP). "
             "Nếu bỏ che mà điểm tăng mạnh, hoặc che port mà điểm giảm mạnh, mô hình đang dựa vào định danh mạng "
             "(IP/port) thay vì đặc điểm lưu lượng.")
    L.append("")

    # ---- pre-training ----
    L.append("## 5. Pre-training (Masked Byte Modeling)")
    L.append("")
    if pretrain:
        L.append("| Biến thể mask | Seed | Epoch | MLM loss (held-out) | Độ chính xác byte bị che (held-out) | Thời gian |")
        L.append("|---|---|---|---|---|---|")
        for r in sorted(pretrain, key=lambda r: (r["mask_variant"], r["seed"])):
            h = r["history"][-1]
            L.append(f"| {r['mask_variant']} | {r['seed']} | {h['epoch']} | {r['initial_monitor']['loss']:.3f} → {h['monitor_loss']:.3f} | "
                     f"{r['initial_monitor']['acc']:.1%} → {h['monitor_acc']:.1%} | {fmt_time(r['train_seconds'])} |")
        L.append("")
        L.append("Pre-training chỉ dùng tập train (bỏ nhãn); 2% tập train được giữ riêng để theo dõi loss.")
        if (fig_dir / "pretrain_mask-ip_seed42.png").exists():
            L.append("")
            L.append("![Pre-training curve](figures/pretrain_mask-ip_seed42.png)")
    else:
        L.append(f"_{NO_RESULT}_")
    L.append("")

    # ---- best model confusion matrix ----
    L.append("## 6. Ma trận nhầm lẫn của mô hình tốt nhất")
    L.append("")
    if rows:
        best = max(rows, key=lambda r: r["macro_f1"][0])
        run = next(r for r in classifier_runs(runs) if r["model"] == best["model"] and r["seed"] == min(best["seeds"])
                   and r["label_fraction"] == 1.0 and is_default_mask(r))
        cm_name = "cm_" + run["_file"].removesuffix(".json") + ".png"
        L.append(f"Mô hình có macro-F1 trung bình cao nhất: **{best['label']}** (seed {run['seed']}).")
        L.append("")
        worst = sorted(run["test"]["per_class_f1"].items(), key=lambda x: x[1])[:4]
        L.append("Các lớp có F1 thấp nhất: " + ", ".join(f"{c} ({v:.3f})" for c, v in worst) + ".")
        if (fig_dir / cm_name).exists():
            L.append("")
            L.append(f"![Confusion matrix](figures/{cm_name})")
    else:
        L.append(f"_{NO_RESULT}_")
    L.append("")

    # ---- limitations ----
    L.append("## 7. Hạn chế")
    L.append("")
    L.append("- **Bộ dữ liệu cũ và dễ**: USTC-TFC2016 (2016) có tỷ lệ plaintext đáng kể; phần lưu lượng lành tính được sinh "
             "bằng IXIA BreakingPoint nên nhiều lớp (BitTorrent, Facetime, MySQL, Outlook, Skype) chỉ có session 1 gói. "
             "Điểm cao trên bộ này không đồng nghĩa với mô hình mạnh trên lưu lượng mã hóa thật (TLS 1.3, QUIC).")
    L.append("- **Chỉ một bộ dữ liệu**: chưa kiểm chứng trên ISCX VPN-nonVPN hay CIC-IoT; lớp Tinba bị loại (gần như toàn DNS), còn 19 lớp.")
    L.append("- **Mô hình thu nhỏ**: Transformer ~3,5 triệu tham số, token là byte đơn (ET-BERT dùng bigram, BERT-base ~110 triệu tham số); "
             "do giới hạn phần cứng, pre-train 5 epoch và fine-tune tối đa 5 epoch (đường cong MLM loss đã đi ngang ở epoch 5).")
    L.append("- **Ít seed**: RQ2 và RQ3 chỉ chạy seed 42; RQ1 tối đa 3 seed. Huấn luyện trên MPS không hoàn toàn tái lập từng bit.")
    L.append("- **Chưa xét** concept drift (lưu lượng thay đổi theo thời gian), bài toán tập mở (ứng dụng chưa từng thấy) và tấn công đối kháng.")
    L.append("")
    return "\n".join(L)


def generate_report(cfg: Config) -> Path:
    results_dir = resolve_path(cfg["paths"]["results_dir"])
    reports_dir = resolve_path(cfg["paths"]["reports_dir"])
    meta_path = resolve_path(cfg["paths"]["processed_dir"]) / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    text = render(cfg, load_runs(results_dir), meta, reports_dir / "figures")
    out = reports_dir / "results.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out
