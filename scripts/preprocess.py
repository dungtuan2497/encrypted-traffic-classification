"""Phase 1 CLI: pcap -> sessions -> representations -> de-dup -> splits.json."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from etc.cli import base_parser, config_from_args
from etc.config import load_dataset_config
from etc.data.build_dataset import build
from etc.utils import get_logger, set_seed, timer


def print_summary(summary: dict) -> None:
    console = Console()
    table = Table(title=f"Dataset: {summary['dataset']} — sessions per class")
    cols = [
        ("sessions_total", "all sessions"),
        ("sessions_eligible", ">= min pkts"),
        ("sessions_sampled", "sampled"),
        ("after_dedup", "after dedup"),
        ("train", "train"),
        ("val", "val"),
        ("test", "test"),
    ]
    table.add_column("class")
    for _, title in cols:
        table.add_column(title, justify="right")
    totals = dict.fromkeys([k for k, _ in cols], 0)
    for lab, row in summary["per_class"].items():
        table.add_row(lab, *[f"{row[k]:,}" for k, _ in cols])
        for k, _ in cols:
            totals[k] += row[k]
    table.add_row("TOTAL", *[f"{totals[k]:,}" for k, _ in cols], style="bold")
    console.print(table)

    pk = Table(title="Packet-level counters")
    pk.add_column("counter")
    pk.add_column("value", justify="right")
    for k, v in summary["packet_stats"].items():
        pk.add_row(k, f"{v:,}")
    console.print(pk)
    if summary["dropped_classes"]:
        console.print(f"[yellow]Excluded classes (too few sessions): {summary['dropped_classes']}[/yellow]")
    console.print(f"Classes kept: {len(summary['classes'])}, sessions: {summary['n_sessions']:,}")
    d = summary["dedup"]
    console.print(
        f"De-dup (hash of '{summary['dedup_variant']}' bytes): "
        f"{d['duplicates_removed']:,} duplicates removed, "
        f"{d['label_conflicts_removed']:,} label-conflicting sessions removed."
    )


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--max-sessions-per-class", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="max packets read per pcap file")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    extra = []
    if args.max_sessions_per_class is not None:
        extra.append(f"preprocess.max_sessions_per_class={args.max_sessions_per_class}")
    if args.limit is not None:
        extra.append(f"preprocess.max_packets_per_file={args.limit}")
    if args.workers is not None:
        extra.append(f"preprocess.num_workers={args.workers}")
    cfg = config_from_args(args, extra=extra)
    set_seed(int(cfg["seed"]))
    log = get_logger()

    with timer() as t:
        summary = build(cfg, load_dataset_config(cfg), log=log)
    print_summary(summary)
    log.info(f"Done in {t['seconds']:.1f}s -> {cfg['paths']['processed_dir']}")


if __name__ == "__main__":
    main()
