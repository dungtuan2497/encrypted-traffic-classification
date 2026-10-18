"""Tests for report aggregation: RQ1 mean ± std, RQ2/RQ3 tables, missing results."""

from pathlib import Path

from etc.config import load_config
from etc.eval.report import NO_RESULT, render, rq1_rows, rq2_table, rq3_table


def run(model: str, seed: int, f1: float, lf: float = 1.0, mask: str | None = "ip") -> dict:
    return {
        "model": model, "seed": seed, "label_fraction": lf, "mask_variant": mask,
        "test": {"accuracy": f1, "macro_f1": f1, "per_class_f1": {"a": f1, "b": 1.0}, "confusion_matrix": [[1]]},
        "n_params": 10, "n_params_kind": "parameters", "train_seconds": 60.0,
        "latency": {"latency_ms_batch1": 1.0, "throughput_per_s": 100.0}, "_file": f"{model}_{seed}.json",
    }


def test_rq1_mean_std_and_filters() -> None:
    runs = [run("cnn1d", 42, 0.90), run("cnn1d", 43, 0.94), run("cnn1d", 44, 0.92, lf=0.1), run("cnn1d", 45, 0.5, mask="none")]
    [row] = rq1_rows(runs, None)
    mean, std, n = row["macro_f1"]
    assert n == 2 and abs(mean - 0.92) < 1e-9 and abs(std - 0.0282842712) < 1e-6
    assert row["seeds"] == [42, 43]


def test_rq2_rq3_tables_and_missing_cells(tmp_path: Path) -> None:
    runs = [
        run("transformer-pt", 42, 0.95, lf=0.01), run("transformer-scratch", 42, 0.93, lf=0.01),
        run("transformer-pt", 42, 0.99), run("transformer-pt", 42, 0.995, mask="none"),
    ]
    assert rq2_table(runs)["transformer-pt"] == {0.01: 0.95, 1.0: 0.99}
    assert rq3_table(runs)["transformer-pt"]["none"]["macro_f1"] == 0.995
    text = render(load_config(), runs, None, tmp_path)
    assert NO_RESULT in text and "+0.0200" in text and "(+0.0050)" in text
