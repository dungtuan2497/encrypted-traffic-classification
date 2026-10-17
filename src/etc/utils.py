"""Shared utilities: seeding, device selection, logging, run metadata."""

from __future__ import annotations

import logging
import os
import random
import subprocess
import time
from contextlib import contextmanager
from collections.abc import Iterator

import numpy as np

from etc.config import PROJECT_ROOT


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (if installed) for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer: str | None = None) -> "torch.device":  # noqa: F821
    """Pick the best available device: cuda -> mps -> cpu (or `prefer` if given)."""
    import torch

    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_logger(name: str = "etc", level: int = logging.INFO) -> logging.Logger:
    """Return a logger with a rich console handler (configured once)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        from rich.logging import RichHandler

        handler = RichHandler(rich_tracebacks=True, show_path=False)
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def git_commit_hash() -> str | None:
    """Current git commit hash, or None if not in a git repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    """Context manager measuring wall-clock seconds into `result["seconds"]`."""
    result: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - start
