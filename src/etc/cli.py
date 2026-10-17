"""Shared argparse helpers: every script accepts `--set key.sub=value` config overrides."""

from __future__ import annotations

import argparse

from etc.config import Config, load_config


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value, e.g. --set preprocess.max_bytes=512 (repeatable)",
    )
    return parser


def config_from_args(args: argparse.Namespace, *names: str, extra: list[str] | None = None) -> Config:
    """Load default + named configs, then `extra` overrides, then --set overrides."""
    return load_config(*names, overrides=(extra or []) + list(args.overrides))
