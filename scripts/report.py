"""Phase 6 CLI: build reports/results.md and figures from results/*.json."""

from __future__ import annotations

from etc.cli import base_parser, config_from_args
from etc.eval.report import generate_report
from etc.utils import get_logger


def main() -> None:
    args = base_parser(__doc__).parse_args()
    path = generate_report(config_from_args(args))
    get_logger().info(f"wrote {path}")


if __name__ == "__main__":
    main()
