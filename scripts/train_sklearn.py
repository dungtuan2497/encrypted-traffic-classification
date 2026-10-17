"""Phase 2 CLI: Random Forest (or optional XGBoost) on statistical features."""

from __future__ import annotations

from etc.cli import base_parser, config_from_args
from etc.train.train_sklearn import train_sklearn
from etc.utils import get_logger, set_seed


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--model", choices=["rf", "xgb"], default="rf")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--label-fraction", type=float, default=1.0)
    args = parser.parse_args()

    cfg = config_from_args(args, "rf")
    seed = int(args.seed if args.seed is not None else cfg["seed"])
    cfg["seed"] = seed
    set_seed(seed)
    train_sklearn(cfg, args.model, seed, args.label_fraction, get_logger())


if __name__ == "__main__":
    main()
