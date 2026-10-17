"""Phase 3 CLI: supervised deep-learning baselines (1D-CNN on bytes, BiGRU on sequences)."""

from __future__ import annotations

from etc.cli import base_parser, config_from_args
from etc.train.train_torch import run_supervised
from etc.utils import get_logger, set_seed


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--model", choices=["cnn", "gru"], required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--label-fraction", type=float, default=1.0)
    parser.add_argument("--mask", choices=["none", "ip", "ip_port"], default=None, help="byte mask variant (RQ3)")
    parser.add_argument("--limit", type=int, default=None, help="cap samples per split for a quick smoke run")
    args = parser.parse_args()

    extra = []
    if args.mask is not None:
        extra += [f"preprocess.mask_ip={args.mask != 'none'}", f"preprocess.mask_port={args.mask == 'ip_port'}"]
    cfg = config_from_args(args, args.model, extra=extra)
    seed = int(args.seed if args.seed is not None else cfg["seed"])
    cfg["seed"] = seed
    set_seed(seed)
    run_supervised(cfg, seed, args.label_fraction, get_logger(), limit=args.limit)


if __name__ == "__main__":
    main()
