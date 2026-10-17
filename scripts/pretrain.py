"""Phase 4 CLI: Masked Byte Modeling pre-training on the training split only."""

from __future__ import annotations

from etc.cli import base_parser, config_from_args
from etc.train.pretrain_mlm import pretrain
from etc.utils import get_logger, set_seed


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mask", choices=["none", "ip", "ip_port"], default=None, help="byte mask variant (RQ3)")
    parser.add_argument("--limit", type=int, default=None, help="cap training sessions for a quick smoke run")
    args = parser.parse_args()

    extra = []
    if args.mask is not None:
        extra += [f"preprocess.mask_ip={args.mask != 'none'}", f"preprocess.mask_port={args.mask == 'ip_port'}"]
    cfg = config_from_args(args, "transformer", extra=extra)
    seed = int(args.seed if args.seed is not None else cfg["seed"])
    cfg["seed"] = seed
    set_seed(seed)
    pretrain(cfg, seed, get_logger(), limit=args.limit)


if __name__ == "__main__":
    main()
