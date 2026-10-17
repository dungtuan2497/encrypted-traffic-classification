"""Phase 4 CLI: fine-tune the byte Transformer (pre-trained or --from-scratch)."""

from __future__ import annotations

from etc.cli import base_parser, config_from_args
from etc.train.finetune import finetune
from etc.utils import get_logger, set_seed


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pretrain-seed", type=int, default=None, help="seed of the pre-trained checkpoint (default: config seed)")
    parser.add_argument("--label-fraction", type=float, default=1.0)
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--mask", choices=["none", "ip", "ip_port"], default=None, help="byte mask variant (RQ3)")
    parser.add_argument("--limit", type=int, default=None, help="cap samples per split for a quick smoke run")
    args = parser.parse_args()

    extra = []
    if args.mask is not None:
        extra += [f"preprocess.mask_ip={args.mask != 'none'}", f"preprocess.mask_port={args.mask == 'ip_port'}"]
    cfg = config_from_args(args, "transformer", extra=extra)
    pretrain_seed = args.pretrain_seed if args.pretrain_seed is not None else int(cfg["seed"])
    seed = int(args.seed if args.seed is not None else cfg["seed"])
    cfg["seed"] = seed
    set_seed(seed)
    finetune(cfg, seed, args.label_fraction, args.from_scratch, get_logger(), limit=args.limit, pretrain_seed=pretrain_seed)


if __name__ == "__main__":
    main()
