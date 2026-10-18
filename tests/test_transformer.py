"""Phase 4 tests: tokenisation, BERT masking rule, model size and heads."""

import numpy as np
import torch

from etc.config import load_config
from etc.data.torch_datasets import LengthBucketSampler, tokenize_session
from etc.models.transformer import CLS, MASK, NUM_BYTES, PAD, SEP, build_transformer, mask_tokens
from etc.train.train_torch import count_params, effective_schedule


def test_tokenize_session_structure() -> None:
    pkts = np.zeros((32, 64), dtype=np.uint8)
    pkts[0, :3] = [1, 2, 3]
    pkts[1, :2] = [4, 5]
    lens = np.zeros(32, dtype=np.uint8)
    lens[:2] = [3, 2]
    tok, pidx = tokenize_session(pkts, lens, max_len=12)
    assert tok.tolist() == [CLS, 1, 2, 3, SEP, 4, 5, SEP, PAD, PAD, PAD, PAD]
    assert pidx.tolist() == [0, 1, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0]
    tok, _ = tokenize_session(pkts, lens, max_len=4)
    assert tok.tolist() == [CLS, 1, 2, 3]  # cut at max_len


def test_mask_tokens_follows_bert_rule() -> None:
    g = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, NUM_BYTES, (64, 512), generator=g)
    tokens[:, 0] = CLS
    tokens[:, 100] = SEP
    tokens[:, 400:] = PAD
    masked, labels = mask_tokens(tokens, generator=g)
    sel = labels != -100
    assert not sel[:, 0].any() and not sel[:, 100].any() and not sel[:, 400:].any()  # specials never chosen
    frac = sel.sum().item() / (64 * 398)
    assert 0.13 < frac < 0.17
    n = sel.sum().item()
    as_mask = (masked[sel] == MASK).sum().item() / n
    unchanged = (masked[sel] == tokens[sel]).sum().item() / n
    assert 0.77 < as_mask < 0.83
    assert 0.08 < unchanged < 0.14  # 10% kept + random draws that hit the same byte
    assert torch.equal(labels[sel], tokens[sel])


def test_transformer_size_and_heads() -> None:
    mcfg = load_config("transformer")["model"]
    model = build_transformer(mcfg, num_classes=19)
    assert count_params(model) < 5_000_000
    x = torch.stack([torch.full((2, 512), PAD), torch.zeros(2, 512, dtype=torch.long)], dim=1)
    x[:, 0, 0] = CLS
    x[:, 0, 1:40] = 7
    assert model(x).shape == (2, 19)
    assert model.mlm_logits(x[..., :40]).shape == (2, 40, 260)
    assert not any(k.startswith("cls_head.") for k in model.encoder_state_dict())


def test_length_bucket_sampler_covers_all() -> None:
    lengths = np.random.default_rng(0).integers(1, 512, 1000)
    batches = list(LengthBucketSampler(lengths, 64, seed=0))
    flat = sorted(i for b in batches for i in b)
    assert flat == list(range(1000))


def test_effective_schedule_stretches_for_few_labels() -> None:
    assert effective_schedule({"epochs": 15, "patience": 5, "min_steps": 1500}, 1000) == (15, 5, 1)
    epochs, patience, every = effective_schedule({"epochs": 15, "patience": 5, "min_steps": 1500}, 10)
    assert epochs * 10 >= 1500 and patience == 5 and every == 10 and epochs == 150
