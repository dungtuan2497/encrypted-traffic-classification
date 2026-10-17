"""Shape tests for torch models and a tiny end-to-end run of the shared training loop."""

import logging

import numpy as np
import torch
from torch import nn

from etc.data.torch_datasets import make_loader
from etc.models.cnn1d import CNN1D
from etc.models.gru import BiGRU
from etc.train.train_torch import class_weights, cosine_schedule, count_params, fit


def test_cnn_and_gru_shapes() -> None:
    cnn = CNN1D(num_classes=19)
    assert cnn(torch.rand(4, 1, 784)).shape == (4, 19)
    gru = BiGRU(num_classes=19)
    assert gru(torch.rand(4, 32, 2)).shape == (4, 19)
    assert count_params(gru) < count_params(cnn)


def test_class_weights_balanced() -> None:
    w = class_weights(np.array([0, 0, 0, 1]), 3)
    assert torch.allclose(w, torch.tensor([4 / 9, 4 / 3, 0.0]))


def test_cosine_schedule_decays_to_zero() -> None:
    opt = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = cosine_schedule(opt, total_steps=10, warmup_steps=2)
    lrs = []
    for _ in range(10):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == 0.5 and lrs[1] == 1.0 and lrs[-1] < 0.05


def test_fit_learns_separable_toy_problem() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    x = (rng.normal(size=(400, 1, 784)) * 0.1 + y[:, None, None]).astype(np.float32)
    idx = np.arange(400)
    model = CNN1D(2, fc_dim=32)
    out = fit(
        model,
        make_loader(x, y, idx[:300], 32, shuffle=True),
        make_loader(x, y, idx[300:], 100, shuffle=False),
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        epochs=3, patience=2, device=torch.device("cpu"),
        criterion=nn.CrossEntropyLoss(), log=logging.getLogger("test"),
    )
    assert out["best_val_macro_f1"] > 0.9
    assert len(out["history"]) >= 1
