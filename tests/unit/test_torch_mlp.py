from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from alert_explainer.detect.lgbm import sample_weights
from alert_explainer.detect.torch_mlp import (
    MLP,
    FlowDataset,
    FlowPreprocessor,
    FocalLoss,
    MLPConfig,
    class_weights,
    fit_mlp,
    load_model,
    make_loader,
    pick_device,
    predict_proba,
    save_model,
)

CPU = torch.device("cpu")


@pytest.mark.parametrize("scheme", ["none", "sqrt", "balanced"])
def test_class_weights_match_lightgbm_row_weights(scheme: str) -> None:
    y = np.array([0] * 900 + [1] * 90 + [2] * 10)
    per_class = class_weights(y, 3, scheme)
    assert np.allclose(per_class[y], sample_weights(y, scheme), rtol=1e-5)


def test_class_weights_absent_class_and_bad_scheme() -> None:
    w = class_weights(np.array([0, 0, 1]), 3, "balanced")
    assert np.isfinite(w).all() and len(w) == 3
    with pytest.raises(ValueError):
        class_weights(np.array([0, 1]), 2, "bogus")


def test_focal_loss_gamma_zero_is_weighted_cross_entropy() -> None:
    torch.manual_seed(0)
    logits, target = torch.randn(64, 5), torch.randint(0, 5, (64,))
    weight = torch.rand(5) + 0.1
    focal = FocalLoss(weight, gamma=0.0)(logits, target)
    assert torch.allclose(focal, F.cross_entropy(logits, target, weight=weight), atol=1e-6)
    unweighted = FocalLoss(None, gamma=0.0)(logits, target)
    assert torch.allclose(unweighted, F.cross_entropy(logits, target), atol=1e-6)


def test_focal_loss_down_weights_easy_examples() -> None:
    easy = torch.tensor([[8.0, 0.0]])  # confidently right
    hard = torch.tensor([[0.2, 0.0]])  # barely right
    target = torch.tensor([0])
    ratio_easy = FocalLoss(gamma=2.0)(easy, target) / F.cross_entropy(easy, target)
    ratio_hard = FocalLoss(gamma=2.0)(hard, target) / F.cross_entropy(hard, target)
    assert ratio_easy < 0.01 < ratio_hard


def _row_indexed_arrays(n: int = 1000, d: int = 4) -> tuple[np.ndarray, np.ndarray]:
    X = np.zeros((n, d), dtype=np.float32)
    X[:, 0] = np.arange(n)  # column 0 = row number, so batches can be checked
    return X, (np.arange(n) % 3).astype(np.int64)


def test_dataset_getitems_returns_aligned_sorted_batches(tmp_path: Path) -> None:
    X, y = _row_indexed_arrays()
    np.save(tmp_path / "train_X.npy", X)
    np.save(tmp_path / "train_y.npy", y)
    ds = FlowDataset.from_npy(tmp_path, "train")
    assert isinstance(ds.X, np.memmap) and len(ds) == 1000
    xb, yb = ds.__getitems__([7, 3, 500])
    assert xb.dtype == torch.float32 and yb.dtype == torch.int64
    assert xb[:, 0].tolist() == [3.0, 7.0, 500.0]
    assert yb.tolist() == [0, 1, 2]
    x1, y1 = ds[10]
    assert x1.shape == (4,) and int(y1) == 1


def test_loader_covers_each_row_once_per_epoch() -> None:
    X, y = _row_indexed_arrays()
    loader = make_loader(FlowDataset(X, y), batch_size=128, seed=0)
    seen = []
    for xb, yb in loader:
        assert xb.shape == (128, 4)
        rows = xb[:, 0].long()
        assert torch.equal(yb, rows % 3)
        seen += rows.tolist()
    assert len(seen) == len(set(seen)) == 1000 - 1000 % 128  # last partial batch dropped
    again = [int(r) for xb, _ in make_loader(FlowDataset(X, y), 128, seed=0) for r in xb[:, 0]]
    assert again == seen  # seeded shuffle is reproducible


def test_dataset_index_subset() -> None:
    X, y = _row_indexed_arrays()
    ds = FlowDataset(X, y, indices=np.array([900, 5, 42]))
    assert len(ds) == 3
    assert ds.labels().tolist() == [5 % 3, 42 % 3, 900 % 3]
    xb, _ = ds.__getitems__([0, 1, 2])
    assert xb[:, 0].tolist() == [5.0, 42.0, 900.0]


def test_preprocessor_is_normal_float32_and_chunk_invariant() -> None:
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.lognormal(5, 3, 20_000), rng.integers(-1, 3, 20_000)]).astype(
        np.float32
    )
    pre = FlowPreprocessor(n_quantiles=200, subsample=10_000).fit(X)
    out = pre.transform(X)
    assert out.dtype == np.float32
    assert abs(out[:, 0].mean()) < 0.05 and abs(out[:, 0].std() - 1) < 0.05
    assert np.array_equal(out, pre.transform(X, chunk_rows=777))


def test_preprocessor_matches_sklearn_transform() -> None:
    rng = np.random.default_rng(1)
    X = np.column_stack(
        [
            rng.lognormal(3, 2, 5000),
            rng.integers(0, 4, 5000),  # heavy ties
            rng.choice([-1.0, 0.0, 65535.0], 5000),  # Init_Win_bytes-style values
            np.full(5000, 7.0),  # constant
        ]
    ).astype(np.float32)
    pre = FlowPreprocessor(n_quantiles=100, subsample=5000).fit(X[:4000])
    unseen = np.vstack([X[4000:], [[1e9, 9, -5, 7]], [[-1e9, -9, 1e6, 8]]]).astype(np.float32)
    assert np.allclose(pre.transform(unseen), pre.qt.transform(unseen), atol=1e-6)


def test_mlp_shapes() -> None:
    model = MLP(70, 7)
    assert model(torch.randn(16, 70)).shape == (16, 7)
    assert sum(isinstance(m, torch.nn.BatchNorm1d) for m in model.modules()) == 3


def _blobs(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.choice(3, size=n, p=[0.8, 0.15, 0.05])
    X = rng.normal(size=(n, 6)).astype(np.float32)
    X[:, 0] += 6.0 * y  # well separated along one feature
    return X, y.astype(np.int64)


@pytest.mark.parametrize("loss", ["ce", "focal"])
def test_fit_mlp_learns_and_restores_best_epoch(loss: str) -> None:
    X, y = _blobs(4000, 0)
    X_va, y_va = _blobs(1000, 1)
    cfg = MLPConfig(hidden=(32, 16), lr=1e-2, batch_size=256, max_epochs=6, patience=2, loss=loss)
    fit = fit_mlp(FlowDataset(X, y), X_va, y_va, cfg, 3, CPU, seed=0, log=lambda _: None)
    assert fit.best_val_macro_f1 > 0.9
    assert fit.best_val_macro_f1 == pytest.approx(
        max(h["val_macro_f1"] for h in fit.history), abs=1e-5
    )
    restored = predict_proba(fit.model, X_va, CPU).argmax(1)
    from alert_explainer.detect.evaluate import macro_f1_fast

    assert macro_f1_fast(y_va, restored, 3) == pytest.approx(fit.best_val_macro_f1, abs=1e-4)


def test_patience_counts_only_after_learning_rate_peak() -> None:
    X, y = _blobs(2000, 0)
    X_va, y_va = _blobs(500, 1)
    quiet = {"log": lambda _: None}
    full = fit_mlp(
        FlowDataset(X, y),
        X_va,
        y_va,
        MLPConfig(hidden=(16,), batch_size=256, max_epochs=10),
        3,
        CPU,
        **quiet,
    )
    assert len(full.history) == 10 and not full.stopped_early  # default: whole schedule
    # patience=1: cannot stop before the peak (epoch 3 of 10), whatever the scores do
    short = fit_mlp(
        FlowDataset(X, y),
        X_va,
        y_va,
        MLPConfig(hidden=(16,), batch_size=256, max_epochs=10, patience=1),
        3,
        CPU,
        **quiet,
    )
    assert len(short.history) >= 4


def test_fit_mlp_is_reproducible() -> None:
    X, y = _blobs(2000, 0)
    X_va, y_va = _blobs(500, 1)
    cfg = MLPConfig(hidden=(16,), batch_size=256, max_epochs=2, patience=2)
    a = fit_mlp(FlowDataset(X, y), X_va, y_va, cfg, 3, CPU, seed=5, log=lambda _: None)
    b = fit_mlp(FlowDataset(X, y), X_va, y_va, cfg, 3, CPU, seed=5, log=lambda _: None)
    assert np.allclose(predict_proba(a.model, X_va, CPU), predict_proba(b.model, X_va, CPU))


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    cfg = MLPConfig(hidden=(8, 4), dropout=0.1)
    model = MLP(5, 3, cfg.hidden, cfg.dropout).eval()
    save_model(model, cfg, ["BENIGN", "DoS", "Bot"], tmp_path / "model.pt")
    loaded, cfg2, classes = load_model(tmp_path / "model.pt")
    x = torch.randn(4, 5)
    assert torch.allclose(model(x), loaded(x))
    assert cfg2 == cfg and classes == ["BENIGN", "DoS", "Bot"]


def test_config_round_trip_and_device() -> None:
    cfg = MLPConfig(hidden=(512, 256, 128), loss="focal")
    assert MLPConfig.from_dict(cfg.to_dict()) == cfg
    assert pick_device("cpu") == CPU
    assert pick_device("auto").type in {"cpu", "mps", "cuda"}
    with pytest.raises(ValueError):
        pick_device("tpu")
