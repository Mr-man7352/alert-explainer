import lightgbm as lgb
import numpy as np
import pytest
import torch

from alert_explainer.detect.attribution import (
    BenignReference,
    integrated_gradients,
    top_features,
    tree_shap,
)
from alert_explainer.detect.torch_mlp import MLP

FEATURES = ["a", "b", "Destination Port"]


def _reference() -> BenignReference:
    rng = np.random.default_rng(0)
    X = np.column_stack(
        [
            rng.normal(0, 1, 5000),
            np.where(rng.random(5000) < 0.4, 0.0, rng.exponential(10, 5000)),  # 40% zeros
            rng.choice([53, 80, 443], 5000, p=[0.2, 0.3, 0.5]),
        ]
    )
    return BenignReference.fit(X, FEATURES)


def test_percentile_mid_rank_and_extremes() -> None:
    ref = _reference()
    p = ref.percentile(np.array([[0.0, 0.0, 80], [10.0, 1e9, 443], [-10.0, -1.0, 53]]))
    assert p[0, 0] == pytest.approx(50, abs=2)  # median of a standard normal
    assert p[0, 1] == pytest.approx(20, abs=1.5)  # zero is shared by 40%: mid-rank = 20
    assert p[1, 0] == 100 and p[1, 1] == 100  # above every benign value
    assert p[2, 0] == 0 and p[2, 1] == 0  # below every benign value
    xs = np.linspace(-3, 3, 50)
    col = ref.percentile(np.column_stack([xs, np.zeros(50), np.full(50, 80)]))[:, 0]
    assert np.all(np.diff(col) >= 0)


def test_share_equal_for_categorical_values() -> None:
    ref = _reference()
    share = ref.share_equal(np.array([[0.0, 0.0, 443], [0.0, 0.0, 21]]))[:, 2]
    assert share[0] == pytest.approx(0.5, abs=0.01)
    assert share[1] == 0.0


def test_reference_save_load(tmp_path) -> None:
    ref = _reference()
    ref.save(tmp_path / "ref.npz")
    back = BenignReference.load(tmp_path / "ref.npz")
    assert back.features == FEATURES and back.n_rows == 5000
    np.testing.assert_array_equal(back.quantiles, ref.quantiles)
    with pytest.raises(ValueError):
        BenignReference.fit(np.empty((0, 3)), FEATURES)


@pytest.fixture(scope="module")
def booster() -> tuple[lgb.Booster, np.ndarray]:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(600, 4))
    y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 1).astype(int)
    params = {"objective": "multiclass", "num_class": 3, "verbose": -1, "num_leaves": 8, "seed": 0}
    return lgb.train(params, lgb.Dataset(X, y), num_boost_round=20), X


def test_tree_shap_is_additive(booster: tuple[lgb.Booster, np.ndarray]) -> None:
    model, X = booster
    target = np.array([0, 1, 2, 1, 0])
    values, expected = tree_shap(model, X[:5], target)
    margin = np.asarray(model.predict(X[:5], raw_score=True))[np.arange(5), target]
    np.testing.assert_allclose(values.sum(axis=1) + expected, margin, atol=1e-9)
    assert values.shape == (5, 4)


def test_tree_shap_matches_shap_tree_explainer(booster: tuple[lgb.Booster, np.ndarray]) -> None:
    shap = pytest.importorskip("shap")
    model, X = booster
    target = np.array([0, 1, 2, 2])
    ours, _ = tree_shap(model, X[:4], target)
    theirs = np.asarray(shap.TreeExplainer(model).shap_values(X[:4]))
    if theirs.ndim == 3 and theirs.shape[0] == 3:  # older shap: (classes, rows, features)
        theirs = theirs.transpose(1, 2, 0)
    np.testing.assert_allclose(ours, theirs[np.arange(4), :, target], atol=1e-6)


def test_integrated_gradients_completeness() -> None:
    torch.manual_seed(0)
    model = MLP(4, 3, (16, 8), 0.0).eval()
    X = np.random.default_rng(2).normal(size=(6, 4)).astype(np.float32)
    baseline = np.zeros(4, dtype=np.float32)
    target = np.array([0, 1, 2, 0, 1, 2])
    values, delta = integrated_gradients(model, X, target, baseline, n_steps=128)
    with torch.no_grad():
        f_x = model(torch.from_numpy(X))[np.arange(6), target].numpy()
        f_b = model(torch.from_numpy(np.tile(baseline, (6, 1))))[np.arange(6), target].numpy()
    np.testing.assert_allclose(values.sum(axis=1), f_x - f_b, atol=1e-3)
    assert np.abs(delta).max() < 1e-3


def test_top_features_prefers_positive_and_handles_categoricals() -> None:
    contributions = np.array([0.1, -2.0, 0.7])
    out = top_features(
        values=np.array([1.5, 3.0, 21.0]),
        contributions=contributions,
        percentiles=np.array([90.0, 5.0, 0.0]),
        features=FEATURES,
        k=3,
        shares=np.array([0.0, 0.0, 0.0]),
    )
    assert [f["name"] for f in out] == ["Destination Port", "a", "b"]  # positives first, then |c|
    port = out[0]
    assert port["benign_percentile"] is None and port["benign_share"] == 0.0
    assert out[1]["benign_percentile"] == 90.0 and out[1]["benign_share"] is None
    assert len(top_features(np.ones(3), contributions, np.ones(3), FEATURES, k=1)) == 1
