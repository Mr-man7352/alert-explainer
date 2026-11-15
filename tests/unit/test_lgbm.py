import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score

from alert_explainer.detect.evaluate import macro_f1_fast
from alert_explainer.detect.lgbm import (
    IsotonicCalibrator,
    decide,
    sample_weights,
    sanitize_feature_names,
    threshold_for_fpr,
    train_lgbm,
)


def test_macro_f1_fast_matches_sklearn() -> None:
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 5, 2000), rng.integers(0, 5, 2000)
    p[y == 4] = 0  # class 4 never predicted
    expected = f1_score(y, p, labels=range(6), average="macro", zero_division=0)  # 5 absent
    assert macro_f1_fast(y, p, 6) == pytest.approx(expected)


@pytest.mark.parametrize("target", [0.0, 0.01, 0.05, 0.2])
def test_threshold_meets_fpr_target(target: float) -> None:
    scores = np.random.default_rng(1).random(5000)
    scores[:500] = 0.3  # ties
    t = threshold_for_fpr(scores, target)
    assert (scores >= t).mean() <= target
    # lowest such threshold: the next lower distinct score breaks the target
    lower = scores[scores < t].max()
    assert (scores >= lower).mean() > target


def test_decide() -> None:
    probs = np.array([[0.9, 0.05, 0.05], [0.4, 0.1, 0.5], [0.6, 0.3, 0.1]])
    assert decide(probs, 0.35).tolist() == [0, 2, 1]
    assert decide(probs, 0.5).tolist() == [0, 2, 0]


def test_sample_weights() -> None:
    y = np.array([0] * 90 + [1] * 10)
    w = sample_weights(y, "balanced")
    assert w.mean() == pytest.approx(1.0)
    assert w[-1] / w[0] == pytest.approx(9.0)
    assert sample_weights(y, "sqrt")[-1] / sample_weights(y, "sqrt")[0] == pytest.approx(3.0)
    with pytest.raises(ValueError):
        sample_weights(y, "bogus")


def test_calibrator_rows_sum_to_one() -> None:
    rng = np.random.default_rng(2)
    raw = rng.dirichlet(np.ones(4), 1000)
    y = raw.argmax(1)
    cal = IsotonicCalibrator().fit(raw, y).transform(raw)
    assert np.allclose(cal.sum(1), 1.0)
    assert ((cal >= 0) & (cal <= 1)).all()


def test_sanitize_feature_names_unique() -> None:
    names = sanitize_feature_names(["Flow Bytes/s", "Flow Bytes s", "Down/Up Ratio"])
    assert names == ["Flow_Bytes_s", "Flow_Bytes_s_2", "Down_Up_Ratio"]


@pytest.mark.slow
def test_train_lgbm_end_to_end(pipeline_dir: Path) -> None:
    metrics = train_lgbm(
        pipeline_dir / "processed",
        pipeline_dir / "models",
        pipeline_dir / "reports",
        grid="quick",
        grid_benign_cap=500,
        max_rounds=60,
        patience=10,
    )
    assert (pipeline_dir / "models" / "lgbm" / "model.txt").exists()
    assert (pipeline_dir / "reports" / "figures" / "lgbm_confusion_matrix.png").exists()
    saved = json.loads((pipeline_dir / "reports" / "lgbm_metrics.json").read_text())
    assert saved["classes"][0] == "BENIGN" and "Other-rare" not in saved["classes"]
    assert set(saved["held_out_families"]) == {"Heartbleed", "Infiltration"}
    assert 0.0 <= metrics["test_at_threshold"]["macro_f1"] <= 1.0
    assert metrics["validation"]["benign_fpr_at_threshold"] <= 0.01


def test_choose_threshold_prefers_precision_within_fpr_budget() -> None:
    from alert_explainer.detect.lgbm import choose_threshold

    rng = np.random.default_rng(4)
    n_benign, n_attack = 20_000, 200
    # BENIGN mostly confident; a 2% tail has small attack scores. Attacks score high.
    p_attack_b = np.where(rng.random(n_benign) < 0.02, rng.uniform(0.001, 0.2, n_benign), 1e-6)
    p_attack_a = rng.uniform(0.6, 1.0, n_attack)
    p_attack = np.concatenate([p_attack_b, p_attack_a])
    probs = np.column_stack([1 - p_attack, p_attack])
    y = np.concatenate([np.zeros(n_benign, int), np.ones(n_attack, int)])

    best, max_recall, curve = choose_threshold(probs, y, 2, target_fpr=0.01)
    assert all(r["benign_fpr"] <= 0.01 for r in curve)
    assert max_recall < best  # does not spend the whole FPR budget
    fpr_best = (decide(probs, best)[y == 0] != 0).mean()
    assert fpr_best < 0.001
    assert (decide(probs, best)[y == 1] != 0).mean() == 1.0
