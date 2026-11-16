"""End-to-end Days 4–5 on the synthetic pipeline: train-mlp, cross-day, compare."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from alert_explainer.detect.data import load_detection_data

TINY = {"max_epochs": 3, "batch_size": 256, "device": "cpu"}


@pytest.fixture(scope="module")
def trained(pipeline_dir: Path) -> Path:
    from alert_explainer.detect.lgbm import train_lgbm
    from alert_explainer.detect.train_mlp import train_mlp

    if not (pipeline_dir / "reports" / "lgbm_metrics.json").exists():
        train_lgbm(
            pipeline_dir / "processed", pipeline_dir / "models", pipeline_dir / "reports",
            grid="quick", grid_benign_cap=500, max_rounds=60, patience=10,
        )  # fmt: skip
    train_mlp(
        pipeline_dir / "processed", pipeline_dir / "models", pipeline_dir / "reports",
        grid="quick", grid_benign_cap=300, **TINY,
    )  # fmt: skip
    return pipeline_dir


def test_load_detection_data_masks(pipeline_dir: Path) -> None:
    d = load_detection_data(pipeline_dir / "processed")
    assert d.classes[0] == "BENIGN" and "Other-rare" not in d.classes
    assert d.X.dtype == np.float32 and d.X.shape == (len(d.y), len(d.features))
    assert not (d.train & d.val).any() and not (d.val & d.test).any()
    assert (d.cal <= d.val).all()  # cal is a subset of val
    assert (d.y[d.held_out] == -1).all() and not (d.held_out & d.train).any()
    assert (d.early ^ d.late).all()
    labels = np.array(d.label_names)[d.label_code]
    flows = pl.read_parquet(pipeline_dir / "processed" / "flows.parquet", columns=["label"])
    assert labels.tolist() == flows["label"].to_list()


@pytest.mark.slow
def test_train_mlp_end_to_end(trained: Path) -> None:
    from alert_explainer.detect.train_mlp import load_mlp_detector

    out = trained / "models" / "mlp"
    for name in ("model.pt", "preprocessor.joblib", "calibrator.joblib", "meta.json"):
        assert (out / name).exists(), name
    for fig in ("mlp_confusion_matrix.png", "mlp_pr_curves.png", "mlp_training_curves.png"):
        assert (trained / "reports" / "figures" / fig).exists(), fig
    assert (trained / "processed" / "mlp" / "train_X.npy").exists()

    m = json.loads((trained / "reports" / "mlp_metrics.json").read_text())
    assert m["classes"][0] == "BENIGN"
    assert len(m["grid"]["results"]) == 3
    assert set(m["ablation_loss"]) == {"ce", "focal"}
    assert sum(r["deployed"] for r in m["ablation_loss"].values()) == 1
    assert m["validation"]["benign_fpr_at_threshold"] <= 0.01
    assert set(m["held_out_families"]) == {"Heartbleed", "Infiltration"}
    assert 0.0 <= m["test_at_threshold"]["macro_f1"] <= 1.0
    assert m["latency"]["single_alert_median_ms"] > 0

    detector = load_mlp_detector(trained / "models")
    d = load_detection_data(trained / "processed")
    pred, probs = detector.predict(d.X[:50])
    assert pred.shape == (50,) and probs.shape == (50, len(detector.classes))
    assert np.allclose(probs.sum(1), 1.0)


@pytest.mark.slow
def test_cross_day_and_compare(trained: Path) -> None:
    from alert_explainer.detect.compare import write_comparison
    from alert_explainer.detect.cross_day import run_cross_day

    report = run_cross_day(
        trained / "processed", trained / "models", trained / "reports",
        max_epochs=3, batch_size=256, device="cpu",
    )  # fmt: skip
    setup = report["setup"]
    assert not set(setup["attack_labels_seen_in_training"]) & set(setup["attack_labels_unseen"])
    assert (
        "Bot" in setup["attack_labels_unseen"]
        and "DoS Hulk" in setup["attack_labels_seen_in_training"]
    )
    for name in ("lgbm", "mlp"):
        r = report["models"][name]
        assert r["in_period_validation"]["at_fpr_0.01"]["benign_fpr"] <= 0.01
        assert set(r["unseen_days"]["at_fpr_0.01"]["by_label"]) == set(
            setup["attack_labels_unseen"]
        )
    assert (trained / "reports" / "figures" / "cross_day_detection.png").exists()

    summary = write_comparison(trained / "models", trained / "reports")
    assert set(summary) == {"lgbm", "mlp", "cross_day"}
    md = (trained / "reports" / "model_comparison.md").read_text()
    assert "LightGBM" in md and "PyTorch MLP" in md and "Cross-day" in md and "focal" in md
    for name in ("lgbm", "mlp"):
        card = (trained / "models" / name / "model_card.md").read_text()
        assert "## Limitations" in card and "Macro-F1" in card
