"""Day 6: schema rules, the builder's helpers, and ``build-alerts`` end to end (synthetic data)."""

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from alert_explainer.detect.alerts import (
    affected_side,
    parse_timestamp,
    protocol_name,
    run_build_alerts,
    stratified_sample,
)
from alert_explainer.explain.attack_kb import FAMILY_TECHNIQUES
from alert_explainer.explain.schemas import CONTEXT_TEXT_MAX, LLM_FIELDS, Alert


@pytest.mark.parametrize(
    ("text", "iso", "afternoon"),
    [
        ("6/7/2017 9:47", "2017-07-06T09:47:00", False),
        ("6/7/2017 3:30", "2017-07-06T15:30:00", True),  # no AM/PM: 3:30 during capture = 15:30
        ("03/07/2017 08:55:58", "2017-07-03T08:55:58", False),
        ("7/7/2017 3:30:12 PM", "2017-07-07T15:30:12", False),
        ("7/7/2017 9:01 AM", "2017-07-07T09:01:00", False),
        ("2017-07-05 10:02:11", "2017-07-05T10:02:11", False),
        ("", None, False),
        (None, None, False),
        ("yesterday", None, False),
    ],
)
def test_parse_timestamp(text: str | None, iso: str | None, afternoon: bool) -> None:
    assert parse_timestamp(text) == (iso, afternoon)


def test_protocol_and_affected_side() -> None:
    assert (
        protocol_name(6.0) == "TCP" and protocol_name(17) == "UDP" and protocol_name(47) == "IP-47"
    )
    assert affected_side(None, None, "172.16.0.1", "192.168.10.50") == "dst"
    assert affected_side(None, None, "192.168.10.9", "205.174.165.73") == "src"  # bot calling out
    assert affected_side(None, None, "8.8.8.8", "1.1.1.1") is None


def test_stratified_sample_balances_families() -> None:
    pred = np.array([0] * 50 + [1] * 100 + [2] * 3 + [3] * 40)
    idx = stratified_sample(pred, ["BENIGN", "A", "B", "C"], 30, np.random.default_rng(0))
    counts = np.bincount(pred[idx], minlength=4)
    assert counts[0] == 0 and counts[2] == 3 and counts.sum() == 30
    assert abs(int(counts[1]) - int(counts[3])) <= 1
    assert len(set(idx.tolist())) == 30
    assert (
        len(stratified_sample(pred, ["BENIGN", "A", "B", "C"], 10_000, np.random.default_rng(0)))
        == 143
    )


def _alert_dict() -> dict:
    return {
        "alert_id": "a-0000123",
        "timestamp": "2017-07-05T10:02:11",
        "src_ip": "172.16.0.1",
        "src_port": 41234,
        "dst_ip": "192.168.10.50",
        "dst_port": 80,
        "protocol": "TCP",
        "family": "DoS",
        "confidence": 0.97,
        "attack_score": 0.99,
        "severity": "High",
        "severity_basis": {
            "base": 2, "confidence_band": 3, "confidence_band_name": "high",
            "asset_criticality": 3, "score": 18, "rule": "…",
        },
        "top_features": [{"name": "Flow IAT Mean", "value": 1520.3, "benign_percentile": 99.6, "contribution": 1.2}],
        "allowed_techniques": ["T1499", "T1499.002"],
        "detector": {"model": "lightgbm", "version": "v", "threshold": 0.2, "attribution_method": "m"},
        "provenance": {"row_id": 123, "true_label": "DoS Hulk"},
    }  # fmt: skip


def test_alert_is_frozen_and_strict() -> None:
    alert = Alert.model_validate(_alert_dict())
    with pytest.raises(ValidationError):
        alert.severity = "Low"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        Alert.model_validate({**_alert_dict(), "unexpected": 1})
    with pytest.raises(ValidationError):
        Alert.model_validate({**_alert_dict(), "severity": "Catastrophic"})
    with pytest.raises(ValidationError):
        Alert.model_validate(
            {**_alert_dict(), "allowed_techniques": ["T9999 Harmless Maintenance"]}
        )
    with pytest.raises(ValidationError):
        Alert.model_validate({**_alert_dict(), "context_text": "x" * (CONTEXT_TEXT_MAX + 1)})
    ok = Alert.model_validate({**_alert_dict(), "candidate_techniques": ["T1046", "T1499"]})
    assert ok.permitted_techniques == ["T1499", "T1499.002", "T1046"]


def test_llm_fields_exclude_ground_truth_and_internals() -> None:
    assert set(Alert.model_fields) >= LLM_FIELDS
    assert {"provenance", "attack_score", "severity_basis", "detector", "src_port"}.isdisjoint(
        LLM_FIELDS
    )


@pytest.fixture(scope="module")
def trained_pipeline(pipeline_dir: Path) -> Path:
    from alert_explainer.detect.lgbm import train_lgbm
    from alert_explainer.detect.train_mlp import train_mlp

    p, m, r = pipeline_dir / "processed", pipeline_dir / "models", pipeline_dir / "reports"
    if not (m / "lgbm" / "meta.json").exists():
        train_lgbm(p, m, r, grid="quick", grid_benign_cap=500, max_rounds=60, patience=10)
    if not (m / "mlp" / "meta.json").exists():
        train_mlp(p, m, r, grid="none", ablation=False, max_epochs=3, batch_size=256, device="cpu")
    return pipeline_dir


@pytest.mark.parametrize("model", ["lgbm", "mlp"])
def test_build_alerts_end_to_end(trained_pipeline: Path, model: str) -> None:
    d = trained_pipeline
    summary = run_build_alerts(
        d / "processed", d / "models", d, d / "reports", model=model, n=24, seed=0
    )
    alerts = [
        Alert.model_validate(a)
        for a in json.loads((d / "alerts" / f"alerts_{model}.json").read_text())
    ]
    assert 0 < len(alerts) <= 24 and summary["n_alerts"] == len(alerts)
    assert len({a.alert_id for a in alerts}) == len(alerts)
    assert (d / "inventory.csv").exists() and (d / "models" / "benign_reference.npz").exists()
    for a in alerts:
        assert a.family != "BENIGN" and a.attack_score >= a.detector.threshold - 1e-4
        assert a.allowed_techniques == list(FAMILY_TECHNIQUES[a.family])
        assert 1 <= len(a.top_features) <= 5
        assert a.severity_basis.score == (
            a.severity_basis.base
            * a.severity_basis.confidence_band
            * a.severity_basis.asset_criticality
        )
        if a.affected_asset == "dst":
            assert a.dst_host is not None
        for f in a.top_features:
            assert (f.benign_percentile is None) == (f.name in ("Destination Port", "Protocol"))
    # stratified: more than one predicted family is represented
    assert len(summary["by_predicted_family"]) > 1
    check = summary["attribution_check"]
    if model == "lgbm":
        assert check["shap_additivity_max_abs_error"] < 1e-6
    else:
        assert check["ig_completeness_median_relative_delta"] < 0.05
    assert summary["latency_per_alert_ms"]["median"] > 0
    report = json.loads((d / "reports" / f"alerts_summary_{model}.json").read_text())
    assert report["n_alerts"] == len(alerts)
