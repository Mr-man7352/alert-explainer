"""Alert builder (blueprint §3 steps 1–2, §7.1; Day 6). Run with ``uv run ae build-alerts``.

For each flow the detector flags as an attack, build a frozen ``Alert``:

- family and **calibrated** confidence from the detector, and the raw attack score that fired it;
- deterministic **severity** from the family, the confidence band and the affected asset's
  criticality (``severity.py``); the full rule is stored on the alert (``severity_basis``);
- the **top-5 features** by attribution towards the predicted family (TreeSHAP or Integrated
  Gradients), each with its raw value and its percentile among BENIGN training flows;
- hostnames and owners from the **synthetic** inventory, and the ATT&CK techniques the mapping
  table allows for the family;
- ``provenance`` (row id, flow id, ground-truth label) for evaluation, never shown to the LLM.

``ae build-alerts`` scores a whole split, samples ``n`` flagged flows stratified by predicted
family (so rare families are represented), builds each alert one at a time (so the per-alert
latency is real) and writes ``data/alerts/alerts_<model>.json`` plus a summary in
``reports/alerts_summary_<model>.json``.

Timestamps: the TrafficLabelling CSVs record local capture time as ``d/m/yyyy h:mm`` (some files
add seconds) on a clock without AM/PM, so 3:30 means 15:30. Capture ran during working hours
(about 08:00–17:00), so an hour below 8 is read as afternoon. The number of timestamps this rule
changed, and of any that failed to parse, is reported in the summary.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from alert_explainer.data.inventory import Asset, Inventory, inventory_from_flows, is_internal
from alert_explainer.detect.attribution import (
    BenignReference,
    build_benign_reference,
    top_features,
    tree_shap,
)
from alert_explainer.detect.detectors import Detector, LGBMDetector, load_detector
from alert_explainer.detect.severity import compute_severity
from alert_explainer.explain.attack_kb import FAMILY_TECHNIQUES
from alert_explainer.explain.schemas import (
    Alert,
    DetectorInfo,
    Provenance,
    SeverityBasis,
    TopFeature,
)

PROTOCOLS = {0: "HOPOPT", 1: "ICMP", 6: "TCP", 17: "UDP", 58: "ICMPv6"}
META_COLUMNS = ("Flow ID", "Source IP", "Source Port", "Destination IP", "Timestamp")
_TIME_FORMATS = ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S")
_TIME_FORMATS_12H = ("%d/%m/%Y %I:%M:%S %p", "%d/%m/%Y %I:%M %p")
FIRST_CAPTURE_HOUR = 8


def parse_timestamp(text: str | None) -> tuple[str | None, bool]:
    """CIC-IDS2017 timestamp → (ISO 8601 local time, whether the afternoon rule was applied)."""
    if not text or not text.strip():
        return None, False
    raw = " ".join(text.split())
    if raw[-2:].upper() in ("AM", "PM"):
        for fmt in _TIME_FORMATS_12H:
            try:
                return datetime.strptime(raw.upper(), fmt).isoformat(), False
            except ValueError:
                continue
        return None, False
    for fmt in _TIME_FORMATS:
        try:
            dt = datetime.strptime(raw, fmt)
        except ValueError:
            continue
        if dt.hour < FIRST_CAPTURE_HOUR:
            return dt.replace(hour=dt.hour + 12).isoformat(), True
        return dt.isoformat(), False
    return None, False


def protocol_name(number: float | int) -> str:
    n = int(number)
    return PROTOCOLS.get(n, f"IP-{n}")


def affected_side(src: Asset | None, dst: Asset | None, src_ip: str, dst_ip: str) -> str | None:
    """The internal asset the alert is about: the destination if internal, else the source."""
    if dst is not None or is_internal(dst_ip):
        return "dst"
    if src is not None or is_internal(src_ip):
        return "src"
    return None


@dataclass
class BuildStats:
    timestamps_afternoon_rule: int = 0
    timestamps_unparsed: int = 0


class AlertBuilder:
    def __init__(
        self,
        detector: Detector,
        inventory: Inventory,
        reference: BenignReference,
        n_features: int = 5,
    ) -> None:
        if reference.features != detector.features:
            raise ValueError("Benign reference and detector use different feature lists")
        self.detector = detector
        self.inventory = inventory
        self.reference = reference
        self.n_features = n_features
        self.features = detector.features
        self.stats = BuildStats()

    def build(self, rows: pl.DataFrame) -> list[Alert]:
        """Alerts for the flagged rows of ``rows`` (which needs metadata + feature columns)."""
        if rows.height == 0:
            return []
        X = rows.select(self.features).to_numpy().astype(np.float64)
        scores = self.detector.score(X)
        flagged = np.flatnonzero(scores.pred != 0)
        if len(flagged) == 0:
            return []
        contributions = self.detector.attribute(X[flagged], scores.pred[flagged])
        percentiles = self.reference.percentile(X[flagged])
        shares = self.reference.share_equal(X[flagged])
        records = rows[flagged.tolist()].to_dicts()
        alerts = []
        for j, (i, rec) in enumerate(zip(flagged, records, strict=True)):
            cls = int(scores.pred[i])
            alerts.append(
                self._alert(
                    rec,
                    family=self.detector.classes[cls],
                    confidence=float(scores.calibrated[i, cls]),
                    attack_score=float(scores.attack_score[i]),
                    values=X[i],
                    contributions=contributions[j],
                    percentiles=percentiles[j],
                    shares=shares[j],
                )
            )
        return alerts

    def _alert(
        self,
        rec: dict[str, Any],
        family: str,
        confidence: float,
        attack_score: float,
        values: np.ndarray,
        contributions: np.ndarray,
        percentiles: np.ndarray,
        shares: np.ndarray,
    ) -> Alert:
        src_ip, dst_ip = str(rec["Source IP"]), str(rec["Destination IP"])
        src, dst = self.inventory.get(src_ip), self.inventory.get(dst_ip)
        side = affected_side(src, dst, src_ip, dst_ip)
        asset = {"src": src, "dst": dst}.get(side) if side else None
        sev = compute_severity(family, confidence, asset.asset_criticality if asset else None)
        timestamp, afternoon = parse_timestamp(rec.get("Timestamp"))
        self.stats.timestamps_afternoon_rule += afternoon
        self.stats.timestamps_unparsed += timestamp is None and bool(rec.get("Timestamp"))
        src_port = rec.get("Source Port")
        return Alert(
            alert_id=f"a-{int(rec['row_id']):07d}",
            timestamp=timestamp,
            src_ip=src_ip,
            src_port=int(src_port) if src_port is not None else None,
            dst_ip=dst_ip,
            dst_port=int(rec["Destination Port"]),
            protocol=protocol_name(rec["Protocol"]),
            src_host=src.hostname if src else None,
            src_owner=src.owner_username if src else None,
            dst_host=dst.hostname if dst else None,
            dst_owner=dst.owner_username if dst else None,
            affected_asset=side,  # type: ignore[arg-type]
            family=family,
            confidence=round(min(max(confidence, 0.0), 1.0), 4),
            attack_score=round(min(max(attack_score, 0.0), 1.0), 4),
            severity=sev.severity,
            severity_basis=SeverityBasis(
                base=sev.base,
                confidence_band=sev.confidence_band,
                confidence_band_name=sev.confidence_band_name,  # type: ignore[arg-type]
                asset_criticality=sev.asset_criticality,
                score=sev.score,
                rule=sev.rule,
            ),
            top_features=[
                TopFeature(**f)
                for f in top_features(
                    values, contributions, percentiles, self.features, self.n_features, shares
                )
            ],
            allowed_techniques=list(FAMILY_TECHNIQUES[family]),
            detector=DetectorInfo(
                model=self.detector.name,
                version=self.detector.version,
                threshold=self.detector.threshold,
                attribution_method=self.detector.attribution_method,
            ),
            provenance=Provenance(
                row_id=int(rec["row_id"]),
                flow_id=rec.get("Flow ID"),
                source_file=rec.get("source_file"),
                true_label=rec.get("label"),
                true_family=rec.get("family"),
            ),
        )


# --------------------------------------------------------------------------- pipeline


def load_split_rows(processed_dir: Path, split: str) -> pl.DataFrame:
    schema = json.loads((processed_dir / "schema.json").read_text())
    columns = ["row_id", *META_COLUMNS, *schema["features"], "label", "family", "source_file"]
    splits = pl.scan_parquet(processed_dir / "splits.parquet").filter(pl.col("split") == split)
    return (
        pl.scan_parquet(processed_dir / "flows.parquet")
        .select(columns)
        .join(splits.select("row_id"), on="row_id", how="semi")
        .sort("row_id")
        .collect()
    )


def stratified_sample(
    pred: np.ndarray, classes: list[str], n: int, rng: np.random.Generator
) -> np.ndarray:
    """Indices of up to ``n`` flagged rows, spread as evenly as possible over predicted families."""
    pools = {c: rng.permutation(np.flatnonzero(pred == c)) for c in range(1, len(classes))}
    pools = {c: p for c, p in pools.items() if len(p)}
    chosen: list[int] = []
    taken = dict.fromkeys(pools, 0)
    while len(chosen) < n and any(taken[c] < len(p) for c, p in pools.items()):
        for c, p in pools.items():  # round robin: one per family per pass
            if taken[c] < len(p) and len(chosen) < n:
                chosen.append(int(p[taken[c]]))
                taken[c] += 1
    return np.sort(np.array(chosen, dtype=int))


def _summary(
    alerts: list[Alert],
    latencies_ms: list[float],
    detector: Detector,
    stats: BuildStats,
    extra: dict[str, Any],
) -> dict[str, Any]:
    def counts(values: list[Any]) -> dict[str, int]:
        return {str(k): v for k, v in Counter(values).most_common()}

    by_family_features: dict[str, Counter[str]] = {}
    for a in alerts:
        by_family_features.setdefault(a.family, Counter()).update(f.name for f in a.top_features)
    return {
        "model": detector.name,
        "model_version": detector.version,
        "attribution_method": detector.attribution_method,
        "n_alerts": len(alerts),
        "by_predicted_family": counts([a.family for a in alerts]),
        "by_severity": counts([a.severity for a in alerts]),
        "by_true_label": counts([a.provenance.true_label for a in alerts]),
        "misclassified_family": sum(a.family != a.provenance.true_family for a in alerts),
        "false_positives_true_benign": sum(a.provenance.true_family == "BENIGN" for a in alerts),
        "by_confidence_band": counts([a.severity_basis.confidence_band_name for a in alerts]),
        "by_affected_asset_criticality": counts(
            [a.severity_basis.asset_criticality for a in alerts]
        ),
        "affected_asset_in_inventory": sum(
            (a.dst_host if a.affected_asset == "dst" else a.src_host) is not None for a in alerts
        ),
        "top_features_by_family": {
            fam: dict(c.most_common(5)) for fam, c in sorted(by_family_features.items())
        },
        "latency_per_alert_ms": {
            "median": round(float(np.median(latencies_ms)), 2) if latencies_ms else None,
            "p95": round(float(np.percentile(latencies_ms, 95)), 2) if latencies_ms else None,
            "includes": "detection + calibration + attribution + severity + Alert validation, "
            "one flow at a time, CPU",
        },
        "timestamps_afternoon_rule": stats.timestamps_afternoon_rule,
        "timestamps_unparsed": stats.timestamps_unparsed,
        **extra,
    }


def attribution_check(detector: Detector, X: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Sanity numbers for the report: SHAP additivity error, or IG completeness (|delta|)."""
    if isinstance(detector, LGBMDetector):
        values, expected = tree_shap(detector.booster, X, pred)
        margin = np.asarray(detector.booster.predict(X, raw_score=True), dtype=float)
        margin = np.atleast_2d(margin)[np.arange(len(X)), pred]
        err = np.abs(values.sum(axis=1) + expected - margin)
        return {"shap_additivity_max_abs_error": float(err.max())}
    values = detector.attribute(X, pred)  # IG on every chosen alert, for its convergence deltas
    delta = getattr(detector, "last_delta", None)
    if delta is None:
        return {}
    # Captum's delta = sum(attributions) − (f(x) − f(baseline)); relative to the logit change
    change = np.abs(values.sum(axis=1) - delta)
    relative = np.abs(delta) / np.where(change > 0, change, 1.0)
    return {
        "ig_completeness_max_abs_delta": round(float(np.abs(delta).max()), 6),
        "ig_completeness_median_relative_delta": round(float(np.median(relative)), 6),
    }


def run_build_alerts(
    processed_dir: Path,
    models_dir: Path,
    data_dir: Path,
    reports_dir: Path,
    model: str = "lgbm",
    n: int = 100,
    split: str = "test",
    seed: int = 42,
    rebuild_reference: bool = False,
    rebuild_inventory: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    rng = np.random.default_rng(seed)

    inventory_path = data_dir / "inventory.csv"
    if rebuild_inventory or not inventory_path.exists():
        inventory = inventory_from_flows(processed_dir / "flows.parquet", seed=seed)
        inventory.save(inventory_path)
        print(f"Wrote synthetic inventory: {len(inventory)} assets → {inventory_path}")
    inventory = Inventory.load(inventory_path)

    reference_path = models_dir / "benign_reference.npz"
    if rebuild_reference or not reference_path.exists():
        reference = build_benign_reference(processed_dir)
        reference.save(reference_path)
        print(f"Benign reference from {reference.n_rows:,} training flows → {reference_path}")
    reference = BenignReference.load(reference_path)

    detector = load_detector(model, models_dir, reference)
    builder = AlertBuilder(detector, inventory, reference)

    rows = load_split_rows(processed_dir, split)
    t0 = time.perf_counter()
    scores = detector.score(rows.select(detector.features).to_numpy().astype(np.float64))
    batch_seconds = time.perf_counter() - t0
    n_flagged = int((scores.pred != 0).sum())
    print(f"{split}: {rows.height:,} flows scored in {batch_seconds:.1f}s, {n_flagged:,} flagged")

    chosen = stratified_sample(scores.pred, detector.classes, n, rng)
    alerts: list[Alert] = []
    latencies: list[float] = []
    for i in chosen:
        t1 = time.perf_counter()
        built = builder.build(rows[int(i) : int(i) + 1])
        latencies.append((time.perf_counter() - t1) * 1000)
        alerts.extend(built)
    X_chosen = rows[chosen.tolist()].select(detector.features).to_numpy().astype(np.float64)
    check = attribution_check(detector, X_chosen, scores.pred[chosen])

    out_dir = data_dir / "alerts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"alerts_{model}.json"
    out_path.write_text(
        json.dumps([a.model_dump(mode="json") for a in alerts], indent=2, ensure_ascii=False) + "\n"
    )
    summary = _summary(
        alerts,
        latencies,
        detector,
        builder.stats,
        {
            "split": split,
            "flows_scored": rows.height,
            "flows_flagged": n_flagged,
            "flagged_by_family": {
                detector.classes[c]: int((scores.pred == c).sum())
                for c in range(1, len(detector.classes))
            },
            "attribution_check": check,
            "inventory_assets": len(inventory),
            "seed": seed,
            "total_seconds": round(time.perf_counter() - started, 1),
        },
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"alerts_summary_{model}.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Built {len(alerts)} alerts → {out_path}")
    print(f"  by family:   {summary['by_predicted_family']}")
    print(f"  by severity: {summary['by_severity']}")
    print(f"  median latency per alert {summary['latency_per_alert_ms']['median']} ms; {check}")
    return summary
