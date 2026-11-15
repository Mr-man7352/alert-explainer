"""LightGBM baseline detector (blueprint §6.1).

Pipeline:
1. Small grid search (on all attack rows + a BENIGN sample, for speed) with early stopping on
   validation macro-F1. The best config is refitted on the full training set.
2. Probability calibration: one-vs-rest isotonic regression on the ``cal`` half of validation.
   Calibrated probabilities are what the alert shows as "confidence".
3. Threshold, tuned on the whole validation set (it acts on raw scores, which the calibrator never
   sees): among thresholds with BENIGN FPR ≤ the target (default 1%), find the best macro-F1, then
   take the *lowest* threshold within ``tolerance`` (0.003) of it. The target is a ceiling, not a
   quota: the lowest threshold that merely meets it spends the whole 1% on false alarms and wrecks
   precision on the small families. Above that, macro-F1 is flat (moving the threshold trades a few
   Bot flows for a few false alarms), so within noise we prefer catching more attacks.
4. Decision rule: attack if ``1 − P_raw(BENIGN) ≥ threshold``; the family is then the most likely
   attack class. Otherwise BENIGN. (Thresholding the raw score keeps the cut-off fine-grained;
   isotonic calibration produces plateaus.)
5. Test metrics at that threshold (and at plain argmax, for comparison), the held-out
   Other-rare detection rate, latency, figures, and saved artefacts.
"""

from __future__ import annotations

import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from alert_explainer.data.labels import HELD_OUT_FAMILIES, TRAIN_FAMILIES
from alert_explainer.detect.evaluate import (
    brier_multiclass,
    detection_metrics,
    macro_f1_fast,
    plot_confusion,
    plot_pr_curves,
)

Config = dict[str, Any]

GRIDS: dict[str, list[Config]] = {
    "quick": [
        {"num_leaves": 63, "learning_rate": 0.1, "min_child_samples": 20, "weights": "sqrt"},
    ],
    "default": [
        {"num_leaves": 31, "learning_rate": 0.1, "min_child_samples": 20, "weights": "sqrt"},
        {"num_leaves": 63, "learning_rate": 0.1, "min_child_samples": 20, "weights": "sqrt"},
        {"num_leaves": 127, "learning_rate": 0.1, "min_child_samples": 20, "weights": "sqrt"},
        {"num_leaves": 63, "learning_rate": 0.05, "min_child_samples": 20, "weights": "sqrt"},
        {"num_leaves": 63, "learning_rate": 0.1, "min_child_samples": 100, "weights": "sqrt"},
        {"num_leaves": 63, "learning_rate": 0.1, "min_child_samples": 20, "weights": "balanced"},
        {"num_leaves": 127, "learning_rate": 0.05, "min_child_samples": 50, "weights": "balanced"},
        {"num_leaves": 31, "learning_rate": 0.1, "min_child_samples": 100, "weights": "none"},
    ],
}


def sanitize_feature_names(names: list[str]) -> list[str]:
    """LightGBM rejects some characters in feature names; map to unique [A-Za-z0-9_] names."""
    out: list[str] = []
    for name in names:
        clean = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_") or "f"
        candidate, i = clean, 1
        while candidate in out:
            i += 1
            candidate = f"{clean}_{i}"
        out.append(candidate)
    return out


def sample_weights(y: np.ndarray, scheme: str) -> np.ndarray:
    """Per-row weights: ``balanced`` (inverse frequency), ``sqrt`` (its square root) or ``none``."""
    if scheme == "none":
        return np.ones(len(y), dtype=np.float32)
    counts = np.maximum(np.bincount(y), 1)
    per_class = len(y) / (len(counts) * counts)
    if scheme == "sqrt":
        per_class = np.sqrt(per_class)
    elif scheme != "balanced":
        raise ValueError(f"Unknown weight scheme {scheme!r}")
    w = per_class[y]
    return (w / w.mean()).astype(np.float32)


class IsotonicCalibrator:
    """One-vs-rest isotonic calibration, renormalised so each row sums to 1."""

    def __init__(self) -> None:
        self.models_: list[IsotonicRegression] = []

    def fit(self, probs: np.ndarray, y: np.ndarray) -> IsotonicCalibrator:
        self.models_ = []
        for k in range(probs.shape[1]):
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(probs[:, k], (y == k).astype(float))
            self.models_.append(iso)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        cal = np.column_stack([m.predict(probs[:, k]) for k, m in enumerate(self.models_)])
        total = cal.sum(axis=1, keepdims=True)
        return np.where(total > 0, cal / np.where(total > 0, total, 1.0), probs)


def threshold_for_fpr(benign_scores: np.ndarray, target_fpr: float) -> float:
    """Smallest threshold t such that mean(benign_scores >= t) <= target_fpr."""
    s = np.sort(np.asarray(benign_scores, dtype=float))[::-1]
    if len(s) == 0:
        return 0.5
    k = int(np.floor(target_fpr * len(s)))
    if k >= len(s):
        return 0.0
    return float(np.nextafter(s[k], np.inf))


def predict_proba(booster: lgb.Booster, X: np.ndarray, iteration: int) -> np.ndarray:
    """Class probabilities as a 2-D float array (LightGBM's own return type is loose)."""
    return np.asarray(booster.predict(X, num_iteration=iteration), dtype=float)


def decide(probs: np.ndarray, threshold: float) -> np.ndarray:
    """BENIGN (0) unless the attack score clears the threshold; then the likeliest attack class."""
    attack_score = 1.0 - probs[:, 0]
    attack_family = probs[:, 1:].argmax(axis=1) + 1
    return np.where(attack_score >= threshold, attack_family, 0)


def choose_threshold(
    probs: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    target_fpr: float,
    tolerance: float = 0.003,
) -> tuple[float, float, list[dict[str, float]]]:
    """Return (chosen threshold, max-recall threshold, curve), all with BENIGN FPR <= target.

    Chosen = the lowest threshold whose macro-F1 is within ``tolerance`` of the best.
    Candidates: thresholds hitting BENIGN FPRs log-spaced from 1e-5 up to the target, and a
    linear grid over (0, 1).
    """
    benign = y == 0
    benign_scores = 1.0 - probs[benign, 0]
    candidates = {threshold_for_fpr(benign_scores, f) for f in np.geomspace(1e-5, target_fpr, 60)}
    candidates |= {float(t) for t in np.linspace(0.02, 0.98, 49)}
    curve = []
    for t in sorted(candidates):
        pred = decide(probs, t)
        fpr = float((pred[benign] != 0).mean())
        if fpr > target_fpr:
            continue
        curve.append(
            {
                "threshold": t,
                "benign_fpr": round(fpr, 6),
                "attack_detection_rate": round(float((pred[~benign] != 0).mean()), 5),
                "macro_f1": round(macro_f1_fast(y, pred, n_classes), 5),
            }
        )
    if not curve:
        t = threshold_for_fpr(benign_scores, target_fpr)
        return t, t, []
    best_f1 = max(r["macro_f1"] for r in curve)
    chosen = min(r["threshold"] for r in curve if r["macro_f1"] >= best_f1 - tolerance)
    max_recall = min(r["threshold"] for r in curve)
    return chosen, max_recall, curve


def train_one(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    cfg: Config,
    n_classes: int,
    feature_names: list[str],
    seed: int,
    max_rounds: int = 2000,
    patience: int = 50,
) -> tuple[lgb.Booster, int, float, float]:
    params = {
        "objective": "multiclass",
        "num_class": n_classes,
        "metric": "None",  # early stopping uses the custom macro-F1 below
        "num_leaves": cfg["num_leaves"],
        "learning_rate": cfg["learning_rate"],
        "min_child_samples": cfg["min_child_samples"],
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "seed": seed,
        "verbose": -1,
    }
    dtrain = lgb.Dataset(
        X_tr, label=y_tr, weight=sample_weights(y_tr, cfg["weights"]), feature_name=feature_names
    )
    dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)

    def macro_f1_eval(preds: np.ndarray, data: lgb.Dataset) -> tuple[str, float, bool]:
        y = np.asarray(data.get_label()).astype(int)
        p = preds if preds.ndim == 2 else preds.reshape(n_classes, -1).T
        return "macro_f1", macro_f1_fast(y, p.argmax(axis=1), n_classes), True

    t0 = time.perf_counter()
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=max_rounds,
        valid_sets=[dval],
        valid_names=["val"],
        feval=macro_f1_eval,
        callbacks=[lgb.early_stopping(patience, verbose=False)],
    )
    seconds = time.perf_counter() - t0
    best_iteration = booster.best_iteration or booster.current_iteration()
    return booster, best_iteration, float(booster.best_score["val"]["macro_f1"]), seconds


def _latency(
    booster: lgb.Booster,
    calibrator: IsotonicCalibrator,
    X: np.ndarray,
    threshold: float,
    best_iteration: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    idx = rng.choice(len(X), size=min(200, len(X)), replace=False)
    times = []
    for i in idx:
        t0 = time.perf_counter()
        raw = predict_proba(booster, X[i : i + 1], best_iteration)
        calibrator.transform(raw)
        decide(raw, threshold)
        times.append((time.perf_counter() - t0) * 1000)
    n = min(100_000, len(X))
    t0 = time.perf_counter()
    predict_proba(booster, X[:n], best_iteration)
    batch_seconds = time.perf_counter() - t0
    return {
        "single_alert_median_ms": round(float(np.median(times)), 3),
        "single_alert_p95_ms": round(float(np.percentile(times, 95)), 3),
        "batch_rows_per_second": round(n / batch_seconds),
    }


def train_lgbm(
    processed_dir: Path,
    models_dir: Path,
    reports_dir: Path,
    grid: str = "default",
    grid_benign_cap: int = 300_000,
    target_fpr: float = 0.01,
    seed: int = 42,
    max_rounds: int = 2000,
    patience: int = 50,
) -> dict[str, Any]:
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    schema = json.loads((processed_dir / "schema.json").read_text())
    features: list[str] = schema["features"]
    lgb_names = sanitize_feature_names(features)

    flows = pl.read_parquet(
        processed_dir / "flows.parquet", columns=["row_id", "label", "family", *features]
    )
    splits = pl.read_parquet(processed_dir / "splits.parquet")
    df = flows.join(splits, on="row_id", how="inner").sort("row_id")
    if df.height != flows.height:
        raise ValueError("splits.parquet does not match flows.parquet; re-run `ae split`")
    del flows

    present = set(df["family"].unique().to_list())
    classes = [f for f in TRAIN_FAMILIES if f in present]
    if classes[0] != "BENIGN":
        raise ValueError("BENIGN must be present")
    k = len(classes)

    y_all = (
        df["family"].replace_strict({c: i for i, c in enumerate(classes)}, default=-1).to_numpy()
    )
    X_all = df.select(features).to_numpy().astype(np.float32, copy=False)
    split = df["split"].to_numpy()
    part = df["val_part"].to_numpy()
    held_out = df["family"].is_in(list(HELD_OUT_FAMILIES)).to_numpy()
    labels_ho = df["label"].to_numpy()[held_out]
    del df  # keep only the numpy arrays from here on (saves ~1 GB on the full dataset)
    known = y_all >= 0
    m_tr = known & (split == "train")
    m_va = known & (split == "val")
    m_cal = m_va & (part == "cal")
    m_te = known & (split == "test")
    X_va, y_va = X_all[m_va], y_all[m_va]
    print(f"Train {m_tr.sum():,} | val {m_va.sum():,} | test {m_te.sum():,} | classes {classes}")

    # 1. Grid search
    tr_idx = np.flatnonzero(m_tr)
    benign_idx = tr_idx[y_all[tr_idx] == 0]
    subsampled = bool(grid_benign_cap) and len(benign_idx) > grid_benign_cap
    if subsampled:
        keep = rng.choice(benign_idx, size=grid_benign_cap, replace=False)
        grid_idx = np.sort(np.concatenate([keep, tr_idx[y_all[tr_idx] != 0]]))
    else:
        grid_idx = tr_idx
    grid_results, best_booster, best = [], None, None
    for i, cfg in enumerate(GRIDS[grid], 1):
        booster, it, score, secs = train_one(
            X_all[grid_idx],
            y_all[grid_idx],
            X_va,
            y_va,
            cfg,
            k,
            lgb_names,
            seed,
            max_rounds,
            patience,
        )
        grid_results.append(
            {
                **cfg,
                "best_iteration": it,
                "val_macro_f1": round(score, 5),
                "seconds": round(secs, 1),
            }
        )
        print(
            f"  grid {i}/{len(GRIDS[grid])} {cfg} → val macro-F1 {score:.4f} @ {it} ({secs:.0f}s)"
        )
        if best is None or score > best["val_macro_f1"]:
            best, best_booster = grid_results[-1], booster
    if best is None or best_booster is None:
        raise ValueError(f"Grid {grid!r} is empty")
    best_cfg = {key: best[key] for key in GRIDS[grid][0]}

    # Refit the best config on the full training set if the grid used a BENIGN sample
    if subsampled:
        print(f"Refitting best config on all {m_tr.sum():,} training rows")
        booster, best_iteration, val_f1, train_seconds = train_one(
            X_all[m_tr],
            y_all[m_tr],
            X_va,
            y_va,
            best_cfg,
            k,
            lgb_names,
            seed,
            max_rounds,
            patience,
        )
    else:
        booster, best_iteration = best_booster, best["best_iteration"]
        val_f1, train_seconds = best["val_macro_f1"], best["seconds"]

    # 2. Calibration on val/cal, 3. threshold on val/thr
    raw_cal = predict_proba(booster, X_all[m_cal], best_iteration)
    calibrator = IsotonicCalibrator().fit(raw_cal, y_all[m_cal])
    raw_thr = predict_proba(booster, X_va, best_iteration)
    y_thr = y_va
    threshold, max_recall_threshold, threshold_curve = choose_threshold(
        raw_thr, y_thr, k, target_fpr
    )
    pred_thr = decide(raw_thr, threshold)
    validation = {
        "macro_f1_argmax": round(val_f1, 4),
        "macro_f1_at_threshold": round(macro_f1_fast(y_thr, pred_thr, k), 4),
        "benign_fpr_at_threshold": round(float((pred_thr[y_thr == 0] != 0).mean()), 5),
        "attack_detection_rate_at_threshold": round(float((pred_thr[y_thr != 0] != 0).mean()), 4),
    }

    # 5. Test
    X_te, y_te = X_all[m_te], y_all[m_te]
    raw_te = predict_proba(booster, X_te, best_iteration)
    cal_te = calibrator.transform(raw_te)
    test_thr = detection_metrics(y_te, decide(raw_te, threshold), cal_te, classes)
    test_thr["brier_uncalibrated"] = round(brier_multiclass(raw_te, y_te), 5)
    test_arg = detection_metrics(y_te, raw_te.argmax(axis=1), cal_te, classes)
    test_max_recall = detection_metrics(y_te, decide(raw_te, max_recall_threshold), cal_te, classes)

    held_out_report: dict[str, Any] = {}
    if held_out.any():
        raw_ho = predict_proba(booster, X_all[held_out], best_iteration)
        pred_ho = decide(raw_ho, threshold)
        for label in sorted(set(labels_ho)):
            sel = labels_ho == label
            predicted = {classes[c]: int((pred_ho[sel] == c).sum()) for c in range(k)}
            held_out_report[label] = {
                "rows": int(sel.sum()),
                "flagged_as_attack": round(float((pred_ho[sel] != 0).mean()), 4),
                "predicted_family_counts": {c: n for c, n in predicted.items() if n},
            }

    gain = booster.feature_importance("gain", iteration=best_iteration)
    order = np.argsort(gain)[::-1][:15]
    top_features = [{"feature": features[i], "gain": round(float(gain[i]), 1)} for i in order]
    latency = _latency(booster, calibrator, X_te, threshold, best_iteration, rng)

    # Artefacts
    out = models_dir / "lgbm"
    out.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out / "model.txt"), num_iteration=best_iteration)
    joblib.dump(calibrator, out / "calibrator.joblib")
    meta = {
        "model": "lightgbm",
        "lightgbm_version": lgb.__version__,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "classes": classes,
        "held_out_families": list(HELD_OUT_FAMILIES),
        "features": features,
        "lightgbm_feature_names": lgb_names,
        "threshold": threshold,
        "target_benign_fpr": target_fpr,
        "decision_rule": "attack if 1 - P_raw(BENIGN) >= threshold, family = argmax over attacks",
        "threshold_rule": "lowest threshold within 0.003 of the best validation macro-F1, BENIGN FPR <= target",
        "confidence": "isotonic-calibrated probability of the predicted family (calibrator.joblib)",
        "params": best_cfg,
        "best_iteration": best_iteration,
        "seed": seed,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    figures = reports_dir / "figures"
    cm = np.array(test_thr["confusion_matrix"]["rows_true_cols_pred"])
    plot_confusion(
        cm,
        classes,
        figures / "lgbm_confusion_matrix.png",
        f"LightGBM, test set, threshold for BENIGN FPR ≤ {target_fpr:.0%}",
    )
    plot_pr_curves(y_te, cal_te, classes, figures / "lgbm_pr_curves.png")

    summary_keys = ("macro_f1", "macro_f1_attacks_only", "benign_fpr", "attack_detection_rate")
    metrics = {
        "model": "lightgbm",
        "generated_at": meta["trained_at"],
        "classes": classes,
        "rows": {
            "train": int(m_tr.sum()),
            "val": int(m_va.sum()),
            "test": int(m_te.sum()),
            "held_out": int(held_out.sum()),
        },
        "best_params": best_cfg,
        "best_iteration": best_iteration,
        "threshold": threshold,
        "target_benign_fpr": target_fpr,
        "validation": validation,
        "test_at_threshold": test_thr,
        "test_argmax": {key: test_arg[key] for key in summary_keys},
        "test_at_max_recall_threshold": {
            "threshold": max_recall_threshold,
            "note": "lowest threshold with BENIGN FPR <= target: maximum recall, poor precision",
            **{key: test_max_recall[key] for key in summary_keys},
            "precision_by_family": {
                c: v["precision"] for c, v in test_max_recall["per_family"].items()
            },
        },
        "threshold_rule": "lowest threshold within 0.003 of the best validation macro-F1, BENIGN FPR <= target",
        "threshold_curve_validation": threshold_curve,
        "held_out_families": held_out_report,
        "top_features_by_gain": top_features,
        "training_seconds_final_fit": round(train_seconds, 1),
        "latency": latency,
        "grid": {
            "name": grid,
            "benign_sample_cap": grid_benign_cap if subsampled else None,
            "results": grid_results,
        },
        "total_seconds": round(time.perf_counter() - started, 1),
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "lgbm_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    t = test_thr
    print(
        f"\nTEST @ threshold {threshold:.4f}: macro-F1 {t['macro_f1']:.4f} | "
        f"BENIGN FPR {t['benign_fpr']:.4%} | attack detection {t['attack_detection_rate']:.4f}"
    )
    for fam, m in t["per_family"].items():
        print(
            f"  {fam:<11} P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f}  n={m['support']:,}"
        )
    for label, r in held_out_report.items():
        print(f"  held-out {label}: {r['flagged_as_attack']:.0%} of {r['rows']} flagged as attack")
    print(f"Median single-alert latency {latency['single_alert_median_ms']} ms")
    print(f"Saved {out}/ and {reports_dir / 'lgbm_metrics.json'}")
    return metrics
