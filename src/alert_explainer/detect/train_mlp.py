"""PyTorch MLP pipeline (blueprint §6.2–6.3; Days 4–5). Run with ``uv run ae train-mlp``.

Steps:
1. Fit ``FlowPreprocessor`` on the training rows; write the transformed train / val / test /
   held-out arrays to ``data/processed/mlp/*.npy``. Training reads them as memory maps.
2. Grid search (Day 5) on all attack rows plus a BENIGN sample, ranked by the best validation
   macro-F1 over each run's full OneCycle schedule. ``--grid none`` skips the search and uses the default config (Day 4).
3. Final runs on the full training set: the best config with cross-entropy, and the same config
   with focal loss (the ablation; ``--no-ablation`` skips it). The deployed model is the one with
   the higher *validation* macro-F1 at its operating threshold, so the test set never picks the
   winner.
4. Exactly the LightGBM treatment from Day 3, for a fair comparison. One-vs-rest isotonic
   calibration is fitted on the ``cal`` half of validation. The threshold rule is the same:
   attack if ``1 − P_raw(BENIGN) ≥ t``, with t = the lowest threshold within 0.003 of the best
   validation macro-F1 while BENIGN FPR ≤ 1%. The test metrics are the same as well.
5. Save ``models/mlp/`` (weights, preprocessor, calibrator, meta), ``reports/mlp_metrics.json``
   and figures.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from alert_explainer.data.labels import HELD_OUT_FAMILIES
from alert_explainer.detect.data import DetectionData, load_detection_data
from alert_explainer.detect.evaluate import (
    brier_multiclass,
    detection_metrics,
    macro_f1_fast,
    plot_confusion,
    plot_pr_curves,
)
from alert_explainer.detect.lgbm import IsotonicCalibrator, choose_threshold, decide
from alert_explainer.detect.torch_mlp import (
    FitResult,
    FlowDataset,
    FlowPreprocessor,
    MLPConfig,
    available_devices,
    fit_mlp,
    load_model,
    n_parameters,
    pick_device,
    predict_proba,
    save_model,
)

BASE = MLPConfig()
GRIDS: dict[str, list[MLPConfig]] = {
    "none": [BASE],
    "quick": [BASE, replace(BASE, lr=1e-3), replace(BASE, weights="balanced")],
    "default": [
        BASE,
        replace(BASE, lr=1e-3),
        replace(BASE, lr=1e-2),
        replace(BASE, dropout=0.1),
        replace(BASE, weights="balanced"),
        replace(BASE, weights="none"),
        replace(BASE, hidden=(512, 256, 128)),
    ],
}
SUMMARY_KEYS = ("macro_f1", "macro_f1_attacks_only", "benign_fpr", "attack_detection_rate")


def export_arrays(
    data: DetectionData,
    pre: FlowPreprocessor,
    out_dir: Path,
    masks: dict[str, np.ndarray],
    chunk_rows: int = 262_144,
) -> dict[str, int]:
    """Write ``<name>_X.npy`` (preprocessed float32) and ``<name>_y.npy`` for each mask.

    X is written straight into a memory-mapped ``.npy`` chunk by chunk, so the full transformed
    copy never sits in RAM next to the raw one.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: dict[str, int] = {}
    for name, mask in masks.items():
        idx = np.flatnonzero(mask)
        X_out = np.lib.format.open_memmap(
            out_dir / f"{name}_X.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(idx), data.X.shape[1]),
        )
        for start in range(0, len(idx), chunk_rows):
            X_out[start : start + chunk_rows] = pre.transform(
                data.X[idx[start : start + chunk_rows]]
            )
        X_out.flush()
        del X_out
        np.save(out_dir / f"{name}_y.npy", data.y[idx].astype(np.int64))
        rows[name] = len(idx)
    return rows


def evaluate_run(
    fit: FitResult,
    device: torch.device,
    X_va: np.ndarray,
    y_va: np.ndarray,
    cal_in_val: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    classes: list[str],
    target_fpr: float,
) -> dict[str, Any]:
    """Calibrate, pick the threshold on validation, and score the test set (as in ``lgbm.py``)."""
    k = len(classes)
    raw_va = predict_proba(fit.model, X_va, device)
    calibrator = IsotonicCalibrator().fit(raw_va[cal_in_val], y_va[cal_in_val])
    threshold, max_recall_threshold, curve = choose_threshold(raw_va, y_va, k, target_fpr)
    pred_va = decide(raw_va, threshold)
    validation = {
        "macro_f1_argmax": round(macro_f1_fast(y_va, raw_va.argmax(axis=1), k), 4),
        "macro_f1_at_threshold": round(macro_f1_fast(y_va, pred_va, k), 4),
        "benign_fpr_at_threshold": round(float((pred_va[y_va == 0] != 0).mean()), 5),
        "attack_detection_rate_at_threshold": round(float((pred_va[y_va != 0] != 0).mean()), 4),
    }
    raw_te = predict_proba(fit.model, X_te, device)
    cal_te = calibrator.transform(raw_te)
    test_thr = detection_metrics(y_te, decide(raw_te, threshold), cal_te, classes)
    test_thr["brier_uncalibrated"] = round(brier_multiclass(raw_te, y_te), 5)
    test_arg = detection_metrics(y_te, raw_te.argmax(axis=1), cal_te, classes)
    test_max_recall = detection_metrics(y_te, decide(raw_te, max_recall_threshold), cal_te, classes)
    return {
        "calibrator": calibrator,
        "threshold": threshold,
        "max_recall_threshold": max_recall_threshold,
        "threshold_curve": curve,
        "validation": validation,
        "test_at_threshold": test_thr,
        "test_argmax": {key: test_arg[key] for key in SUMMARY_KEYS},
        "test_at_max_recall_threshold": {
            "threshold": max_recall_threshold,
            "note": "lowest threshold with BENIGN FPR <= target: maximum recall, poor precision",
            **{key: test_max_recall[key] for key in SUMMARY_KEYS},
        },
        "cal_te": cal_te,
    }


def _latency(
    model_cpu: torch.nn.Module,
    pre: FlowPreprocessor,
    calibrator: IsotonicCalibrator,
    X_raw: np.ndarray,
    threshold: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """CPU latency including preprocessing (how a single-machine API would serve it)."""
    model_cpu.eval()
    idx = rng.choice(len(X_raw), size=min(200, len(X_raw)), replace=False)
    times = []
    with torch.inference_mode():
        for i in idx:
            t0 = time.perf_counter()
            x = torch.from_numpy(pre.transform(X_raw[i : i + 1]))
            raw = torch.softmax(model_cpu(x), dim=1).numpy().astype(np.float64)
            calibrator.transform(raw)
            decide(raw, threshold)
            times.append((time.perf_counter() - t0) * 1000)
        n = min(100_000, len(X_raw))
        t0 = time.perf_counter()
        Xp = pre.transform(X_raw[:n])
        for start in range(0, n, 65_536):
            torch.softmax(model_cpu(torch.from_numpy(Xp[start : start + 65_536])), dim=1)
        batch_seconds = time.perf_counter() - t0
    return {
        "device": "cpu",
        "single_alert_median_ms": round(float(np.median(times)), 3),
        "single_alert_p95_ms": round(float(np.percentile(times, 95)), 3),
        "batch_rows_per_second": round(n / batch_seconds),
    }


def plot_training_curves(runs: dict[str, list[dict[str, float]]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(11, 4))
    for name, history in runs.items():
        epochs = [h["epoch"] for h in history]
        ax_loss.plot(epochs, [h["val_log_loss"] for h in history], marker="o", ms=3, label=name)
        ax_f1.plot(epochs, [h["val_macro_f1"] for h in history], marker="o", ms=3, label=name)
    ax_loss.set_title("Validation log-loss (unweighted)")
    ax_f1.set_title("Validation macro-F1 (argmax)")
    for ax in (ax_loss, ax_f1):
        ax.set_xlabel("Epoch")
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train_mlp(
    processed_dir: Path,
    models_dir: Path,
    reports_dir: Path,
    grid: str = "default",
    grid_benign_cap: int = 300_000,
    target_fpr: float = 0.01,
    seed: int = 42,
    device: str = "auto",
    max_epochs: int | None = None,
    batch_size: int | None = None,
    ablation: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    dev = pick_device(device)
    overrides: dict[str, Any] = {}
    if max_epochs is not None:
        overrides["max_epochs"] = max_epochs
    if batch_size is not None:
        overrides["batch_size"] = batch_size
    configs = [replace(c, **overrides) for c in GRIDS[grid]]
    print(f"PyTorch {torch.__version__} on {dev}")

    # 1. Data, preprocessing, memory-mapped arrays
    data = load_detection_data(processed_dir)
    classes, k = data.classes, data.n_classes
    t0 = time.perf_counter()
    pre = FlowPreprocessor(seed=seed).fit(data.X[data.train])
    arrays_dir = processed_dir / "mlp"
    rows = export_arrays(
        data,
        pre,
        arrays_dir,
        {"train": data.train, "val": data.val, "test": data.test, "held_out": data.held_out},
    )
    cal_in_val = data.cal[data.val]
    X_te_raw = data.X[data.test]
    labels_ho = np.array(data.label_names, dtype=object)[data.label_code[data.held_out]]
    del data  # free the raw feature matrix (~0.7 GB on the full dataset)
    print(
        f"Preprocessed and wrote {arrays_dir}/ in {time.perf_counter() - t0:.0f}s: "
        f"train {rows['train']:,} | val {rows['val']:,} | test {rows['test']:,} | classes {classes}"
    )

    train_ds = FlowDataset.from_npy(arrays_dir, "train")
    val_ds = FlowDataset.from_npy(arrays_dir, "val", mmap=False)
    test_ds = FlowDataset.from_npy(arrays_dir, "test", mmap=False)
    X_va, y_va = val_ds.X, np.asarray(val_ds.y)
    X_te, y_te = test_ds.X, np.asarray(test_ds.y)
    y_tr = train_ds.labels()

    # 2. Grid search on a BENIGN sample + all attacks
    grid_results: list[dict[str, Any]] = []
    best_cfg = configs[0]
    benign_idx = np.flatnonzero(y_tr == 0)
    subsampled = bool(grid_benign_cap) and len(benign_idx) > grid_benign_cap
    if len(configs) > 1:
        if subsampled:
            keep = rng.choice(benign_idx, size=grid_benign_cap, replace=False)
            grid_ds = FlowDataset(
                train_ds.X, train_ds.y, np.concatenate([keep, np.flatnonzero(y_tr != 0)])
            )
        else:
            grid_ds = train_ds
        print(f"Grid '{grid}': {len(configs)} configs on {len(grid_ds):,} training rows")
        best_score = -1.0
        for i, cfg in enumerate(configs, 1):
            print(f"  grid {i}/{len(configs)} {cfg.short()}")
            fit = fit_mlp(grid_ds, X_va, y_va, cfg, k, dev, seed, log=print)
            grid_results.append(
                {
                    **cfg.to_dict(),
                    "best_epoch": fit.best_epoch,
                    "epochs_run": len(fit.history),
                    "val_macro_f1": round(fit.best_val_macro_f1, 5),
                    "seconds": round(fit.seconds, 1),
                }
            )
            print(
                f"  → val macro-F1 {fit.best_val_macro_f1:.4f} @ epoch {fit.best_epoch} ({fit.seconds:.0f}s)"
            )
            if fit.best_val_macro_f1 > best_score:
                best_score, best_cfg = fit.best_val_macro_f1, cfg

    # 3. Final runs on all training rows: cross-entropy, and focal loss as the ablation
    variants = {"ce": replace(best_cfg, loss="ce")}
    if ablation:
        variants["focal"] = replace(best_cfg, loss="focal")
    runs: dict[str, dict[str, Any]] = {}
    fits: dict[str, FitResult] = {}
    for name, cfg in variants.items():
        print(f"Final fit [{name}] on all {len(train_ds):,} training rows: {cfg.short()}")
        fit = fit_mlp(train_ds, X_va, y_va, cfg, k, dev, seed, log=print)
        result = evaluate_run(fit, dev, X_va, y_va, cal_in_val, X_te, y_te, classes, target_fpr)
        fits[name], runs[name] = fit, {"config": cfg, **result}
        t = result["test_at_threshold"]
        print(
            f"  [{name}] best epoch {fit.best_epoch}/{len(fit.history)} in {fit.seconds:.0f}s | "
            f"val macro-F1 @thr {result['validation']['macro_f1_at_threshold']:.4f} | "
            f"test macro-F1 @thr {t['macro_f1']:.4f}"
        )
    deployed = max(runs, key=lambda n: (runs[n]["validation"]["macro_f1_at_threshold"], n == "ce"))
    run, fit = runs[deployed], fits[deployed]
    cfg = run["config"]
    print(f"Deploying [{deployed}] (higher validation macro-F1 at threshold)")

    # 4. Held-out families, latency
    held_out_report: dict[str, Any] = {}
    X_ho = np.load(arrays_dir / "held_out_X.npy")
    if len(X_ho):
        pred_ho = decide(predict_proba(fit.model, X_ho, dev), run["threshold"])
        for label in sorted(set(labels_ho)):
            sel = labels_ho == label
            predicted = {classes[c]: int((pred_ho[sel] == c).sum()) for c in range(k)}
            held_out_report[label] = {
                "rows": int(sel.sum()),
                "flagged_as_attack": round(float((pred_ho[sel] != 0).mean()), 4),
                "predicted_family_counts": {c: n for c, n in predicted.items() if n},
            }

    # 5. Artefacts
    out = models_dir / "mlp"
    save_model(fit.model, cfg, classes, out / "model.pt")
    joblib.dump(pre, out / "preprocessor.joblib")
    joblib.dump(run["calibrator"], out / "calibrator.joblib")
    model_cpu, _, _ = load_model(out / "model.pt")  # also proves the saved weights reload
    latency = _latency(model_cpu, pre, run["calibrator"], X_te_raw, run["threshold"], rng)
    trained_at = datetime.now(UTC).isoformat(timespec="seconds")
    meta = {
        "model": "pytorch-mlp",
        "torch_version": torch.__version__,
        "trained_at": trained_at,
        "device": str(dev),
        "classes": classes,
        "held_out_families": list(HELD_OUT_FAMILIES),
        "features": json.loads((processed_dir / "schema.json").read_text())["features"],
        "preprocessing": "QuantileTransformer(normal, 1000 quantiles) fitted on train (preprocessor.joblib)",
        "config": cfg.to_dict(),
        "loss": deployed,
        "n_parameters": n_parameters(fit.model),
        "best_epoch": fit.best_epoch,
        "threshold": run["threshold"],
        "target_benign_fpr": target_fpr,
        "decision_rule": "attack if 1 - P_raw(BENIGN) >= threshold, family = argmax over attacks",
        "threshold_rule": "lowest threshold within 0.003 of the best validation macro-F1, BENIGN FPR <= target",
        "confidence": "isotonic-calibrated probability of the predicted family (calibrator.joblib)",
        "seed": seed,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    figures = reports_dir / "figures"
    test_thr = run["test_at_threshold"]
    plot_confusion(
        np.array(test_thr["confusion_matrix"]["rows_true_cols_pred"]),
        classes,
        figures / "mlp_confusion_matrix.png",
        f"PyTorch MLP, test set, threshold for BENIGN FPR ≤ {target_fpr:.0%}",
    )
    plot_pr_curves(y_te, run["cal_te"], classes, figures / "mlp_pr_curves.png")
    plot_training_curves(
        {n: f.history for n, f in fits.items()}, figures / "mlp_training_curves.png"
    )

    ablation_report = {
        name: {
            "config": r["config"].to_dict(),
            "deployed": name == deployed,
            "best_epoch": fits[name].best_epoch,
            "epochs_run": len(fits[name].history),
            "training_seconds": round(fits[name].seconds, 1),
            "threshold": r["threshold"],
            "validation": r["validation"],
            "test_at_threshold": {
                key: r["test_at_threshold"][key] for key in (*SUMMARY_KEYS, "brier")
            },
            "test_f1_by_family": {
                c: v["f1"] for c, v in r["test_at_threshold"]["per_family"].items()
            },
            "history": fits[name].history,
        }
        for name, r in runs.items()
    }
    metrics = {
        "model": "pytorch-mlp",
        "generated_at": trained_at,
        "device": str(dev),
        "torch_version": torch.__version__,
        "classes": classes,
        "rows": {
            "train": rows["train"],
            "val": rows["val"],
            "test": rows["test"],
            "held_out": rows["held_out"],
        },
        "config": cfg.to_dict(),
        "loss": deployed,
        "n_parameters": meta["n_parameters"],
        "best_epoch": fit.best_epoch,
        "threshold": run["threshold"],
        "target_benign_fpr": target_fpr,
        "validation": run["validation"],
        "test_at_threshold": test_thr,
        "test_argmax": run["test_argmax"],
        "test_at_max_recall_threshold": run["test_at_max_recall_threshold"],
        "threshold_rule": meta["threshold_rule"],
        "threshold_curve_validation": run["threshold_curve"],
        "held_out_families": held_out_report,
        "training_seconds_final_fit": round(fit.seconds, 1),
        "latency": latency,
        "ablation_loss": ablation_report,
        "grid": {
            "name": grid,
            "benign_sample_cap": grid_benign_cap if (subsampled and len(configs) > 1) else None,
            "results": grid_results,
        },
        "total_seconds": round(time.perf_counter() - started, 1),
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "mlp_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    print(
        f"\nTEST @ threshold {run['threshold']:.4f}: macro-F1 {test_thr['macro_f1']:.4f} | "
        f"BENIGN FPR {test_thr['benign_fpr']:.4%} | attack detection {test_thr['attack_detection_rate']:.4f}"
    )
    for fam, m in test_thr["per_family"].items():
        print(
            f"  {fam:<11} P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f}  n={m['support']:,}"
        )
    for label, r in held_out_report.items():
        print(f"  held-out {label}: {r['flagged_as_attack']:.0%} of {r['rows']} flagged as attack")
    print(f"Median single-alert latency (CPU) {latency['single_alert_median_ms']} ms")
    print(f"Saved {out}/ and {reports_dir / 'mlp_metrics.json'}")
    return metrics


def bench_devices(
    processed_dir: Path, rows: int = 300_000, epochs: int = 2, seed: int = 42
) -> dict[str, Any]:
    """Time training epochs on every available device (CPU, MPS, CUDA) on a sample of rows.

    The last epoch's training time is used (the first includes warm-up such as MPS kernel
    compilation), then scaled to the full training set.
    """
    data = load_detection_data(processed_dir)
    rng = np.random.default_rng(seed)
    train_idx, val_idx = np.flatnonzero(data.train), np.flatnonzero(data.val)
    pick = rng.choice(train_idx, size=min(rows, len(train_idx)), replace=False)
    val_pick = rng.choice(val_idx, size=min(50_000, len(val_idx)), replace=False)
    pre = FlowPreprocessor(seed=seed).fit(data.X[pick])
    ds = FlowDataset(pre.transform(data.X[pick]), data.y[pick])
    X_va, y_va = pre.transform(data.X[val_pick]), data.y[val_pick]
    cfg = MLPConfig(max_epochs=epochs, patience=epochs)
    print(f"Benchmark: {len(pick):,} rows, batch {cfg.batch_size}, {epochs} epochs per device")
    results: dict[str, Any] = {}
    for name in available_devices():
        fit = fit_mlp(
            ds, X_va, y_va, cfg, data.n_classes, torch.device(name), seed, log=lambda _: None
        )
        steady = fit.history[-1]["train_seconds"]
        results[name] = {
            "seconds_per_epoch_on_sample": steady,
            "estimated_seconds_per_full_epoch": round(steady * len(train_idx) / len(pick), 1),
        }
        print(
            f"  {name:<5} {steady:6.2f} s/epoch on the sample → about "
            f"{results[name]['estimated_seconds_per_full_epoch']:.0f} s per full epoch "
            f"({len(train_idx):,} rows)"
        )
    fastest = min(results, key=lambda d: results[d]["seconds_per_epoch_on_sample"])
    print(f"Fastest: {fastest}. Put AE_DEVICE={fastest} in .env (or pass --device {fastest}).")
    return {"rows": len(pick), "epochs": epochs, "results": results, "fastest": fastest}


def load_mlp_detector(models_dir: Path) -> MLPDetector:
    return MLPDetector(models_dir / "mlp")


class MLPDetector:
    """The saved MLP as one object, for the alert builder (Day 6): raw features in, decision out."""

    def __init__(self, model_dir: Path) -> None:
        self.model, self.config, self.classes = load_model(model_dir / "model.pt")
        self.preprocessor: FlowPreprocessor = joblib.load(model_dir / "preprocessor.joblib")
        self.calibrator: IsotonicCalibrator = joblib.load(model_dir / "calibrator.joblib")
        self.meta: dict[str, Any] = json.loads((model_dir / "meta.json").read_text())
        self.threshold = float(self.meta["threshold"])

    def predict(self, X_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (class index per row, calibrated probabilities)."""
        raw = predict_proba(self.model, self.preprocessor.transform(X_raw), torch.device("cpu"))
        return decide(raw, self.threshold), self.calibrator.transform(raw)
