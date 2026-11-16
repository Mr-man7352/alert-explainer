"""Cross-day robustness check (blueprint §5, revised on Day 3). Run with ``uv run ae cross-day``.

The blueprint planned to train on Mon–Wed and test on Thu–Fri *for families present on both
sides*. In CIC-IDS2017 there are none: BruteForce is Tuesday, DoS Wednesday, WebAttack Thursday,
and Bot, PortScan and DDoS Friday. So this is a **binary unseen-attack test** instead:

- Train an attack-vs-BENIGN detector on Mon–Wed only (BENIGN, BruteForce, DoS). An 85/15 stratified
  split gives the validation rows for early stopping and the threshold. Other-rare is excluded
  from training, as in the main models.
- On Thu–Fri, every attack is an attack type the model has never seen. Measure how many of each
  are flagged at all, and how much BENIGN FPR rises on the new days' normal traffic.
- Two operating points: the threshold for BENIGN FPR ≤ 1% on Mon–Wed validation (the most
  permissive within the false-alarm budget), and a plain 0.5. Also threshold-free AUROC and
  average precision.
- Both detectors use their Day 3 / Day 4–5 settings (LightGBM params from ``models/lgbm/meta.json``,
  MLP config from ``models/mlp/meta.json``; defaults if those don't exist yet). The MLP's quantile
  transform is refitted on Mon–Wed training rows only, so nothing about Thu–Fri leaks in.

The contrast with the random-split scores is the point: random splits flatter IDS models.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from alert_explainer.detect.data import load_detection_data
from alert_explainer.detect.lgbm import (
    GRIDS as LGBM_GRIDS,
)
from alert_explainer.detect.lgbm import (
    predict_proba as lgbm_predict_proba,
)
from alert_explainer.detect.lgbm import (
    sanitize_feature_names,
    threshold_for_fpr,
    train_one,
)
from alert_explainer.detect.torch_mlp import (
    FlowDataset,
    FlowPreprocessor,
    MLPConfig,
    fit_mlp,
    pick_device,
    predict_proba,
)

EARLY_DAYS = ["Monday", "Tuesday", "Wednesday"]
LATE_DAYS = ["Thursday", "Friday"]


def _operating_point(
    scores: np.ndarray, is_attack: np.ndarray, labels: np.ndarray, threshold: float
) -> dict[str, Any]:
    flagged = scores >= threshold
    by_label = {
        str(label): {
            "rows": int((labels == label).sum()),
            "flagged": round(float(flagged[labels == label].mean()), 4),
        }
        for label in sorted(set(labels[is_attack].tolist()))
    }
    return {
        "threshold": round(float(threshold), 6),
        "benign_fpr": round(float(flagged[~is_attack].mean()), 5) if (~is_attack).any() else None,
        "attack_detection_rate": round(float(flagged[is_attack].mean()), 4)
        if is_attack.any()
        else None,
        "by_label": by_label,
    }


def _score_report(
    val_scores: np.ndarray,
    val_attack: np.ndarray,
    val_labels: np.ndarray,
    late_scores: np.ndarray,
    late_attack: np.ndarray,
    late_labels: np.ndarray,
    target_fpr: float,
) -> dict[str, Any]:
    t_fpr = threshold_for_fpr(val_scores[~val_attack], target_fpr)
    both = late_attack.any() and (~late_attack).any()
    return {
        "in_period_validation": {
            f"at_fpr_{target_fpr:g}": _operating_point(val_scores, val_attack, val_labels, t_fpr),
            "at_0.5": _operating_point(val_scores, val_attack, val_labels, 0.5),
        },
        "unseen_days": {
            "auroc": round(float(roc_auc_score(late_attack, late_scores)), 4) if both else None,
            "average_precision": (
                round(float(average_precision_score(late_attack, late_scores)), 4) if both else None
            ),
            f"at_fpr_{target_fpr:g}": _operating_point(
                late_scores, late_attack, late_labels, t_fpr
            ),
            "at_0.5": _operating_point(late_scores, late_attack, late_labels, 0.5),
        },
    }


def plot_cross_day(report: dict[str, Any], key: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(report["models"])
    labels = sorted(
        {lab for m in models for lab in report["models"][m]["unseen_days"][key]["by_label"]}
    )
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(labels) * len(models) + 1.5))
    height = 0.8 / len(models)
    y = np.arange(len(labels))
    for i, m in enumerate(models):
        by_label = report["models"][m]["unseen_days"][key]["by_label"]
        values = [by_label.get(lab, {}).get("flagged", 0.0) for lab in labels]
        name = {"lgbm": "LightGBM", "mlp": "PyTorch MLP"}.get(m, m)
        bars = ax.barh(y + i * height, values, height, label=name)
        ax.bar_label(bars, labels=[f"{v:.0%}" for v in values], fontsize=7, padding=2)
    ax.set_yticks(y + height * (len(models) - 1) / 2, labels)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Share flagged as attack")
    fpr = report["setup"]["target_benign_fpr"]
    ax.set_title(
        "Unseen Thu–Fri attacks flagged by a detector trained on Mon–Wed\n"
        f"(threshold: BENIGN FPR ≤ {fpr:.0%} on Mon–Wed validation)",
        fontsize=10,
    )
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_cross_day(
    processed_dir: Path,
    models_dir: Path,
    reports_dir: Path,
    models: tuple[str, ...] = ("lgbm", "mlp"),
    target_fpr: float = 0.01,
    seed: int = 42,
    device: str = "auto",
    max_epochs: int | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    data = load_detection_data(processed_dir)
    label_names = np.array(data.label_names, dtype=object)
    is_attack_all = data.y != 0  # held-out rows (y = -1) count as attacks too

    pool = np.flatnonzero(data.early & ~data.held_out)
    tr_idx, va_idx = train_test_split(
        pool, test_size=0.15, stratify=data.y[pool], random_state=seed
    )
    late_idx = np.flatnonzero(data.late)
    y_tr = is_attack_all[tr_idx].astype(np.int64)
    y_va = is_attack_all[va_idx].astype(np.int64)
    va_attack, late_attack = y_va == 1, is_attack_all[late_idx]
    va_labels = label_names[data.label_code[va_idx]]
    late_labels = label_names[data.label_code[late_idx]]
    seen = sorted(set(va_labels[va_attack].tolist()))
    unseen = sorted(set(late_labels[late_attack].tolist()))
    print(
        f"Cross-day: train {len(tr_idx):,} / val {len(va_idx):,} Mon–Wed rows "
        f"(attacks seen: {seen}); test {len(late_idx):,} Thu–Fri rows (attacks unseen: {unseen})"
    )

    report: dict[str, Any] = {
        "setup": {
            "train_days": EARLY_DAYS,
            "test_days": LATE_DAYS,
            "task": "binary attack vs BENIGN; every Thu–Fri attack type is absent from training",
            "rows": {"train": len(tr_idx), "val": len(va_idx), "test": len(late_idx)},
            "attack_labels_seen_in_training": seen,
            "attack_labels_unseen": unseen,
            "target_benign_fpr": target_fpr,
            "excluded": "Other-rare (Heartbleed on Wednesday) left out of training, as in the main models",
            "seed": seed,
        },
        "models": {},
    }

    if "lgbm" in models:
        meta_path = models_dir / "lgbm" / "meta.json"
        params = (
            json.loads(meta_path.read_text())["params"]
            if meta_path.exists()
            else LGBM_GRIDS["quick"][0]
        )
        print(f"LightGBM (binary) with {params}")
        booster, it, _, secs = train_one(
            data.X[tr_idx], y_tr, data.X[va_idx], y_va, params, 2,
            sanitize_feature_names(data.features), seed,
        )  # fmt: skip
        va_scores = lgbm_predict_proba(booster, data.X[va_idx], it)[:, 1]
        late_scores = lgbm_predict_proba(booster, data.X[late_idx], it)[:, 1]
        report["models"]["lgbm"] = {
            "params": params,
            "best_iteration": it,
            "training_seconds": round(secs, 1),
            **_score_report(
                va_scores, va_attack, va_labels, late_scores, late_attack, late_labels, target_fpr
            ),
        }

    if "mlp" in models:
        meta_path = models_dir / "mlp" / "meta.json"
        cfg = (
            MLPConfig.from_dict(json.loads(meta_path.read_text())["config"])
            if meta_path.exists()
            else MLPConfig()
        )
        overrides: dict[str, Any] = {}
        if max_epochs is not None:
            overrides["max_epochs"] = max_epochs
        if batch_size is not None:
            overrides["batch_size"] = batch_size
        cfg = MLPConfig.from_dict({**cfg.to_dict(), **overrides})
        dev = pick_device(device)
        print(f"PyTorch MLP (binary) on {dev}: {cfg.short()}")
        pre = FlowPreprocessor(seed=seed).fit(data.X[tr_idx])
        X_va = pre.transform(data.X[va_idx])
        fit = fit_mlp(
            FlowDataset(pre.transform(data.X[tr_idx]), y_tr), X_va, y_va, cfg, 2, dev, seed
        )
        va_scores = predict_proba(fit.model, X_va, dev)[:, 1]
        late_scores = predict_proba(fit.model, pre.transform(data.X[late_idx]), dev)[:, 1]
        report["models"]["mlp"] = {
            "config": cfg.to_dict(),
            "best_epoch": fit.best_epoch,
            "training_seconds": round(fit.seconds, 1),
            **_score_report(
                va_scores, va_attack, va_labels, late_scores, late_attack, late_labels, target_fpr
            ),
        }

    report["total_seconds"] = round(time.perf_counter() - started, 1)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "cross_day.json").write_text(json.dumps(report, indent=2) + "\n")
    key = f"at_fpr_{target_fpr:g}"
    plot_cross_day(report, key, reports_dir / "figures" / "cross_day_detection.png")

    for name, r in report["models"].items():
        u, v = r["unseen_days"], r["in_period_validation"][key]
        print(
            f"\n[{name}] Mon–Wed val: detection {v['attack_detection_rate']} at FPR {v['benign_fpr']} | "
            f"Thu–Fri: AUROC {u['auroc']}, detection {u[key]['attack_detection_rate']} "
            f"at FPR {u[key]['benign_fpr']}"
        )
        for label, s in u[key]["by_label"].items():
            print(f"  {label:<28} {s['flagged']:>7.1%} of {s['rows']:,}")
    print(f"Saved {reports_dir / 'cross_day.json'}")
    return report
