"""Detection metrics and figures (blueprint §6.3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)


def macro_f1_fast(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Macro-F1 over all ``n_classes`` (absent classes score 0), via one bincount.

    Same result as ``f1_score(average="macro", labels=range(n), zero_division=0)``, but fast
    enough to run every boosting round for early stopping.
    """
    cm = np.bincount(y_true * n_classes + y_pred, minlength=n_classes * n_classes).reshape(
        n_classes, n_classes
    )
    tp = np.diag(cm).astype(float)
    denom = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1.mean())


def brier_multiclass(probs: np.ndarray, y_true: np.ndarray) -> float:
    onehot = np.eye(probs.shape[1])[y_true]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def detection_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray, classes: list[str]
) -> dict[str, Any]:
    """Metrics for one set of predictions. Class 0 must be BENIGN."""
    labels = list(range(len(classes)))
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=classes, output_dict=True, zero_division=0
    )
    benign = y_true == 0
    per_family = {
        c: {
            "precision": round(report[c]["precision"], 4),
            "recall": round(report[c]["recall"], 4),
            "f1": round(report[c]["f1-score"], 4),
            "support": int(report[c]["support"]),
            "pr_auc": (
                round(float(average_precision_score(y_true == i, probs[:, i])), 4)
                if (y_true == i).any()
                else None
            ),
        }
        for i, c in enumerate(classes)
    }
    return {
        "macro_f1": round(
            float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4
        ),
        "macro_f1_attacks_only": round(
            float(f1_score(y_true, y_pred, labels=labels[1:], average="macro", zero_division=0)), 4
        ),
        "accuracy": round(float((y_true == y_pred).mean()), 4),
        "benign_fpr": round(float((y_pred[benign] != 0).mean()), 5) if benign.any() else None,
        "attack_detection_rate": (
            round(float((y_pred[~benign] != 0).mean()), 4) if (~benign).any() else None
        ),
        "brier": round(brier_multiclass(probs, y_true), 5),
        "per_family": per_family,
        "confusion_matrix": {
            "labels": classes,
            "rows_true_cols_pred": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        },
    }


def plot_confusion(cm: np.ndarray, classes: list[str], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm)
    norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=40, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted family")
    ax.set_ylabel("True family")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                colour = "white" if norm[i, j] > 0.6 else "black"
                ax.text(
                    j,
                    i,
                    f"{norm[i, j]:.2f}\n{cm[i, j]:,}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=colour,
                )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Share of true family")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pr_curves(y_true: np.ndarray, probs: np.ndarray, classes: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, c in enumerate(classes):
        if not (y_true == i).any():
            continue
        precision, recall, _ = precision_recall_curve(y_true == i, probs[:, i])
        ap = average_precision_score(y_true == i, probs[:, i])
        ax.plot(recall, precision, label=f"{c} (AP {ap:.3f})", linewidth=1.4)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–recall per family (test set, calibrated)")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.01)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
