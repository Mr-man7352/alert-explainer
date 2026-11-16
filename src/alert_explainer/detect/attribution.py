"""Per-alert feature attributions and benign-traffic percentiles (blueprint §6–7; Day 6).

- **LightGBM → TreeSHAP.** Exact SHAP values from LightGBM's built-in TreeSHAP
  (``predict(pred_contrib=True)``), the same algorithm as ``shap.TreeExplainer`` (Lundberg et al.
  2020) without the ``shap``/``numba`` runtime dependency. ``tests/unit/test_attribution.py``
  checks the two agree, and that the values add up to the model's raw margin (SHAP's additivity
  property). Attributions are for the *predicted* family's raw score (log-odds scale).
- **PyTorch MLP → Integrated Gradients** (Captum). The target is the predicted family's logit, and
  the baseline is the median BENIGN training flow (in the model's preprocessed space), so an
  attribution reads as "how much this feature moved the decision away from typical normal
  traffic". IG's completeness property (attributions sum to f(x) − f(baseline)) is reported as the
  convergence delta and tested. The quantile transform is per-feature and monotone, so each input
  dimension is still one named feature.
- **Benign percentile.** For each feature, 1,001 quantiles of the BENIGN *training* rows are stored
  (``BenignReference``). An alert's value is placed among them with a mid-rank rule, so a value
  shared by 40% of benign flows (e.g. zero) reads as the 20th percentile rather than 0 or 40.

``top_features`` keeps the features that pushed hardest *towards* the predicted family (largest
positive attribution). If fewer than ``k`` are positive, the rest are filled by absolute size.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import torch
from torch import nn

QUANTILES = np.linspace(0.0, 1.0, 1001)
# Numbers that are really labels: a percentile of a port number means nothing. For these the alert
# reports how common the exact value is among BENIGN flows instead.
CATEGORICAL_FEATURES: frozenset[str] = frozenset({"Destination Port", "Protocol"})


# --------------------------------------------------------------------------- benign reference


@dataclass
class BenignReference:
    features: list[str]
    quantiles: np.ndarray  # (len(QUANTILES), n_features), float64
    n_rows: int

    @classmethod
    def fit(cls, X_benign: np.ndarray, features: list[str]) -> BenignReference:
        if len(X_benign) == 0:
            raise ValueError("No BENIGN training rows to build the reference from")
        # One column at a time: on the full dataset (~1.5M BENIGN training rows) a float64 copy
        # of the whole matrix would cost ~0.8 GB for no benefit.
        q = np.column_stack(
            [
                np.quantile(X_benign[:, j].astype(np.float64), QUANTILES)
                for j in range(X_benign.shape[1])
            ]
        )
        return cls(features=list(features), quantiles=q, n_rows=len(X_benign))

    @property
    def median(self) -> np.ndarray:
        return self.quantiles[len(QUANTILES) // 2]

    def percentile(self, X: np.ndarray) -> np.ndarray:
        """Mid-rank percentile (0–100) of each value among the benign quantiles. Shape of X."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        out = np.empty_like(X)
        last = len(QUANTILES) - 1
        for j in range(X.shape[1]):
            grid = self.quantiles[:, j]
            lo = np.searchsorted(grid, X[:, j], side="left")  # grid points below the value
            hi = np.searchsorted(grid, X[:, j], side="right")  # ... below or equal
            # Mid-rank: the value sits halfway through the grid points it ties with, or halfway
            # between its two neighbours when it ties with none.
            out[:, j] = (lo + hi - 1) / 2 / last * 100
        # values above every benign quantile are the 100th percentile, below every one the 0th
        out = np.where(self.quantiles[-1] < X, 100.0, np.where(self.quantiles[0] > X, 0.0, out))
        return np.clip(out, 0.0, 100.0)

    def share_equal(self, X: np.ndarray) -> np.ndarray:
        """Approximate share (0–1) of benign flows with exactly this value, from the quantiles."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        out = np.empty_like(X)
        for j in range(X.shape[1]):
            grid = self.quantiles[:, j]
            ties = np.searchsorted(grid, X[:, j], side="right") - np.searchsorted(
                grid, X[:, j], side="left"
            )
            out[:, j] = ties / len(QUANTILES)
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            quantiles=self.quantiles,
            features=np.array(self.features),
            n_rows=np.array(self.n_rows),
        )

    @classmethod
    def load(cls, path: Path) -> BenignReference:
        blob = np.load(path, allow_pickle=False)
        return cls(
            features=[str(f) for f in blob["features"]],
            quantiles=blob["quantiles"],
            n_rows=int(blob["n_rows"]),
        )


def build_benign_reference(processed_dir: Path) -> BenignReference:
    """Quantiles of every feature over the BENIGN rows of the training split."""
    import polars as pl

    schema = json.loads((processed_dir / "schema.json").read_text())
    features: list[str] = schema["features"]
    splits = pl.scan_parquet(processed_dir / "splits.parquet").filter(pl.col("split") == "train")
    X = (
        pl.scan_parquet(processed_dir / "flows.parquet")
        .filter(pl.col("family") == "BENIGN")
        .join(splits.select("row_id"), on="row_id", how="semi")
        .select(features)
        .collect()
        .to_numpy()
    )
    return BenignReference.fit(X, features)


# --------------------------------------------------------------------------- attributions


def tree_shap(
    booster: lgb.Booster, X: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """SHAP values for class ``target[i]`` of row i. Returns (values (n, f), expected value (n,))."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    n, f = X.shape
    contrib = np.asarray(booster.predict(X, pred_contrib=True), dtype=np.float64)
    k = contrib.shape[1] // (f + 1)
    per_class = contrib.reshape(n, k, f + 1)
    chosen = per_class[np.arange(n), np.asarray(target, dtype=int)]
    return chosen[:, :f], chosen[:, f]


class _Logits(nn.Module):
    """Wrap the MLP so Captum sees a plain (batch, features) → logits function."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.model(x)
        return out


def integrated_gradients(
    model: nn.Module,
    X_pre: np.ndarray,
    target: np.ndarray,
    baseline_pre: np.ndarray,
    n_steps: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """IG attributions of the target logit. Returns (values (n, f), convergence delta (n,))."""
    from captum.attr import IntegratedGradients

    model = model.to("cpu").eval()
    x = torch.from_numpy(np.ascontiguousarray(np.atleast_2d(X_pre), dtype=np.float32))
    base = torch.from_numpy(np.ascontiguousarray(baseline_pre, dtype=np.float32)).reshape(1, -1)
    ig = IntegratedGradients(_Logits(model))
    attr, delta = ig.attribute(
        x,
        baselines=base.expand_as(x),
        target=torch.as_tensor(np.asarray(target, dtype=np.int64)),
        n_steps=n_steps,
        method="gausslegendre",
        return_convergence_delta=True,
    )
    return attr.detach().numpy().astype(np.float64), delta.detach().numpy().astype(np.float64)


def top_features(
    values: np.ndarray,
    contributions: np.ndarray,
    percentiles: np.ndarray,
    features: list[str],
    k: int = 5,
    shares: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """The k features pushing hardest towards the predicted class, for one row.

    Categorical features get ``benign_percentile = None`` and ``benign_share`` (how common the
    exact value is among benign flows) instead.
    """
    contributions = np.asarray(contributions, dtype=np.float64)
    positive = [int(i) for i in np.argsort(-contributions) if contributions[i] > 0][:k]
    if len(positive) < k:
        rest = [int(i) for i in np.argsort(-np.abs(contributions)) if i not in positive]
        positive += rest[: k - len(positive)]
    out = []
    for i in positive:
        categorical = features[i] in CATEGORICAL_FEATURES
        share = None if shares is None or not categorical else round(float(shares[i]), 4)
        out.append(
            {
                "name": features[i],
                "value": round(float(values[i]), 4),
                "benign_percentile": None if categorical else round(float(percentiles[i]), 1),
                "benign_share": share,
                "contribution": round(float(contributions[i]), 4),
            }
        )
    return out
