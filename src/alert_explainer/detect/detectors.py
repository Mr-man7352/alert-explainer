"""The saved detectors behind one interface, for the alert builder (Day 6).

``load_detector("lgbm" | "mlp", models_dir, reference)`` returns an object that

- ``score(X_raw)`` → raw probabilities, calibrated probabilities and the decision (the Day 3/5
  rule: attack if ``1 − P_raw(BENIGN) ≥ threshold``, family = likeliest attack class), and
- ``attribute(X_raw, target)`` → one attribution per feature for the target class
  (TreeSHAP for LightGBM, Integrated Gradients for the MLP; see ``attribution.py``).

Both read only files written by training: LightGBM's text model, joblib calibrators and
preprocessors (this project's own files), and PyTorch weights loaded with ``weights_only=True``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import joblib
import lightgbm as lgb
import numpy as np

from alert_explainer.detect.attribution import BenignReference, integrated_gradients, tree_shap
from alert_explainer.detect.lgbm import IsotonicCalibrator, decide


@dataclass
class Scores:
    raw: np.ndarray  # (n, k) uncalibrated probabilities
    calibrated: np.ndarray  # (n, k)
    pred: np.ndarray  # (n,) class index; 0 = BENIGN

    @property
    def attack_score(self) -> np.ndarray:
        return 1.0 - self.raw[:, 0]


class Detector(Protocol):
    name: str
    version: str
    attribution_method: str
    classes: list[str]
    features: list[str]
    threshold: float
    meta: dict[str, Any]

    def score(self, X: np.ndarray) -> Scores: ...

    def attribute(self, X: np.ndarray, target: np.ndarray) -> np.ndarray: ...


class LGBMDetector:
    name = "lightgbm"
    attribution_method = "TreeSHAP (LightGBM pred_contrib), raw score of the predicted family"

    def __init__(self, model_dir: Path) -> None:
        self.meta: dict[str, Any] = json.loads((model_dir / "meta.json").read_text())
        self.booster = lgb.Booster(model_file=str(model_dir / "model.txt"))
        self.calibrator: IsotonicCalibrator = joblib.load(model_dir / "calibrator.joblib")
        self.classes: list[str] = list(self.meta["classes"])
        self.features: list[str] = list(self.meta["features"])
        self.threshold = float(self.meta["threshold"])
        self.version = f"lightgbm-{self.meta['trained_at']}"

    def score(self, X: np.ndarray) -> Scores:
        raw = np.asarray(self.booster.predict(np.asarray(X, dtype=np.float64)), dtype=float)
        raw = np.atleast_2d(raw)
        return Scores(raw, self.calibrator.transform(raw), decide(raw, self.threshold))

    def attribute(self, X: np.ndarray, target: np.ndarray) -> np.ndarray:
        values, _ = tree_shap(self.booster, X, target)
        return values


class MLPAttributionDetector:
    name = "pytorch-mlp"
    attribution_method = (
        "Integrated Gradients (Captum), predicted-family logit, median-BENIGN baseline"
    )

    def __init__(self, model_dir: Path, reference: BenignReference) -> None:
        import torch

        from alert_explainer.detect.torch_mlp import FlowPreprocessor, load_model, predict_proba

        self._predict_proba = predict_proba
        self._cpu = torch.device("cpu")
        self.model, self.config, self.classes = load_model(model_dir / "model.pt")
        self.preprocessor: FlowPreprocessor = joblib.load(model_dir / "preprocessor.joblib")
        self.calibrator: IsotonicCalibrator = joblib.load(model_dir / "calibrator.joblib")
        self.meta = json.loads((model_dir / "meta.json").read_text())
        self.features = list(self.meta["features"])
        self.threshold = float(self.meta["threshold"])
        self.version = f"pytorch-mlp-{self.meta['trained_at']}"
        if reference.features != self.features:
            raise ValueError("Benign reference and MLP were built on different feature lists")
        self.baseline_pre = self.preprocessor.transform(reference.median.reshape(1, -1))[0]
        self.last_delta: np.ndarray | None = None

    def score(self, X: np.ndarray) -> Scores:
        X_pre = self.preprocessor.transform(np.atleast_2d(X))
        raw = self._predict_proba(self.model, X_pre, self._cpu)
        return Scores(raw, self.calibrator.transform(raw), decide(raw, self.threshold))

    def attribute(self, X: np.ndarray, target: np.ndarray) -> np.ndarray:
        X_pre = self.preprocessor.transform(np.atleast_2d(X))
        values, delta = integrated_gradients(self.model, X_pre, target, self.baseline_pre)
        self.last_delta = delta
        return values


def load_detector(name: str, models_dir: Path, reference: BenignReference) -> Detector:
    if name == "lgbm":
        det: Detector = LGBMDetector(models_dir / "lgbm")
    elif name == "mlp":
        det = MLPAttributionDetector(models_dir / "mlp", reference)
    else:
        raise ValueError(f"Unknown detector {name!r}; use lgbm or mlp")
    if det.features != reference.features:
        raise ValueError(
            f"{name} was trained on different features from the benign reference; "
            "rebuild it with `ae build-alerts --rebuild-reference`"
        )
    return det
