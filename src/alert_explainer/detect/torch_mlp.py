"""PyTorch MLP detector: the building blocks (blueprint §6.2). The pipeline is in ``train_mlp.py``.

- ``FlowPreprocessor``: ``QuantileTransformer(output_distribution="normal")``, fitted on training
  rows only. There is no ``log1p`` before it, unlike the blueprint's first plan. The quantile
  transform is rank-based, so any monotone transform before it (``log1p`` included) changes nothing
  except the interpolation between quantiles. It also handles the negative values some features
  have (e.g. ``Init_Win_bytes_forward = -1``), which ``log1p`` does not.
- ``FlowDataset``: a map-style Dataset over (memory-mapped) NumPy arrays. It implements
  ``__getitems__``, so the DataLoader fetches a whole batch with one fancy index instead of 2,048
  ``__getitem__`` calls. With ~1.76M training rows, per-row Python indexing would cost more time
  than the GPU work. Indices are sorted within each batch for sequential reads from the memory
  map; order inside a batch does not affect the gradient or BatchNorm statistics.
- ``MLP``: [Linear → BatchNorm → ReLU → Dropout] × 3, then a linear head that outputs logits.
- ``FocalLoss``: class-weighted focal loss for the Day 5 ablation (γ = 0 is weighted cross-entropy).
- ``fit_mlp``: the hand-written training loop. AdamW, a OneCycle learning rate, and the weights
  of the epoch with the best validation macro-F1 restored at the end. By default the full OneCycle
  schedule runs (``patience=None``). On the first real run, patience 4 stopped every run at epochs
  6–11, while the learning rate was still at its peak (OneCycle warms up for the first 30% of
  epochs). That left validation macro-F1 jumping by ±0.1 between epochs and the model never
  annealed. If ``patience`` is set, it only counts epochs after the learning-rate peak.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import ndtri
from sklearn.preprocessing import QuantileTransformer
from torch import nn
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler

from alert_explainer.detect.evaluate import macro_f1_fast

Batch = tuple[torch.Tensor, torch.Tensor]


# --------------------------------------------------------------------------- device and seeds


def available_devices() -> list[str]:
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def pick_device(name: str = "auto") -> torch.device:
    """``auto`` prefers CUDA, then Apple's MPS, then CPU. Run ``ae bench-mlp`` to check which is
    actually faster for this small model on your machine."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name not in available_devices():
        raise ValueError(
            f"Device {name!r} is not available here; choose from {available_devices()}"
        )
    return torch.device(name)


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)  # seeds CPU, CUDA and MPS generators


# --------------------------------------------------------------------------- preprocessing


_BOUNDS = 1e-7  # sklearn's BOUNDS_THRESHOLD
_CLIP = float(ndtri(1 - (_BOUNDS - np.spacing(1))))  # ≈ 5.2, the value sklearn clips to


class FlowPreprocessor:
    """Per-feature quantile transform to a standard normal, clipped to about ±5.2.

    ``fit`` uses sklearn's ``QuantileTransformer``. ``transform`` re-implements its forward
    transform (same interpolation, bounds and clipping; unit-tested to match it to 1e-6) using
    ``scipy.special.ndtri``. sklearn's own ``transform`` takes ~11 ms for a single row, mostly
    per-call overhead in ``scipy.stats.norm.ppf``; this one takes well under 1 ms, which matters
    for per-alert latency.
    """

    def __init__(self, n_quantiles: int = 1000, subsample: int = 200_000, seed: int = 42) -> None:
        self.qt = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution="normal",
            subsample=subsample,
            random_state=seed,
        )

    def fit(self, X: np.ndarray) -> FlowPreprocessor:
        self.qt.fit(X)
        return self

    def transform(self, X: np.ndarray, chunk_rows: int = 262_144) -> np.ndarray:
        """float32 in, float32 out, in chunks.

        As in sklearn on float32 input, the ±1e-7 bound checks happen in float32 (so tied values
        at the edge quantiles are interpolated, not pushed to ±5.2); interpolation is float64.
        """
        quantiles: np.ndarray = self.qt.quantiles_  # (n_quantiles, n_features)
        refs: np.ndarray = self.qt.references_  # (n_quantiles,)
        out = np.empty(X.shape, dtype=np.float32)
        for start in range(0, len(X), chunk_rows):
            block = np.asarray(X[start : start + chunk_rows], dtype=np.float32)
            u = np.empty(block.shape, dtype=np.float64)
            for j in range(block.shape[1]):
                col, q = block[:, j], quantiles[:, j]
                # Interpolate both ways and average, so repeated quantiles (ties) map to the middle
                u[:, j] = 0.5 * (np.interp(col, q, refs) - np.interp(-col, -q[::-1], -refs[::-1]))
                u[col + _BOUNDS > q[-1], j] = 1.0
                u[col - _BOUNDS < q[0], j] = 0.0
            out[start : start + chunk_rows] = np.clip(ndtri(u), -_CLIP, _CLIP)
        return out


# --------------------------------------------------------------------------- dataset


class FlowDataset(Dataset[Batch]):
    """Rows of ``X`` (float32) and ``y`` (int class index), optionally restricted to ``indices``.

    ``X`` and ``y`` can be ordinary arrays or ``np.load(..., mmap_mode="r")`` memory maps.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray | None = None) -> None:
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)}")
        self.X, self.y = X, y
        self.indices = None if indices is None else np.sort(np.asarray(indices, dtype=np.int64))

    @classmethod
    def from_npy(cls, directory: Path, name: str, mmap: bool = True) -> FlowDataset:
        mode: Literal["r"] | None = "r" if mmap else None
        X = np.load(directory / f"{name}_X.npy", mmap_mode=mode)
        y = np.load(directory / f"{name}_y.npy", mmap_mode=mode)
        return cls(X, y)

    def __len__(self) -> int:
        return len(self.y) if self.indices is None else len(self.indices)

    def _rows(self, positions: np.ndarray) -> np.ndarray:
        return positions if self.indices is None else self.indices[positions]

    def __getitem__(self, i: int) -> Batch:
        row = int(self._rows(np.array([i]))[0])
        x = torch.from_numpy(np.array(self.X[row], dtype=np.float32))
        return x, torch.tensor(int(self.y[row]), dtype=torch.int64)

    def __getitems__(self, positions: Sequence[int]) -> Batch:
        rows = np.sort(self._rows(np.asarray(positions, dtype=np.int64)))
        x = torch.from_numpy(np.ascontiguousarray(self.X[rows], dtype=np.float32))
        y = torch.from_numpy(np.asarray(self.y[rows], dtype=np.int64))
        return x, y

    def labels(self) -> np.ndarray:
        y = np.asarray(self.y)
        return y if self.indices is None else y[self.indices]


def _keep_batch(batch: Any) -> Any:
    """collate_fn: ``__getitems__`` already returns stacked tensors."""
    return batch


def make_loader(dataset: FlowDataset, batch_size: int, seed: int) -> DataLoader[Batch]:
    """Shuffled batches; the last partial batch is dropped so BatchNorm never sees a batch of 1."""
    if len(dataset) < 2:
        raise ValueError("Need at least 2 training rows")
    generator = torch.Generator().manual_seed(seed)
    sampler = BatchSampler(
        RandomSampler(dataset, generator=generator),
        batch_size=min(batch_size, len(dataset)),
        drop_last=True,
    )
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=_keep_batch, num_workers=0)


# --------------------------------------------------------------------------- model and loss


class MLP(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden: Sequence[int] = (256, 128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = n_features
        for h in hidden:
            # No bias: BatchNorm's shift makes it redundant
            layers += [
                nn.Linear(width, h, bias=False),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            width = h
        layers.append(nn.Linear(width, n_classes))
        self.net = nn.Sequential(*layers)
        self.n_features, self.n_classes = n_features, n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(x)
        return out


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def class_weights(y: np.ndarray, n_classes: int, scheme: str) -> np.ndarray:
    """Per-class weights; the same schemes as LightGBM's ``sample_weights`` (mean row weight 1)."""
    counts = np.bincount(y, minlength=n_classes)
    if scheme == "none":
        per_class = np.ones(n_classes)
    elif scheme in ("balanced", "sqrt"):
        per_class = len(y) / (n_classes * np.maximum(counts, 1))
        if scheme == "sqrt":
            per_class = np.sqrt(per_class)
    else:
        raise ValueError(f"Unknown weight scheme {scheme!r}")
    mean_row_weight = float((counts * per_class).sum() / max(len(y), 1))
    return (per_class / mean_row_weight).astype(np.float32)


class FocalLoss(nn.Module):
    """Class-weighted focal loss (Lin et al. 2017): ``-w_y (1 - p_y)^γ log p_y``.

    Normalised like ``F.cross_entropy(weight=...)`` (divided by the summed weights), so γ = 0
    gives exactly weighted cross-entropy.
    """

    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight: torch.Tensor | None
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=1).gather(1, target[:, None]).squeeze(1)
        loss = -((1.0 - log_p.exp()) ** self.gamma) * log_p
        if self.weight is None:
            return loss.mean()
        w = self.weight[target]
        return (w * loss).sum() / w.sum()


# --------------------------------------------------------------------------- config and training


@dataclass(frozen=True)
class MLPConfig:
    hidden: tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.2
    lr: float = 3e-3  # OneCycle peak learning rate
    weight_decay: float = 1e-4
    weights: str = "sqrt"  # class weighting: none | sqrt | balanced
    loss: str = "ce"  # ce | focal
    gamma: float = 2.0  # focal loss only
    batch_size: int = 2048
    max_epochs: int = 20
    # None: run the whole OneCycle schedule and keep the best epoch. An int: stop after that many
    # epochs without improvement, counted only once the learning rate has peaked.
    patience: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["hidden"] = list(self.hidden)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MLPConfig:
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "hidden" in known:
            known["hidden"] = tuple(known["hidden"])
        return cls(**known)

    def short(self) -> str:
        loss = f"focal(γ={self.gamma})" if self.loss == "focal" else "ce"
        return (
            f"hidden={'-'.join(map(str, self.hidden))} drop={self.dropout} lr={self.lr:g} "
            f"w={self.weights} loss={loss}"
        )


@dataclass
class FitResult:
    model: MLP
    best_epoch: int
    best_val_macro_f1: float
    seconds: float
    stopped_early: bool
    history: list[dict[str, float]] = field(default_factory=list)


@torch.inference_mode()
def predict_proba(
    model: MLP, X: np.ndarray, device: torch.device, chunk_rows: int = 65_536
) -> np.ndarray:
    """Softmax probabilities as float64 (computed in float32 on ``device``)."""
    model.eval()
    out = np.empty((len(X), model.n_classes), dtype=np.float64)
    for start in range(0, len(X), chunk_rows):
        xb = torch.from_numpy(np.ascontiguousarray(X[start : start + chunk_rows], dtype=np.float32))
        probs = torch.softmax(model(xb.to(device)), dim=1)
        out[start : start + chunk_rows] = probs.cpu().numpy()
    return out


def fit_mlp(
    train: FlowDataset,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: MLPConfig,
    n_classes: int,
    device: torch.device,
    seed: int = 42,
    log: Callable[[str], None] = print,
) -> FitResult:
    """Train with early stopping on validation macro-F1 (argmax); returns the best epoch's model."""
    seed_everything(seed)
    y_train = train.labels()
    model = MLP(train.X.shape[1], n_classes, cfg.hidden, cfg.dropout).to(device)
    weight = torch.from_numpy(class_weights(y_train, n_classes, cfg.weights)).to(device)
    criterion: nn.Module
    if cfg.loss == "ce":
        criterion = nn.CrossEntropyLoss(weight=weight)
    elif cfg.loss == "focal":
        criterion = FocalLoss(weight, cfg.gamma).to(device)
    else:
        raise ValueError(f"Unknown loss {cfg.loss!r}")

    loader = make_loader(train, cfg.batch_size, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    pct_start = 0.3  # OneCycle default: learning rate rises for the first 30% of steps
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, total_steps=cfg.max_epochs * len(loader), pct_start=pct_start
    )
    peak_epoch = int(np.ceil(pct_start * cfg.max_epochs))
    y_val = np.asarray(y_val, dtype=np.int64)

    best_f1, best_epoch, bad_epochs = -1.0, 0, 0
    best_state: dict[str, torch.Tensor] = {}
    history: list[dict[str, float]] = []
    stopped_early = False
    started = time.perf_counter()
    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.perf_counter()
        model.train()
        loss_sum = torch.zeros((), device=device)
        rows = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.detach() * len(yb)  # stays on the device: no sync per step
            rows += len(yb)
        train_seconds = time.perf_counter() - t0

        probs = predict_proba(model, X_val, device)
        val_f1 = macro_f1_fast(y_val, probs.argmax(axis=1), n_classes)
        val_log_loss = float(
            -np.log(np.clip(probs[np.arange(len(y_val)), y_val], 1e-12, None)).mean()
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(float(loss_sum) / max(rows, 1), 6),
                "val_log_loss": round(val_log_loss, 6),
                "val_macro_f1": round(val_f1, 5),
                "lr_end": float(scheduler.get_last_lr()[0]),
                "train_seconds": round(train_seconds, 2),
                "epoch_seconds": round(time.perf_counter() - t0, 2),
            }
        )
        improved = val_f1 > best_f1 + 1e-4
        log(
            f"    epoch {epoch:>2}  train loss {history[-1]['train_loss']:.4f}  "
            f"val log-loss {val_log_loss:.4f}  val macro-F1 {val_f1:.4f}"
            f"{' *' if improved else ''}  ({history[-1]['epoch_seconds']:.1f}s)"
        )
        if improved:
            best_f1, best_epoch, bad_epochs = val_f1, epoch, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        elif epoch > peak_epoch:
            bad_epochs += 1
            if cfg.patience is not None and bad_epochs >= cfg.patience:
                stopped_early = True
                break

    model.load_state_dict(best_state)
    return FitResult(
        model=model,
        best_epoch=best_epoch,
        best_val_macro_f1=best_f1,
        seconds=time.perf_counter() - started,
        stopped_early=stopped_early,
        history=history,
    )


# --------------------------------------------------------------------------- save / load


def save_model(model: MLP, cfg: MLPConfig, classes: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(
        {
            "state_dict": state,
            "n_features": model.n_features,
            "classes": list(classes),
            "config": cfg.to_dict(),
        },
        path,
    )


def load_model(path: Path, device: str | torch.device = "cpu") -> tuple[MLP, MLPConfig, list[str]]:
    """Load weights only (``weights_only=True``: no arbitrary pickle code runs)."""
    blob = torch.load(path, map_location="cpu", weights_only=True)
    cfg = MLPConfig.from_dict(blob["config"])
    classes = list(blob["classes"])
    model = MLP(int(blob["n_features"]), len(classes), cfg.hidden, cfg.dropout)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model.to(device), cfg, classes
