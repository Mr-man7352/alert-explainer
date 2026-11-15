"""Train/validation/test splits (blueprint §5, "Splits").

- ``split``: stratified 70/15/15 on family, fixed seed.
- ``val_part``: the validation set halved (stratified). The ``cal`` half fits probability
  calibration. Early stopping and the attack threshold use the whole validation set (the threshold
  acts on raw scores, which calibration never touches). ``thr`` is kept for ablations.
- ``day_split``: Mon–Wed → train, Thu–Fri → test, for the cross-day robustness check.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from alert_explainer.data.labels import HELD_OUT_FAMILIES

EARLY_DAYS = ("Monday", "Tuesday", "Wednesday")


def assign_splits(
    families: np.ndarray, seed: int = 42, train: float = 0.70, val: float = 0.15
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(families))
    train_idx, rest = train_test_split(idx, train_size=train, stratify=families, random_state=seed)
    val_share = val / (1.0 - train)
    val_idx, test_idx = train_test_split(
        rest, train_size=val_share, stratify=families[rest], random_state=seed
    )
    cal_idx, thr_idx = train_test_split(
        val_idx, train_size=0.5, stratify=families[val_idx], random_state=seed
    )
    split = np.empty(len(families), dtype=object)
    split[train_idx], split[val_idx], split[test_idx] = "train", "val", "test"
    val_part = np.full(len(families), None, dtype=object)
    val_part[cal_idx], val_part[thr_idx] = "cal", "thr"
    return split, val_part


def run_splits(processed_dir: Path, reports_dir: Path, seed: int = 42) -> dict[str, Any]:
    flows = pl.read_parquet(processed_dir / "flows.parquet", columns=["row_id", "family", "day"])
    split, val_part = assign_splits(flows["family"].to_numpy(), seed=seed)
    out = flows.with_columns(
        pl.Series("split", split.tolist(), dtype=pl.String),
        pl.Series("val_part", val_part.tolist(), dtype=pl.String),
        pl.when(pl.col("day").is_in(EARLY_DAYS))
        .then(pl.lit("train"))
        .otherwise(pl.lit("test"))
        .alias("day_split"),
    )
    out.select("row_id", "split", "val_part", "day_split").write_parquet(
        processed_dir / "splits.parquet"
    )

    def table(column: str) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for fam, part, n in out.group_by("family", column).len().iter_rows():
            result.setdefault(fam, {})[part] = int(n)
        return result

    by_split, by_day_split = table("split"), table("day_split")
    shared = sorted(
        f
        for f, parts in by_day_split.items()
        if f != "BENIGN" and f not in HELD_OUT_FAMILIES and len(parts) == 2
    )
    summary = {
        "seed": seed,
        "proportions": {"train": 0.70, "val": 0.15, "test": 0.15},
        "family_by_split": by_split,
        "family_by_day_split": by_day_split,
        "attack_families_in_both_day_splits": shared,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "splits_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
