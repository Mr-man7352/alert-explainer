"""Load the processed flows and splits as NumPy arrays for the detectors (Days 4–5).

The LightGBM baseline (Day 3) has its own inline copy of this logic; new code uses this module.
Split columns are turned into boolean masks inside Polars, so no per-row Python strings are kept:
on the full dataset that saves several hundred MB.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from alert_explainer.data.labels import HELD_OUT_FAMILIES, TRAIN_FAMILIES


@dataclass
class DetectionData:
    features: list[str]
    classes: list[str]  # training classes, BENIGN first
    X: np.ndarray  # (rows, features) float32, raw feature values
    y: np.ndarray  # class index into ``classes``; -1 for held-out families
    label_code: np.ndarray  # raw CIC-IDS2017 label as an index into ``label_names``
    label_names: list[str]
    train: np.ndarray  # boolean masks; train/val/cal/test exclude held-out families
    val: np.ndarray
    cal: np.ndarray  # the calibration half of val
    test: np.ndarray
    held_out: np.ndarray  # Other-rare rows (all splits)
    early: np.ndarray  # Mon–Wed (cross-day train side), all families
    late: np.ndarray  # Thu–Fri (cross-day test side), all families

    @property
    def n_classes(self) -> int:
        return len(self.classes)


def load_detection_data(processed_dir: Path) -> DetectionData:
    schema = json.loads((processed_dir / "schema.json").read_text())
    features: list[str] = schema["features"]
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
    if not classes or classes[0] != "BENIGN":
        raise ValueError("BENIGN must be present")
    label_names = sorted(df["label"].unique().to_list())

    masks = df.select(
        (pl.col("split") == "train").fill_null(False).alias("train"),
        (pl.col("split") == "val").fill_null(False).alias("val"),
        ((pl.col("split") == "val") & (pl.col("val_part") == "cal")).fill_null(False).alias("cal"),
        (pl.col("split") == "test").fill_null(False).alias("test"),
        pl.col("family").is_in(list(HELD_OUT_FAMILIES)).alias("held_out"),
        (pl.col("day_split") == "train").fill_null(False).alias("early"),
    )
    y = (
        df["family"]
        .replace_strict({c: i for i, c in enumerate(classes)}, default=-1, return_dtype=pl.Int64)
        .to_numpy()
    )
    label_code = (
        df["label"]
        .replace_strict({name: i for i, name in enumerate(label_names)}, return_dtype=pl.Int32)
        .to_numpy()
    )
    X = df.select(features).to_numpy().astype(np.float32, copy=False)
    del df

    known = y >= 0
    early = masks["early"].to_numpy()
    return DetectionData(
        features=features,
        classes=classes,
        X=X,
        y=y,
        label_code=label_code,
        label_names=label_names,
        train=masks["train"].to_numpy() & known,
        val=masks["val"].to_numpy() & known,
        cal=masks["cal"].to_numpy() & known,
        test=masks["test"].to_numpy() & known,
        held_out=masks["held_out"].to_numpy(),
        early=early,
        late=~early,
    )
