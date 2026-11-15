import json
from pathlib import Path

import numpy as np
import polars as pl

from alert_explainer.data.splits import assign_splits


def test_assign_splits_proportions_and_determinism() -> None:
    rng = np.random.default_rng(0)
    fam = rng.choice(["BENIGN", "DoS", "Bot"], size=10_000, p=[0.8, 0.15, 0.05])
    split, part = assign_splits(fam, seed=1)
    for name, share in [("train", 0.70), ("val", 0.15), ("test", 0.15)]:
        assert abs((split == name).mean() - share) < 0.005
    for f in ["BENIGN", "DoS", "Bot"]:  # stratified
        assert abs((split[fam == f] == "train").mean() - 0.70) < 0.01
    assert set(part[split == "val"]) == {"cal", "thr"}
    assert all(p is None for p in part[split != "val"])
    split2, _ = assign_splits(fam, seed=1)
    assert (split == split2).all()


def test_split_file(pipeline_dir: Path) -> None:
    splits = pl.read_parquet(pipeline_dir / "processed" / "splits.parquet")
    flows = pl.read_parquet(pipeline_dir / "processed" / "flows.parquet", columns=["row_id"])
    assert splits["row_id"].to_list() == flows["row_id"].to_list()
    assert set(splits["split"].unique()) == {"train", "val", "test"}
    summary = json.loads((pipeline_dir / "reports" / "splits_summary.json").read_text())
    # In CIC-IDS2017 every attack family sits on one day only
    assert summary["attack_families_in_both_day_splits"] == []
