import json
from pathlib import Path

import polars as pl
import pytest

from alert_explainer.data.etl import clean_header, day_from_filename, process_file
from alert_explainer.data.synthetic import HEADER


def test_synthetic_header_matches_real_layout() -> None:
    assert len(HEADER) == 85  # TrafficLabelling: 84 columns + Label


def test_clean_header_strips_and_marks_duplicates() -> None:
    assert clean_header(["﻿Flow ID", " A", " B", " A"]) == ["Flow ID", "A", "B", "A.1"]


def test_day_from_filename() -> None:
    assert day_from_filename("Wednesday-workingHours.pcap_ISCX.csv") == "Wednesday"
    with pytest.raises(ValueError):
        day_from_filename("random.csv")


def test_thursday_web_attacks_file(pipeline_dir: Path) -> None:
    path = pipeline_dir / "raw" / "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
    df, stats, _ = process_file(path)
    assert stats.encoding == "latin-1"
    assert stats.empty_rows_dropped == 50
    assert stats.duplicate_columns_dropped == ["Fwd Header Length.1"]
    web = {lab for lab in df["label"].unique() if lab.startswith("Web")}
    assert web <= {"Web Attack - Brute Force", "Web Attack - XSS", "Web Attack - Sql Injection"}
    assert len(web) == 3


def test_parquet_is_clean(pipeline_dir: Path) -> None:
    df = pl.read_parquet(pipeline_dir / "processed" / "flows.parquet")
    schema = json.loads((pipeline_dir / "processed" / "schema.json").read_text())
    features = schema["features"]

    assert "Fwd Header Length" in df.columns and "Fwd Header Length.1" not in df.columns
    assert "Bwd PSH Flags" not in df.columns  # constant
    assert all(df.schema[c] == pl.Float32 for c in features)
    assert df.select(pl.col(features).null_count().sum()).sum_horizontal().item() == 0
    assert df.select(pl.any_horizontal(pl.col(features).is_infinite()).sum()).item() == 0
    assert not set(schema["metadata_columns"]) & set(features)  # IPs etc. never features
    assert {"Destination Port", "Protocol"} <= set(features)
    assert set(df["day"].unique()) == {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday"}
    assert df["row_id"].to_list() == list(range(df.height))
    # no duplicate flow statistics + label left
    assert df.select([*features, "label"]).is_duplicated().sum() == 0


def test_stats_and_data_card(pipeline_dir: Path) -> None:
    stats = json.loads((pipeline_dir / "reports" / "etl_stats.json").read_text())
    t = stats["totals"]
    assert t["empty_rows_dropped"] == 50
    assert t["duplicates_removed"] > 0
    assert t["rows_with_non_finite_dropped"] > 0
    assert set(stats["non_finite_by_column"]) == {"Flow Bytes/s", "Flow Packets/s"}
    assert len(stats["constant_columns_dropped"]) == 8
    assert (
        t["rows_read"] - t["empty_rows_dropped"] - t["rows_with_non_finite_dropped"]
        == t["rows_before_dedupe"]
    )
    assert t["rows_before_dedupe"] - t["duplicates_removed"] == t["rows_final"]
    card = (pipeline_dir / "docs" / "data_card.md").read_text()
    assert "Final rows" in card and "Each attack family occurs on a single day" in card
