"""Polars ETL: CIC-IDS2017 TrafficLabelling CSVs → one clean Parquet file (blueprint §5).

Every step is counted in ``reports/etl_stats.json`` and summarised in ``docs/data_card.md``:

1. Read each CSV as text. The Thursday web-attack file is not valid UTF-8, so fall back to latin-1.
2. Clean the header: strip whitespace/BOM, and drop the duplicated ``Fwd Header Length`` column.
3. Drop empty rows (the Thursday web-attack file ends in ~288k rows of bare commas).
4. Repair the garbled dash in "Web Attack – …" labels and map the 15 labels to 8 families.
5. Cast features to float; treat Infinity/NaN as missing and drop those rows.
6. Drop exact duplicate rows (all features + label, ignoring IPs/ports/timestamps) *before*
   splitting, so the same flow cannot sit in both train and test. Features are then stored as
   float32 (duplicates are found on the exact float64 values first).
7. Drop constant (zero-variance) feature columns.
"""

from __future__ import annotations

import csv
import io
import json
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from alert_explainer.data.labels import RAW_TO_FAMILY, normalize_label

ID_COLUMNS: tuple[str, ...] = (
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Destination Port",
    "Protocol",
    "Timestamp",
)
# Identifier columns that are also model inputs (the public MachineLearningCVE CSVs keep them too).
ID_FEATURES: tuple[str, ...] = ("Destination Port", "Protocol")
# Identifier columns kept only as metadata (for alerts and pseudonymisation), never as features.
METADATA_COLUMNS: tuple[str, ...] = tuple(c for c in ID_COLUMNS if c not in ID_FEATURES)
LABEL_COLUMN = "Label"
DERIVED_COLUMNS: tuple[str, ...] = ("label", "family", "day", "source_file")
FEATURE_HASH, ROW_HASH = "_feature_hash", "_row_hash"
NON_FINITE_TOKENS = ["infinity", "+infinity", "-infinity", "inf", "+inf", "-inf", "nan"]

_DAY = re.compile(r"(monday|tuesday|wednesday|thursday|friday)", re.IGNORECASE)
_DUPLICATE_SUFFIX = re.compile(r"^(?P<base>.+)\.(?P<n>\d+)$")


@dataclass
class FileStats:
    file: str
    day: str
    encoding: str
    rows_read: int
    empty_rows_dropped: int
    rows_with_non_finite_dropped: int
    rows_out: int
    duplicate_columns_dropped: list[str] = field(default_factory=list)
    labels: dict[str, int] = field(default_factory=dict)


def clean_header(names: list[str]) -> list[str]:
    """Strip names and make repeats unique (``X``, ``X.1``, …) so they can be dropped."""
    seen: Counter[str] = Counter()
    out = []
    for raw in names:
        name = raw.replace("﻿", "").strip()
        out.append(f"{name}.{seen[name]}" if seen[name] else name)
        seen[name] += 1
    return out


def day_from_filename(name: str) -> str:
    match = _DAY.search(name)
    if not match:
        raise ValueError(f"Cannot tell the weekday from file name {name!r}")
    return match.group(1).capitalize()


def _read_text_csv(path: Path, columns: list[str]) -> tuple[pl.DataFrame, str]:
    def read(source: Path | io.BytesIO) -> pl.DataFrame:
        return pl.read_csv(
            source,
            has_header=False,
            skip_rows=1,
            new_columns=columns,
            infer_schema=False,  # read everything as text; we cast explicitly below
            truncate_ragged_lines=True,
            encoding="utf8",
        )

    try:
        return read(path), "utf-8"
    except pl.exceptions.PolarsError:
        text = path.read_bytes().decode("latin-1")
        return read(io.BytesIO(text.encode("utf-8"))), "latin-1"


def _numeric(column: str) -> pl.Expr:
    """Text → Float64, with Infinity/NaN/blank/unparseable all mapped to null."""
    text = pl.col(column).str.strip_chars()
    missing = text.is_null() | (text == "") | text.str.to_lowercase().is_in(NON_FINITE_TOKENS)
    value = pl.when(missing).then(None).otherwise(text).cast(pl.Float64, strict=False)
    return pl.when(value.is_infinite()).then(None).otherwise(value).alias(column)


def process_file(path: Path) -> tuple[pl.DataFrame, FileStats, dict[str, int]]:
    """Clean one CSV. Returns the frame, its stats and per-column non-finite counts."""
    with path.open("r", encoding="latin-1", newline="") as fh:
        header = clean_header(next(csv.reader(fh)))
    missing = [c for c in (*ID_COLUMNS, LABEL_COLUMN) if c not in header]
    if missing:
        raise ValueError(
            f"{path.name} lacks {missing}. Use the TrafficLabelling CSVs "
            "(GeneratedLabelledFlows.zip), not MachineLearningCVE, which has IPs removed."
        )

    df, encoding = _read_text_csv(path, header)
    rows_read = df.height

    duplicate_columns = [
        c for c in header if (m := _DUPLICATE_SUFFIX.match(c)) and m["base"] in header
    ]
    df = df.drop(duplicate_columns)

    label = pl.col(LABEL_COLUMN)
    df = df.filter(label.is_not_null() & (label.str.strip_chars() != ""))
    empty_rows = rows_read - df.height

    mapping = {raw: normalize_label(raw) for raw in df[LABEL_COLUMN].unique().to_list()}
    unknown = sorted({v for v in mapping.values() if v not in RAW_TO_FAMILY})
    if unknown:
        raise ValueError(f"{path.name}: unknown labels {unknown}; add them to RAW_TO_FAMILY")

    features = [c for c in df.columns if c not in ID_COLUMNS and c != LABEL_COLUMN]
    features += list(ID_FEATURES)
    text_ids = [c for c in METADATA_COLUMNS if c != "Source Port"]
    df = df.with_columns(
        label.replace_strict(mapping).alias("label"),
        *[_numeric(c) for c in features],
        _numeric("Source Port").cast(pl.Int32),
        *[pl.col(c).str.strip_chars() for c in text_ids],
    )

    non_finite = {
        c: n for c, n in df.select(pl.col(features).null_count()).row(0, named=True).items() if n
    }
    has_missing = pl.any_horizontal(pl.col(features).is_null())
    rows_with_non_finite = int(df.select(has_missing.sum()).item())
    df = df.filter(~has_missing)

    day = day_from_filename(path.name)
    # Hash the exact float64 values now, so de-duplication stays exact after the float32 cast.
    feature_hash = pl.struct(features).hash(seed=0)
    df = (
        df.with_columns(
            pl.col("label").replace_strict(RAW_TO_FAMILY).alias("family"),
            pl.lit(day).alias("day"),
            pl.lit(path.name).alias("source_file"),
            feature_hash.alias(FEATURE_HASH),
        )
        .with_columns(pl.struct(FEATURE_HASH, "label").hash(seed=1).alias(ROW_HASH))
        .with_columns(pl.col(features).cast(pl.Float32))
        .select([*METADATA_COLUMNS, *features, *DERIVED_COLUMNS, FEATURE_HASH, ROW_HASH])
    )

    stats = FileStats(
        file=path.name,
        day=day,
        encoding=encoding,
        rows_read=rows_read,
        empty_rows_dropped=empty_rows,
        rows_with_non_finite_dropped=rows_with_non_finite,
        rows_out=df.height,
        duplicate_columns_dropped=duplicate_columns,
        labels=_counts(df, "label"),
    )
    return df, stats, non_finite


def _counts(df: pl.DataFrame, column: str) -> dict[str, int]:
    rows = df.group_by(column).len().sort("len", descending=True).iter_rows()
    return {str(k): int(n) for k, n in rows}


_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")


def _file_order(path: Path) -> tuple[int, str]:
    match = _DAY.search(path.name)
    day = match.group(1).capitalize() if match else ""
    return (_WEEKDAYS.index(day) if day in _WEEKDAYS else len(_WEEKDAYS), path.name)


def run_etl(
    raw_dir: Path,
    processed_dir: Path,
    reports_dir: Path,
    docs_dir: Path | None = None,
) -> dict[str, Any]:
    files = sorted(raw_dir.rglob("*.csv"), key=_file_order)
    if not files:
        raise FileNotFoundError(
            f"No CSV files under {raw_dir}. Download GeneratedLabelledFlows.zip from "
            "https://www.unb.ca/cic/datasets/ids-2017.html and run scripts/get_data.sh."
        )
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    frames, file_stats = [], []
    non_finite_by_column: Counter[str] = Counter()
    for path in files:
        t0 = time.perf_counter()
        df, file_stat, non_finite = process_file(path)
        frames.append(df)
        file_stats.append(file_stat)
        non_finite_by_column.update(non_finite)
        print(
            f"  {path.name}: {file_stat.rows_out:,} rows [{file_stat.encoding}] "
            f"{time.perf_counter() - t0:.1f}s"
        )

    columns = frames[0].columns
    for path, frame in zip(files, frames, strict=True):
        if frame.columns != columns:
            raise ValueError(f"{path.name} has different columns from {files[0].name}")
    df = pl.concat(frames, how="vertical_relaxed")
    del frames

    internal = {*METADATA_COLUMNS, *DERIVED_COLUMNS, FEATURE_HASH, ROW_HASH}
    feature_columns = [c for c in columns if c not in internal]
    rows_before_dedupe = df.height
    labels_before_dedupe = _counts(df, "label")
    # Exact duplicates = same float64 feature values and label (64-bit hash; collision odds for
    # 3M rows are ~1e-7). Hashing avoids building a 79-column row index for 2.8M rows.
    df = df.filter(pl.col(ROW_HASH).is_first_distinct())
    duplicates_removed = rows_before_dedupe - df.height

    # Identical feature vectors with different labels: label noise, reported not removed.
    conflict_keys = (
        df.group_by(FEATURE_HASH).agg(pl.col("label").n_unique().alias("n")).filter(pl.col("n") > 1)
    )
    conflicting_rows = df.filter(pl.col(FEATURE_HASH).is_in(conflict_keys[FEATURE_HASH])).height
    df = df.drop(FEATURE_HASH, ROW_HASH)

    n_unique = df.select(pl.col(feature_columns).n_unique()).row(0, named=True)
    constant_columns = [c for c, n in n_unique.items() if n <= 1]
    features = [c for c in feature_columns if c not in constant_columns]
    df = df.drop(constant_columns).with_row_index("row_id")

    flows_path = processed_dir / "flows.parquet"
    df.write_parquet(flows_path, compression="zstd")
    t0 = time.perf_counter()
    pl.read_parquet(flows_path)
    load_seconds = time.perf_counter() - t0

    schema = {
        "features": features,
        "metadata_columns": list(METADATA_COLUMNS),
        "id_features": list(ID_FEATURES),
        "label_columns": ["label", "family"],
    }
    (processed_dir / "schema.json").write_text(json.dumps(schema, indent=2) + "\n")

    family_by_day: dict[str, dict[str, int]] = {}
    for family, day, n in df.group_by("family", "day").len().iter_rows():
        family_by_day.setdefault(family, {})[day] = int(n)

    stats: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": "CIC-IDS2017 TrafficLabelling (GeneratedLabelledFlows.zip)",
        "files": [asdict(s) for s in file_stats],
        "totals": {
            "rows_read": sum(s.rows_read for s in file_stats),
            "empty_rows_dropped": sum(s.empty_rows_dropped for s in file_stats),
            "rows_with_non_finite_dropped": sum(s.rows_with_non_finite_dropped for s in file_stats),
            "rows_before_dedupe": rows_before_dedupe,
            "duplicates_removed": duplicates_removed,
            "rows_final": df.height,
        },
        "non_finite_by_column": dict(non_finite_by_column.most_common()),
        "duplicate_columns_dropped": sorted(
            {c for s in file_stats for c in s.duplicate_columns_dropped}
        ),
        "constant_columns_dropped": constant_columns,
        "conflicting_label_vectors": conflict_keys.height,
        "conflicting_label_rows": conflicting_rows,
        "labels_before_dedupe": labels_before_dedupe,
        "labels_final": _counts(df, "label"),
        "families_final": _counts(df, "family"),
        "family_by_day": family_by_day,
        "n_features": len(features),
        "features": features,
        "parquet_bytes": flows_path.stat().st_size,
        "parquet_load_seconds": round(load_seconds, 2),
        "etl_seconds": round(time.perf_counter() - started, 1),
    }
    (reports_dir / "etl_stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    if docs_dir is not None:
        from alert_explainer.data.data_card import render_data_card

        docs_dir.mkdir(parents=True, exist_ok=True)
        (docs_dir / "data_card.md").write_text(render_data_card(stats))
    return stats
