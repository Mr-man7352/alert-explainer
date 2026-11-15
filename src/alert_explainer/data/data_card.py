"""Render docs/data_card.md from reports/etl_stats.json."""

from __future__ import annotations

from typing import Any

from alert_explainer.data.labels import FAMILIES, HELD_OUT_FAMILIES, RAW_TO_FAMILY

DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
EARLY_DAYS = DAYS[:3]


def _n(x: int) -> str:
    return f"{x:,}"


def render_data_card(stats: dict[str, Any]) -> str:
    t = stats["totals"]
    lines = [
        "# Data card — CIC-IDS2017 (TrafficLabelling), cleaned",
        "",
        f"_Generated automatically by `ae etl` on {stats['generated_at']}. "
        "Numbers come from `reports/etl_stats.json`._",
        "",
        "## Source",
        "",
        "- Sharafaldin, Lashkari & Ghorbani (2018), Canadian Institute for Cybersecurity, UNB: "
        "<https://www.unb.ca/cic/datasets/ids-2017.html>",
        "- Files: the **TrafficLabelling** CSVs inside `GeneratedLabelledFlows.zip`, which keep "
        "Flow ID, source/destination IP and port, and timestamp (needed for pseudonymisation and "
        "realistic alerts). Five working days, 3–7 July 2017, simulated enterprise network.",
        "- Public research data. No real personal data; hostnames and usernames used later are synthetic.",
        "",
        "## Cleaning steps and row counts",
        "",
        "| Step | Rows |",
        "|---|---:|",
        f"| Rows read from {len(stats['files'])} CSVs | {_n(t['rows_read'])} |",
        f"| Empty rows dropped (bare commas) | −{_n(t['empty_rows_dropped'])} |",
        f"| Rows with Infinity/NaN in a feature dropped | −{_n(t['rows_with_non_finite_dropped'])} |",
        f"| Rows before de-duplication | {_n(t['rows_before_dedupe'])} |",
        f"| Exact duplicates removed (features + label) | −{_n(t['duplicates_removed'])} |",
        f"| **Final rows** | **{_n(t['rows_final'])}** |",
        "",
        f"- Duplicate columns dropped: {', '.join(f'`{c}`' for c in stats['duplicate_columns_dropped']) or 'none'}",
        f"- Constant (zero-variance) columns dropped ({len(stats['constant_columns_dropped'])}): "
        + (", ".join(f"`{c}`" for c in stats["constant_columns_dropped"]) or "none"),
        "- Non-finite values by column: "
        + (", ".join(f"`{c}` {_n(n)}" for c, n in stats["non_finite_by_column"].items()) or "none"),
        f"- Identical feature vectors carrying different labels (kept, reported as label noise): "
        f"{_n(stats['conflicting_label_vectors'])} vectors, {_n(stats['conflicting_label_rows'])} rows",
        f"- Features kept: **{stats['n_features']}** (all flow statistics plus `Destination Port` and "
        "`Protocol`). IPs, source port, Flow ID and timestamp are kept as metadata only, never as features.",
        f"- Parquet size {stats['parquet_bytes'] / 1e6:.0f} MB; loads in {stats['parquet_load_seconds']} s.",
        "",
        "## Per-file summary",
        "",
        "| File | Day | Encoding | Rows read | Empty | Non-finite | Rows out |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for f in stats["files"]:
        lines.append(
            f"| `{f['file']}` | {f['day']} | {f['encoding']} | {_n(f['rows_read'])} | "
            f"{_n(f['empty_rows_dropped'])} | {_n(f['rows_with_non_finite_dropped'])} | {_n(f['rows_out'])} |"
        )

    before, after = stats["labels_before_dedupe"], stats["labels_final"]
    lines += [
        "",
        "## Labels → families",
        "",
        "| Raw label | Family | Before dedupe | Final | Removed |",
        "|---|---|---:|---:|---:|",
    ]
    for label, family in RAW_TO_FAMILY.items():
        b, a = before.get(label, 0), after.get(label, 0)
        lines.append(f"| {label} | {family} | {_n(b)} | {_n(a)} | {_n(b - a)} |")

    by_day = stats["family_by_day"]
    lines += [
        "",
        "## Families by day",
        "",
        "| Family | " + " | ".join(DAYS) + " | Total |",
        "|---|" + "---:|" * (len(DAYS) + 1),
    ]
    for fam in FAMILIES:
        row = by_day.get(fam, {})
        cells = [_n(row.get(d, 0)) for d in DAYS]
        lines.append(f"| {fam} | " + " | ".join(cells) + f" | {_n(sum(row.values()))} |")

    shared = sorted(
        fam
        for fam, row in by_day.items()
        if fam != "BENIGN"
        and fam not in HELD_OUT_FAMILIES
        and any(row.get(d, 0) for d in EARLY_DAYS)
        and any(row.get(d, 0) for d in DAYS[3:])
    )
    lines += [
        "",
        "## Known issues and limitations",
        "",
        "- **Labelling errors in the original release.** Engelen, Rimmer & Joosen (2021) and Lanvin et "
        "al. (2023) found mislabelled flows, duplicated packets and feature-calculation bugs; the "
        "KU Leuven study reconstructed or relabelled more than 20% of traces. This card describes the "
        "*original* release; a corrected-version rerun is a stretch goal.",
        "- **Random splits flatter IDS models.** Neighbouring flows from the same attack session are "
        "near-identical, so a random split puts near-duplicates in train and test even after exact "
        "de-duplication.",
        "- **Each attack family occurs on a single day.** Attack families present both Mon–Wed and "
        f"Thu–Fri: {', '.join(shared) if shared else '**none**'}. So the blueprint's cross-day check "
        "(train Mon–Wed, test Thu–Fri on shared families) can only be run as a *binary* "
        "unseen-attack test: does a detector trained on Mon–Wed attacks flag Thu–Fri attacks at all?",
        "- **Other-rare** (Infiltration, Heartbleed) has too few flows to train on; it is held out "
        "of training and its detection rate is reported separately.",
        "- **Timestamps** are kept as raw text: the release writes them as day/month/year, and "
        "afternoon times appear to use a 12-hour clock without AM/PM (check before parsing). "
        "The `day` column comes from the file name.",
        "- **Thursday web-attack file** is cp1252-encoded (read as latin-1) and its labels contain a "
        "garbled en dash, repaired to `Web Attack - …`.",
        "",
    ]
    return "\n".join(lines)
