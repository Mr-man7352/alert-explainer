"""Every row of the severity rule table, checked against a hand-written copy (blueprint §6.4)."""

import itertools

import pytest

from alert_explainer.detect.severity import (
    BASE_SEVERITY,
    SEVERITIES,
    compute_severity,
    confidence_band,
    score_to_severity,
    severity_table,
    severity_table_markdown,
)

L, M, H, C = "Low", "Medium", "High", "Critical"
# family → rows (confidence low, medium, high) × columns (asset criticality 1, 2, 3).
# Written out by hand from the rule in severity.py's docstring, not generated from the code.
EXPECTED = {
    "PortScan": [[L, L, L], [L, M, M], [L, M, M]],
    "DoS": [[L, M, M], [M, M, H], [M, H, H]],
    "BruteForce": [[L, M, M], [M, M, H], [M, H, H]],
    "DDoS": [[L, M, M], [M, H, H], [M, H, C]],
    "WebAttack": [[L, M, M], [M, H, H], [M, H, C]],
    "Bot": [[L, M, M], [M, H, H], [M, H, C]],
    "Other-rare": [[L, M, M], [M, H, H], [M, H, C]],
}
BAND_CONFIDENCE = {0: 0.3, 1: 0.75, 2: 0.95}  # a confidence inside each band


@pytest.mark.parametrize(
    ("family", "band", "crit"),
    [(f, b, c) for f in EXPECTED for b in range(3) for c in range(3)],
)
def test_every_row_of_the_table(family: str, band: int, crit: int) -> None:
    result = compute_severity(family, BAND_CONFIDENCE[band], crit + 1)
    assert result.severity == EXPECTED[family][band][crit]
    assert result.score == BASE_SEVERITY[family] * (band + 1) * (crit + 1)


def test_table_covers_every_family_and_matches_compute() -> None:
    rows = severity_table()
    assert len(rows) == len(BASE_SEVERITY) * 3 * 3
    assert {r["family"] for r in rows} == set(EXPECTED)
    bands = {"low": 0, "medium": 1, "high": 2}
    for r in rows:
        assert (
            r["severity"]
            == EXPECTED[r["family"]][bands[r["confidence_band"]]][r["asset_criticality"] - 1]
        )
    md = severity_table_markdown()
    assert md.count("\n") == 2 + len(BASE_SEVERITY) * 3


@pytest.mark.parametrize(
    ("confidence", "band"),
    [(0.0, 1), (0.5999, 1), (0.6, 2), (0.8999, 2), (0.9, 3), (1.0, 3)],
)
def test_confidence_band_edges(confidence: float, band: int) -> None:
    assert confidence_band(confidence)[0] == band


def test_documented_consequences() -> None:
    # the blueprint's example: DDoS on a criticality-3 server at confidence >= 0.9 → Critical
    assert compute_severity("DDoS", 0.93, 3).severity == "Critical"
    # a port scan never rises above Medium
    assert max(
        SEVERITIES.index(compute_severity("PortScan", c, k).severity)
        for c, k in itertools.product((0.1, 0.7, 1.0), (1, 2, 3))
    ) == SEVERITIES.index("Medium")
    # Critical needs the maximum score
    assert all((r["severity"] == "Critical") == (r["score"] == 27) for r in severity_table())


def test_monotone_in_confidence_and_criticality() -> None:
    for family in BASE_SEVERITY:
        for crit in (1, 2, 3):
            levels = [
                SEVERITIES.index(compute_severity(family, c, crit).severity)
                for c in (0.1, 0.7, 0.95)
            ]
            assert levels == sorted(levels)
        for conf in (0.1, 0.7, 0.95):
            levels = [
                SEVERITIES.index(compute_severity(family, conf, k).severity) for k in (1, 2, 3)
            ]
            assert levels == sorted(levels)


def test_unknown_asset_defaults_to_criticality_2() -> None:
    assert compute_severity("DoS", 0.95, None).asset_criticality == 2
    assert compute_severity("DoS", 0.95, None).severity == compute_severity("DoS", 0.95, 2).severity


def test_rule_text_explains_the_result() -> None:
    r = compute_severity("WebAttack", 0.97, 3)
    assert r.rule == "base 3 × confidence band 3 (high) × asset criticality 3 = 27 → Critical"


@pytest.mark.parametrize(
    ("family", "confidence", "crit"),
    [
        ("BENIGN", 0.9, 1),
        ("Nonsense", 0.9, 1),
        ("DoS", 1.2, 1),
        ("DoS", -0.1, 1),
        ("DoS", 0.9, 4),
        ("DoS", 0.9, 0),
    ],
)
def test_invalid_inputs_raise(family: str, confidence: float, crit: int) -> None:
    with pytest.raises(ValueError):
        compute_severity(family, confidence, crit)


def test_score_to_severity_levels() -> None:
    assert [score_to_severity(s) for s in (1, 2, 3, 4, 6, 8, 9, 12, 18, 27)] == [
        L, L, L, M, M, M, M, H, H, C,
    ]  # fmt: skip
    with pytest.raises(ValueError):
        score_to_severity(0)
