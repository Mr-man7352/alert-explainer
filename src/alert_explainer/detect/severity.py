"""Deterministic alert severity (blueprint §6.4). The LLM never sets or changes it.

    score = base_severity[family] × confidence_band × asset_criticality      (each 1–3)

    score  1–3 → Low     4–9 → Medium     12–18 → High     27 → Critical

(The product of three numbers in 1–3 can only be 1, 2, 3, 4, 6, 8, 9, 12, 18 or 27.)

- **base_severity** reflects what the attack does if it succeeds: reconnaissance (PortScan) 1;
  single-source DoS and password guessing 2; distributed DoS, exploitation of a public-facing
  application, botnet command and control and the rare families 3.
- **confidence_band** comes from the *calibrated* probability of the predicted family:
  < 0.6 → 1, 0.6–0.9 → 2, ≥ 0.9 → 3.
- **asset_criticality** is the affected internal asset's criticality from the synthetic inventory
  (1 standard, 2 sensitive, 3 business-critical). An internal destination is the affected asset;
  if the destination is external (e.g. a bot calling out), the internal source is. An address that
  is in neither the inventory nor the internal range gets 2: unknown assets are treated as
  sensitive until someone says otherwise.

Consequences worth stating: only a confident, high-impact attack on a business-critical asset is
Critical (e.g. a DDoS on the web server at confidence ≥ 0.9), and a port scan never rises above
Medium. ``severity_table()`` lists every combination; ``tests/unit/test_severity.py`` checks each
row against a hand-written copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from alert_explainer.data.labels import FAMILIES

Severity = Literal["Low", "Medium", "High", "Critical"]
SEVERITIES: tuple[Severity, ...] = ("Low", "Medium", "High", "Critical")

BASE_SEVERITY: dict[str, int] = {
    "PortScan": 1,
    "DoS": 2,
    "BruteForce": 2,
    "DDoS": 3,
    "WebAttack": 3,
    "Bot": 3,
    "Other-rare": 3,
}
BASE_REASON: dict[str, str] = {
    "PortScan": "reconnaissance",
    "DoS": "single-source denial of service",
    "BruteForce": "password guessing",
    "DDoS": "distributed denial of service",
    "WebAttack": "exploitation of a public-facing application",
    "Bot": "botnet command and control",
    "Other-rare": "rare, high-impact attack type",
}
CONFIDENCE_BANDS: tuple[tuple[float, int, str], ...] = (
    # (lower bound, band, name); checked from the top
    (0.9, 3, "high"),
    (0.6, 2, "medium"),
    (0.0, 1, "low"),
)
DEFAULT_CRITICALITY = 2
LEVELS: tuple[tuple[int, Severity], ...] = (
    (27, "Critical"),
    (12, "High"),
    (4, "Medium"),
    (1, "Low"),
)

if set(BASE_SEVERITY) != set(FAMILIES) - {"BENIGN"}:  # pragma: no cover
    raise RuntimeError("every attack family needs a base severity")


@dataclass(frozen=True)
class SeverityResult:
    severity: Severity
    score: int
    base: int
    confidence_band: int
    confidence_band_name: str
    asset_criticality: int

    @property
    def rule(self) -> str:
        return (
            f"base {self.base} × confidence band {self.confidence_band} "
            f"({self.confidence_band_name}) × asset criticality {self.asset_criticality} "
            f"= {self.score} → {self.severity}"
        )


def confidence_band(confidence: float) -> tuple[int, str]:
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be in [0, 1], got {confidence}")
    for lower, band, name in CONFIDENCE_BANDS:
        if confidence >= lower:
            return band, name
    raise RuntimeError("unreachable")  # pragma: no cover


def score_to_severity(score: int) -> Severity:
    for lower, level in LEVELS:
        if score >= lower:
            return level
    raise ValueError(f"score must be ≥ 1, got {score}")


def compute_severity(
    family: str, confidence: float, asset_criticality: int | None
) -> SeverityResult:
    if family not in BASE_SEVERITY:
        raise ValueError(f"No severity for family {family!r} (BENIGN flows are not alerts)")
    crit = DEFAULT_CRITICALITY if asset_criticality is None else asset_criticality
    if crit not in (1, 2, 3):
        raise ValueError(f"asset criticality must be 1, 2 or 3, got {crit}")
    band, band_name = confidence_band(confidence)
    base = BASE_SEVERITY[family]
    score = base * band * crit
    return SeverityResult(score_to_severity(score), score, base, band, band_name, crit)


def severity_table() -> list[dict[str, Any]]:
    """Every (family, confidence band, criticality) combination, for the docs and the tests."""
    rows = []
    for family in BASE_SEVERITY:
        for lower, _, name in reversed(CONFIDENCE_BANDS):
            for crit in (1, 2, 3):
                r = compute_severity(family, lower, crit)
                rows.append(
                    {
                        "family": family,
                        "confidence_band": name,
                        "asset_criticality": crit,
                        "score": r.score,
                        "severity": r.severity,
                    }
                )
    return rows


def severity_table_markdown() -> str:
    lines = [
        "| Family (base) | Confidence | Criticality 1 | Criticality 2 | Criticality 3 |",
        "|---|---|---|---|---|",
    ]
    rows = severity_table()
    for family, base in BASE_SEVERITY.items():
        for _, _, band in reversed(CONFIDENCE_BANDS):
            cells = [
                f"{r['severity']} ({r['score']})"
                for r in rows
                if r["family"] == family and r["confidence_band"] == band
            ]
            lines.append(f"| {family} ({base}) | {band} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
