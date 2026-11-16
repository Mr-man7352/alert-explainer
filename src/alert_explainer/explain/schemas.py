"""Pydantic schemas shared by the alert builder, the explainer and the API (blueprint §7.1).

``Alert`` is **frozen**: once deterministic code has set the family, confidence and severity,
nothing downstream (the LLM least of all) can change them. ``extra="forbid"`` rejects unknown
fields, so a malformed payload fails loudly instead of being half-read.

At this stage the alert holds the real (synthetic) identifiers: IPs from the dataset and
hostnames/owners from the inventory. The Day 10 pseudonymiser turns it into the LLM payload by
replacing them with tokens, and only the fields listed in ``LLM_FIELDS`` are ever sent.
``provenance`` (row id, flow id, ground-truth label) is for evaluation only and is never sent.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Severity = Literal["Low", "Medium", "High", "Critical"]
CONTEXT_TEXT_MAX = 300  # characters of untrusted free text kept per alert (blueprint §7.3)
TECHNIQUE_ID = re.compile(r"^T\d{4}(\.\d{3})?$")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TopFeature(_Frozen):
    name: str
    value: float
    # Percentile of the value among BENIGN training flows (numeric features) ...
    benign_percentile: float | None = Field(default=None, ge=0, le=100)
    # ... or, for categorical ones (port, protocol), the share of BENIGN flows with this value.
    benign_share: float | None = Field(default=None, ge=0, le=1)
    contribution: float  # attribution towards the predicted family (SHAP value or IG)


class SeverityBasis(_Frozen):
    base: int = Field(ge=1, le=3)
    confidence_band: int = Field(ge=1, le=3)
    confidence_band_name: Literal["low", "medium", "high"]
    asset_criticality: int = Field(ge=1, le=3)
    score: int = Field(ge=1, le=27)
    rule: str


class DetectorInfo(_Frozen):
    model: str
    version: str
    threshold: float
    attribution_method: str


class Provenance(_Frozen):
    """Evaluation-only metadata. Never sent to the LLM or shown in the manager view."""

    row_id: int
    flow_id: str | None = None
    source_file: str | None = None
    true_label: str | None = None
    true_family: str | None = None


class Alert(_Frozen):
    alert_id: str = Field(pattern=r"^a-\d{7,}$")
    timestamp: str | None  # ISO 8601, local capture time as recorded in the CSV
    src_ip: str
    src_port: int | None = Field(default=None, ge=0, le=65535)
    dst_ip: str
    dst_port: int = Field(ge=0, le=65535)
    protocol: str
    src_host: str | None = None
    src_owner: str | None = None
    dst_host: str | None = None
    dst_owner: str | None = None
    affected_asset: Literal["src", "dst"] | None = None
    family: str
    confidence: float = Field(ge=0, le=1)  # calibrated probability of the predicted family
    attack_score: float = Field(
        ge=0, le=1
    )  # 1 − raw P(BENIGN); the alert fired because ≥ threshold
    severity: Severity
    severity_basis: SeverityBasis
    top_features: list[TopFeature] = Field(min_length=1, max_length=10)
    allowed_techniques: list[str] = Field(min_length=1)
    candidate_techniques: list[str] = Field(default_factory=list)
    context_text: str | None = Field(default=None, max_length=CONTEXT_TEXT_MAX)
    detector: DetectorInfo
    provenance: Provenance

    @field_validator("allowed_techniques", "candidate_techniques")
    @classmethod
    def _technique_ids(cls, ids: list[str]) -> list[str]:
        bad = [t for t in ids if not TECHNIQUE_ID.match(t)]
        if bad:
            raise ValueError(f"not ATT&CK technique IDs: {bad}")
        return ids

    @property
    def permitted_techniques(self) -> list[str]:
        """Technique IDs an explanation may cite: the mapping table's, then retrieval candidates."""
        return list(dict.fromkeys([*self.allowed_techniques, *self.candidate_techniques]))


# Fields the LLM may see (after pseudonymisation). Everything else stays in the database.
LLM_FIELDS: frozenset[str] = frozenset(
    {
        "alert_id",
        "timestamp",
        "src_ip",
        "dst_ip",
        "dst_port",
        "protocol",
        "src_host",
        "src_owner",
        "dst_host",
        "dst_owner",
        "affected_asset",
        "family",
        "confidence",
        "severity",
        "top_features",
        "allowed_techniques",
        "candidate_techniques",
        "context_text",
    }
)
