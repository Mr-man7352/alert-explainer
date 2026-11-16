"""Day 7: deterministic retrieval, the embedding fallback, and the retrieval check."""

import json
from pathlib import Path

import numpy as np
import pytest

from alert_explainer.explain.attack_kb import FAMILY_TECHNIQUES, parse_bundle
from alert_explainer.explain.retrieval_check import run_retrieval_check
from alert_explainer.explain.retrieve import (
    N_CANDIDATES,
    Retriever,
    TechniqueIndex,
    TfidfEmbedder,
    alert_query,
    family_techniques,
    is_ambiguous,
)
from alert_explainer.explain.schemas import Alert
from tests.unit.test_attack_kb import mini_bundle


@pytest.fixture(scope="module")
def kb():
    b = mini_bundle()
    # give a few techniques distinctive text so lexical retrieval has something to find
    for o in b["objects"]:
        if o.get("name") == "Technique T1110.001":
            o["name"], o["description"] = (
                "Password Guessing",
                "Adversaries guess passwords at login prompts.",
            )
        if o.get("name") == "Technique T1498":
            o["name"], o["description"] = (
                "Network Denial of Service",
                "Floods exhaust network bandwidth.",
            )
    return parse_bundle(b)


def _alert(family: str = "PortScan", confidence: float = 0.95, **kw) -> Alert:
    base = {
        "alert_id": "a-0000042",
        "timestamp": None,
        "src_ip": "172.16.0.1",
        "dst_ip": "192.168.10.50",
        "dst_port": 22,
        "protocol": "TCP",
        "dst_host": "web-01",
        "dst_owner": "j.smith",
        "family": family,
        "confidence": confidence,
        "attack_score": 0.99,
        "severity": "Medium",
        "severity_basis": {
            "base": 1, "confidence_band": 3, "confidence_band_name": "high",
            "asset_criticality": 3, "score": 9, "rule": "…",
        },
        "top_features": [
            {"name": "Destination Port", "value": 22, "benign_share": 0.001, "contribution": 2.0},
            {"name": "Flow Duration", "value": 12.0, "benign_percentile": 3.0, "contribution": 1.0},
        ],
        "allowed_techniques": list(FAMILY_TECHNIQUES[family]),
        "detector": {"model": "lightgbm", "version": "v", "threshold": 0.2, "attribution_method": "m"},
        "provenance": {"row_id": 42, "true_label": "PortScan"},
    }  # fmt: skip
    return Alert.model_validate({**base, **kw})


def test_deterministic_path_returns_mapped_techniques(kb) -> None:
    r = Retriever(kb).retrieve(_alert())
    assert not r.ambiguous and r.query is None
    assert r.mapped_ids == ["T1046"] and r.candidate_ids == []
    ref = r.techniques[0]
    assert ref.source == "mapping" and ref.name == "Network Service Discovery"
    assert [m["id"] for m in ref.mitigations] == ["M1030", "M1042"]
    assert ref.detection_strategies == ("Scan detection",)
    for family, ids in FAMILY_TECHNIQUES.items():
        assert [t.id for t in family_techniques(kb, family)] == list(ids)


def test_ambiguous_alert_gets_embedding_candidates(kb) -> None:
    index = TechniqueIndex(kb, TfidfEmbedder([t.embedding_text() for t in kb.techniques.values()]))
    alert = _alert(family="BruteForce", confidence=0.45)
    assert is_ambiguous(alert) and is_ambiguous(_alert(family="Other-rare"))
    r = Retriever(kb, index).retrieve(alert)
    assert r.ambiguous and r.query is not None
    assert r.mapped_ids == list(FAMILY_TECHNIQUES["BruteForce"])
    assert len(r.candidate_ids) == N_CANDIDATES
    assert not set(r.candidate_ids) & set(r.mapped_ids)  # mapped IDs are never duplicated
    assert all(t.similarity is not None for t in r.techniques if t.source == "embedding")
    updated = Retriever(kb, index).with_candidates(alert, r)
    assert updated.candidate_techniques == r.candidate_ids
    assert updated.permitted_techniques == r.allowed_ids
    # confident alerts never use the fallback unless forced
    assert Retriever(kb, index).retrieve(_alert()).candidate_ids == []
    assert Retriever(kb, index).retrieve(_alert(), force_fallback=True).candidate_ids


def test_query_never_contains_identifiers() -> None:
    a = _alert()
    q = alert_query(a)
    for value in (a.src_ip, a.dst_ip, a.dst_host, a.dst_owner):
        assert value not in q
    assert "port 22/TCP (SSH)" in q and "lower than 97% of normal flows" in q


def test_tfidf_search_finds_the_obvious_technique(kb) -> None:
    index = TechniqueIndex(kb, TfidfEmbedder([t.embedding_text() for t in kb.techniques.values()]))
    assert index.search("guess passwords at the login prompt", k=1)[0][0] == "T1110.001"
    assert index.rank_of("flood that exhausts network bandwidth", {"T1498"}) == 1
    hits = index.search("password", k=3, exclude={"T1110.001"})
    assert "T1110.001" not in [t for t, _ in hits] and len(hits) == 3


class _CountingEmbedder:
    name = "fake-model"

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        rng = np.random.default_rng(len(texts))
        m = rng.normal(size=(len(texts), 8)).astype(np.float32)
        return m / np.linalg.norm(m, axis=1, keepdims=True)


def test_embeddings_are_cached(kb, tmp_path: Path) -> None:
    first = _CountingEmbedder()
    TechniqueIndex(kb, first, cache_dir=tmp_path)
    second = _CountingEmbedder()
    index = TechniqueIndex(kb, second, cache_dir=tmp_path)
    assert first.calls == 1 and second.calls == 0
    assert index.matrix.shape == (len(kb), 8) and len(list(tmp_path.glob("embeddings-*.npy"))) == 1


def test_retrieval_check_report(kb, tmp_path: Path) -> None:
    alerts = [
        _alert(),
        _alert(family="BruteForce", confidence=0.4, alert_id="a-0000043",
               provenance={"row_id": 43, "true_label": "SSH-Patator"}),
    ]  # fmt: skip
    path = tmp_path / "alerts.json"
    path.write_text(json.dumps([a.model_dump(mode="json") for a in alerts]))
    result = run_retrieval_check(
        kb, tmp_path, path, tmp_path / "reports", embedders=("tfidf", "bogus")
    )
    assert result["all_mapped_ids_resolve"]
    assert "error" in result["embedding"]["bogus"]
    tf = result["embedding"]["tfidf"]
    assert tf["family_description"]["n_queries"] == len(FAMILY_TECHNIQUES)
    assert tf["features_only"]["n_queries"] == 2  # only families that have alerts
    a = result["alerts"]
    assert a["n_alerts"] == 2 and a["ambiguous"] == 1
    assert a["mapped_technique_correct_for_true_label"] == 1.0
    md = (tmp_path / "reports" / "retrieval_check.md").read_text()
    assert "## 1. Mapping table" in md and "T1046 Network Service Discovery" in md
