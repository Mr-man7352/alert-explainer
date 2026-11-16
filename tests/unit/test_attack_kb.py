"""Day 7: ATT&CK parsing on a small hand-built STIX bundle, plus the real pinned bundle if present."""

import itertools
import json
from pathlib import Path
from typing import Any

import pytest

from alert_explainer.config import get_settings
from alert_explainer.data.labels import RAW_TO_FAMILY, TRAIN_FAMILIES
from alert_explainer.explain.attack_kb import (
    ATTACK_VERSION,
    FAMILY_TECHNIQUES,
    LABEL_EXPECTED_TECHNIQUES,
    MAPPING_REASONS,
    KnowledgeBase,
    build_kb,
    clean_text,
    parse_bundle,
    trim_words,
    verify_mapping,
)

MAPPED_IDS = sorted({t for ids in FAMILY_TECHNIQUES.values() for t in ids})


def _technique(tid: str, name: str, **extra: Any) -> dict[str, Any]:
    return {
        "type": "attack-pattern",
        "id": f"attack-pattern--{tid}",
        "name": name,
        "description": extra.pop("description", f"{name} description. (Citation: Someone 2020)"),
        "kill_chain_phases": [{"kill_chain_name": "mitre-attack", "phase_name": "impact"}],
        "x_mitre_platforms": ["Linux"],
        "x_mitre_is_subtechnique": "." in tid,
        "external_references": [
            {
                "source_name": "mitre-attack",
                "external_id": tid,
                "url": f"https://attack.mitre.org/x/{tid}",
            }
        ],
        **extra,
    }


def _mitigation(mid: str, name: str, **extra: Any) -> dict[str, Any]:
    return {
        "type": "course-of-action",
        "id": f"course-of-action--{mid}",
        "name": name,
        "external_references": [{"source_name": "mitre-attack", "external_id": mid}],
        **extra,
    }


_REL_IDS = itertools.count()


def _rel(kind: str, source: str, target: str, **extra: Any) -> dict[str, Any]:
    return {
        "type": "relationship",
        "id": f"relationship--{next(_REL_IDS)}",  # STIX ids are unique per object
        "relationship_type": kind,
        "source_ref": source,
        "target_ref": target,
        **extra,
    }


def mini_bundle(include_all_mapped: bool = True) -> dict[str, Any]:
    objects: list[dict[str, Any]] = []
    if include_all_mapped:
        objects += [_technique(t, f"Technique {t}") for t in MAPPED_IDS if t != "T1046"]
    objects += [
        _technique(
            "T1046",
            "Network Service Discovery",
            description="Adversaries may scan [ports](https://example.org) for <code>services</code>."
            "(Citation: A) Second sentence here.\n\nThird paragraph.",
            kill_chain_phases=[{"kill_chain_name": "mitre-attack", "phase_name": "discovery"}],
        ),
        _technique("T0001", "Revoked One", revoked=True),
        _technique("T0002", "Deprecated One", x_mitre_deprecated=True),
        _technique("T0003", "Other Matrix", kill_chain_phases=[{"kill_chain_name": "ics", "phase_name": "x"}]),
        _mitigation("M1030", "Network Segmentation"),
        _mitigation("M1042", "Disable or Remove Feature or Program"),
        _mitigation("M9999", "Old Mitigation", x_mitre_deprecated=True),
        {"type": "x-mitre-detection-strategy", "id": "x-mitre-detection-strategy--d1", "name": "Scan detection"},
        _rel("mitigates", "course-of-action--M1042", "attack-pattern--T1046", description="Turn off unused services. (Citation: B)"),
        _rel("mitigates", "course-of-action--M1030", "attack-pattern--T1046", description="Segment the network."),
        _rel("mitigates", "course-of-action--M9999", "attack-pattern--T1046", description="Deprecated."),
        _rel("mitigates", "course-of-action--M1030", "attack-pattern--T1046", description="Revoked rel.", revoked=True),
        _rel("detects", "x-mitre-detection-strategy--d1", "attack-pattern--T1046"),
        _rel("mitigates", "course-of-action--M1030", "attack-pattern--T0001", description="To a revoked technique."),
    ]  # fmt: skip
    return {"type": "bundle", "id": "bundle--test", "objects": objects}


def test_parse_filters_revoked_and_deprecated() -> None:
    kb = parse_bundle(mini_bundle())
    assert "T1046" in kb and "T0001" not in kb and "T0002" not in kb
    t = kb["T1046"]
    assert t.name == "Network Service Discovery" and t.tactics == ("discovery",)
    assert kb["T0003"].tactics == ()  # only mitre-attack kill-chain phases count
    assert [m.id for m in t.mitigations] == [
        "M1030",
        "M1042",
    ]  # sorted; deprecated + revoked rel dropped
    assert t.mitigations[1].guidance == "Turn off unused services."
    assert t.detection_strategies == ("Scan detection",)
    assert "Citation" not in t.description and "<code>" not in t.description
    assert "scan ports for services." in t.description
    assert kb["T1499.002"].is_subtechnique and kb["T1499.002"].parent_id == "T1499"
    assert kb["T1499"].parent_id is None


def test_clean_and_trim_text() -> None:
    assert clean_text("A [link](http://x) (Citation: Y)  here.") == "A link here."
    text = "One two three. Four five six seven. Eight nine."
    assert trim_words(text, 100) == text
    assert trim_words(text, 8) == "One two three. Four five six seven."
    assert trim_words("a b c d e f", 3) == "a b c …"


def test_verify_mapping() -> None:
    verify_mapping(parse_bundle(mini_bundle()))
    with pytest.raises(ValueError, match="missing"):
        verify_mapping(parse_bundle(mini_bundle(include_all_mapped=False)))


def test_build_kb_from_local_bundle_round_trip(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps(mini_bundle()))
    kb = build_kb(tmp_path / "attack", bundle=bundle)
    assert kb.meta["n_techniques"] == len(kb) and len(kb.meta["bundle_sha256"]) == 64
    back = KnowledgeBase.load(tmp_path / "attack" / "kb.json")
    assert back.techniques == kb.techniques and back.meta == kb.meta


def test_mapping_tables_are_complete() -> None:
    attack_families = [f for f in TRAIN_FAMILIES if f != "BENIGN"] + ["Other-rare"]
    assert set(FAMILY_TECHNIQUES) == set(attack_families) == set(MAPPING_REASONS)
    attack_labels = {label for label, fam in RAW_TO_FAMILY.items() if fam != "BENIGN"}
    assert set(LABEL_EXPECTED_TECHNIQUES) == attack_labels
    for label, expected in LABEL_EXPECTED_TECHNIQUES.items():
        # a correct technique is always one the family's mapping allows
        assert expected & set(FAMILY_TECHNIQUES[RAW_TO_FAMILY[label]]), label


# --- the real pinned bundle, when `ae attack-kb` has been run (skipped in CI) -----------------

EXPECTED_NAMES = {
    "T1499": "Endpoint Denial of Service",
    "T1499.002": "Service Exhaustion Flood",
    "T1498": "Network Denial of Service",
    "T1498.001": "Direct Network Flood",
    "T1046": "Network Service Discovery",
    "T1110": "Brute Force",
    "T1110.001": "Password Guessing",
    "T1190": "Exploit Public-Facing Application",
    "T1071": "Application Layer Protocol",
    "T1071.001": "Web Protocols",
    "T1204": "User Execution",
}


@pytest.fixture(scope="module")
def real_kb() -> KnowledgeBase:
    path = get_settings().attack_dir / "kb.json"
    if not path.exists():
        pytest.skip("real ATT&CK KB not built; run `uv run ae attack-kb`")
    return KnowledgeBase.load(path)


def test_real_kb_mapped_techniques_have_expected_names(real_kb: KnowledgeBase) -> None:
    assert real_kb.meta["attack_version"] == ATTACK_VERSION
    assert set(EXPECTED_NAMES) == set(MAPPED_IDS)
    for tid, name in EXPECTED_NAMES.items():
        assert real_kb[tid].name == name
    assert len(real_kb) > 500
    assert all(real_kb[t].mitigations for t in MAPPED_IDS)
