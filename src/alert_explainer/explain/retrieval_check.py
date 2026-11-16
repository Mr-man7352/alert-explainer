"""Day 7 check: does each family retrieve the right ATT&CK techniques? (``uv run ae retrieval-check``)

Three parts, written to ``reports/retrieval_check.{json,md}``:

1. **Mapping table.** Every family's techniques resolve in the pinned KB (ID, name, tactic,
   mitigations, detection strategies). This is the path every alert takes.
2. **Embedding retrieval on its own**, as a sanity check of the fallback. For each family, two
   queries: a short description of the attack, and a *features-only* description (port and
   unusual flow features from that family's alerts, with no attack words). The rank of the
   family's best mapped technique among ~700 is reported, with recall@3/@5 and MRR. The first
   query is easy by construction; the second shows how much the fallback can infer from numbers
   alone. Both embedders are run when available (MiniLM and the TF-IDF baseline).
3. **The built alerts** (if ``ae build-alerts`` has run): how many are ambiguous and got
   candidates, and the share whose mapped techniques include a correct one for the flow's *true*
   label (``LABEL_EXPECTED_TECHNIQUES``). Because the validator (Day 9) restricts explanations to
   these IDs, that share is the ceiling for the "correct ATT&CK technique" metric.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from alert_explainer.explain.attack_kb import (
    FAMILY_TECHNIQUES,
    LABEL_EXPECTED_TECHNIQUES,
    MAPPING_REASONS,
    KnowledgeBase,
)
from alert_explainer.explain.retrieve import (
    FAMILY_DESCRIPTION,
    SERVICES,
    Embedder,
    Retriever,
    TechniqueIndex,
    _describe_feature,
    make_embedder,
)
from alert_explainer.explain.schemas import Alert


def _family_query(family: str) -> str:
    return f"Network traffic resembling {FAMILY_DESCRIPTION[family]}."


def _features_only_query(family: str, alerts: list[Alert]) -> str | None:
    """Port and the most common top features of this family's alerts, without attack words."""
    mine = [a for a in alerts if a.family == family]
    if not mine:
        return None
    port = Counter(a.dst_port for a in mine).most_common(1)[0][0]
    proto = Counter(a.protocol for a in mine).most_common(1)[0][0]
    feats: dict[str, list[float]] = {}
    for a in mine:
        for f in a.top_features[:3]:
            if f.benign_percentile is not None:  # the port is already in the query
                feats.setdefault(f.name, []).append(f.benign_percentile)
    top = sorted(feats.items(), key=lambda kv: -len(kv[1]))[:3]
    service = SERVICES.get(port)
    described = "; ".join(_describe_feature(name, float(np.median(p))) for name, p in top)
    port_text = f"port {port}/{proto}" + (f" ({service})" if service else "")
    return f"Network traffic sent to {port_text}. Unusual flow features: {described}."


def _rank_block(index: TechniqueIndex, queries: dict[str, str | None]) -> dict[str, Any]:
    per_family: dict[str, Any] = {}
    ranks: list[int] = []
    for family, query in queries.items():
        if query is None:
            continue
        wanted = set(FAMILY_TECHNIQUES[family])
        rank = index.rank_of(query, wanted)
        top5 = index.search(query, k=5)
        per_family[family] = {
            "query": query,
            "rank_of_mapped_technique": rank,
            "top5": [
                {"id": t, "name": index.kb[t].name, "similarity": round(s, 3)} for t, s in top5
            ],
        }
        ranks.append(rank or len(index.ids) + 1)
    r = np.array(ranks, dtype=float)
    return {
        "per_family": per_family,
        "recall_at_3": round(float((r <= 3).mean()), 3) if len(r) else None,
        "recall_at_5": round(float((r <= 5).mean()), 3) if len(r) else None,
        "mrr": round(float((1 / r).mean()), 3) if len(r) else None,
        "n_queries": len(r),
    }


def _load_alerts(path: Path) -> list[Alert]:
    if not path.exists():
        return []
    return [Alert.model_validate(a) for a in json.loads(path.read_text())]


def run_retrieval_check(
    kb: KnowledgeBase,
    attack_dir: Path,
    alerts_path: Path,
    reports_dir: Path,
    embedders: tuple[str, ...] = ("minilm", "tfidf"),
) -> dict[str, Any]:
    alerts = _load_alerts(alerts_path)

    mapping = {}
    for family, ids in FAMILY_TECHNIQUES.items():
        mapping[family] = {
            "reason": MAPPING_REASONS[family],
            "techniques": [
                {
                    "id": t,
                    "name": kb[t].name,
                    "tactics": list(kb[t].tactics),
                    "n_mitigations": len(kb[t].mitigations),
                    "mitigations": [m.name for m in kb[t].mitigations[:3]],
                    "detection_strategies": list(kb[t].detection_strategies[:1]),
                }
                for t in ids
            ],
        }

    embedding: dict[str, Any] = {}
    indices: dict[str, TechniqueIndex] = {}
    for kind in embedders:
        try:
            embedder: Embedder = make_embedder(kind, kb)
        except Exception as exc:  # model download blocked, package missing, …
            embedding[kind] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  embedder {kind} unavailable: {exc}")
            continue
        index = TechniqueIndex(kb, embedder, cache_dir=attack_dir)
        indices[kind] = index
        embedding[kind] = {
            "model": embedder.name,
            "family_description": _rank_block(
                index, {f: _family_query(f) for f in FAMILY_TECHNIQUES}
            ),
            "features_only": _rank_block(
                index, {f: _features_only_query(f, alerts) for f in FAMILY_TECHNIQUES}
            ),
        }

    alerts_block: dict[str, Any] = {"n_alerts": len(alerts)}
    if alerts:
        best: TechniqueIndex | None = indices.get("minilm") or indices.get("tfidf")
        retriever = Retriever(kb, best)
        ambiguous, candidate_hits = 0, 0
        correct_mapped = 0
        examples: list[dict[str, Any]] = []
        for a in alerts:
            r = retriever.retrieve(a)
            expected = LABEL_EXPECTED_TECHNIQUES.get(a.provenance.true_label or "", frozenset())
            correct_mapped += bool(expected & set(r.mapped_ids))
            if r.ambiguous:
                ambiguous += 1
                candidate_hits += bool(expected & set(r.candidate_ids))
                if len(examples) < 5:
                    examples.append(
                        {
                            "alert_id": a.alert_id,
                            "family": a.family,
                            "confidence": a.confidence,
                            "true_label": a.provenance.true_label,
                            "candidates": r.candidate_ids,
                        }
                    )
        alerts_block.update(
            {
                "embedder": best.embedder.name if best else None,
                "ambiguous": ambiguous,
                "ambiguous_with_a_correct_candidate": candidate_hits,
                "mapped_technique_correct_for_true_label": round(correct_mapped / len(alerts), 4),
                "ambiguous_examples": examples,
            }
        )

    result = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "kb": kb.meta,
        "mapping": mapping,
        "all_mapped_ids_resolve": True,
        "embedding": embedding,
        "alerts": alerts_block,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "retrieval_check.json").write_text(json.dumps(result, indent=2) + "\n")
    (reports_dir / "retrieval_check.md").write_text(render_markdown(result))
    return result


def render_markdown(r: dict[str, Any]) -> str:
    kb = r["kb"]
    lines = [
        "# ATT&CK retrieval check",
        "",
        f"Generated by `uv run ae retrieval-check`. Knowledge base: Enterprise ATT&CK "
        f"v{kb.get('attack_version')} ({kb.get('n_techniques')} active techniques, "
        f"SHA-256 `{str(kb.get('bundle_sha256', ''))[:12]}…`).",
        "",
        "## 1. Mapping table (deterministic path, used by every alert)",
        "",
        "| Family | Techniques | Tactic | Mitigations (first 3) |",
        "|---|---|---|---|",
    ]
    for family, block in r["mapping"].items():
        for i, t in enumerate(block["techniques"]):
            lines.append(
                f"| {family if i == 0 else ''} | {t['id']} {t['name']} | {', '.join(t['tactics'])} "
                f"| {'; '.join(t['mitigations']) or '—'} |"
            )
    lines += ["", "Why each mapping:", ""]
    lines += [f"- **{f}**: {b['reason']}" for f, b in r["mapping"].items()]

    lines += [
        "",
        "## 2. Embedding retrieval on its own (fallback sanity check)",
        "",
        "Rank of the family's best mapped technique among all techniques. *Family description* "
        "queries name the attack, so they are easy; *features only* queries give just the port "
        "and the unusual flow features of that family's alerts.",
        "",
        "| Embedder | Query | Recall@3 | Recall@5 | MRR | Queries |",
        "|---|---|---|---|---|---|",
    ]
    for kind, block in r["embedding"].items():
        if "error" in block:
            lines.append(f"| {kind} | unavailable: {block['error'][:80]} | | | | |")
            continue
        for qname in ("family_description", "features_only"):
            b = block[qname]
            lines.append(
                f"| {block['model']} | {qname.replace('_', ' ')} | {b['recall_at_3']} | "
                f"{b['recall_at_5']} | {b['mrr']} | {b['n_queries']} |"
            )
    for block in r["embedding"].values():
        if "error" in block:
            continue
        lines += ["", f"Per family ({block['model']}, family-description query): rank, top-3", ""]
        for family, f in block["family_description"]["per_family"].items():
            top = ", ".join(f"{t['id']} {t['name']}" for t in f["top5"][:3])
            lines.append(f"- {family}: rank {f['rank_of_mapped_technique']} — {top}")

    a = r["alerts"]
    lines += ["", "## 3. Built alerts", ""]
    if a["n_alerts"]:
        lines += [
            f"- Alerts checked: {a['n_alerts']} (embedder for candidates: {a['embedder']})",
            f"- Mapped techniques include a correct one for the true label: "
            f"**{a['mapped_technique_correct_for_true_label']:.0%}** (the ceiling for the "
            "correct-technique metric, since explanations may only cite these IDs)",
            f"- Ambiguous (Other-rare or confidence < 0.6), so candidates were retrieved: "
            f"{a['ambiguous']}; of these, with a correct candidate: "
            f"{a['ambiguous_with_a_correct_candidate']}",
        ]
    else:
        lines.append("No alerts found; run `uv run ae build-alerts` first.")
    return "\n".join(lines) + "\n"
