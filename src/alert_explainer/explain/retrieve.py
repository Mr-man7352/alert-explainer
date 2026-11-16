"""ATT&CK retrieval for one alert (blueprint §7.2; Day 7).

1. **Deterministic first.** The alert's ``allowed_techniques`` come from the family mapping table
   (``attack_kb.FAMILY_TECHNIQUES``). Their names, trimmed descriptions, mitigations and
   detection strategies are looked up in the KB.
2. **Embedding fallback.** If the alert is *ambiguous* (family Other-rare, or calibrated
   confidence < 0.6), the alert is described in words and the top-3 techniques by cosine
   similarity are added as **candidates**. Candidates are allowed in the explanation (so the
   validator accepts them) but are labelled as such, so an analyst can tell a mapped technique from
   a suggested one.

Grounding in a fixed list stops the LLM inventing technique IDs; the fallback still covers cases
the table does not.

Embedders:
- ``SentenceTransformerEmbedder``: ``sentence-transformers/all-MiniLM-L6-v2`` by default, run
  locally on CPU (downloaded once from Hugging Face, then cached).
- ``TfidfEmbedder``: scikit-learn TF-IDF, fitted on the technique texts. No download, so tests
  and CI use it, and it doubles as a lexical baseline for the retrieval check.

The technique matrix (~700 × 384 floats) is small enough for a NumPy dot product; no vector
database is needed. It is cached in ``data/attack/`` keyed on the KB hash and the model name.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from alert_explainer.explain.attack_kb import (
    FAMILY_TECHNIQUES,
    KnowledgeBase,
    Technique,
)
from alert_explainer.explain.schemas import Alert

AMBIGUOUS_CONFIDENCE = 0.6
FALLBACK_FAMILIES = frozenset({"Other-rare"})
N_CANDIDATES = 3
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

FAMILY_DESCRIPTION: dict[str, str] = {
    "DoS": "a denial-of-service flood of requests exhausting a single server's service",
    "DDoS": "a distributed denial-of-service flood saturating the target's network bandwidth",
    "PortScan": "a port scan probing many ports to discover listening network services",
    "BruteForce": "repeated login attempts guessing passwords against an authentication service",
    "WebAttack": "malicious requests exploiting a public-facing web application",
    "Bot": "a compromised host beaconing to a command-and-control server over web protocols",
    "Other-rare": "a rare attack: exploiting a vulnerable public-facing TLS service, or a user "
    "opening a malicious file followed by internal network scanning",
}
SERVICES: dict[int, str] = {
    21: "FTP",
    22: "SSH",
    23: "Telnet",
    25: "SMTP",
    53: "DNS",
    80: "HTTP",
    110: "POP3",
    139: "NetBIOS",
    143: "IMAP",
    443: "HTTPS",
    444: "SNPP",
    445: "SMB",
    3389: "RDP",
    8080: "HTTP alternate",
}


# --------------------------------------------------------------------------- embedders


class Embedder(Protocol):
    name: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


def _normalise(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms > 0, norms, 1.0)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list[str]) -> np.ndarray:
        return _normalise(self._model.encode(texts, batch_size=64, show_progress_bar=False))


class TfidfEmbedder:
    """Lexical baseline: TF-IDF (unigrams + bigrams, English stop words) fitted on the corpus."""

    def __init__(self, corpus: list[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.name = "tfidf"
        self._vec = TfidfVectorizer(
            ngram_range=(1, 2), stop_words="english", sublinear_tf=True, min_df=1
        ).fit(corpus)

    def encode(self, texts: list[str]) -> np.ndarray:
        return _normalise(self._vec.transform(texts).toarray())


def make_embedder(kind: str, kb: KnowledgeBase, model_name: str = DEFAULT_MODEL) -> Embedder:
    if kind == "tfidf":
        return TfidfEmbedder([t.embedding_text() for t in kb.techniques.values()])
    if kind == "minilm":
        return SentenceTransformerEmbedder(model_name)
    raise ValueError(f"Unknown embedder {kind!r}; use minilm or tfidf")


# --------------------------------------------------------------------------- index


class TechniqueIndex:
    def __init__(
        self, kb: KnowledgeBase, embedder: Embedder, cache_dir: Path | None = None
    ) -> None:
        self.kb = kb
        self.embedder = embedder
        self.ids = sorted(kb.techniques)
        texts = [kb[t].embedding_text() for t in self.ids]
        key = hashlib.sha256(
            json.dumps([embedder.name, self.ids, texts]).encode("utf-8")
        ).hexdigest()[:16]
        cached = (
            cache_dir / f"embeddings-{key}.npy" if cache_dir and embedder.name != "tfidf" else None
        )
        if cached is not None and cached.exists():
            self.matrix = np.load(cached, allow_pickle=False)
        else:
            self.matrix = embedder.encode(texts)
            if cached is not None:
                cached.parent.mkdir(parents=True, exist_ok=True)
                np.save(cached, self.matrix)

    def search(
        self, query: str, k: int = 5, exclude: set[str] | None = None
    ) -> list[tuple[str, float]]:
        q = self.embedder.encode([query])[0]
        sims = self.matrix @ q
        out: list[tuple[str, float]] = []
        for i in np.argsort(-sims):
            tid = self.ids[i]
            if exclude and tid in exclude:
                continue
            out.append((tid, float(sims[i])))
            if len(out) == k:
                break
        return out

    def rank_of(self, query: str, technique_ids: set[str]) -> int | None:
        """1-based rank of the best-ranked of ``technique_ids`` for this query."""
        sims = self.matrix @ self.embedder.encode([query])[0]
        for rank, i in enumerate(np.argsort(-sims), 1):
            if self.ids[i] in technique_ids:
                return rank
        return None


# --------------------------------------------------------------------------- retrieval


@dataclass(frozen=True)
class TechniqueRef:
    id: str
    name: str
    tactics: tuple[str, ...]
    description: str  # ~120 words, for the prompt's REFERENCE_DATA block
    mitigations: tuple[dict[str, str], ...]
    detection_strategies: tuple[str, ...]
    url: str
    source: Literal["mapping", "embedding"]
    similarity: float | None = None

    @classmethod
    def from_technique(
        cls,
        t: Technique,
        source: Literal["mapping", "embedding"],
        similarity: float | None = None,
        n_mitigations: int = 3,
    ) -> TechniqueRef:
        return cls(
            id=t.id,
            name=t.name,
            tactics=t.tactics,
            description=t.short_description,
            mitigations=tuple(
                {"id": m.id, "name": m.name, "guidance": m.guidance}
                for m in t.mitigations[:n_mitigations]
            ),
            detection_strategies=t.detection_strategies[:2],
            url=t.url,
            source=source,
            similarity=None if similarity is None else round(similarity, 4),
        )


@dataclass(frozen=True)
class Retrieval:
    techniques: tuple[TechniqueRef, ...]
    ambiguous: bool
    query: str | None = None

    @property
    def mapped_ids(self) -> list[str]:
        return [t.id for t in self.techniques if t.source == "mapping"]

    @property
    def candidate_ids(self) -> list[str]:
        return [t.id for t in self.techniques if t.source == "embedding"]

    @property
    def allowed_ids(self) -> list[str]:
        return [t.id for t in self.techniques]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ambiguous": self.ambiguous,
            "query": self.query,
            "techniques": [asdict(t) for t in self.techniques],
        }


def _describe_feature(name: str, percentile: float | None, share: float | None = None) -> str:
    if percentile is None:
        if share is None:
            return f"unusual {name}"
        common = "rare" if share < 0.01 else "uncommon" if share < 0.1 else "common"
        return f"{name} value {common} in normal traffic"
    if percentile >= 50:
        return f"{name} higher than {percentile:.0f}% of normal flows"
    return f"{name} lower than {100 - percentile:.0f}% of normal flows"


def alert_query(alert: Alert) -> str:
    """The alert in plain words, as the embedding query. Identifiers are never included."""
    service = SERVICES.get(alert.dst_port)
    port = f"port {alert.dst_port}/{alert.protocol}" + (f" ({service})" if service else "")
    features = "; ".join(
        _describe_feature(f.name, f.benign_percentile, f.benign_share)
        for f in alert.top_features[:3]
    )
    family = FAMILY_DESCRIPTION.get(alert.family, alert.family)
    return (
        f"Network traffic resembling {family}, sent to {port}. Unusual flow features: {features}."
    )


def is_ambiguous(alert: Alert) -> bool:
    return alert.family in FALLBACK_FAMILIES or alert.confidence < AMBIGUOUS_CONFIDENCE


class Retriever:
    def __init__(self, kb: KnowledgeBase, index: TechniqueIndex | None = None) -> None:
        self.kb = kb
        self.index = index

    def retrieve(self, alert: Alert, force_fallback: bool = False) -> Retrieval:
        mapped = [
            TechniqueRef.from_technique(self.kb[t], "mapping")
            for t in alert.allowed_techniques
            if t in self.kb
        ]
        missing = [t for t in alert.allowed_techniques if t not in self.kb]
        if missing:
            raise KeyError(f"alert {alert.alert_id}: techniques {missing} are not in the KB")
        ambiguous = is_ambiguous(alert)
        if not (ambiguous or force_fallback) or self.index is None:
            return Retrieval(tuple(mapped), ambiguous)
        query = alert_query(alert)
        hits = self.index.search(query, k=N_CANDIDATES, exclude=set(alert.allowed_techniques))
        candidates = [TechniqueRef.from_technique(self.kb[t], "embedding", s) for t, s in hits]
        return Retrieval(tuple(mapped + candidates), ambiguous, query)

    def with_candidates(self, alert: Alert, retrieval: Retrieval) -> Alert:
        """A copy of the alert whose ``candidate_techniques`` are the retrieval's candidates."""
        # Re-validate rather than model_copy(update=...), which would skip the field validators.
        return Alert.model_validate(
            {**alert.model_dump(), "candidate_techniques": retrieval.candidate_ids}
        )


def family_techniques(kb: KnowledgeBase, family: str) -> list[TechniqueRef]:
    return [TechniqueRef.from_technique(kb[t], "mapping") for t in FAMILY_TECHNIQUES[family]]
