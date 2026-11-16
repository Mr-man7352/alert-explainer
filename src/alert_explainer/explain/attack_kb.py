"""MITRE ATT&CK knowledge base: download, parse, map families to techniques (blueprint §7.2; Day 7).

Source: MITRE's ``attack-stix-data`` GitHub repository, Enterprise domain, **pinned** to one
release (``ATTACK_VERSION``), with the file's SHA-256 recorded. ``ae attack-kb`` downloads it to
``data/attack/`` (git-ignored) and writes a compact ``kb.json``.

Parsing is plain JSON rather than ``mitreattack-python``: we need three object types and two
relationship types, and that library brings about twenty transitive dependencies (Excel, SVG,
TAXII clients) into a project whose supply chain is audited. The same filters apply: revoked and
deprecated objects are dropped, and so are relationships that are themselves revoked/deprecated.

Per technique we keep: ID, name, tactics, platforms, a cleaned description (citations and
markdown links removed) and a ~120-word trimmed version for prompts, its mitigations with the
technique-specific guidance text, and the names of ATT&CK v18+ detection strategies that cover it.

The **mapping table** (``FAMILY_TECHNIQUES``) is the deterministic first step of retrieval. Every
ID was checked against ATT&CK v20.1, and ``verify_mapping`` re-checks it whenever the KB is built,
so a technique that MITRE revokes or deprecates in a later release fails the build instead of
reaching an explanation.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ATTACK_VERSION = "20.1"
BUNDLE_URL = (
    "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/"
    "enterprise-attack/enterprise-attack-{version}.json"
)
# SHA-256 of enterprise-attack-20.1.json as downloaded on 16 Nov 2026. A mismatch means the
# pinned file changed upstream; the build stops rather than silently using different data.
KNOWN_SHA256 = {"20.1": "539a843a7b53ed9749740b2123dddae3726b1a4831841c32aca740b011b910b5"}
SHORT_DESCRIPTION_WORDS = 120
MITIGATION_WORDS = 45

# Family → ATT&CK technique IDs, most specific first. The first ID is the family's primary
# technique. Reasons are recorded so the table can be defended (and appear in the docs).
FAMILY_TECHNIQUES: dict[str, tuple[str, ...]] = {
    "DoS": ("T1499", "T1499.002"),
    "DDoS": ("T1498", "T1498.001"),
    "PortScan": ("T1046",),
    "BruteForce": ("T1110", "T1110.001"),
    "WebAttack": ("T1190", "T1110.001"),
    "Bot": ("T1071", "T1071.001"),
    "Other-rare": ("T1190", "T1204", "T1046"),
}
MAPPING_REASONS: dict[str, str] = {
    "DoS": "Hulk/GoldenEye flood a web server with HTTP requests and slowloris/Slowhttptest hold its "
    "connections open: denial of service against a service on one endpoint (Service Exhaustion Flood).",
    "DDoS": "LOIC traffic from several sources saturates the target's network capacity "
    "(Network Denial of Service, Direct Network Flood).",
    "PortScan": "nmap probes many ports to find listening services (Network Service Discovery).",
    "BruteForce": "FTP-Patator and SSH-Patator try many passwords against a login service "
    "(Brute Force: Password Guessing).",
    "WebAttack": "SQL injection and XSS exploit the public web application (Exploit "
    "Public-Facing Application); the web brute-force variant guesses its login (Password Guessing).",
    "Bot": "Ares bots poll their controller over HTTP (Application Layer Protocol: Web Protocols).",
    "Other-rare": "Heartbleed exploits a public-facing TLS service (T1190). Infiltration starts "
    "with a victim opening a malicious file (User Execution) and continues with an internal "
    "network scan (T1046). Never trained on: alerts are rare and retrieval falls back to embeddings.",
}
# Raw CIC-IDS2017 label → technique IDs an explanation may correctly cite (for the "correct
# technique" metric, blueprint §2). Stricter than the family table for WebAttack.
LABEL_EXPECTED_TECHNIQUES: dict[str, frozenset[str]] = {
    **dict.fromkeys(
        ("DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest"),
        frozenset({"T1499", "T1499.002"}),
    ),
    "DDoS": frozenset({"T1498", "T1498.001"}),
    "PortScan": frozenset({"T1046"}),
    "FTP-Patator": frozenset({"T1110", "T1110.001"}),
    "SSH-Patator": frozenset({"T1110", "T1110.001"}),
    "Web Attack - Brute Force": frozenset({"T1110", "T1110.001", "T1190"}),
    "Web Attack - XSS": frozenset({"T1190"}),
    "Web Attack - Sql Injection": frozenset({"T1190"}),
    "Bot": frozenset({"T1071", "T1071.001"}),
    "Infiltration": frozenset({"T1204", "T1046"}),
    "Heartbleed": frozenset({"T1190"}),
}


# --------------------------------------------------------------------------- data classes


@dataclass(frozen=True)
class Mitigation:
    id: str
    name: str
    guidance: str  # technique-specific text from the "mitigates" relationship, trimmed


@dataclass(frozen=True)
class Technique:
    id: str
    name: str
    tactics: tuple[str, ...]
    platforms: tuple[str, ...]
    description: str
    short_description: str
    url: str
    is_subtechnique: bool
    mitigations: tuple[Mitigation, ...] = ()
    detection_strategies: tuple[str, ...] = ()

    @property
    def parent_id(self) -> str | None:
        return self.id.split(".")[0] if self.is_subtechnique else None

    def embedding_text(self) -> str:
        tactics = ", ".join(t.replace("-", " ") for t in self.tactics)
        return f"{self.name}. Tactic: {tactics}. {self.short_description}"


@dataclass
class KnowledgeBase:
    techniques: dict[str, Technique]
    meta: dict[str, Any] = field(default_factory=dict)

    def __contains__(self, technique_id: object) -> bool:
        return technique_id in self.techniques

    def __getitem__(self, technique_id: str) -> Technique:
        return self.techniques[technique_id]

    def __len__(self) -> int:
        return len(self.techniques)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "meta": self.meta,
            "techniques": [asdict(t) for t in sorted(self.techniques.values(), key=_id_key)],
        }
        path.write_text(json.dumps(blob, indent=1, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: Path) -> KnowledgeBase:
        blob = json.loads(path.read_text())
        techniques = {}
        for t in blob["techniques"]:
            techniques[t["id"]] = Technique(
                **{
                    **t,
                    "tactics": tuple(t["tactics"]),
                    "platforms": tuple(t["platforms"]),
                    "mitigations": tuple(Mitigation(**m) for m in t["mitigations"]),
                    "detection_strategies": tuple(t["detection_strategies"]),
                }
            )
        return cls(techniques, blob.get("meta", {}))


def _id_key(t: Technique) -> tuple[int, int]:
    main, _, sub = t.id[1:].partition(".")
    return (int(main), int(sub or 0))


# --------------------------------------------------------------------------- text cleaning

_CITATION = re.compile(r"\s*\(Citation:[^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_SPACES = re.compile(r"[ \t]+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def clean_text(text: str) -> str:
    """Drop ATT&CK citations, markdown links (keep their text), HTML tags and extra spaces."""
    text = _CITATION.sub("", text or "")
    text = _MD_LINK.sub(r"\1", text)
    text = _HTML_TAG.sub("", text)
    paragraphs = [_SPACES.sub(" ", p).strip() for p in text.split("\n")]
    return "\n".join(p for p in paragraphs if p)


def trim_words(text: str, max_words: int) -> str:
    """Whole sentences up to ``max_words``; if the first sentence is longer, cut it with '…'."""
    flat = " ".join(text.split())
    if len(flat.split()) <= max_words:
        return flat
    kept: list[str] = []
    count = 0
    for sentence in _SENTENCE_END.split(flat):
        n = len(sentence.split())
        if count + n > max_words:
            break
        kept.append(sentence)
        count += n
    if kept:
        return " ".join(kept)
    return " ".join(flat.split()[:max_words]) + " …"


# --------------------------------------------------------------------------- parsing


def _attack_id(obj: dict[str, Any]) -> tuple[str | None, str | None]:
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("external_id"), ref.get("url")
    return None, None


def _active(obj: dict[str, Any]) -> bool:
    return not obj.get("revoked", False) and not obj.get("x_mitre_deprecated", False)


def parse_bundle(bundle: dict[str, Any]) -> KnowledgeBase:
    """STIX 2.1 bundle (Enterprise ATT&CK) → techniques with mitigations and detection strategies."""
    objects: dict[str, dict[str, Any]] = {o["id"]: o for o in bundle.get("objects", [])}
    stix_to_tid: dict[str, str] = {}
    base: dict[str, dict[str, Any]] = {}
    for obj in objects.values():
        if obj.get("type") != "attack-pattern" or not _active(obj):
            continue
        tid, url = _attack_id(obj)
        if not tid:
            continue
        stix_to_tid[obj["id"]] = tid
        description = clean_text(obj.get("description", ""))
        base[tid] = {
            "id": tid,
            "name": obj["name"],
            "tactics": tuple(
                p["phase_name"]
                for p in obj.get("kill_chain_phases", [])
                if p.get("kill_chain_name") == "mitre-attack"
            ),
            "platforms": tuple(obj.get("x_mitre_platforms", [])),
            "description": description,
            "short_description": trim_words(description, SHORT_DESCRIPTION_WORDS),
            "url": url or f"https://attack.mitre.org/techniques/{tid.replace('.', '/')}/",
            "is_subtechnique": bool(obj.get("x_mitre_is_subtechnique", "." in tid)),
        }

    mitigations: dict[str, list[Mitigation]] = {}
    detections: dict[str, list[str]] = {}
    for rel in objects.values():
        if rel.get("type") != "relationship" or not _active(rel):
            continue
        tid = stix_to_tid.get(rel.get("target_ref", ""))
        source = objects.get(rel.get("source_ref", ""))
        if tid is None or source is None or not _active(source):
            continue
        if rel["relationship_type"] == "mitigates" and source["type"] == "course-of-action":
            mid, _ = _attack_id(source)
            if mid:
                guidance = trim_words(clean_text(rel.get("description", "")), MITIGATION_WORDS)
                mitigations.setdefault(tid, []).append(Mitigation(mid, source["name"], guidance))
        elif (
            rel["relationship_type"] == "detects" and source["type"] == "x-mitre-detection-strategy"
        ):
            detections.setdefault(tid, []).append(source["name"])

    techniques = {
        tid: Technique(
            **fields_,
            mitigations=tuple(sorted(mitigations.get(tid, []), key=lambda m: m.id)),
            detection_strategies=tuple(sorted(set(detections.get(tid, [])))),
        )
        for tid, fields_ in base.items()
    }
    return KnowledgeBase(techniques)


def verify_mapping(kb: KnowledgeBase) -> None:
    """Every mapped or expected technique ID must exist (active) in the KB."""
    wanted = {t for ids in FAMILY_TECHNIQUES.values() for t in ids}
    wanted |= {t for ids in LABEL_EXPECTED_TECHNIQUES.values() for t in ids}
    missing = sorted(t for t in wanted if t not in kb)
    if missing:
        raise ValueError(
            f"Mapped ATT&CK techniques missing, revoked or deprecated in this release: {missing}. "
            "Check them on attack.mitre.org and update FAMILY_TECHNIQUES."
        )


# --------------------------------------------------------------------------- download + build


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_bundle(dest_dir: Path, version: str = ATTACK_VERSION, attempts: int = 3) -> Path:
    """Download the pinned Enterprise ATT&CK bundle once; later calls reuse the file.

    A connection can end early without raising, which leaves a truncated file (seen on the first
    real run). So the download goes to a ``.part`` file and is kept only if its size matches
    ``Content-Length`` and, for a pinned version, its SHA-256 matches ``KNOWN_SHA256``.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"enterprise-attack-{version}.json"
    if path.exists():
        return path
    url = BUNDLE_URL.format(version=version)
    if not url.startswith("https://"):  # pragma: no cover - the constant is https
        raise ValueError("refusing a non-HTTPS download")
    tmp = path.with_suffix(".part")
    expected = KNOWN_SHA256.get(version)
    problem = ""
    for attempt in range(1, attempts + 1):
        print(f"Downloading ATT&CK {version} (~55 MB) from {url} (attempt {attempt}/{attempts})")
        with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as fh:  # noqa: S310
            length = resp.headers.get("Content-Length")
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
        size = tmp.stat().st_size
        if length is not None and size != int(length):
            problem = f"truncated download: {size:,} of {int(length):,} bytes"
        elif expected and sha256_file(tmp) != expected:
            problem = "SHA-256 does not match the pinned value"
        else:
            tmp.replace(path)
            return path
        print(f"  {problem}; retrying")
    tmp.unlink(missing_ok=True)
    raise OSError(f"Could not download ATT&CK {version}: {problem}")


def build_kb(
    attack_dir: Path, version: str = ATTACK_VERSION, bundle: Path | None = None
) -> KnowledgeBase:
    """Download (or reuse) the bundle, check its hash, parse it, verify the mapping, save kb.json."""
    path = bundle or download_bundle(attack_dir, version)
    digest = sha256_file(path)
    expected = KNOWN_SHA256.get(version)
    if bundle is None and expected and digest != expected:
        raise ValueError(
            f"{path.name}: SHA-256 {digest} does not match the pinned {expected}. "
            "It may be a truncated download: delete it and run `ae attack-kb` again. If the "
            "upstream file really changed, review the change before updating KNOWN_SHA256."
        )
    kb = parse_bundle(json.loads(path.read_text(encoding="utf-8")))
    verify_mapping(kb)
    kb.meta = {
        "attack_version": version,
        "domain": "enterprise-attack",
        "source": BUNDLE_URL.format(version=version),
        "bundle_sha256": digest,
        "n_techniques": len(kb),
        "n_subtechniques": sum(t.is_subtechnique for t in kb.techniques.values()),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "license_note": "ATT&CK® content © The MITRE Corporation, reproduced under MITRE's terms "
        "of use (https://attack.mitre.org/resources/legal-and-branding/terms-of-use/).",
    }
    kb.save(attack_dir / "kb.json")
    return kb


def load_kb(attack_dir: Path) -> KnowledgeBase:
    path = attack_dir / "kb.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `uv run ae attack-kb` first")
    return KnowledgeBase.load(path)
