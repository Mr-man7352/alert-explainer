"""CIC-IDS2017 raw labels → 8 detection families (blueprint §5)."""

from __future__ import annotations

import re

FAMILIES: tuple[str, ...] = (
    "BENIGN",
    "DoS",
    "DDoS",
    "PortScan",
    "BruteForce",
    "WebAttack",
    "Bot",
    "Other-rare",
)
# Families the detector is trained on. Other-rare (Infiltration, Heartbleed: ~50 flows) is too
# small to learn and is reported separately as a known limitation.
TRAIN_FAMILIES: tuple[str, ...] = FAMILIES[:-1]
HELD_OUT_FAMILIES: tuple[str, ...] = ("Other-rare",)

RAW_TO_FAMILY: dict[str, str] = {
    "BENIGN": "BENIGN",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DDoS": "DDoS",
    "PortScan": "PortScan",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    "Web Attack - Brute Force": "WebAttack",
    "Web Attack - XSS": "WebAttack",
    "Web Attack - Sql Injection": "WebAttack",
    "Bot": "Bot",
    "Infiltration": "Other-rare",
    "Heartbleed": "Other-rare",
}

# "Web Attack – Brute Force" is stored with a cp1252 en dash (byte 0x96), which shows up as
# "\x96", "�" or "â€“" depending on how the file is decoded. Accept any non-letter run.
_WEB_ATTACK = re.compile(r"^web attack[^a-z]*(?P<rest>.+)$", re.IGNORECASE)
_WEB_ATTACK_CANONICAL = {
    "brute force": "Brute Force",
    "xss": "XSS",
    "sql injection": "Sql Injection",
}


def normalize_label(raw: str) -> str:
    """Strip whitespace and repair the garbled dash in web-attack labels."""
    label = raw.strip()
    match = _WEB_ATTACK.match(label)
    if match:
        rest = match["rest"].strip()
        return f"Web Attack - {_WEB_ATTACK_CANONICAL.get(rest.lower(), rest)}"
    return label


def to_family(label: str) -> str:
    try:
        return RAW_TO_FAMILY[label]
    except KeyError as exc:
        raise KeyError(f"Unknown CIC-IDS2017 label {label!r}; add it to RAW_TO_FAMILY") from exc
