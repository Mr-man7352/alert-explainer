import pytest

from alert_explainer.data.labels import (
    FAMILIES,
    RAW_TO_FAMILY,
    TRAIN_FAMILIES,
    normalize_label,
    to_family,
)


@pytest.mark.parametrize(
    "raw",
    [
        "Web Attack \x96 Brute Force",  # cp1252 byte read as latin-1
        "Web Attack � Brute Force",  # replacement character
        "Web Attack â€“ Brute Force",  # mojibake
        "Web Attack – Brute Force",  # proper en dash
        " Web Attack - brute force ",
    ],
)
def test_web_attack_dash_repaired(raw: str) -> None:
    assert normalize_label(raw) == "Web Attack - Brute Force"


def test_web_attack_variants() -> None:
    assert normalize_label("Web Attack \x96 XSS") == "Web Attack - XSS"
    assert normalize_label("Web Attack \x96 Sql Injection") == "Web Attack - Sql Injection"


def test_other_labels_only_stripped() -> None:
    assert normalize_label("  DoS Hulk ") == "DoS Hulk"


def test_fifteen_labels_eight_families() -> None:
    assert len(RAW_TO_FAMILY) == 15
    assert set(RAW_TO_FAMILY.values()) == set(FAMILIES)
    assert TRAIN_FAMILIES[0] == "BENIGN" and "Other-rare" not in TRAIN_FAMILIES


def test_unknown_label_raises() -> None:
    with pytest.raises(KeyError, match="RAW_TO_FAMILY"):
        to_family("Totally New Attack")
