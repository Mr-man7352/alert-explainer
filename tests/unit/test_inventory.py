from pathlib import Path

import pytest

from alert_explainer.data.inventory import (
    EMAIL_DOMAIN,
    KNOWN_HOSTS,
    Inventory,
    build_inventory,
    inventory_from_flows,
    is_internal,
)

IPS = [
    "192.168.10.50", "192.168.10.3", "192.168.10.5", "192.168.10.8", "192.168.10.9",
    "192.168.10.255", "192.168.10.0", "172.16.0.1", "8.8.8.8", "205.174.165.73", "not-an-ip",
]  # fmt: skip


def test_internal_addresses_only() -> None:
    inv = build_inventory(IPS)
    assert "8.8.8.8" not in inv and "205.174.165.73" not in inv and "not-an-ip" not in inv
    assert "192.168.10.255" not in inv and "192.168.10.0" not in inv  # broadcast / network
    assert {"192.168.10.5", "192.168.10.8", "192.168.10.9"} <= set(inv.assets)
    assert set(KNOWN_HOSTS) <= set(inv.assets)  # known hosts are always present


@pytest.mark.parametrize(
    ("ip", "internal"),
    [("192.168.10.14", True), ("172.16.0.1", True), ("192.168.11.4", False), ("10.0.0.1", False)],
)
def test_is_internal(ip: str, internal: bool) -> None:
    assert is_internal(ip) is internal


def test_known_roles_and_criticality() -> None:
    inv = build_inventory(IPS)
    web, dc = inv.get("192.168.10.50"), inv.get("192.168.10.3")
    assert web is not None and web.hostname == "web-01" and web.asset_criticality == 3
    assert dc is not None and dc.role.startswith("domain controller")
    ws = inv.get("192.168.10.5")
    assert ws is not None and ws.role == "user workstation" and ws.asset_criticality in (1, 2)
    assert "-ws-" in ws.hostname
    assert inv.get(None) is None and inv.get("8.8.8.8") is None


def test_deterministic_unique_and_clearly_synthetic() -> None:
    a, b = build_inventory(IPS, seed=1), build_inventory(list(reversed(IPS)), seed=1)
    assert a.assets == b.assets
    assert build_inventory(IPS, seed=2).assets != a.assets
    users = [x.owner_username for x in a.assets.values()]
    hosts = [x.hostname for x in a.assets.values()]
    assert len(set(users)) == len(users) and len(set(hosts)) == len(hosts)
    for asset in a.assets.values():
        assert asset.synthetic is True
        assert asset.owner_email == f"{asset.owner_username}@{EMAIL_DOMAIN}"
        first, _, last = asset.owner_username.partition(".")
        assert len(first) == 1 and last.isalpha() and last.islower()  # no titles like "mrs"


def test_many_workstations_get_unique_names() -> None:
    ips = [f"192.168.10.{i}" for i in range(2, 254)]
    inv = build_inventory(ips)
    assert len(inv) == len(ips) + 2  # + the gateway (.1) and the firewall
    assert len({a.owner_username for a in inv.assets.values()}) == len(inv)


def test_save_load_round_trip(tmp_path: Path) -> None:
    inv = build_inventory(IPS)
    inv.save(tmp_path / "inventory.csv")
    assert Inventory.load(tmp_path / "inventory.csv").assets == inv.assets


def test_inventory_from_synthetic_flows(pipeline_dir: Path) -> None:
    inv = inventory_from_flows(pipeline_dir / "processed" / "flows.parquet")
    assert "192.168.10.50" in inv and "172.16.0.1" in inv
    assert all(is_internal(ip) for ip in inv.assets)
