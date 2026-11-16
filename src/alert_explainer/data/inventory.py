"""Synthetic asset inventory (blueprint §5, "Synthetic asset inventory"; Day 6).

CIC-IDS2017 has IP addresses but no hostnames or usernames. This module gives each internal
address a **synthetic** hostname, owner, department and asset criticality (1–3), so that:

- the alert builder can compute a deterministic severity from the affected asset's criticality, and
- the pseudonymiser (Day 10) has realistic identifiers to protect.

Every person, username and e-mail here is invented by Faker with a fixed seed. E-mails use the
reserved ``corp.example`` domain (RFC 2606), so none can belong to a real person. The file is
git-ignored (``data/inventory.csv``) and carries a ``synthetic`` column set to ``true``.

Server roles follow the victim network described on the CIC-IDS2017 dataset page: 192.168.10.3
is the domain controller and DNS server, 192.168.10.50 the public web server and 192.168.10.51 a
second Ubuntu server. Every other 192.168.10.x address seen in the flows is treated as a user
workstation. 172.16.0.1 is the firewall's inside address, behind which external attack traffic is
NATed in the capture.
"""

from __future__ import annotations

import csv
import ipaddress
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from faker import Faker

INTERNAL_NETWORK = ipaddress.ip_network("192.168.10.0/24")
EMAIL_DOMAIN = "corp.example"

# ip → (hostname, role, operating system, department, criticality)
KNOWN_HOSTS: dict[str, tuple[str, str, str, str, int]] = {
    "192.168.10.3": ("dc-01", "domain controller + DNS", "Windows Server 2016", "IT", 3),
    "192.168.10.50": ("web-01", "public web server", "Ubuntu 16.04", "IT", 3),
    "192.168.10.51": ("app-01", "application server", "Ubuntu 12.04", "IT", 2),
    "192.168.10.1": ("gw-01", "gateway router", "network appliance", "IT", 3),
    "172.16.0.1": (
        "edge-fw-01",
        "firewall (NAT address for external traffic)",
        "firewall",
        "IT",
        3,
    ),
}
# Department → (hostname prefix, criticality of its workstations). Finance, HR and Legal hold the
# most sensitive data, so their machines rank higher.
DEPARTMENTS: dict[str, tuple[str, int]] = {
    "Finance": ("fin", 2),
    "HR": ("hr", 2),
    "Legal": ("leg", 2),
    "Engineering": ("eng", 1),
    "Sales": ("sal", 1),
    "Operations": ("ops", 1),
    "Marketing": ("mkt", 1),
}
CRITICALITY_MEANING = {1: "standard", 2: "sensitive", 3: "business-critical"}


@dataclass(frozen=True)
class Asset:
    ip: str
    hostname: str
    owner_username: str
    owner_name: str
    owner_email: str
    department: str
    role: str
    os: str
    asset_criticality: int
    synthetic: bool = True


class Inventory:
    """Lookup by IP. Unknown addresses (external hosts) return ``None``."""

    def __init__(self, assets: Iterable[Asset]) -> None:
        self.assets: dict[str, Asset] = {a.ip: a for a in assets}

    def get(self, ip: str | None) -> Asset | None:
        return self.assets.get(ip) if ip else None

    def __len__(self) -> int:
        return len(self.assets)

    def __contains__(self, ip: object) -> bool:
        return ip in self.assets

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        names = [f.name for f in fields(Asset)]
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=names)
            writer.writeheader()
            for asset in sorted(self.assets.values(), key=lambda a: _ip_key(a.ip)):
                writer.writerow(asdict(asset))

    @classmethod
    def load(cls, path: Path) -> Inventory:
        with path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        return cls(
            Asset(
                ip=row["ip"],
                hostname=row["hostname"],
                owner_username=row["owner_username"],
                owner_name=row["owner_name"],
                owner_email=row["owner_email"],
                department=row["department"],
                role=row["role"],
                os=row["os"],
                asset_criticality=int(row["asset_criticality"]),
                synthetic=row["synthetic"].strip().lower() == "true",
            )
            for row in rows
        )


def _ip_key(ip: str) -> tuple[int, int]:
    addr = ipaddress.ip_address(ip)
    return (addr.version, int(addr))


def is_internal(ip: str) -> bool:
    """True for addresses that belong in the inventory (victim LAN + firewall inside address)."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return ip in KNOWN_HOSTS or (
        addr in INTERNAL_NETWORK
        and addr not in (INTERNAL_NETWORK.network_address, INTERNAL_NETWORK.broadcast_address)
    )


def _username(first: str, last: str, taken: set[str]) -> str:
    """``j.smith``-style login from a first and last name; digits are added on a clash."""
    first_clean = re.sub(r"[^a-z]", "", first.lower()) or "x"
    last_clean = re.sub(r"[^a-z]", "", last.lower()) or "user"
    base = f"{first_clean[0]}.{last_clean}"
    candidate, n = base, 1
    while candidate in taken:
        n += 1
        candidate = f"{base}{n}"
    taken.add(candidate)
    return candidate


def build_inventory(ips: Iterable[str], seed: int = 42) -> Inventory:
    """One synthetic asset per internal IP, deterministic for a given seed and IP set."""
    fake = Faker("en_GB")
    fake.seed_instance(seed)
    internal = sorted({ip for ip in ips if is_internal(ip)} | set(KNOWN_HOSTS), key=_ip_key)
    departments = list(DEPARTMENTS)
    taken: set[str] = set()
    counters: dict[str, int] = {}
    assets = []
    for i, ip in enumerate(internal):
        first, last = fake.first_name(), fake.last_name()  # no titles ("Mrs", "Dr") or suffixes
        name = f"{first} {last}"
        username = _username(first, last, taken)
        if ip in KNOWN_HOSTS:
            hostname, role, os_name, dept, crit = KNOWN_HOSTS[ip]
        else:
            dept = departments[i % len(departments)]
            prefix, crit = DEPARTMENTS[dept]
            counters[prefix] = counters.get(prefix, 0) + 1
            hostname = f"{prefix}-ws-{counters[prefix]:03d}"
            role, os_name = "user workstation", "unspecified"
        assets.append(
            Asset(
                ip=ip,
                hostname=hostname,
                owner_username=username,
                owner_name=name,
                owner_email=f"{username}@{EMAIL_DOMAIN}",
                department=dept,
                role=role,
                os=os_name,
                asset_criticality=crit,
            )
        )
    return Inventory(assets)


def inventory_from_flows(flows_path: Path, seed: int = 42) -> Inventory:
    """Build the inventory from every source/destination IP in ``flows.parquet``."""
    import polars as pl

    ips = (
        pl.scan_parquet(flows_path)
        .select(pl.concat_list("Source IP", "Destination IP").alias("ip"))
        .explode("ip")
        .unique()
        .collect()["ip"]
        .to_list()
    )
    return build_inventory((ip for ip in ips if ip), seed=seed)
