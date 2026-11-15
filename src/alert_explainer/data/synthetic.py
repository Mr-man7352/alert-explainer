"""Small synthetic CSVs in the exact TrafficLabelling layout, for tests and ``ae demo``.

They reproduce the quirks the ETL must handle: leading spaces in names, the duplicated
``Fwd Header Length`` column, Infinity/NaN in the rate columns, a cp1252 en dash in the Thursday
web-attack labels, trailing empty rows, exact and IP-only-different duplicates, and the
constant bulk/flag columns. The numbers are random and mean nothing about real traffic.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

HEADER = [
    "Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port", "Protocol",
    "Timestamp", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std",
    "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max",
    "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length",
    "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length",
    "Max Packet Length", "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
    "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Header Length", "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets",
    "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Std", "Active Max",
    "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "Label",
]  # fmt: skip
CONSTANT = {
    "Bwd PSH Flags", "Bwd URG Flags", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
}  # fmt: skip
NUMERIC = list(HEADER[7:-1])
DUP_INDEX = len(NUMERIC) - 1 - NUMERIC[::-1].index("Fwd Header Length")  # 2nd occurrence
FEATURE_SLOTS = [i for i, c in enumerate(NUMERIC) if c not in CONSTANT and i != DUP_INDEX]

# file name, day of month, attack share, {label: weight}
FILES: list[tuple[str, int, float, dict[str, float]]] = [
    ("Monday-WorkingHours.pcap_ISCX.csv", 3, 0.0, {}),
    ("Tuesday-WorkingHours.pcap_ISCX.csv", 4, 0.3, {"FTP-Patator": 1, "SSH-Patator": 1}),
    (
        "Wednesday-workingHours.pcap_ISCX.csv", 5, 0.35,
        {"DoS Hulk": 1, "DoS GoldenEye": 1, "DoS slowloris": 1, "DoS Slowhttptest": 1,
         "Heartbleed": 0.1},
    ),
    (
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", 6, 0.3,
        {"Web Attack – Brute Force": 1, "Web Attack – XSS": 1, "Web Attack – Sql Injection": 1},
    ),
    ("Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv", 6, 0.04, {"Infiltration": 1}),
    ("Friday-WorkingHours-Morning.pcap_ISCX.csv", 7, 0.3, {"Bot": 1}),
    ("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", 7, 0.35, {"PortScan": 1}),
    ("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", 7, 0.35, {"DDoS": 1}),
]  # fmt: skip

FAMILY_OF = {
    "BENIGN": 0, "DoS": 1, "DDoS": 2, "PortScan": 3, "BruteForce": 4, "WebAttack": 5, "Bot": 6,
    "Other-rare": 7,
}  # fmt: skip
PORTS = {
    "FTP-Patator": 21, "SSH-Patator": 22, "Bot": 8080, "Infiltration": 444, "Heartbleed": 444,
}  # fmt: skip
INTERNAL = [f"192.168.10.{h}" for h in (3, 5, 8, 9, 12, 14, 15, 16, 17, 19, 25, 50, 51)]
EXTERNAL = ["8.8.8.8", "23.15.4.24", "52.84.20.10", "104.16.24.35", "172.217.6.226"]


def _family(label: str) -> str:
    from alert_explainer.data.labels import RAW_TO_FAMILY, normalize_label

    return RAW_TO_FAMILY[normalize_label(label)]


def make_synthetic(out_dir: Path, rows_per_file: int = 2000, seed: int = 7) -> list[Path]:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_mu = rng.uniform(0, 8, len(NUMERIC))
    signatures = {f: rng.choice(FEATURE_SLOTS, 8, replace=False) for f in FAMILY_OF.values()}
    signatures[2][:4] = signatures[1][:4]  # DDoS looks partly like DoS: some honest confusion
    paths = []
    for name, dom, attack_share, labels in FILES:
        n_attack = int(rows_per_file * attack_share)
        weights = np.array(list(labels.values()), dtype=float)
        attack_labels = (
            rng.choice(list(labels), n_attack, p=weights / weights.sum()).tolist() if labels else []
        )
        row_labels = ["BENIGN"] * (rows_per_file - n_attack) + attack_labels
        rng.shuffle(row_labels)

        rows = []
        for label in row_labels:
            fam = FAMILY_OF[_family(label)]
            mu = base_mu.copy()
            mu[signatures[fam]] += rng.normal(1.6, 0.6) if fam else 0.0
            values = np.round(rng.lognormal(mu, 1.0), 3)
            for c in CONSTANT:
                values[NUMERIC.index(c)] = 0
            values[DUP_INDEX] = values[NUMERIC.index("Fwd Header Length")]  # duplicated column
            nums = [f"{v:g}" for v in values]
            if rng.random() < 0.006:
                bad = "Infinity" if rng.random() < 0.6 else "NaN"
                nums[NUMERIC.index("Flow Bytes/s")] = bad
                nums[NUMERIC.index("Flow Packets/s")] = bad

            if fam == 0:
                src, dst = rng.choice(INTERNAL), rng.choice(EXTERNAL)
                dport = int(
                    rng.choice([53, 80, 443, 123, 137, 8080, int(rng.integers(1024, 65535))])
                )
            else:
                src, dst = "172.16.0.1", "192.168.10.50"
                dport = PORTS.get(label, int(rng.integers(1, 1024)) if fam == 3 else 80)
            proto = 17 if dport in (53, 123, 137) else 6
            sport = int(rng.integers(1024, 65535))
            hour, minute = int(rng.integers(8, 18)), int(rng.integers(0, 60))
            ts = f"{dom}/7/2017 {hour - 12 if hour > 12 else hour}:{minute:02d}"
            flow_id = f"{dst}-{src}-{dport}-{sport}-{proto}"
            rows.append([flow_id, src, sport, dst, dport, proto, ts, *nums, label])

        # ~3% duplicates: half exact copies, half same flow statistics with different IDs/ports
        for i in rng.choice(len(rows), max(1, len(rows) // 33), replace=False):
            dup = list(rows[i])
            if rng.random() < 0.5:
                dup[2] = int(rng.integers(1024, 65535))
                dup[0] = f"{dup[3]}-{dup[1]}-{dup[4]}-{dup[2]}-{dup[5]}"
            rows.append(dup)

        path = out_dir / name
        encoding = "cp1252" if "WebAttacks" in name else "utf-8"
        with path.open("w", newline="", encoding=encoding) as fh:
            writer = csv.writer(fh)
            writer.writerow([HEADER[0]] + [f" {c}" for c in HEADER[1:]])
            writer.writerows(rows)
            if "WebAttacks" in name:
                writer.writerows([[""] * len(HEADER)] * 50)  # trailing empty rows
        paths.append(path)
    return paths
