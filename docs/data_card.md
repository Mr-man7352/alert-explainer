# Data card — CIC-IDS2017 (TrafficLabelling), cleaned

_Generated automatically by `ae etl` on 2026-11-15T14:04:53+00:00. Numbers come from `reports/etl_stats.json`._

## Source

- Sharafaldin, Lashkari & Ghorbani (2018), Canadian Institute for Cybersecurity, UNB: <https://www.unb.ca/cic/datasets/ids-2017.html>
- Files: the **TrafficLabelling** CSVs inside `GeneratedLabelledFlows.zip`, which keep Flow ID, source/destination IP and port, and timestamp (needed for pseudonymisation and realistic alerts). Five working days, 3–7 July 2017, simulated enterprise network.
- Public research data. No real personal data; hostnames and usernames used later are synthetic.

## Cleaning steps and row counts

| Step | Rows |
|---|---:|
| Rows read from 8 CSVs | 3,119,345 |
| Empty rows dropped (bare commas) | −288,602 |
| Rows with Infinity/NaN in a feature dropped | −2,867 |
| Rows before de-duplication | 2,827,876 |
| Exact duplicates removed (features + label) | −306,848 |
| **Final rows** | **2,521,028** |

- Duplicate columns dropped: `Fwd Header Length.1`
- Constant (zero-variance) columns dropped (8): `Bwd PSH Flags`, `Bwd URG Flags`, `Fwd Avg Bytes/Bulk`, `Fwd Avg Packets/Bulk`, `Fwd Avg Bulk Rate`, `Bwd Avg Bytes/Bulk`, `Bwd Avg Packets/Bulk`, `Bwd Avg Bulk Rate`
- Non-finite values by column: `Flow Bytes/s` 2,867, `Flow Packets/s` 2,867
- Identical feature vectors carrying different labels (kept, reported as label noise): 696 vectors, 1,392 rows
- Features kept: **70** (all flow statistics plus `Destination Port` and `Protocol`). IPs, source port, Flow ID and timestamp are kept as metadata only, never as features.
- Parquet size 235 MB; loads in 0.62 s.

## Per-file summary

| File | Day | Encoding | Rows read | Empty | Non-finite | Rows out |
|---|---|---|---:|---:|---:|---:|
| `Monday-WorkingHours.pcap_ISCX.csv` | Monday | utf-8 | 529,918 | 0 | 437 | 529,481 |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | Tuesday | utf-8 | 445,909 | 0 | 264 | 445,645 |
| `Wednesday-workingHours.pcap_ISCX.csv` | Wednesday | utf-8 | 692,703 | 0 | 1,297 | 691,406 |
| `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | Thursday | utf-8 | 288,602 | 0 | 207 | 288,395 |
| `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | Thursday | latin-1 | 458,968 | 288,602 | 135 | 170,231 |
| `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | Friday | utf-8 | 225,745 | 0 | 34 | 225,711 |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | Friday | utf-8 | 286,467 | 0 | 371 | 286,096 |
| `Friday-WorkingHours-Morning.pcap_ISCX.csv` | Friday | utf-8 | 191,033 | 0 | 122 | 190,911 |

## Labels → families

| Raw label | Family | Before dedupe | Final | Removed |
|---|---|---:|---:|---:|
| BENIGN | BENIGN | 2,271,320 | 2,095,287 | 176,033 |
| DoS Hulk | DoS | 230,124 | 172,846 | 57,278 |
| DoS GoldenEye | DoS | 10,293 | 10,286 | 7 |
| DoS slowloris | DoS | 5,796 | 5,385 | 411 |
| DoS Slowhttptest | DoS | 5,499 | 5,228 | 271 |
| DDoS | DDoS | 128,025 | 128,014 | 11 |
| PortScan | PortScan | 158,804 | 90,694 | 68,110 |
| FTP-Patator | BruteForce | 7,935 | 5,931 | 2,004 |
| SSH-Patator | BruteForce | 5,897 | 3,219 | 2,678 |
| Web Attack - Brute Force | WebAttack | 1,507 | 1,470 | 37 |
| Web Attack - XSS | WebAttack | 652 | 652 | 0 |
| Web Attack - Sql Injection | WebAttack | 21 | 21 | 0 |
| Bot | Bot | 1,956 | 1,948 | 8 |
| Infiltration | Other-rare | 36 | 36 | 0 |
| Heartbleed | Other-rare | 11 | 11 | 0 |

## Families by day

| Family | Monday | Tuesday | Wednesday | Thursday | Friday | Total |
|---|---:|---:|---:|---:|---:|---:|
| BENIGN | 502,800 | 412,435 | 408,567 | 396,277 | 375,208 | 2,095,287 |
| DoS | 0 | 0 | 193,745 | 0 | 0 | 193,745 |
| DDoS | 0 | 0 | 0 | 0 | 128,014 | 128,014 |
| PortScan | 0 | 0 | 0 | 0 | 90,694 | 90,694 |
| BruteForce | 0 | 9,150 | 0 | 0 | 0 | 9,150 |
| WebAttack | 0 | 0 | 0 | 2,143 | 0 | 2,143 |
| Bot | 0 | 0 | 0 | 0 | 1,948 | 1,948 |
| Other-rare | 0 | 0 | 11 | 36 | 0 | 47 |

## Known issues and limitations

- **Labelling errors in the original release.** Engelen, Rimmer & Joosen (2021) and Lanvin et al. (2023) found mislabelled flows, duplicated packets and feature-calculation bugs; the KU Leuven study reconstructed or relabelled more than 20% of traces. This card describes the *original* release; a corrected-version rerun is a stretch goal.
- **Random splits flatter IDS models.** Neighbouring flows from the same attack session are near-identical, so a random split puts near-duplicates in train and test even after exact de-duplication.
- **Each attack family occurs on a single day.** Attack families present both Mon–Wed and Thu–Fri: **none**. So the blueprint's cross-day check (train Mon–Wed, test Thu–Fri on shared families) can only be run as a *binary* unseen-attack test: does a detector trained on Mon–Wed attacks flag Thu–Fri attacks at all?
- **Other-rare** (Infiltration, Heartbleed) has too few flows to train on; it is held out of training and its detection rate is reported separately.
- **Timestamps** are kept as raw text: the release writes them as day/month/year, and afternoon times appear to use a 12-hour clock without AM/PM (check before parsing). The `day` column comes from the file name.
- **Thursday web-attack file** is cp1252-encoded (read as latin-1) and its labels contain a garbled en dash, repaired to `Web Attack - …`.
