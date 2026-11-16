# data/ (git-ignored)

- `raw/` — CIC-IDS2017 **TrafficLabelling** CSVs (from `GeneratedLabelledFlows.zip`). Put the zip's
  contents here with `scripts/get_data.sh ~/Downloads/GeneratedLabelledFlows.zip`. Any folder layout
  works; the ETL reads every `*.csv` under `raw/`.
- `processed/` — `flows.parquet`, `schema.json`, `splits.parquet` (created by `ae etl` and `ae split`).
- `demo/` — synthetic data from `ae demo` (for testing without the real dataset).
- `inventory.csv` — **synthetic** asset inventory (Faker, fixed seed): hostname, owner, department and
  criticality per internal IP (`ae inventory`). No real people; e-mails use the reserved `corp.example`.
- `alerts/` — `alerts_<model>.json` from `ae build-alerts` (real dataset IPs + synthetic identities).
- `attack/` — the pinned MITRE ATT&CK STIX bundle, the compact `kb.json` and cached technique
  embeddings (`ae attack-kb`, `ae retrieval-check`).
