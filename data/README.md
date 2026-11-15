# data/ (git-ignored)

- `raw/` — CIC-IDS2017 **TrafficLabelling** CSVs (from `GeneratedLabelledFlows.zip`). Put the zip's
  contents here with `scripts/get_data.sh ~/Downloads/GeneratedLabelledFlows.zip`. Any folder layout
  works; the ETL reads every `*.csv` under `raw/`.
- `processed/` — `flows.parquet`, `schema.json`, `splits.parquet` (created by `ae etl` and `ae split`).
- `demo/` — synthetic data from `ae demo` (for testing without the real dataset).
