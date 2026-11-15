# Alert Explainer

**LLM explanations for network intrusion detection, built secure by design.**

An ML detector flags attacks in network-flow data (CIC-IDS2017). Each alert is then explained in plain
English at two levels: a short **manager view** (what happened, how serious, what to do) and a detailed
**analyst view** (evidence, MITRE ATT&CK technique, investigation steps). Identities are pseudonymised
before the LLM sees them, the API is authenticated and role-based, every action is audit-logged, and a
prompt-injection suite runs in CI.

> **Status (15 Nov 2026):** Days 1–3 done: repo, data pipeline, LightGBM baseline.
> Detection, explanation, privacy and API layers follow (see `docs/devlog.md`).

## Design principles

1. **The ML model detects; the LLM only explains.** Family, confidence and severity come from
   deterministic code; the LLM cannot change them.
2. **The LLM is an untrusted component.** Its input contains attacker-controlled text, so its output is
   validated before anyone sees it, and it has no tools or write access.
3. **Data minimisation first.** The LLM receives only the fields it needs, with identities pseudonymised.
4. **Local by default.** An open-weights model via Ollama, so security logs never leave the machine.
5. **Every claim is measured.** Detection, faithfulness, readability and injection robustness all have numbers.

## Quick start (macOS)

```bash
# one-off tools
brew install uv libomp          # libomp is needed by LightGBM on macOS (see Troubleshooting)
# Docker Desktop: https://www.docker.com/products/docker-desktop/

make setup                      # uv sync + pre-commit hooks
make test                       # should pass with no data downloaded
make demo                       # whole pipeline on small synthetic data (data/demo/)

cp .env.example .env            # then edit the placeholders
make db-up                      # PostgreSQL 16 on localhost

# real data: download GeneratedLabelledFlows.zip (TrafficLabelling CSVs) from
# https://www.unb.ca/cic/datasets/ids-2017.html, then
scripts/get_data.sh ~/Downloads/GeneratedLabelledFlows.zip
make pipeline                   # ETL → splits → LightGBM
```

| Command | Output |
|---|---|
| `uv run ae etl` | `data/processed/flows.parquet`, `schema.json`, `reports/etl_stats.json`, `docs/data_card.md` |
| `uv run ae split` | `data/processed/splits.parquet`, `reports/splits_summary.json` |
| `uv run ae train-lgbm` | `models/lgbm/`, `reports/lgbm_metrics.json`, `reports/figures/lgbm_*.png` |
| `uv run ae train-lgbm --grid quick` | One config instead of eight (fast check) |

## Troubleshooting

**`Library not loaded: @rpath/libomp.dylib` when importing LightGBM (macOS).** LightGBM needs an
OpenMP runtime built for the same CPU as your Python. A common cause on Apple Silicon Macs is an old
Intel Homebrew in `/usr/local` (left over from Migration Assistant): its `libomp` is x86_64, and newer
Intel Homebrew can no longer install it. Quick fix, no Homebrew needed:

```bash
scripts/fix_libomp_macos.sh     # copies scikit-learn's bundled libomp next to the project's Python
```

Long-term fix: install native Homebrew (it lives in `/opt/homebrew`) and run `brew install libomp`.

## Results

Test set: 378,148 flows, never used for training or tuning.

| Metric | LightGBM | PyTorch MLP |
|---|---|---|
| Macro-F1 (7 families) | **0.972** | _Day 4–5_ |
| BENIGN false-positive rate at the chosen threshold | **0.10%** | |
| Attacks detected (any family) | **99.96%** | |
| Median detection latency per alert | 0.49 ms | |
| Rows after cleaning | 2,521,028 (from 2,830,743) | |

Per-family results, the threshold curve and held-out Heartbleed/Infiltration detection are in
`reports/lgbm_metrics.json`; figures are in `reports/figures/`. Random splits flatter IDS models (see
`docs/data_card.md`); the cross-day test comes on Day 5.

## Layout

```
src/alert_explainer/
  data/     labels.py  etl.py  data_card.py  splits.py  synthetic.py
  detect/   lgbm.py  evaluate.py            (torch_mlp, severity, attribution: Days 4–6)
  explain/  privacy/  api/  ui/             (Days 7–13)
tests/unit/                                  ETL, labels, splits, LightGBM
docs/       devlog.md  data_card.md
```

## Data

CIC-IDS2017 (Sharafaldin, Lashkari & Ghorbani, 2018), Canadian Institute for Cybersecurity, UNB. Public
research data; any hostnames and usernames in this project are **synthetic**. Known labelling problems in
the original release are documented in `docs/data_card.md`.
