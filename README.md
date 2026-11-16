# Alert Explainer

**LLM explanations for network intrusion detection, built secure by design.**

An ML detector flags attacks in network-flow data (CIC-IDS2017). Each alert is then explained in plain
English at two levels: a short **manager view** (what happened, how serious, what to do) and a detailed
**analyst view** (evidence, MITRE ATT&CK technique, investigation steps). Identities are pseudonymised
before the LLM sees them, the API is authenticated and role-based, every action is audit-logged, and a
prompt-injection suite runs in CI.

> **Status (16 Nov 2026):** Days 1–7 done: repo, data pipeline, LightGBM baseline, PyTorch MLP with
> tuning and a focal-loss ablation, cross-day robustness test, model comparison and model cards;
> synthetic asset inventory, deterministic severity, per-alert SHAP / Integrated Gradients
> attributions and the `Alert` builder; a pinned MITRE ATT&CK knowledge base with a verified
> family → technique mapping and an embedding fallback. The LLM, privacy and API layers follow
> (see `docs/devlog.md`).

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
make bench-mlp                  # which PyTorch device is faster here (CPU or Apple MPS)?
make pipeline                   # ETL → splits → LightGBM → MLP → cross-day → comparison
make alerts                     # Days 6–7: inventory, 100 alerts, ATT&CK KB, retrieval check
```

| Command | Output |
|---|---|
| `uv run ae etl` | `data/processed/flows.parquet`, `schema.json`, `reports/etl_stats.json`, `docs/data_card.md` |
| `uv run ae split` | `data/processed/splits.parquet`, `reports/splits_summary.json` |
| `uv run ae train-lgbm` | `models/lgbm/`, `reports/lgbm_metrics.json`, `reports/figures/lgbm_*.png` |
| `uv run ae train-lgbm --grid quick` | One config instead of eight (fast check) |
| `uv run ae bench-mlp` | Seconds per MLP epoch on each device; set `AE_DEVICE` in `.env` to the fastest |
| `uv run ae train-mlp --grid none --no-ablation` | One MLP run with the default config (Day 4 check) |
| `uv run ae train-mlp` | 7-config grid, then CE vs focal on all rows → `models/mlp/`, `reports/mlp_metrics.json`, `reports/figures/mlp_*.png` |
| `uv run ae cross-day` | Train on Mon–Wed, test on unseen Thu–Fri attacks → `reports/cross_day.json`, `reports/figures/cross_day_detection.png` |
| `uv run ae compare` | `reports/model_comparison.md`, `models/lgbm/model_card.md`, `models/mlp/model_card.md` |
| `uv run ae inventory` | `data/inventory.csv`: a **synthetic** hostname, owner, department and criticality per internal IP |
| `uv run ae build-alerts` | 100 alerts from the test set (`--model lgbm` or `mlp`) → `data/alerts/alerts_<model>.json`, `reports/alerts_summary_<model>.json` |
| `uv run ae severity-table` | `docs/severity_rules.md`: the deterministic severity rule and every combination |
| `uv run ae attack-kb` | Downloads Enterprise ATT&CK v20.1 (pinned, SHA-256 checked) → `data/attack/kb.json`; fails if a mapped technique is missing or revoked |
| `uv run ae retrieval-check` | `reports/retrieval_check.md`: mapping table, embedding retrieval (MiniLM and TF-IDF), and the built alerts |

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
| Macro-F1 (7 families) | **0.972** | _run `make train-mlp`_ |
| BENIGN false-positive rate at the chosen threshold | **0.10%** | |
| Attacks detected (any family) | **99.96%** | |
| Median detection latency per alert | 0.49 ms | |
| Rows after cleaning | 2,521,028 (from 2,830,743) | |

Per-family results, the threshold curve and held-out Heartbleed/Infiltration detection are in
`reports/lgbm_metrics.json`; figures are in `reports/figures/`. The full LightGBM vs MLP comparison,
the loss ablation and the cross-day test are in `reports/model_comparison.md`. Random splits flatter IDS
models: trained on Mon–Wed only, a detector meets every Thu–Fri attack type for the first time.

## Layout

```
src/alert_explainer/
  data/     labels.py  etl.py  data_card.py  splits.py  synthetic.py  inventory.py
  detect/   data.py  lgbm.py  torch_mlp.py  train_mlp.py  cross_day.py  compare.py  evaluate.py
            severity.py  attribution.py  detectors.py  alerts.py
  explain/  schemas.py  attack_kb.py  retrieve.py  retrieval_check.py
            (prompts, LLM, validator, fallback: Days 8–9)
  privacy/  api/  ui/                       (Days 10–13)
tests/unit/                                  ETL, labels, splits, LightGBM, MLP, cross-day, comparison,
                                             severity, inventory, attribution, alerts, ATT&CK KB, retrieval
docs/       devlog.md  data_card.md  severity_rules.md
```

## Data

CIC-IDS2017 (Sharafaldin, Lashkari & Ghorbani, 2018), Canadian Institute for Cybersecurity, UNB. Public
research data; any hostnames and usernames in this project are **synthetic**. Known labelling problems in
the original release are documented in `docs/data_card.md`.

MITRE ATT&CK® Enterprise v20.1 (STIX 2.1, from MITRE's `attack-stix-data` repository), © The MITRE
Corporation, used under MITRE's terms of use. It is downloaded by `ae attack-kb` and not stored in the repo.
