# Devlog

One entry per build day: what was built, decisions and why, what is still open.

## Days 1–3 (built Sat 14 Nov 2026)

The blueprint planned Days 1–3 for 12–14 Nov; all three were built on the 14th.

### Built
- **Day 1, scaffold:** uv project (Python 3.12, `uv.lock`), Ruff (lint + format), mypy, pytest,
  pre-commit (Ruff, Gitleaks, large-file and private-key guards), Docker Compose with PostgreSQL 16
  (bound to localhost only), `.env.example` with placeholders only, minimal GitHub Actions CI, `ae` CLI.
  Dependencies are added day by day, not all up front, to keep the lock file and the attack surface small.
- **Day 2, ETL** (`data/etl.py`): reads each CSV as text, with a latin-1 fallback for the Thursday
  web-attack file. Strips header names and drops the duplicated `Fwd Header Length`. Drops the empty
  rows, repairs the garbled "Web Attack –" labels and maps the 15 labels to 8 families. Infinity/NaN
  rows are dropped, exact duplicates removed, constant columns dropped, and the result written to
  Parquet. `docs/data_card.md` is generated from the counts.
- **Day 3, splits + baseline:** stratified 70/15/15 split, LightGBM grid search with early stopping on
  macro-F1, isotonic calibration, threshold for BENIGN FPR ≤ 1%, test metrics, confusion matrix,
  PR curves, latency, saved model.
- 29 tests, run on synthetic CSVs that copy the real files' quirks (`data/synthetic.py`, `ae demo`).

### Decisions
- **Duplicates are found on features + label, ignoring IPs/ports/timestamps.** The leakage risk is
  identical flow statistics in train and test, whatever the Flow ID. Rows are hashed on the exact
  float64 values, then features are stored as float32. This halved peak ETL memory to roughly
  4 GB for the full dataset, by extrapolation.
- **Infinity/NaN rows are dropped, not imputed.** Median imputation before splitting leaks test
  statistics; there are only a few thousand such rows. The count goes in the data card.
- **`Destination Port` and `Protocol` are features; IPs, source port, Flow ID and timestamp are not.**
  This matches the public MachineLearningCVE feature set. The identifiers stay as metadata for alerts
  and pseudonymisation.
- **Validation is halved for calibration only.** The `cal` half fits the isotonic calibrator.
  Early stopping and the threshold use all of validation, because the threshold acts on raw scores.
- **Threshold on the raw attack score; show calibrated confidence.** Isotonic calibration is a step
  function, so thresholding its plateaus gives a coarse FPR. Decision: attack if
  `1 − P_raw(BENIGN) ≥ t`, family = most likely attack class.
- **Threshold rule, revised after the first real run.** The first rule ("lowest threshold with
  BENIGN FPR ≤ 1%") gave t = 0.0001. That spent the whole 1% budget on about 3,000 false alarms:
  Bot precision fell to 0.28 and macro-F1 to 0.885, worse than plain argmax (0.971). The 1% is a
  ceiling, not a quota. Above it, macro-F1 is flat (0.971–0.972 on test for t from 0.1 to 0.9): moving
  t only trades a few Bot flows for a few false alarms, and the best validation macro-F1 is noise
  from about 290 Bot flows. New rule: the lowest threshold within 0.003 of the best validation
  macro-F1, which catches the most attacks. The validation curve is saved in `lgbm_metrics.json`, and
  the old rule's result is kept there as `test_at_max_recall_threshold` for the report.
- **Grid search uses all attack rows plus 300k sampled BENIGN rows**, then the best config is refitted
  on the full training set. Eight configs over leaves, learning rate, min child samples and
  class-weighting (balanced / sqrt-balanced / none).
- **Other-rare (Infiltration + Heartbleed) is held out of training.** Its detection rate at the threshold
  is reported separately.

### Finding to discuss in the report
- **No attack family appears both Mon–Wed and Thu–Fri** (BruteForce Tue, DoS Wed, WebAttack Thu,
  Bot/PortScan/DDoS Fri). The blueprint's cross-day check on shared families is therefore impossible.
  Day 5 plan: run it as a **binary unseen-attack test**. Train on Mon–Wed, then measure how many
  Thu–Fri attacks are flagged at all. `splits.parquet` already has the `day_split` column.

### Open
- Run `make pipeline` on the real data and copy the numbers into the README and section 2 of the
  blueprint.
- Check the real files' timestamp format before Day 6 (alert builder).
