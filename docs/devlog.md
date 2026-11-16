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

## Days 4–5 (built Sun 15 Nov 2026)

### Built
- **Day 4, PyTorch MLP** (`detect/torch_mlp.py`, `detect/train_mlp.py`): `FlowPreprocessor` (quantile
  transform to a normal distribution, fitted on train only). Train/val/test arrays are exported to
  `data/processed/mlp/*.npy` and read as memory maps by a custom `FlowDataset`. The `MLP` is
  70 → 256 → 128 → 64 → 7 with BatchNorm, ReLU and Dropout 0.2, about 60k parameters. The training
  loop is hand-written: AdamW, OneCycle LR, class-weighted cross-entropy, and early stopping
  (patience 4) on validation macro-F1, restoring the best epoch. The device is picked
  automatically (CUDA > MPS > CPU); `ae bench-mlp` times both CPU and MPS so the faster can be set
  in `AE_DEVICE`.
- **Day 5, tuning and evaluation:** a 7-config grid (learning rate, dropout, class weighting, width) on all attack rows
  plus 300k sampled BENIGN rows. Then two final runs on the full training set: the best config with
  cross-entropy, and the same config with **focal loss (γ = 2)** as a controlled ablation.
  `ae cross-day` runs the unseen-attack test for both LightGBM and the MLP. `ae compare` writes
  `reports/model_comparison.md` plus generated model cards for both models.
- `detect/data.py`: one loader for the new code that turns split columns into boolean masks inside
  Polars, with no 2.5M-row arrays of Python strings.
- 21 new tests (50 in total): Dataset/loader batching and reproducibility, focal loss = weighted CE at
  γ = 0, class weights equal LightGBM's row weights, the fast preprocessor matches sklearn, save/load
  round trip, and end-to-end `train-mlp`, `cross-day` and `compare` on the synthetic data.

### Decisions
- **Same treatment as LightGBM, so the comparison is fair.** The MLP uses the same split, the
  same isotonic calibration on the `cal` half of validation, the same threshold rule and decision
  rule, the same metric code and the same class-weighting schemes. The only differences are the
  model and its preprocessing.
- **No `log1p` before the quantile transform** (the blueprint had both). A quantile transform is
  rank-based, so any monotone transform before it changes nothing. It also copes with the
  `-1` values in `Init_Win_bytes_*`, which `log1p` would not.
- **A batch-level `__getitems__` instead of per-row `__getitem__`.** With 1.76M rows, 2,048 Python
  calls per batch would cost more than the forward pass. Indices are sorted inside each batch so
  memory-map reads are sequential, and `drop_last=True` stops BatchNorm meeting a batch of 1.
  `num_workers=0`, because macOS spawns worker processes slowly.
- **Our own `transform` for the quantile transformer.** sklearn's took **11 ms for a single row**
  (per-call overhead in `scipy.stats.norm.ppf`), against 0.07 ms for the network itself. That would
  have made the MLP's per-alert latency ~20× LightGBM's for no real reason. The replacement uses the
  same algorithm (two-way interpolation, the same float32 bound checks, `ndtri`, the same
  clipping), is unit-tested to match sklearn to 1e-6, and brings preprocessing under 1 ms. Fitting
  still uses sklearn.
- **Focal loss is selected on validation, never on test.** Both variants are reported on test.
- **OneCycle plus early stopping: fixed after the first real run.** The first full run used
  patience 4 with a 30-epoch OneCycle schedule. Every run stopped at epochs 6–11, while the learning
  rate was still at its peak (0.003; OneCycle warms up for the first 30% of epochs). Validation macro-F1
  jumped by ±0.1 between epochs (0.79 → 0.93 → 0.79), the model never annealed, and the grid ranked
  configs on unfinished runs. Result: test macro-F1 0.929 vs LightGBM's 0.972, all of the gap in Bot
  (F1 0.57 vs 0.81); the other six families were 0.96–1.00. Fix: run the full schedule (now 20
  epochs) and keep the best epoch; if patience is set, it only counts after the learning-rate peak.
  This costs about 1.5 extra minutes per full-data run. Lesson: early stopping assumes a decaying
  learning rate.
- **Cross-day test, binary.** No attack family is present on both sides of the day split, so the
  detectors are retrained as attack-vs-BENIGN on Mon–Wed (85/15 for early stopping and threshold).
  On Thu–Fri every attack type is unseen. The MLP's quantile transform is refitted on Mon–Wed rows
  only, so nothing from Thu–Fri leaks in. Reported: per-label share flagged, Thu–Fri BENIGN FPR (how
  far normal traffic drifts) and AUROC.
- **CPU-only PyTorch on Linux** (`[tool.uv.sources]` in `pyproject.toml`), so CI and Docker don't
  download ~3 GB of CUDA libraries. macOS uses the normal wheel, which includes MPS.
- **Day 6 interface ready:** `train_mlp.MLPDetector` loads weights, preprocessor, calibrator and
  threshold, and maps raw features to (family, calibrated probabilities). Weights load with
  `torch.load(weights_only=True)`, so no pickled code runs.

### First real-data cross-day result (Mon–Wed → Thu–Fri, threshold for BENIGN FPR ≤ 1% on Mon–Wed)
- LightGBM: Thu–Fri AUROC 0.859, 60.1% of unseen attacks flagged. The MLP (before the fix above):
  AUROC 0.847, 50.5%. BENIGN FPR rose to 1.48% / 1.13% on the new days: normal traffic drifts too.
- **Unseen attacks that resemble seen ones are caught; quiet ones are not.** DDoS (like the Wednesday
  DoS floods) 100% / 84%; the web attacks (brute force, XSS and SQL injection, like the Tuesday
  Patator brute force and the DoS traffic on port 80) 90–100%. PortScan was 3–4%, Bot 0% and
  Infiltration 0%: low-volume traffic that looks like normal connections. The detection rate on the
  random split (99.96%) says nothing about unseen attack types.

### Open
- Re-run `make train-mlp cross-day compare` with the schedule fix, and copy the numbers
  into the README results table and section 2 of the blueprint.
- Day 6: SHAP for LightGBM and Captum Integrated Gradients for the MLP, against the deployed model.
