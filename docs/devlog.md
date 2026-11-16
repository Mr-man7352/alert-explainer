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

## Days 6–7 (built Mon 16 Nov 2026)

The blueprint planned Day 6 for 17 Nov and Day 7 for 18 Nov; both were built on the 16th.

### Built
- **Day 6, synthetic inventory** (`data/inventory.py`, `ae inventory`): one asset per internal IP in
  the flows (192.168.10.x plus the firewall's NAT address 172.16.0.1). The domain controller (.3),
  web server (.50) and second server (.51) follow the roles on the CIC-IDS2017 dataset page and get
  criticality 3/3/2. Every other address is a workstation in one of seven departments (Finance,
  HR and Legal machines are criticality 2, the rest 1). Names come from Faker with a fixed seed, and
  e-mails use the reserved `corp.example` domain. Each row is marked `synthetic=true`.
- **Day 6, deterministic severity** (`detect/severity.py`, `ae severity-table` →
  `docs/severity_rules.md`): `base[family] × confidence band × asset criticality`, each 1–3, so the
  score is 1–27. Scores 1–3 are Low, 4–9 Medium, 12–18 High and 27 Critical. All 63 combinations are
  unit-tested against a hand-written copy of the table.
- **Day 6, attributions** (`detect/attribution.py`, `detect/detectors.py`): TreeSHAP for LightGBM and
  Captum Integrated Gradients for the MLP, both towards the predicted family. The top 5 features carry
  their value and their percentile among BENIGN training flows (`models/benign_reference.npz`).
- **Day 6, alert builder** (`detect/alerts.py`, `explain/schemas.py`, `ae build-alerts`): a frozen
  Pydantic `Alert` per flagged flow. `ae build-alerts` scores the whole test split and samples 100
  flagged flows stratified by predicted family. It builds them one at a time (so latency is per
  alert) and writes `data/alerts/alerts_<model>.json` plus `reports/alerts_summary_<model>.json`.
- **Day 7, ATT&CK knowledge base** (`explain/attack_kb.py`, `ae attack-kb`): Enterprise ATT&CK
  v20.1 from MITRE's `attack-stix-data` repository, pinned by version and SHA-256. It parses to 706
  active techniques (479 sub-techniques), each with its mitigations (with the technique-specific
  guidance text) and the v18+ detection strategies that cover it.
- **Day 7, mapping table + retrieval** (`explain/retrieve.py`): family → technique IDs first, then an
  embedding fallback (MiniLM, or TF-IDF offline) that adds 3 *candidate* techniques for ambiguous
  alerts. `ae retrieval-check` writes `reports/retrieval_check.md`.
- 125 new tests (176 in total; one is skipped until the real KB is built), still on synthetic data
  only: no downloads in CI.

### Decisions
- **SHAP from LightGBM itself.** `booster.predict(pred_contrib=True)` is LightGBM's built-in
  TreeSHAP, the same algorithm as `shap.TreeExplainer`. Using it keeps `shap` and `numba` out of the
  runtime. `shap` is a dev dependency: a test checks the two give the same values (to 1e-6) and that
  they add up to the model's raw margin. On the demo data the additivity error is ~1e-14.
- **Integrated Gradients baseline = the median BENIGN training flow.** A zero baseline would mean
  "every feature at the training mean", which is not a real flow. With the benign median, an
  attribution reads as "how far this feature moved the decision away from typical normal traffic",
  which matches the benign percentile shown beside it. The target is the predicted family's
  logit, computed with 64 Gauss–Legendre steps. Completeness error is reported per run: the median
  relative error was 0.1% on the demo data.
- **Percentiles are mid-rank, and ports get a share instead.** About 40% of benign flows share some
  values (often 0). Mid-rank puts such a value at the 20th percentile, not 0 or 40. A percentile
  of a port number means nothing, so for `Destination Port` and `Protocol` the alert gives the share of
  benign flows with that exact value ("port 21: <0.1% of normal traffic").
- **Affected asset:** the destination if it is internal, otherwise the internal source (a bot
  calling out). Unknown addresses get criticality 2: an unknown asset counts as sensitive until
  someone says otherwise.
- **Severity consequences, by design:** only a confident (≥ 0.9), high-impact (base 3) attack on a
  business-critical asset is Critical, and a port scan never goes above Medium. A confident DDoS on
  a standard workstation is Medium. That is debatable, but the table is tunable, documented and tested.
- **`Alert` is frozen and `extra="forbid"`.** Nothing downstream, the LLM least of all, can change
  family, confidence or severity. `LLM_FIELDS` is an explicit whitelist of what the Day 8
  prompt may contain: no ground truth, no attack score, no source port. `provenance` (row id, flow id,
  true label) is for evaluation only.
- **Timestamps:** the CSVs use `d/m/yyyy h:mm` on a clock with no AM/PM, and capture ran during
  working hours, so an hour below 8 is read as afternoon (3:30 → 15:30). The summary counts how
  often this rule fired and how many timestamps failed to parse. *To check on the real files*
  (the Days 1–3 open item).
- **Plain JSON instead of `mitreattack-python`.** We need three object types and two relationship
  types (~60 lines), and that library pulls in about twenty transitive dependencies (Excel, SVG,
  TAXII). Same filters: revoked and deprecated objects are dropped, and so are revoked
  relationships. The bundle is pinned to v20.1 and its SHA-256 is checked, so the KB cannot change
  under us.
- **Mapping verified against v20.1.** Every ID in `FAMILY_TECHNIQUES` exists and is active.
  `verify_mapping` re-checks this on every `ae attack-kb`, so a later revocation fails the build.
  `LABEL_EXPECTED_TECHNIQUES` lists which IDs are *correct* for each raw label; it is stricter than
  the family table (XSS and SQL injection accept only T1190). Because the validator (Day 9) will
  restrict explanations to the alert's allowed IDs, the share of alerts whose mapped techniques
  include a correct one is the **ceiling** for the "correct ATT&CK technique" metric. The retrieval
  check reports it (87% on the demo data, which has deliberate confusions).
- **Ambiguous = Other-rare or calibrated confidence < 0.6.** Only then are embedding candidates
  added, labelled `source="embedding"`. The embedding query is built from the family, the
  port/service and the top features, and never contains IPs, hostnames or usernames.
- **Two retrieval checks, one easy and one hard.** The easy query names the attack. The hard one
  ("features only") gives just the port and the unusual flow statistics of that family's alerts, which
  shows how little the fallback can infer from numbers alone. That is why it is only a fallback.
- **Downloads are verified before they are kept.** The first real download on the Mac ended ~14 KB
  short without any error (a truncated JSON file); the SHA-256 pin caught it. `download_bundle`
  now writes to a `.part` file, checks its size against `Content-Length` and its hash against the
  pin, retries up to 3 times, and only then renames it into place.
- Fixed five misplaced `# fmt: skip` comments in `detect/compare.py` (ruff RUF028), which would
  have failed CI's lint step.

### Open
- Run `make alerts` on the real data (after `make pipeline`), then copy the numbers into the README
  and blueprint section 2: alert latency, MiniLM recall@3 and the correct-technique ceiling.
- `uv.lock` needs regenerating (`uv lock`) for the new dependencies: faker, captum,
  sentence-transformers, and dev-only shap.
- Check the afternoon-timestamp rule against the real files (`timestamps_afternoon_rule` in the
  alert summary should be roughly the share of afternoon alerts).
