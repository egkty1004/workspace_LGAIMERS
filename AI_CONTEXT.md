# AI Technical Context

## Problem and metric

The task is binary probability prediction for `control_success`: for each pitch row, predict the probability that pitch control succeeds. `row_id` is the row identity used to preserve input/output correspondence; `control_success` is the training target and submission probability column.

The official evaluation is based on probability quality:

- Brier Score is the mean squared error between predicted probabilities and binary labels. Lower is better.
- Brier Skill Score (BSS) compares the prediction Brier Score with the constant-rate reference `r * (1 - r)`, where `r` is the evaluation label mean. The competition reports a non-negative scaled skill score; higher is better.

The repository definition is:

```text
Brier = mean((p - y)^2)
BSS = max(0, 100000 * (1 - Brier / (r * (1 - r))))
```

Here, `r` is the evaluation-label mean. `repro_979/common.py` is the implementation authority for this score.

Always distinguish raw model predictions/scores from deployed-form predictions/scores. A deployed form may apply a logit offset and clipping before Brier/BSS calculation or submission. A raw local BSS and a deployed-form or Public score are not directly comparable unless the exact formula, offset, clipping, fold, and label scope match.

Active deployed-form constants, fold definitions, and gates are authoritative in code/policy, especially:

- `repro_979/recovery_policy.py`
- `repro_979/common.py`

Numeric values repeated in documentation are snapshots only. If documentation, structured state, evidence, and active code/policy conflict, report the conflict and stop any action whose correctness depends on it until the conflict is resolved. Active policy describes current executable behavior, but conflicting evidence must not be silently overwritten or reinterpreted.

## Architecture and history

The repository contains several generations of work:

1. `baseline/`: official RandomForest training and inference notebooks.
2. `experiments/`: early HGB/LGBM, feature-engineering, EDA, drift, and diagnostic work. These experiments established that improvements on a single 2024 holdout did not reliably transfer to 2025 Public results.
3. GIHO 979 reproduction: independent reproduction of an Entity-Embedding MLP plus LightGBM approach, including multi-seed blending and submission-alignment work.
4. Legacy/top-100 pipeline under `repro_979/`: feature/model qualification, calibration, primary-fold blend selection, R-only transfer gates, deployment, package validation, and submission-decision tooling.
5. Current recovery pipeline under `repro_979/`: a bounded candidate registry, rolling-origin selection, terminal-label firewall, frozen promotion decision, deployment replay, and package validation.

The recovery pipeline is current. Legacy/top-100 runners remain useful historical evidence and utilities, but must not silently replace the recovery selection policy.

## Validation concepts

The current recovery policy defines chronological origins in code. Conceptually:

- `r2022`: train through 2021 regular-season rows; validate on 2022 regular-season rows.
- `r2023`: train through 2022 regular-season rows; validate on 2023 regular-season rows.
- `primary`: train through 2023 rows; evaluate on all 2024 rows.
- `r2024`: train through 2023 regular-season rows; evaluate on 2024 regular-season rows.

Under the active recovery protocol:

- `r2022` and `r2023` are selection origins.
- `primary` is a terminal/stress origin used only after candidate selection is frozen.
- `r2024` overlaps `primary` because it is the regular-season subset of 2024; it is diagnostic-only and must not be a selection or sorting key.
- `primary` has been reused historically. Do not describe it as a pristine, independent holdout.
- The terminal-label firewall structurally prevents primary-label access before the freeze state allows it. Candidate screening must not bypass this boundary.

Exact masks, thresholds, bootstrap settings, and permitted sort keys must be read from `repro_979/recovery_policy.py`, not reconstructed from prose.

## Leakage prohibitions

The following are blockers:

- Direct or indirect target leakage.
- Fitting preprocessing, imputation, calibration, feature selection, early stopping, or hyperparameters on data outside the permitted training/inner-selection window.
- Target encoding that exposes validation/outer/terminal labels.
- Future-row or future-history leakage in as-of, rolling, game, pitcher, or batter features.
- Reading test distribution statistics, reconstructing test sequences, or using one test row to adjust another.
- Using terminal labels, `r2024`, Public scores, or leaderboard feedback to select recovery candidates when the active policy forbids it.
- Changing split geometry or scoring/deployment form without an explicit, recorded protocol change.

Test rows must remain independent, and final inference must preserve exact `row_id` values and order.

## Execution and environment

Final submission testing should reproduce the official environment constraints documented in `docs/dacon_aimers9_pitching_control_hackathon.md`, including offline operation, 245,789-row inference, resource ceilings, required output shape/order, and time limits.

The school GPU is development compute and is not equivalent to the competition evaluation hardware. Never infer target runtime or compatibility from school-GPU speed alone. Check GPU availability before use, avoid assuming device 0 is free, and use smoke then limited validation before any approved full run.

Full CV, multi-seed training, 10,000-resample bootstrap, deployment training, and full package validation are expensive. They require explicit user approval. Scripts such as full deployment or adopted multi-seed workflows are not safe default diagnostics.

For reproducibility, record where applicable:

- Git commit and code/config hashes.
- Candidate/config identity and parent baseline.
- Data scope, fold masks, and labels read.
- Random seeds and model/library versions.
- Raw versus deployed scoring formula, offset, and clipping.
- Model, cache, evidence, and package hashes.
- Runtime and CPU/GPU/memory environment.

Git-tracked state and evidence are the technical record. Notion may mirror experiment and leaderboard information for people when available, but it is not required to interpret or execute the pipeline.
