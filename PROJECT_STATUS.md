# Project Status

Updated: 2026-08-21 (repository-evidence snapshot)

Evidence base: `master` at `ab13a7862b7f985adb0a9b5e87cc71d0203e675e`.

Update this document whenever the active ML policy, champion, leaderboard state, or recovery decision materially changes.

## Current baseline

The current technical champion is the reconciled **v93 6-leg** baseline.

- Public score: **1001.7444910721**
- Recorded package identity: `submit_v93_6leg_r0477`
- Recorded deployed-form snapshot: `C_LOGIT = -0.0461645795229729`, clip `[0.30, 0.70]`
- Composition recorded in repository evidence: LGB/MLP base plus FTT, ArmB, and CatBoost logit-delta legs.

`repro_979/recovery_policy.py` is authoritative for active deployed-form constants and gates. Values above are a status snapshot; if they conflict with active policy code, report the conflict rather than silently choosing the prose value.

The preserved rollback baseline is **lgb_mlp_cat**:

- Public score: **992.8390640403**
- Recorded composition: LGB 0.30 + MLP 0.35 + CatBoost 0.35.

## Current recovery result

The latest bounded recovery round reproduced the reconciled baseline on its selection origins and screened the registered CatBoost, causal residual, and causal calibration candidates. The candidates were rejected by the active gates.

- Freeze verdict: `NO_PROMOTION`
- Terminal check: `SKIPPED`
- Deployment replay: `SKIPPED_NO_PROMOTION`
- Package validation: valid `SKIPPED`
- New recovery package: none
- Current champion remains v93 6-leg

This result must not be reinterpreted as a validator failure or permission to try an unregistered fallback candidate.

## Data-integrity and temporal audit status

PR #6 merged the reviewed audit implementation into `master`:

- `scripts/audit_data_integrity_temporal.py`
- `tests/test_data_integrity_temporal.py`
- `docs/data_integrity_temporal_audit.md`

The final committed-head official MEDIUM audit was generated from PR head `7c40e2994e077fa01ebc8bdc1878a15496274a09`, not from the later merge commit. Its `canonical_report_sha256` is `bd3ea2f7bd8a720ca4de32d998742f81991f25b60384ba5b169a07fc68234b0c`, and the audit-script SHA-256 is `7e22b8ff96e5b435bdceddf5fddb9897b9c61cb91f14023f6ab5fcdc4720e2e2`. Substantive results were deterministically reproduced before commit; the committed-head rerun differed only in `git_sha` and the resulting canonical report hash.

Main-data conclusions:

- The 2019-2023 structural row and target-integrity checks passed.
- Exact intra-month chronology remains `NOT_PROVEN`.
- Exact as-of reconstruction and feature-generation provenance remain `NOT_PROVEN`.
- No audited as-of count/rate domain violations were observed, and cold-start missingness behavior is consistent with the documented zero-count semantics.

First-30k recovery evidence limitation:

- Current bounded selectors take the first 30,000 true positions in DataFrame/source order. The audit found substantial composition distortion relative to the full rolling-origin panels.
- The bounded-minus-full validation target-rate difference is approximately `-0.02195` (-2.20 percentage points) for r2023 and `-0.00292` (-0.29 percentage points) for r2022. `game_month` total variation is very large on multiple bounded panels.
- The current recovery `NO_PROMOTION` remains the recorded policy result, but first-30k evidence must not be treated as a representative proxy for full-origin model quality.
- Do not silently change the validation protocol. Any redesign of bounded validation requires a separate Experiment Brief, plan review, and explicit policy change.

Trackman conclusions:

- The original 622,737 `invalid_game_dates` result was an audit-parser artifact. In the audited official 2019-2023 Trackman data, `game_date` values were empirically observed in two valid formats: `MM/DD/YYYY` (including variable-width month/day) and `YYYY-MM-DD`.
- The corrected audit reports `invalid_game_dates = 0`, `game_date_season_mismatches = 0`, and `trackman_structural_contract = BENIGN`. The `trackman_id` missing and duplicate checks pass.
- `(trackman_game_id, pitch_no)` has two observed duplicates and remains `UNKNOWN` because official uniqueness is not documented.
- Raw main/Trackman player IDs do not establish a crosswalk. Exact main-row joining and safe Trackman feature-usage levels remain `NOT_PROVEN`.
- Do not infer Trackman unusability or authorize Trackman modeling from this audit alone.

The isolated `abs-2024-features` audit has **not** been run, so no ABS causal claim exists. Any later 2024 feature-only structural result must remain branch-inert and cannot directly tune or select features, thresholds, transformations, models, calibration, or recovery policy.

## Active recovery pipeline

Primary entry points and roles:

- `repro_979/recovery_live_state.py`: reconcile user-observed leaderboard state and package provenance.
- `repro_979/recovery_policy.py`: active recovery fold, scoring, gate, firewall, and audit definitions.
- `repro_979/recovery_evaluator.py`: baseline reproduction, screening evidence, freeze, terminal evaluation, and audits.
- `repro_979/recovery_catboost_runner.py`: bounded CatBoost candidates.
- `repro_979/recovery_residual_runner.py`: causal residual candidates.
- `repro_979/recovery_calibration_runner.py`: causal calibration candidates.
- `repro_979/recovery_deploy.py`: replay/package only an eligible frozen candidate.
- `repro_979/package_validator_recovery.py`: package parity, independence, offline, resource, and runtime gates.
- `repro_979/submission_decision.py`: submission-readiness and state gates.

Important policy/state records:

- `repro_979/recovery_candidate_registry.json`
- `repro_979/leaderboard_state.json`
- `.omo/evidence/aimers9-top100-recovery/`

These files have different roles: policy code defines active behavior, JSON records structured configuration/state, and evidence records historical executions and audits. Do not mutate state merely to make documentation agree with it.

## GPU bootstrap operational status

The reproducible school-GPU development bootstrap is now present in the repository. The fresh-environment correction that explicitly uses the supported libmamba solver was merged via PR #4.

Final school-GPU MEDIUM acceptance completed successfully:

- Fresh `apply`: rc=0, elapsed 203 seconds.
- Subsequent `check`: rc=0.
- Second `apply`: rc=0, elapsed 4 seconds, confirming the tested idempotent path.
- Direct environment audit: rc=0; `pip check`: rc=0.
- Environment identity: Python 3.11.15, Torch 2.7.1+cu128, CUDA build 12.8, and `torch.cuda.is_available()` true.
- Bridges: exactly `repro_979/open/data/train.csv`, `repro_979/open/data/test.csv`, `repro_979/open/data/sample_submission.csv`, and `repro_979/cache/v93_extract_verify` were present; no `trackman_history.csv` bridge was created.
- The main repository was clean after acceptance.

Two earlier fresh-bootstrap attempts timed out at 600 seconds during Conda environment creation. Later direct shell creation, Python-subprocess creation, and the full acceptance run all succeeded. The available evidence does not establish a definitive root cause for the earlier timeouts.

This acceptance validates the current school-GPU development bootstrap under the tested, cache-warmed environment. It does **not** establish cold-cache fresh-machine reproducibility or official submission/package readiness.

## Known blockers and technical debt

- The v93 full retraining/reproduction path is not fully self-contained from the Git repository alone. Repository evidence reconciles package hashes, configuration, and methodology, but external share-package provenance/materials were involved.
- Historical evidence under `.omo/evidence` is partly Git-tracked even though `.omo/` is ignored for new artifacts. Do not assume the directory is uniformly tracked or uniformly disposable.
- A full-directory recovery audit has historical Task 1 evidence-hygiene conflicts, including forbidden-token scanning; later Task 2-10 evidence passed in isolation. This is an audit-record hygiene issue that should be reported, not silently erased.
- Some historical plan text names fixtures or CLI entry points differently from their actual implementations. In particular, active audit commands live in `recovery_evaluator.py`, while `recovery_policy.py` is primarily the policy module.
- Multiple historical scoring conventions coexist: raw prediction BSS, older deployed offsets, the legacy/top-100 deployed form, and the current recovery deployed form. Every future experiment record must identify its formula, offset, clip, fold, label scope, and raw/deployed status.
- Some legacy scripts use old absolute paths, fixed GPU assumptions, or thread counts that do not match the 6-vCPU evaluation environment. Their existence is not evidence that they are safe to run.
- Requirements and packaged inference files exist in several historical locations; the candidate-bound package validator, not directory age or filename, should establish submission readiness.

## Next Actions

1. Consider the isolated 2024 feature-only ABS audit under its branch-inert information boundary.
2. Separately prepare an Experiment Brief and plan review for any validation-protocol experiment addressing first-30k representativeness.
3. Do not resume ordinary model-selection experiments under a silently altered validation protocol.
4. Before future GPU compute, inspect live GPU occupancy and explicitly choose an available device; the completed bootstrap acceptance does not reserve or select a GPU.
5. Choose the next experiment only after external strategy review; do not infer a new ML experiment from the recovery `NO_PROMOTION` result alone.

## Do Not Use As Current Selection Policy

The legacy/top-100 workflows—such as primary-fold blend selection, broad feature/model qualification, CatBoost weight sweeps, older calibration selection, and their associated deployment runners—are historical and may remain useful for diagnosis or provenance. They must not silently replace the active recovery policy.

Before running any selection workflow, verify that its fold roles, allowed label sources, deployed scoring form, candidate registry, and promotion gates match `repro_979/recovery_policy.py`. If repository sources conflict, stop and describe the conflict rather than resolving it by choosing whichever runner produces the best score.
