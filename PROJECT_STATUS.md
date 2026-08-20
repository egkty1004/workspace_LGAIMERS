# Project Status

Updated: 2026-08-20 (repository-evidence snapshot)

Evidence base: `master` at `26879a392996ad878ed330da858f49fdb9abab78`.

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

1. Keep the repository context and status documentation synchronized without changing ML policy or state.
2. Before future GPU compute, inspect live GPU occupancy and explicitly choose an available device; the completed bootstrap acceptance does not reserve or select a GPU.
3. Choose the next experiment only after external strategy review; do not infer a new ML experiment from the recovery `NO_PROMOTION` result alone.

## Do Not Use As Current Selection Policy

The legacy/top-100 workflows—such as primary-fold blend selection, broad feature/model qualification, CatBoost weight sweeps, older calibration selection, and their associated deployment runners—are historical and may remain useful for diagnosis or provenance. They must not silently replace the active recovery policy.

Before running any selection workflow, verify that its fold roles, allowed label sources, deployed scoring form, candidate registry, and promotion gates match `repro_979/recovery_policy.py`. If repository sources conflict, stop and describe the conflict rather than resolving it by choosing whichever runner produces the best score.
