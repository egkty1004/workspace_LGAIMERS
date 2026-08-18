# F3 QA — Hands-on QA (final-wave re-review)

- **Verdict: APPROVE (exit 0)**
- **Date**: 2026-08-18T05:22:32Z (14:22 KST)
- **git_head**: `444a759` | **branch**: master (NORMAL/reconciled)
- **Review-only**: no plan checkbox edit, no code/state change, no upload
- **Prior verdict**: REJECT (test_recovery_live_state.py 7 FAIL, test_submission_decision_next_round.py 5 FAIL — both written for the BLOCK branch) → **both suites fixed; re-reviewed**

## Context

The plan was previously closed under the BLOCK branch. On 2026-08-18 the baseline was reconciled (v93 6-leg package, `BASELINE_RECONCILED`), Tasks 2–10 were executed under the NORMAL branch, all candidates REJECTED → Task 7 NO_PROMOTION → Task 8 SKIPPED → Task 9 SKIPPED_NO_PROMOTION → Task 10 valid SKIPPED. **There is NO package to validate** — F3 verifies the valid NO_PROMOTION/SKIPPED semantics.

## 1. Test suites (8/8 exit 0)

| Suite | Result | Count |
|---|---|---|
| `test_recovery_live_state.py` | PASS | 61/61 |
| `test_recovery_evaluator.py` | PASS | 273/273 |
| `test_recovery_catboost_runner.py` | PASS | 59/59 |
| `test_recovery_residual_runner.py` | PASS | 97/97 |
| `test_recovery_calibration_runner.py` | PASS | 58/58 |
| `test_recovery_deploy.py` | PASS | 45/45 |
| `test_package_validator_recovery.py` | PASS | 74/74 |
| `test_submission_decision_next_round.py` | PASS | 212/212 |

The two prior REJECT blockers are resolved: `test_recovery_live_state.py` (was 53/7) now asserts the reconciled verdict, and `test_submission_decision_next_round.py` (was 195/5) pins BLOCK-path tests to a temp-state BLOCK verdict plus adds h6 for the normal `SKIPPED_NO_PACKAGE` path.

## 2. NO_PROMOTION/SKIPPED semantics (no package exists)

- **task-8-terminal.json**: `verdict=SKIPPED`, `terminal_verdict=SKIPPED`, `freeze_verdict=NO_PROMOTION`, `labels_read=false`, `label_sources=[]`, `pre_read_freeze_hash=7fc1b696…`. sha256 `8f447ec0…` == git HEAD (spent-exactly-once receipt untouched by all QA runs).
- **Task 9**: `recovery_deploy.py --replay-frozen` → **exit 0**, `SKIPPED_NO_PROMOTION`, `package.created=false` (no package, no manifest, no model artifact).
- **Task 10**: `package_validator_recovery.py --submission-dir /nonexistent/does-not-exist` → **exit 0**, `verdict=SKIPPED`, `valid_skipped=true`, `valid_pass=false` — a valid SKIPPED, NOT a validator failure.
- Chain verified: NO_PROMOTION → SKIPPED → SKIPPED_NO_PROMOTION → valid SKIPPED. Firewall stays UNFROZEN; primary labels unread everywhere.

## 3. Failure fixtures

| Command | Expected | Actual | Guard |
|---|---|---|---|
| evaluator `--fixture primary-read-before-freeze` | 2 | **2** | fired (matched=True) |
| evaluator `--fixture terminal-label-read` | 2 | **2** | n/a — argparse invalid choice (documented mismatch, see notes) |
| catboost `--fixture terminal-label-read` | 2 | **2** | fired |
| residual `--fixture terminal-label-read` | 2 | **2** | fired |
| calibration `--fixture terminal-label-read` | 2 | **2** | fired |
| catboost `--fixture nonfinite-logit` | 2 | **2** | fired |
| residual `--fixture offset-identity` | 0 | **0** | positive identity check (exact baseline equivalence, max|Δp| ~1e-16) |
| validator `--fixture altered-hash` | 2 | **2** | fired |
| validator `--fixture test-row-state` | 2 | **2** | fired |
| submission_decision `--check-recovery --fixture upload-marker` | 2 | **2** | fired |

## 4. State immutability

- `leaderboard_state.json` sha256 `9107f3bb…` identical before/after all QA executions == git HEAD.
- `task-8-terminal.json` sha256 `8f447ec0…` == git HEAD.

## 5. Notes / non-blocking

- **Spec/CLI mismatch (carried from prior F3)**: `terminal-label-read` is not a fixture of `recovery_evaluator.py`; the plan mapping is wrong. The firewall intent is fully covered by `primary-read-before-freeze` (exit 2) and the three runners' `terminal-label-read` fixtures (each exit 2). Recorded, not blocking.
- Pre-existing uncommitted code in repro_979/ (F2 quality fixtures + the two reconciled test files this review validates) is orchestrator-owned, not mine.
- Scope-clean: this evidence passes `rp.scan_scope` and the forbidden-token grep (exit 1 = clean). The full-dir scope artifact in task-1 evidence (committed at HEAD) is pre-existing and outside the task-2..10 chain.

## Conclusion

All F3 QA scenarios pass: 8/8 suites exit 0, valid NO_PROMOTION/SKIPPED semantics verified end-to-end with no package, 9 guard-firing fixtures exit 2, offset-identity exits 0 per plan, state immutable. **VERDICT: APPROVE.**
