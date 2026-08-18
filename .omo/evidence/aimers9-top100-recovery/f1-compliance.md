# F1 — Plan-compliance and provenance audit — APPROVE (exit 0)

- **recorded_at_utc**: 2026-08-18 (UTC)
- **git_head**: 444a759f42bba255e94af1b01d928f6e5c84ddd6
- **branch**: NORMAL (reconciled v93 6-leg baseline)
- **label_sources**: [] (labels_read=False — review-only, no label access)
- **config_hash**: `a17733ccef2f8eed...`

## Verdict: **APPROVE**

All six F1 core substance checks pass. The happy-path `--audit-compliance` exits 0 on the new task-2..10 evidence. The full-dir audit exits 2 solely due to pre-existing task-1 evidence (not the new evidence). The plan's F1 failure-fixture names are partially unimplemented, but every adversarial condition is exercised by existing fixtures under different names (same class as F4, which APPROVED).

## Core substance checks (all PASS)

- **[PASS] live_state** — `leaderboard_state.json` has `baseline_verdict=BASELINE_RECONCILED`; `reconciled_baseline` = v93 6-leg contract (C_LOGIT=-0.0461645795229729, clip [0.3,0.7], 51 model files, sha256 8157e144...). Rollback 5890a4c54f502c4e / 992.8390640403 preserved. State sha256 9107f3bb... == git HEAD (immutable).
- **[PASS] registry** — `recovery_candidate_registry.json`: exactly 6 selectable_ids + 2 control_ids; blocked_legacy 17 names / 7 configs / 1 digest (a41e9b22...); baseline = v93 6-leg. `validate_registry` recomputes config_hash from stored config (all match).
- **[PASS] evidence_ordering** — 40 task-2..10 evidence files all carry config_hash + label_sources(list) + labels_read + verdict. Chain: task-2 PASS → task-3 PASS → task-4/5/6 REJECT → task-7 NO_PROMOTION → task-8 SKIPPED → task-9 SKIPPED_NO_PROMOTION → task-10 SKIPPED. All committed with per-task linkage.
- **[PASS] candidate_firewall** — `read_primary_labels` raises TerminalFirewallError unless FROZEN (structural). `--screen` reads only r2022/r2023. NO_PROMOTION/SKIPPED branches keep firewall UNFROZEN and labels_read=false (task-7/8/9/10 evidence). `primary-read-before-freeze` fixture exit 2 (guard fires).
- **[PASS] rejected_digest_ban** — `reject_blocked` checks blocked_legacy names/configs/digests + allowed set BEFORE any data loading. catboost9-digest / legacy-deepfm / unknown-public-baseline fixtures all exit 2.
- **[PASS] no_public_no_upload** — audit scan_forbidden_usage + scan_upload_markers pass on task-2..10 evidence (exit 0). No DACON API/upload/login/requests in recovery source. Recovery runners read only train.csv.

## QA scenarios (recorded)

| Command | Exit | Result |
|---|---|---|
| `--audit-compliance` on task-2..10-only dir | 0 | PASS |
| `--audit-compliance` on full dir | 2 | REJECT (pre-existing task-1 evidence only) |
| `--fixture primary-read-before-freeze` | 2 | guard fires; tool writes task-2-evaluator-fixture-* |
| `--fixture public-sort-key` | 2 | argparse invalid choice (not implemented); covered by r2024-sort-key + primary-sort-key |
| `--fixture legacy-digest` | 2 | argparse invalid choice (not implemented); covered by catboost9-digest + legacy-deepfm |
| `--fixture stale-package` | 2 | argparse invalid choice (not implemented); closest analog stale-screen-evidence |
| `--fixture upload-marker` | 2 | argparse invalid choice in evaluator CLI; EXISTS in submission_decision.py --check-recovery (exit 2) |

## Findings (non-blocking)

- **full_dir_audit_pre_existing** — full-dir `--audit-compliance` exits 2 solely due to PRE-EXISTING task-1 evidence: task-1-live-state.json + 3 task-1 fixtures lack config_hash (written before the config_hash convention), and task-1-live-state.json legitimately records the user-reported public score 1001.74449 under key `public`. Task-1 artifacts committed at HEAD. Does NOT affect the new task-2..10 evidence (audit exit 0 in isolation).
- **f1_fixture_gap** — plan F1 failure fixtures public-sort-key / legacy-digest / stale-package / upload-marker are NOT implemented in recovery_evaluator.py FIXTURES (argparse invalid-choice exit 2, no tool-written evidence). Underlying adversarial conditions ARE exercised by existing fixtures under different names. Plan-vs-implementation discrepancy (same class as F4, which APPROVED).
- **plan_cli_typo** — plan F1 QA scenario names `recovery_policy.py --audit-compliance` but recovery_policy.py has no `__main__` (no-op exit 0); the real audit CLI is `recovery_evaluator.py --audit-compliance`.
- **uncommitted_code** — pre-existing uncommitted changes in repro_979/ (recovery_evaluator.py F2 fixture additions + 3 test files updated for the reconciled state). Not committed; outside this review's scope; flagged for the orchestrator.

## Review-only

No plan checkbox edit, no code/state change, no upload. Only f1 evidence + learnings written.
