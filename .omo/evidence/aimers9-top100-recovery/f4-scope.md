# F4 — scope & records audit (final-wave reviewer, NORMAL branch) — APPROVE (exit 0)

- **recorded_at_utc**: 2026-08-18T05:20:00+00:00
- **git_head**: 444a759f42bba255e94af1b01d928f6e5c84ddd6
- **config_hash**: 158be42401fc5ce3fec78af960900a4d7c040fa08d960a4849113608eeb4b420
- **branch**: NORMAL (reconciled v93 6-leg baseline; Tasks 2-10 executed; all candidates REJECTED -> NO_PROMOTION -> SKIPPED)
- **label_sources**: [] | **labels_read**: false

## VERDICT: APPROVE

All five substantive F4 rejection criteria pass. The only automated audit failure is a
pre-existing Task-1 evidence token (scope-scan hygiene), which does not affect the new
task-2..10 evidence.

## Checks (all PASS)

1. **No unstaged excluded directories** — `git diff --cached` empty (0 staged files).
   Untracked dirs (`.omo.gpu01-backup-20260814-030428/`, the user share dir, and the two
   baseline copy dirs) are NOT staged. Plan "no git add -A" rule holds.
2. **No unrecorded label-scored CV** — every label-scored screen has a Notion local-CV
   receipt: task-4 row `3c05ed6b-28d5-8134-8718-c9211dc0748e`, task-5 row
   `3c05ed6b-28d5-8165-9c54-c02719a542e5`, task-6 two rows (calibration_beta +
   calibration_isotonic); all `row_written=true`.
3. **No fabricated leaderboard records** — `leaderboard_state.json` holds only the
   user-reported 1001.74449 (reconciled to the v93 6-leg package, BASELINE_RECONCILED)
   and the real rollback 992.8390640403. No invented results. Task-1 Notion leaderboard
   row `3bf5ed6b` verified.
4. **No broad candidate expansion** — registry has exactly 6 selectable + 2 control IDs;
   task-4/5/6/7 evidence screens only registry candidates. No broad expansion.
5. **Complete commit/evidence linkage** — all 10 task commits include their evidence
   files (fb2f49c->task-2, f9e447f->task-3, 6eb2d11->task-4, 4a87878->task-5,
   7bd68d9->task-6, 3b60c2c->type fix, 48bad32->task-7, 902237f->task-8, 9d52e33->task-9,
   444a759->task-10).
6. **State immutability** — `leaderboard_state.json` sha256 `9107f3bb...` identical to
   git HEAD before/after review (review-only, no state/code change).

## QA scenario results

- **Happy path (full dir)**: `recovery_evaluator.py --audit-scope` exits **2** (REJECT)
  solely because the pre-existing `task-1-live-state.json` contains the package-archive
  extension token (a literal forbidden path token) in its provenance evidence. This is a
  Task-1 artifact committed at HEAD, NOT caused by the new task-2..10 evidence.
- **Happy path (task-2..10 isolation)**: `--audit-scope` on a temp dir with ONLY
  task-2..10 evidence exits **0** (PASS) — the new evidence is scope-clean.
- **Fixtures**: `legacy-deepfm`, `catboost9-digest`, `unknown-public-baseline`
  (unapproved-candidate concern), `manifest-mutation` (evidence integrity) — each exits **2**.

## Findings (non-blocking)

- **Pre-existing task-1 token**: the full-dir audit failure is a Task-1 artifact
  (committed at HEAD), documented in the notepad as a pre-existing issue to clean up.
  It is a scope-scan hygiene issue, not a substantive F4 rejection criterion.
- **Plan-vs-implementation fixture names**: the plan F4 spec names fixtures
  `forbidden-path`, `unmatched-notion-receipt`, `unapproved-candidate`,
  `unauthorized-state-mutation` — these do NOT exist in the implementation (argparse
  invalid-choice exit 2, no evidence written). The concerns are covered by existing
  fixtures and the audit's own scans.
- **Plan command typo**: the plan F4 spec command `recovery_policy.py --audit-scope` is
  a no-op (no `__main__`); the real audit CLI is `recovery_evaluator.py --audit-scope`.

## Review-only discipline

No plan checkbox edit, no code/state change, no upload, no DACON call. Only this
evidence and the notepad learnings were written.
