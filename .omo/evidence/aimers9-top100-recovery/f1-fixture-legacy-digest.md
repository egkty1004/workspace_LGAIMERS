# F1 fixture record — legacy-digest — FIXTURE_UNIMPLEMENTED (exit 2)

- **recorded_at_utc**: 2026-08-18 (UTC)
- **git_head**: 444a759f42bba255e94af1b01d928f6e5c84ddd6
- **fixture**: legacy-digest-guard
- **guard_triggered**: false
- **tool_written_evidence**: false

## Detail

Fixture NOT in `recovery_evaluator.py` FIXTURES → argparse invalid-choice exit 2 (incidental, NOT a guard firing), no tool-written evidence. The adversarial condition (rejected-digest ban) is exercised by the existing `catboost9-digest` and `legacy-deepfm` fixtures (both exit 2, verified).

## Covered by

`catboost9-digest`, `legacy-deepfm` (blocked-legacy rejection before data load).
