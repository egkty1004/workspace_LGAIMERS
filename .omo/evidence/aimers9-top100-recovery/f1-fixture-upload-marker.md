# F1 fixture record — upload-marker — FIXTURE_UNIMPLEMENTED (exit 2)

- **recorded_at_utc**: 2026-08-18 (UTC)
- **git_head**: 444a759f42bba255e94af1b01d928f6e5c84ddd6
- **fixture**: upload-marker-guard
- **guard_triggered**: false
- **tool_written_evidence**: false

## Detail

Fixture NOT in `recovery_evaluator.py` FIXTURES → argparse invalid-choice exit 2 (incidental, NOT a guard firing), no tool-written evidence. The no-upload condition IS exercised by `submission_decision.py --check-recovery --fixture upload-marker-guard` (exit 2, verified).

## Covered by

`submission_decision.py --check-recovery --fixture upload-marker-guard` (upload-token rejection).
