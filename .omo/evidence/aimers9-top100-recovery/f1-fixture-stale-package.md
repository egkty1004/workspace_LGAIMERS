# F1 fixture record — stale-package — FIXTURE_UNIMPLEMENTED (exit 2)

- **recorded_at_utc**: 2026-08-18 (UTC)
- **git_head**: 444a759f42bba255e94af1b01d928f6e5c84ddd6
- **fixture**: stale-package-guard
- **guard_triggered**: false
- **tool_written_evidence**: false

## Detail

Fixture NOT in `recovery_evaluator.py` FIXTURES → argparse invalid-choice exit 2 (incidental, NOT a guard firing), no tool-written evidence. The closest analog `stale-screen-evidence` (exit 2, verified) covers the staleness-rejection condition.

## Covered by

`stale-screen-evidence` (stale/non-terminal evidence rejection in freeze).
