# F1 fixture record — public-sort-key — FIXTURE_UNIMPLEMENTED (exit 2)

- **recorded_at_utc**: 2026-08-18 (UTC)
- **git_head**: 444a759f42bba255e94af1b01d928f6e5c84ddd6
- **fixture**: public-sort-key-guard
- **guard_triggered**: false
- **tool_written_evidence**: false

## Detail

Fixture NOT in `recovery_evaluator.py` FIXTURES → argparse invalid-choice exit 2 (incidental, NOT a guard firing), no tool-written evidence. The adversarial condition (no-Public sort) is exercised by the existing `r2024-sort-key` and `primary-sort-key` fixtures (both exit 2, verified).

## Covered by

`r2024-sort-key`, `primary-sort-key` (forbidden sort-key guards).
