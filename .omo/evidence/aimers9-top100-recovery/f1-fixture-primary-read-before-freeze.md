# F1 fixture record — primary-read-before-freeze — FIXTURE_REJECT (exit 2)

- **recorded_at_utc**: 2026-08-18 (UTC)
- **git_head**: 444a759f42bba255e94af1b01d928f6e5c84ddd6
- **fixture**: primary-read-before-freeze-guard
- **guard_triggered**: true
- **tool_written_evidence**: true

## Detail

Fixture EXISTS in `recovery_evaluator.py` FIXTURES. Guard fires (matched=True): `read_primary_labels` raises TerminalFirewallError while the firewall is UNFROZEN. Exit 2. The tool writes `task-2-evaluator-fixture-primary-read-before-freeze.{json,md}` (the plan asked for `f1-fixture-*` naming — naming discrepancy only).

## Covered by

Structural firewall (`recovery_policy.py` `read_primary_labels` raises unless FROZEN).
