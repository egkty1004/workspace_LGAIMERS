# Todo 11 fixture — stale-live-state (recovery readiness, adversarial) — REJECT (exit 2)

- **recorded_at_utc**: 2026-08-17T04:24:18.998354+00:00
- **git_head**: 3ebbfb4dbd4d39033c6abe227a9ada74ce42e967
- **label_sources**: [] (labels_read=False)

## Findings

- **[PASS]** no_state_mutation: 실제 상태 파일 sha256 불변 (in-memory copy 만 변형)
- **[PASS]** no_main_evidence_clobber: fixture 는 task-11-readiness-fixture-<name>.{json,md} 만 기록

## Verdict: **REJECT**
