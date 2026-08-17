# Todo 2 fixture — primary-read-before-freeze (adversarial, exit 2) — FIXTURE_REJECT (exit 2)

- **recorded_at_utc**: 2026-08-17T23:58:41+00:00
- **git_head**: a89b1d086ee3f0c755e8f498f4b4c442f677c8d4
- **config_hash**: `158be42401fc5ce3fec78af960900a4d7c040fa08d960a4849113608eeb4b420`
- **label_sources**: [] (labels_read=False)

## Checks

- **[PASS]** fixture_primary-read-before-freeze: 파이어월이 primary 라벨 읽기를 차단: [FIREWALL] primary 라벨은 동결 전에 읽을 수 없습니다 (firewall_state='UNFROZEN', 필요 'FROZEN') — exit 2

## Findings

- **[PASS]** no_main_evidence_clobber: fixture 는 task-2-evaluator-fixture-<name>.{json,md} 만 기록

## Verdict: **FIXTURE_REJECT** (exit 2)
