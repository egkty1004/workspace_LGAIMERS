# Todo 2 — immutable rolling-origin recovery evaluator (smoke) — PASS (exit 0)

- **recorded_at_utc**: 2026-08-17T23:58:41+00:00
- **git_head**: a89b1d086ee3f0c755e8f498f4b4c442f677c8d4
- **config_hash**: `158be42401fc5ce3fec78af960900a4d7c040fa08d960a4849113608eeb4b420`
- **label_sources**: [] (labels_read=False)

## Checks

- **[PASS]** policy_config_hash: config_hash=158be42401fc5ce3… (라벨 무관)
- **[PASS]** manifest_valid: manifest_hash=d725b0d73ce32a28… violations=[]
- **[PASS]** origin_masks: origins=['r2022', 'r2023', 'primary', 'r2024'] leak=[]
- **[PASS]** row_disjointness: r2022/r2023 disjoint from primary; r2024 overlap documented: {"r2022": {"overlap_rows_with_primary": 0, "expected": true, "reason": "row-ID disjoint from primary"}, "r2023": {"overlap_rows_with_primary": 0, "expected": true, "reason": "row-ID disjoint from primary"}, "r2024": {"overlap_rows_with_primary": 133, "expected": true, "reason": "r2024 val = (season==2024) & R subset of primary val (overlapping diagnostic origin, branch-inert)"}}
- **[PASS]** selection_labels_only: labels=['r2022', 'r2023'] — primary/r2024 미로드
- **[PASS]** terminal_firewall: primary 라벨 읽기 차단 (firewall_state='UNFROZEN'): [FIREWALL] primary 라벨은 동결 전에 읽을 수 없습니다 (firewall_state='UNFROZEN', 필요 'FROZEN') — exit 2
- **[PASS]** screen_gate_path: gate verdict=REJECT (합성 경로 검증)
- **[PASS]** bootstrap_definition: paired row-bootstrap LB5(r2022)=+0.0000 (10,000 resamples, rng=default_rng(20260817+outer_year))

## Findings

- **[PASS]** manifest_before_labels: 매니페스트는 라벨 로드 이전에 해시/기원/산식/부트스트랩/파이어월 상태를 기록

## Verdict: **PASS** (exit 0)
