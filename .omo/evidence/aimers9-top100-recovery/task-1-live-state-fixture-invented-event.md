# Todo 1 fixture — invented-event (adversarial, exit 2) — FIXTURE_REJECT (exit 2)

- **recorded_at_utc**: 2026-08-17T03:36:17+00:00
- **git_head**: c191a5864749aabedd9c3166475c52d59cddd21e
- **state**:  (sha256 …)

## Facts

### user_observed (사용자 제공)


### worker_reported_at (worker 캡처 — 관측 시각 아님)


### package_proven (아티팩트에서 검증된 사실)


## Baseline verdict: **FIXTURE_REJECT**

- 
- rollback preserved: None / None
- task routing: blocked=None task11=None

## Notion

- status: `None`

## Checks

- **[PASS]** user_value_rank_100_cutoff: rank_100_cutoff = 1090.64249 (사용자 보고값 일치)
- **[PASS]** user_value_current_best_public_score: current_best_public_score = 1001.74449 (사용자 보고값 일치)
- **[PASS]** user_value_team_rank: team_rank = 287 (사용자 보고값 일치)
- **[PASS]** date_captured: date_captured = 2026-08-17 (보고일; 관측 시각은 UNKNOWN)
- **[FAIL]** observation_timestamp_unknown: observation_timestamp = '2026-08-17T10:00:00+09:00' — 사용자가 관측 시각을 제공하지 않았으므로 'UNKNOWN' 이어야 함
- **[PASS]** reported_at_kst: reported_at = 2026-08-17T12:34:43+09:00 (+09:00, worker-captured)
- **[PASS]** rollback_qualified_package: qualified_package 유지: 5890a4c54f502c4e
- **[PASS]** rollback_qualification_digest: qualification_digest 유지
- **[PASS]** score_package_mismatch: qualified_candidate_public_score = 992.8390640403 (rollback 실점수 유지)
- **[FAIL]** submissions_by_date_preserved: submissions_by_date = {'2026-08-14': 2, '2026-08-17': 1} (필요 {'2026-08-14': 2}) — 제출 횟수 변경 금지 (실제 제출 이벤트만)
- **[PASS]** last_submission_preserved: last_submission 유지 (date=2026-08-14)
- **[PASS]** baseline_verdict_facts_consistency: baseline_verdict = BASELINE_PROVENANCE_BLOCK (provenance 필드와 일관)
- **[PASS]** rollback_package_hashes: rollback 패키지 모델 파일 33건 디스크 재해시 일치

## Verdict: **FIXTURE_REJECT** (exit 2)
