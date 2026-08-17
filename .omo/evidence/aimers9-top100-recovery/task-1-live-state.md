# Todo 1 — reconcile live leaderboard state & 1001.74449 provenance — BASELINE_PROVENANCE_BLOCK (exit 0)

- **recorded_at_utc**: 2026-08-17T03:36:11+00:00
- **git_head**: c191a5864749aabedd9c3166475c52d59cddd21e
- **state**: /home/2022113165/workspace_LGAIMERS/repro_979/leaderboard_state.json (sha256 b1aacb7cadf22bb2…)

## Facts

### user_observed (사용자 제공)

- **rank_100_cutoff**: `1090.64249`
- **current_best_public_score**: `1001.74449`
- **team_rank**: `287`
- **reported_date**: `2026-08-17`
- **observation_timestamp**: `UNKNOWN`
- **source**: `user-reported DACON leaderboard`
- **note**: `사용자가 제공한 값 (2026-08-17 세션). 관측 시각은 미제공 → UNKNOWN.`

### worker_reported_at (worker 캡처 — 관측 시각 아님)

- **reported_at**: `2026-08-17T12:34:43+09:00`
- **timezone**: `Asia/Seoul`
- **note**: `worker-captured 현재 시각 — 사용자의 관측/제출 시각이 아님 (observation_timestamp=UNKNOWN 유지).`

### package_proven (아티팩트에서 검증된 사실)

- **retrainable_source**: [FAIL] no submit package or training source tied to 1001.74449 found; searched repro_979/submit_*, aimers9_final_blend_team_share_20260814, .omo/evidence, docs/exp_hgb_validation.md, git log --all -S 1001.74449 (no commits)
- **preprocessing**: [FAIL] no preprocessing manifest for any 1001.74449 package; the only proven preprocessing contract (common.py byte-identical to GIHO extract) belongs to 5890a4c54f502c4e (Public 992.8390640403)
- **configuration**: [FAIL] no config/feature list/blend config for a 1001.74449 package; best proven config is the .30/.35/.35 logit blend of 5890a4c54f502c4e (C_LOGIT=-0.0404, clip .30/.70)
- **model_hashes**: [FAIL] no model files/provenance.json/sha256 for a 1001.74449 package; rollback 5890a4c54f502c4e model hashes are verified and intact (qualification_digest e3e047dd...)
- **per_origin_refit**: [FAIL] no per-origin refit capability recorded for any 1001.74449 package; for rollback only CatBoost is refittable in-workspace (deploy_train_catboost_full.py), LGB/MLP are byte-frozen GIHO copies
- **weight_contract**: [FAIL] no script.py/provenance.json to check the .30/.35/.35 LGB/MLP/Cat logit contract against; the contract is proven ONLY for 5890a4c54f502c4e

## Baseline verdict: **BASELINE_PROVENANCE_BLOCK**

- retrainable_source/preprocessing/configuration/model_hashes/per_origin_refit 중 미증명 요소 존재
- rollback preserved: 5890a4c54f502c4e / 992.8390640403
- task routing: blocked=['2', '3', '4', '5', '6', '7', '8', '9', '10'] task11=SKIPPED_BASELINE_BLOCK

## Notion

- status: `available`
- receipt: `{"status": "available", "table": "aimers9 실험 기록 - 리더보드 제출 기록 (block 2a000a06-d653-4989-8c92-112993d69962)", "appended_row_block_id": "3bf5ed6b-28d5-8105-bd4b-da500e622e00", "row": {"submission_date": "UNKNOWN (사용자 보고 2026-08-17)", "file_model": "UNKNOWN - provenance 미확정 (BASELINE_PROVENANCE_BLOCK)", "public": "1001.74449", "private": "(미공개)", "note": "사용자 보고 (2026-08-17 세션), observation_timestamp=UNKNOWN, reported_at=2026-08-17T12:34:43+09:00 (worker capture). 제출일/파일/모델/프로비넌스 UNKNOWN - 워크스페이스 아티팩트에서 출처 미확인 -> BASELINE_PROVENANCE_BLOCK, Tasks 2-10 중단. rollback 5890a4c54f502c4e / 992.8390640403 유지, delta 계산 불가, C_LOGIT=-0.0404 불변."}, "appended_at_utc": "2026-08-17T03:35:00+00:00", "by": "opencode worker (Aimers9 recovery plan Task 1)"}`

## Checks

- **[PASS]** user_value_rank_100_cutoff: rank_100_cutoff = 1090.64249 (사용자 보고값 일치)
- **[PASS]** user_value_current_best_public_score: current_best_public_score = 1001.74449 (사용자 보고값 일치)
- **[PASS]** user_value_team_rank: team_rank = 287 (사용자 보고값 일치)
- **[PASS]** date_captured: date_captured = 2026-08-17 (보고일; 관측 시각은 UNKNOWN)
- **[PASS]** observation_timestamp_unknown: observation_timestamp = UNKNOWN (사용자 미제공)
- **[PASS]** reported_at_kst: reported_at = 2026-08-17T12:34:43+09:00 (+09:00, worker-captured)
- **[PASS]** rollback_qualified_package: qualified_package 유지: 5890a4c54f502c4e
- **[PASS]** rollback_qualification_digest: qualification_digest 유지
- **[PASS]** score_package_mismatch: qualified_candidate_public_score = 992.8390640403 (rollback 실점수 유지)
- **[PASS]** submissions_by_date_preserved: submissions_by_date 유지 ({'2026-08-14': 2})
- **[PASS]** last_submission_preserved: last_submission 유지 (date=2026-08-14)
- **[PASS]** baseline_verdict_facts_consistency: baseline_verdict = BASELINE_PROVENANCE_BLOCK (provenance 필드와 일관)
- **[PASS]** rollback_package_hashes: rollback 패키지 모델 파일 33건 디스크 재해시 일치

## Verdict: **BASELINE_PROVENANCE_BLOCK** (exit 0)
