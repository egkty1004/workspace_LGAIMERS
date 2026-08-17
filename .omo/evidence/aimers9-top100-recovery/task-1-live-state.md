# Todo 1 — reconcile live leaderboard state & 1001.74449 provenance — BASELINE_RECONCILED (exit 0)

- **recorded_at_utc**: 2026-08-17T17:09:57+00:00
- **git_head**: 7445a0681ed4261df70537b6ea3f850a9b831cb5
- **state**: /home/2022113165/workspace_LGAIMERS/repro_979/leaderboard_state.json (sha256 9107f3bb82988d39…)

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

- **retrainable_source**: [PROVEN] 8.16 1001.7444910721 share/submit_v93_r0476.zip (SHA256 8157e144090bcccbf1c44367c75a2d2427e040b8353c8c1e17334b41324ac5fb) = submit_v93_6leg_r0477_20260815_204917.zip; methodology v93_6leg_r0477_코드_방법론_설명.md (2026-08-17)
- **preprocessing**: [PROVEN] script.py common.preprocess_for_submission (49 features: platoon+count_state); common.py byte-consistent
- **configuration**: [PROVEN] W_LGB=0.65, LAM_FTT=0.17991944576662527, LAM_ARMB=0.43481381354434545, LAM_CAT=0.0701066994221915, C_LOGIT=-0.0461645795229729, clip[.30,.70], SEEDS 42..51, FTT_SEEDS 42..44
- **model_hashes**: [PROVEN] 51 model files in zip (f3_s42..51.txt, mlp_s42..51.pt, ftt_s42..44.pt, armb_s42..51.pt, catboost_s42..51.cbm, prep pkl, train_meta.json); zip sha256 8157e144...
- **per_origin_refit**: [PROVEN] methodology documents exp93/exp95/exp95c generation (train_v93_5leg_blend.py, train_v95_catboost_deploy.py, compute_v95_pbar.py, build_v93_6leg_pkg.py); primary-optimized lambda*, frozen R-fold eval
- **weight_contract**: [PROVEN] 6-leg v93 formula (NOT the .30/.35/.35 rollback contract): z_base=0.65LGB+0.35MLP + 0.1799(FTT-base)+0.4348(ArmB-base)+0.0701(Cat-base); C_LOGIT=-0.0461645795229729

## Baseline verdict: **BASELINE_RECONCILED**

- 전체 retrainable provenance + .30/.35/.35 계약 증명
- rollback preserved: 5890a4c54f502c4e / 992.8390640403
- task routing: blocked=[] task11=normal

## Notion

- status: `unavailable`

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
- **[PASS]** baseline_verdict_facts_consistency: baseline_verdict = BASELINE_RECONCILED (provenance 필드와 일관)
- **[PASS]** rollback_package_hashes: rollback 패키지 모델 파일 33건 디스크 재해시 일치

## Verdict: **BASELINE_RECONCILED** (exit 0)
