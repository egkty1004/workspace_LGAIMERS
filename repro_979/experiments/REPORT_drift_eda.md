# Task 3 — Drift-aware EDA + feature hypothesis registry

> 날짜: 2026-08-13T20:01:36.672516+00:00 | smoke=False | 데이터: train 1,475,092행 (2019~2024) + test 형식 샘플

## 1. 누수 가드

- 2025 라벨 미사용 (모든 통계/변환 상수는 학습 윈도우 전용)
- test 행 간 파생 없음 (`test_row_dependency=false` 강제)
- 외부 데이터 없음

## 2. 수치형 드리프트 (2024 vs 2019, |SMD| 상위)

| feature | 2019 mean | 2024 mean | Δ | SMD | n19 | n24 |
|---|---|---|---|---|---|---|
| asof_batter_n | 720.7569 | 5402.5488 | +4681.7919 | +1.608 | 237,413 | 253,507 |
| asof_pitcher_n | 768.7283 | 3928.2145 | +3159.4862 | +1.216 | 237,413 | 253,507 |
| asof_pitcher_pitchmix_n | 768.7283 | 3928.2145 | +3159.4862 | +1.216 | 237,413 | 253,507 |
| asof_pitcher_prev5_game_middle_rate | 0.1266 | 0.1724 | +0.0459 | +0.986 | 225,135 | 249,869 |
| asof_pitcher_reverse_rate | 0.1823 | 0.2372 | +0.0550 | +0.915 | 237,058 | 253,426 |
| asof_batter_middle_rate | 0.1238 | 0.1520 | +0.0282 | +0.902 | 237,013 | 253,419 |
| asof_pitcher_middle_rate | 0.1249 | 0.1557 | +0.0308 | +0.871 | 237,058 | 253,426 |
| asof_pitcher_prev5_game_success_rate | 0.5587 | 0.4909 | -0.0679 | -0.869 | 225,135 | 249,869 |
| asof_pitcher_prev3_game_middle_rate | 0.1267 | 0.1739 | +0.0472 | +0.867 | 225,135 | 249,869 |
| asof_pitcher_prev3_game_success_rate | 0.5592 | 0.4895 | -0.0697 | -0.790 | 225,135 | 249,869 |
| asof_batter_success_rate | 0.5615 | 0.5175 | -0.0441 | -0.787 | 237,013 | 253,419 |
| asof_pitcher_success_rate | 0.5598 | 0.5114 | -0.0484 | -0.713 | 237,058 | 253,426 |

## 3. 레짐 드리프트 (R vs F, |SMD| 상위)

| feature | mean_R | mean_F | SMD | n_R | n_F |
|---|---|---|---|---|---|
| asof_batter_success_rate | 0.5330 | 0.5946 | -1.284 | 1,313,630 | 160,632 |
| asof_batter_n | 3549.5257 | 996.3210 | +0.867 | 1,314,088 | 161,004 |
| asof_pitcher_ball_rate | 0.3685 | 0.4031 | -0.857 | 1,313,586 | 160,714 |
| asof_pitcher_strike_rate | 0.4454 | 0.4135 | +0.831 | 1,313,586 | 160,714 |
| asof_pitcher_fastball_rate | 0.5473 | 0.6123 | -0.670 | 1,313,586 | 160,714 |
| asof_pitcher_n | 2842.1322 | 1185.5883 | +0.627 | 1,314,088 | 161,004 |
| asof_pitcher_pitchmix_n | 2842.1322 | 1185.5883 | +0.627 | 1,314,088 | 161,004 |
| asof_pitcher_prev5_game_success_rate | 0.5192 | 0.5684 | -0.620 | 1,293,925 | 151,982 |

## 4. 범주 지원

| feature | train cats | test cats | train-only | low<1k |
|---|---|---|---|---|
| top_bottom | 2 | 2 | 0 | 0 |
| game_type | 2 | 1 | 1 | 0 |
| base_state | 8 | 2 | 6 | 0 |
| season | 6 | 1 | 6 | 0 |
| game_month | 8 | 3 | 5 | 0 |
| game_dayofweek | 7 | 4 | 3 | 0 |
| inning | 13 | 3 | 10 | 1 |
| balls_before | 4 | 3 | 1 | 0 |
| strikes_before | 3 | 1 | 2 | 0 |
| outs_before | 3 | 3 | 0 | 0 |
| runner_on_1b | 2 | 2 | 0 | 0 |
| runner_on_2b | 2 | 1 | 1 | 0 |
| runner_on_3b | 2 | 1 | 1 | 0 |
| num_runners_on | 4 | 2 | 2 | 0 |

## 5. 결측 패턴 (asof 16종)

| feature | missing n | rate |
|---|---|---|
| asof_batter_middle_rate | 830 | 0.0006 |
| asof_batter_success_rate | 830 | 0.0006 |
| asof_pitcher_ball_rate | 792 | 0.0005 |
| asof_pitcher_breaking_rate | 792 | 0.0005 |
| asof_pitcher_fastball_rate | 792 | 0.0005 |
| asof_pitcher_middle_rate | 792 | 0.0005 |
| asof_pitcher_offspeed_rate | 792 | 0.0005 |
| asof_pitcher_prev1_game_middle_rate | 29,185 | 0.0198 |
| asof_pitcher_prev1_game_success_rate | 29,185 | 0.0198 |
| asof_pitcher_prev3_game_middle_rate | 29,185 | 0.0198 |
| asof_pitcher_prev3_game_success_rate | 29,185 | 0.0198 |
| asof_pitcher_prev5_game_middle_rate | 29,185 | 0.0198 |
| asof_pitcher_prev5_game_success_rate | 29,185 | 0.0198 |
| asof_pitcher_reverse_rate | 792 | 0.0005 |
| asof_pitcher_strike_rate | 792 | 0.0005 |
| asof_pitcher_success_rate | 792 | 0.0005 |

## 6. 콜드 스타트

- pitcher_debut(asof_pitcher_n==0): 0.0005 (n=1,475,092)
- batter_debut(asof_batter_n==0): 0.0006
- G1(prev1 결측): 0.0198

## 7. 가설 레지스트리 요약

- proposed(신규): 8 → season_dev_success_pitcher, league_trend_ratio_middle, usage_rank_pitcher, log1p_usage_pitcher, count_state_abs_regime, debut_league_impute_success, batter_debut_league_impute_success, f_game_post2023_label_mapping
- blocked(기각 반영): 9 → count_platoon_3b2_same, asof_n_bucket, score_diff_binary, recent_gap_success, return_gap, pitcher_debut, batter_debut, li_risp_flag, outs_count_3b2_2out

## 8. 검증

- 레지스트리 검증(--validate-registry): PASS (problems: 0)
- 스모크(--smoke): n/a
