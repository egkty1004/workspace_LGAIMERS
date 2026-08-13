# REPORT — Todo 4: 드리프트 인지 전처리/피처 후보 게이팅

- 실행: 2026-08-13T23:13:19.604586+00:00 | git: 57eb10dc828ff21c784c87d8613bf7cc26aaaa5a | seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51] | smoke: False
- 게이트: primary Δ ≥ +15 & R-only 3/3 & max|Δmean| ≤ 0.005 & bootstrap primary Δ 5%하한 > 0 → 채택, 그 외 기각

## 기준선 (동결 49피처 LGB 10시드 × 4폴드)

| fold | n_rows | BSS | Task2 문서화 | Δ | pred_mean |
|---|---|---|---|---|---|
| primary | 253507 | 729.680 | 729.680 | +0.000 | 0.4973 |
| r2022 | 217024 | 578.409 | 578.409 | +0.000 | 0.5073 |
| r2023 | 219839 | 550.953 | 550.953 | +0.000 | 0.5033 |
| r2024 | 223497 | 711.398 | 711.397 | +0.000 | 0.5009 |

- 기준선 assert: within_tolerance=True (max|Δ| 0.000, 허용 ±1.0)
- 기준선+MLP 블렌드 primary 재현: 챔피언 774.159 → 774.159 (Δ +0.000)

## 후보 판정

| candidate | primary Δ | R-only | maxΔmean | boot 5% | blend Δ | verdict | reasons |
|---|---|---|---|---|---|---|---|
| season_dev_success_pitcher | -3.9 | 1/3 | 0.0007 | -8.26 | -0.96 | rejected | primary -3.9 < +15; R-only 1/3; bootstrap primary Δ 5%하한 -8.260 ≤ 0 |
| league_trend_ratio_middle | -2.7 | 2/3 | 0.0003 | -7.06 | -0.98 | rejected | primary -2.7 < +15; R-only 2/3; bootstrap primary Δ 5%하한 -7.058 ≤ 0 |
| usage_rank_pitcher | -4.8 | 2/3 | 0.0003 | -8.88 | +0.54 | rejected | primary -4.8 < +15; R-only 2/3; bootstrap primary Δ 5%하한 -8.881 ≤ 0 |
| log1p_usage_pitcher | -5.5 | 3/3 | 0.0003 | -9.52 | +0.59 | rejected | primary -5.5 < +15; bootstrap primary Δ 5%하한 -9.520 ≤ 0 |
| count_state_abs_regime | -5.1 | 3/3 | 0.0005 | -9.37 | +0.18 | rejected | primary -5.1 < +15; bootstrap primary Δ 5%하한 -9.365 ≤ 0 |
| debut_league_impute_success | -4.1 | 1/3 | 0.0005 | -8.10 | -0.54 | rejected | primary -4.1 < +15; R-only 1/3; bootstrap primary Δ 5%하한 -8.095 ≤ 0 |
| batter_debut_league_impute_success | -10.1 | 1/3 | 0.0005 | -14.50 | -2.00 | rejected | primary -10.1 < +15; R-only 1/3; bootstrap primary Δ 5%하한 -14.497 ≤ 0 |
| f_game_post2023_label_mapping | -7.0 | 0/3 | 0.0002 | -11.39 | -0.58 | rejected | primary -7.0 < +15; R-only 0/3; bootstrap primary Δ 5%하한 -11.394 ≤ 0 |

## 요약: 8 후보 중 채택 0

참고: OOF 로짓은 `repro_979/cache/qualification/task4_*` (git 제외). 채택 후보는 Todo 7 홀드아웃 블렌드 선택 입력.
