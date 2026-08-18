# Todo 5 — screen causal OOF residual-ridge correction — REJECT (exit 0)

- **recorded_at_utc**: 2026-08-18T01:25:12+00:00
- **git_head**: f9e447fe904adf6c826b459ea160f321725c7fbc
- **config_hash**: `158be42401fc5ce3fec78af960900a4d7c040fa08d960a4849113608eeb4b420`
- **label_sources**: ['r2022', 'r2023'] (labels_read=True)

## Checks

- **[PASS]** origin_masks: r2022/r2023/primary/r2024 마스크 + row-ID 분리 검증
- **[PASS]** base_oof_causal: one-year-ahead base OOF (train mask ends t-1), R-only, bounded 30000 rows/year, 경량 LGB 프록시 (스크린 단순화)
- **[PASS]** correction_config_before_outer_labels_r2022: C=0.01 선택은 outer(r2022) 라벨 로드 이전에 완료 (fit_years=[2020, 2021])
- **[PASS]** inner_outer_disjoint_r2022: inner-OOF/outer 마스크 분리 + 해시 (outer_year=2022)
- **[PASS]** correction_cap_r2022: correction cap = ±0.05 (z_corrected clip)
- **[PASS]** residual_complementarity_r2022: corr(correction, base_residual) = +0.0236
- **[PASS]** correction_config_before_outer_labels_r2023: C=0.001 선택은 outer(r2023) 라벨 로드 이전에 완료 (fit_years=[2020, 2021, 2022])
- **[PASS]** inner_outer_disjoint_r2023: inner-OOF/outer 마스크 분리 + 해시 (outer_year=2023)
- **[PASS]** correction_cap_r2023: correction cap = ±0.05 (z_corrected clip)
- **[PASS]** residual_complementarity_r2023: corr(correction, base_residual) = +0.0308

## Violations

- **[FAIL]** screen_gate: r2022: ΔBSS=-1929.7791(>1.0?False) Brier 0.258996<0.245175?False LB5=-2086.4445(>0?False) mean_shift=0.097017(<=0.005?False) finite=True
- **[FAIL]** screen_gate: r2023: ΔBSS=-1884.7226(>1.0?False) Brier 0.262810<0.244940?False LB5=-2060.1910(>0?False) mean_shift=0.108999(<=0.005?False) finite=True

## Findings

- **[PASS]** no_primary_read: --screen 경로는 read_selection_labels 만 호출 — primary 라벨 구조적으로 미로드 (파이어월)
- **[PASS]** no_target_encoding: 보정 피처는 [base_logit,balls_before,strikes_before,outs_before] 표준화 + [game_type,platoon,base_state] one-hot 만 — target encoding/타깃 파생 그룹/터미널 라벨 미사용
- **[PASS]** bounded_subset_consistency: Task 3 과 동일 bounded 30000 행 부분집합 사용 (문서화된 편차)

## Verdict: **REJECT** (exit 0)
