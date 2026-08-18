# Todo 4 — screen bounded recovery CatBoost variants — REJECT (exit 0)

- **recorded_at_utc**: 2026-08-18T01:51:43+00:00
- **git_head**: 7bd68d9bf3697cecd2912b1c085e35e4a08607b2
- **config_hash**: `158be42401fc5ce3fec78af960900a4d7c040fa08d960a4849113608eeb4b420`
- **label_sources**: ['r2022', 'r2023'] (labels_read=True)

## Checks

- **[PASS]** futility_catboost_c2_lossguide: mean ΔBSS vs C1=+1.1880 survives=True
- **[PASS]** futility_catboost_c3_ordered: mean ΔBSS vs C1=+0.3928 survives=False
- **[PASS]** futility_catboost_c4_rmse: mean ΔBSS vs C1=-6.2595 survives=False
- **[FAIL]** full_catboost_c2_lossguide: verdict=REJECT violations=['r2022: ΔBSS=-95.3517(>1.0?False) Brier 0.245413<0.245175?False LB5=-106.4481(>0?False) mean_shift=0.001487(<=0.005?True) finite=True', 'r2023: ΔBSS=-95.9618(>1.0?False) Brier 0.245180<0.244940?False LB5=-108.6261(>0?False) mean_shift=0.001004(<=0.005?True) finite=True']

## Violations

- **[FAIL]** futility_catboost_c3_ordered: mean ΔBSS vs C1=+0.3928 — futility REJECT (필요 mean>0 & 각 기원>-1.0)
- **[FAIL]** futility_catboost_c4_rmse: mean ΔBSS vs C1=-6.2595 — futility REJECT (필요 mean>0 & 각 기원>-1.0)
- **[FAIL]** full_catboost_c2_lossguide: full screen REJECT: ['r2022: ΔBSS=-95.3517(>1.0?False) Brier 0.245413<0.245175?False LB5=-106.4481(>0?False) mean_shift=0.001487(<=0.005?True) finite=True', 'r2023: ΔBSS=-95.9618(>1.0?False) Brier 0.245180<0.244940?False LB5=-108.6261(>0?False) mean_shift=0.001004(<=0.005?True) finite=True']

## Findings

- **[PASS]** selection_labels_only: r2022/r2023 선택 라벨만 읽음 — primary 미로드 (구조적 파이어월)
- **[PASS]** bounded_subset_consistency: baseline 캐시와 동일한 30000 행 부분집합 사용 — 공정 비교
- **[PASS]** reconciliation_v93: Cat 후보는 v93 6-레그 블렌드의 CatBoost 레그만 대체

## Verdict: **REJECT** (exit 0)
