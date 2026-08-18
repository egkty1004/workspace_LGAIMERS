# F2 — code-quality, temporal-leakage, data-scope audit — PASS (exit 0)

- **recorded_at_utc**: 2026-08-18T05:17:05+00:00
- **git_head**: 444a759f42bba255e94af1b01d928f6e5c84ddd6
- **config_hash**: `158be42401fc5ce3fec78af960900a4d7c040fa08d960a4849113608eeb4b420`
- **label_sources**: [] (labels_read=False)

## Checks

- **[PASS]** origin_leakage: 기원 마스크에 2025 시즌 없음 (누수 없음)
- **[PASS]** row_disjointness: r2022/r2023 disjoint; r2024 overlap 문서화: {"r2022": {"overlap_rows_with_primary": 0, "expected": true, "reason": "row-ID disjoint from primary"}, "r2023": {"overlap_rows_with_primary": 0, "expected": true, "reason": "row-ID disjoint from primary"}, "r2024": {"overlap_rows_with_primary": 133, "expected": true, "reason": "r2024 val = (season==2024) & R subset of primary val (overlapping diagnostic origin, branch-inert)"}}
- **[PASS]** deployed_formula: scoring=common.score(clip(sigmoid(z+C_LOGIT),.30,.70),y) C_LOGIT=-0.0461645795229729 clip=[0.3,0.7] — 불변
- **[PASS]** terminal_firewall: read_primary_labels 는 FROZEN 전 TerminalFirewallError — --screen 경로는 read_selection_labels 만 호출

## Findings

- **[PASS]** origin_masks: r2022/r2023/primary/r2024 마스크 정의 검증 (합성 데이터)
- **[PASS]** deployed_formula_immutable: 배포 산식 불변 — raw-logit Brier 는 진단 전용, 선택 키 아님
- **[PASS]** no_holdout_misrepresentation: primary 는 역사적으로 재사용된 스트레스 체크 — 독립 홀드아웃 아님

## Verdict: **PASS** (exit 0)

## Reviewer Review (F2 final-wave re-run)

- **overall_verdict**: **APPROVE** (branch NORMAL, git_head 444a759)
- **reviewer**: F2 final-wave reviewer (re-run after the 4 plan-named F2 fixtures were implemented)

### Review Checks

- **[PASS]** origin_masks: r2022=(train<=2021,R)->(2022,R); r2023=(train<=2022,R)->(2023,R); primary=(train<=2023,all)->(2024,all); r2024=(train<=2023,R)->(2024,R). Real-data verification: train max season 2024, zero 2025 rows in any val mask; r2022/r2023 row-ID disjoint from primary (0 overlap); r2024 overlap with primary documented branch-inert.
- **[PASS]** inner_outer_fit_boundary: inner selection uses od_type=Iter, od_wait=100, use_best_model=True and records each seed best_iteration+1 before outer labels read; outer refits use origin-specific frozen count with no eval set and use_best_model=False; inner val strictly before outer year; no outer labels leak into inner selection.
- **[PASS]** residual_causality: one-year-ahead OOF (train mask ends t-1), prior-OOF fit only, C selected before outer labels read, assert_inner_outer_disjoint raises on overlap, no target encoding, no terminal labels.
- **[PASS]** calibration_causality: each outer year Y fit solely on OOF years <Y, r2022 cold-start identity documented, z_candidate=logit(p_transform)-C_LOGIT preserves the deployed formula, no standalone outer/Public/test-fitted offset.
- **[PASS]** trackman_exclusion: zero Trackman imports in recovery source; the token appears only inside the _assert_no_trackman_import guard and its fixture.
- **[PASS]** test_row_exclusion: recovery source never reads the test data file (only the official train file); no test-row aggregation/order/sequence features; _assert_no_test_row_statistic guard rejects test-derived feature norms.
- **[PASS]** deployed_formula: common.score(np.clip(common.sigmoid(z + C_LOGIT), .30, .70), y) with C_LOGIT=-0.0461645795229729, clip [.30,.70]; git diff vs HEAD empty for the C_LOGIT/CLIP constants.

### QA Results (recorded exit codes)

- **[PASS]** recovery_evaluator.py --audit-quality → exit 0 (expected 0)
- **[PASS]** recovery_evaluator.py --fixture outer-label-fit → exit 2 (expected 2, matched=True)
- **[PASS]** recovery_evaluator.py --fixture test-row-statistic → exit 2 (expected 2, matched=True)
- **[PASS]** recovery_evaluator.py --fixture trackman-import → exit 2 (expected 2, matched=True)
- **[PASS]** recovery_evaluator.py --fixture altered-c-logit-clip → exit 2 (expected 2, matched=True)
- **[PASS]** recovery_evaluator.py --audit-quality --fixture outer-label-fit (combined) → exit 2 (expected 2, matched=True)
- **[PASS]** test_recovery_evaluator.py → exit 0 (273 PASS / 0 FAIL)

### Review Notes

The 4 plan-named F2 fixtures (outer-label-fit, test-row-statistic, trackman-import, altered-c-logit-clip) are now implemented as real guard-firing fixtures (exit 2) writing separate f2-fixture-<name>.{json,md} evidence with matched=True. The previous REJECT (f2_failure_fixtures_missing) is resolved. Review-only: no plan checkbox edit, no code/state change, no upload.
