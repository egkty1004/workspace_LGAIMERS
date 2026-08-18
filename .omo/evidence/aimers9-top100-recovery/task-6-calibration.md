# Todo 6 — screen causal Brier calibration transforms — REJECT (exit 0)

- **recorded_at_utc**: 2026-08-18T01:22:02+00:00
- **git_head**: f9e447fe904adf6c826b459ea160f321725c7fbc
- **config_hash**: `a8d344c7a98fd9571c556c322c9f428dbb6ab940484d8951d08c8c19c96a84e5`
- **label_sources**: ['r2022', 'r2023'] (labels_read=True)

## Checks

- **[PASS]** registry_valid: calibration_beta/calibration_isotonic 허용, identity control
- **[PASS]** origin_masks: r2022/r2023 마스크, 2025 부재, primary 와 row-ID 분리
- **[PASS]** selection_labels_only: r2022/r2023 선택 라벨만 — primary/터미널 미로드
- **[PASS]** frozen_baseline_oof: 동결 v93 베이스라인 OOF 로짓 로드 (bounded 30000 행/기원, 캐시 키 ee8ea9579969deaa…)
- **[PASS]** screen_calibration_beta: verdict=REJECT violations=['r2022: ΔBSS=+0.0000(>1.0?False) Brier 0.245175<0.245175?False LB5=+0.0000(>0?False) mean_shift=0.000000(<=0.005?True) finite=True', 'r2023: ΔBSS=+239.4435(>1.0?True) Brier 0.244342<0.244940?True LB5=+104.2242(>0?True) mean_shift=0.007101(<=0.005?False) finite=True']
- **[PASS]** screen_calibration_isotonic: verdict=REJECT violations=['r2022: ΔBSS=+0.0000(>1.0?False) Brier 0.245175<0.245175?False LB5=+0.0000(>0?False) mean_shift=0.000000(<=0.005?True) finite=True', 'r2023: ΔBSS=+164.5563(>1.0?True) Brier 0.244529<0.244940?True LB5=+29.1502(>0?True) mean_shift=0.006351(<=0.005?False) finite=True']
- **[PASS]** screen_calibration_identity: verdict=REJECT violations=['r2022: ΔBSS=+0.0000(>1.0?False) Brier 0.245175<0.245175?False LB5=+0.0000(>0?False) mean_shift=0.000000(<=0.005?True) finite=True', 'r2023: ΔBSS=+0.0000(>1.0?False) Brier 0.244940<0.244940?False LB5=+0.0000(>0?False) mean_shift=0.000000(<=0.005?True) finite=True']

## Violations

- **[FAIL]** screen_calibration_beta: r2022: ΔBSS=+0.0000(>1.0?False) Brier 0.245175<0.245175?False LB5=+0.0000(>0?False) mean_shift=0.000000(<=0.005?True) finite=True
- **[FAIL]** screen_calibration_beta: r2023: ΔBSS=+239.4435(>1.0?True) Brier 0.244342<0.244940?True LB5=+104.2242(>0?True) mean_shift=0.007101(<=0.005?False) finite=True
- **[FAIL]** screen_calibration_isotonic: r2022: ΔBSS=+0.0000(>1.0?False) Brier 0.245175<0.245175?False LB5=+0.0000(>0?False) mean_shift=0.000000(<=0.005?True) finite=True
- **[FAIL]** screen_calibration_isotonic: r2023: ΔBSS=+164.5563(>1.0?True) Brier 0.244529<0.244940?True LB5=+29.1502(>0?True) mean_shift=0.006351(<=0.005?False) finite=True
- **[FAIL]** screen_calibration_identity: r2022: ΔBSS=+0.0000(>1.0?False) Brier 0.245175<0.245175?False LB5=+0.0000(>0?False) mean_shift=0.000000(<=0.005?True) finite=True
- **[FAIL]** screen_calibration_identity: r2023: ΔBSS=+0.0000(>1.0?False) Brier 0.244940<0.244940?False LB5=+0.0000(>0?False) mean_shift=0.000000(<=0.005?True) finite=True

## Findings

- **[PASS]** time_causal_fitting: 각 outer year Y 의 변환은 OOF 연도 < Y 에서만 피팅 후 1회 적용; r2022(Y=2022) 는 prior OOF 부재 cold-start identity (문서화)
- **[PASS]** frozen_formula_preserved: 배포 산식 불변 — z_candidate=logit(p_transform)-C_LOGIT, clip(sigmoid(z_candidate+C_LOGIT),.30,.70)
- **[PASS]** no_global_offset_retune: C_LOGIT/clip 상수 불변 — 전역 오프셋 재튜닝 없음
- **[PASS]** no_terminal_read: 스크린 경로는 r2022/r2023 선택 라벨만 — primary/터미널 미로드

## Verdict: **REJECT** (exit 0)
