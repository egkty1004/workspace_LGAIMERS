# Todo 6 — season-balanced training & nested slope-only calibration (compact report)

- **Plan**: `aimers9-top100-score-improvement` Todo 6
- **Runner**: `repro_979/calibration_runner.py`
- **Evidence**: `.omo/evidence/aimers9-top100/task-6-calibration.{json,log}` (+ `task-6-leakage-failure.log`)
- **OOF logits (Todo 7 inputs, git-excluded cache)**: `repro_979/cache/calibration/task6/{schedule}/{fold}.npy` + `result.json`
- **Run date**: 2026-08-14 | **git**: `6e33cc1` | 10 seeds (42..51) × 4 folds × 4 schedules × 3 methods, ~29 min (1763 s)

## Protocol

- Frozen champion contract reused from Task 2 (`qualification_runner`): 49-feature `CHAMPION_FEATURES`,
  `LGB_CATS`(5), fold masks == `screen_all_10seed.py:build_folds`, `C_LOGIT = -0.0404` (b pinned — never fitted).
- **Nested design (leak-free)**: for each outer fold (primary/r2022/r2023/r2024), the last season of the
  training window is the `inner_val` (fitting/selection set only); the scored fold is never touched.
  Self-checks assert `inner_val ∩ outer_val = ∅`, inner splits ⊆ outer window, no 2025 rows anywhere (4/4 PASS).
- **Training**: LGB on the full outer window with schedule weights, **fixed 109 rounds**
  (`train_meta.json` documented `num_boost_round`; no early stopping → no parameter selected on any fold).
  Measured necessity: training on window-minus-last-season collapses outer BSS to 0 (drift), so the last
  season is reserved strictly for calibration/selection while the model trains on the full window.
- **Schedule selection**: argmax `inner_val` raw BSS per fold (never the scored fold).
- **Slope-only calibration**: fit `a` only (Brier-min grid, b=C_LOGIT pinned) on `inner_val` logits.
- Champion reference: raw blend BSS from `cache/qualification/champion/*.npy` (cross-checked to `result.json`).

## Season-weight schedules (predeclared constants, formula per `SEASON_SCHEDULES`)

| schedule | formula (t = (s−s_min)/(s_max−s_min)) |
|---|---|
| `uniform` | w(s)=1.0 (champion-equivalent baseline) |
| `linear_recent` | w(s)=1+t → [1.0, 2.0] |
| `exp_recent` | w(s)=2^t → [1.0, 2.0] |
| `regime_step` | w(s)=0.5 if s<mid else 1.5, mid=(s_min+s_max)/2 |

## Outer-fold BSS (10-seed logit mean; champion blend = LGB×MLP w=0.51)

| treatment | primary | r2022 | r2023 | r2024 |
|---|---|---|---|---|
| **uniform raw** | 724.8 | **574.6** | **526.3** | **706.6** |
| linear_recent raw | 714.4 | 556.5 | 520.6 | 692.4 |
| exp_recent raw | 715.5 | 555.8 | 518.0 | 682.7 |
| regime_step raw | 706.9 | 560.3 | 523.2 | 679.1 |
| uniform policy (a=1,b=-0.0404) | **777.8** | 559.9 | 466.2 | **762.6** |
| uniform slope_only | 718.3 | 499.3 | 345.8 | 692.4 |
| champion blend (reference) | 774.2 | 607.2 | 571.8 | 724.1 |
| champion lgb (reference) | 729.7 | 578.4 | 551.0 | 711.4 |

Full 12-treatment table in the evidence JSON (`per_fold.*.treatments.*.methods.*`).

## Verdicts (acceptance gate, 10-seed)

Gate: all 3 R-only ΔBSS vs champion raw blend > +1.0 AND max|Δmean| ≤ 0.005 → **accepted**, else rejected with reasons.

- **Accepted: none (0/12).** Uniform raw is closest but R-only deltas are −32.6 / −45.5 / −17.5 vs blend.
- All `policy`/`slope_only` also fail mean-alignment (|Δmean| ≈ 0.010 > 0.005) — the pinned intercept
  deliberately shifts the mean toward the 2025 target (0.477), which is a submission-policy behavior, not a
  validation mean-preserving transform.

## Findings

1. **Season weighting does not help.** `uniform` (no weighting) is the best schedule on all 4 outer folds.
   The nested selector (inner raw BSS argmax) picked `exp_recent`/`regime_step` — the *worst* outer schedules —
   because recent-emphasis schedules overfit the in-sample last season. Inner-split selection here is
   anti-correlated with outer generalization (drift makes the last season an unreliable proxy for the next year).
2. **Frozen policy (C_LOGIT alignment) is the strongest calibration** on primary (+53.0) and r2024 (+56.0),
   but hurts r2022 (−14.7) / r2023 (−60.1): C_LOGIT was tuned to the 2025 target mean (r≈0.477); only
   primary/r2024 rates sit near that target. Net effect is fold-dependent.
3. **Nested slope-only is non-identifiable and degrades BSS.** Brier profile over `a∈[0.6,1.4]` is flat
   (range ≈ 0.0017–0.0023, `flat_non_identifiable=true` for every fold/schedule) because inner-val predictions
   come from a model trained on that same season (in-window). The fitted `a` hits the grid edge (1.4), and
   applying it drops r2023 to 345.8 (raw 526.3). The leak-free inner split cannot identify a reliability slope.
4. **Fixed-109 cost vs champion LGB**: uniform raw is 4.9/3.8/24.7/4.8 BSS below the champion LGB per fold —
   the price of replacing scored-fold early stopping with the documented fixed-round constant (largest on r2023).
5. **Mean-shift guard** behaves as designed: raw methods preserve mean (Δmean≈0), calibration methods shift it
   ~0.010, enforcing the 0.005 alignment bound.

## Failure QA (captured in `task-6-leakage-failure.log`)

`python3 repro_979/calibration_runner.py --leakage-injection` → injects the scored fold's rows into slope
fitting → built-in guard raises `LeakageError`, **exit code 2** (nonzero), message:
`[LEAKAGE] fold=primary: 캘리브레이션/선택 피팅 셋에 스코어 폴드 행 포함 ...`.

## Reproduce

```bash
python3 repro_979/calibration_runner.py --smoke          # PASS, ~1 min
python3 repro_979/calibration_runner.py                  # full grid, ~29 min
python3 repro_979/calibration_runner.py --leakage-injection   # exit 2 (leakage failure QA)
```

Outputs: evidence JSON/log under `.omo/evidence/aimers9-top100/`; OOF logits (raw z, mean-of-seed) under
`repro_979/cache/calibration/task6/{schedule}/{fold}.npy` — calibration re-applied from `result.json` a/b.
No treatment accepted → no promoted candidate to Todo 7; champion control remains the Todo 7 baseline.
