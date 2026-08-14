# Todo 7b — CatBoost-weight blend sweep (compact report)

- **Plan**: `aimers9-top100-score-improvement` Todo 7b (extension of Todo 7 accepted blends)
- **Runner**: `repro_979/blend_weight_sweep.py`
- **Evidence**: `.omo/evidence/aimers9-top100/task-7b-sweep.{json,log}`
- **Run date**: 2026-08-14 | **Selection fold**: `primary` (2024 validation) | **Held-out**: r2022/r2023/r2024 | 725 s

## Protocol

- **Inputs (qualified OOF matrices, sha256-verified)**: champion blend + LGB/MLP components
  (Task 2 `task-2-control.json` cache_provenance), catboost family (Task 5 `task-5-models-catboost.json`). No training — pure logit arithmetic.
- **Weight selection**: pre-specified grid points on **primary only** (Grid B refine anchors = top coarse by SELECTION-fold BSS). Held-out labels are used
  ONLY for evaluating fixed weights (leakage assertion in evidence).
- **Transfer gate (same thresholds as Todo 7)**: (a) no held-out labels in selection;
  (b) ΔBSS > +1.0 on all 3 R-only folds; (c) max|Δmean| ≤ 0.005; (d) max member corr < 0.99 or residual align ≥ 0.02; (e) pooled bootstrap (1000, seed 42) 5% lower bound > 0.
- **Candidate ID**: `sha256(canonical{{members, weights, selection_fold, seed, step}})[:16]`
  (mirrors `blend_selector._candidate_id`).

## Grids

| grid | members | ranges | step | n_points |
|---|---|---|---|---|
| A (champ_cat family) | champion × catboost | champion 0.80..0.95 (catboost 0.05..0.20) | 0.01 | 16 |
| B (lgb_mlp_cat family) | lgb × mlp × catboost | lgb 0.20..0.40, mlp 0.35..0.55, catboost 0.15..0.35 | 0.05 coarse + 0.02 refine | 46 |
| C (lgb==mlp) | lgb × mlp × catboost | lgb==mlp=t, catboost=1-2t, t∈[0.325,0.425] | 0.01 | 11 |

### Grid A — champ_cat family — top-3 by held-out bootstrap LB (5%)

| rank | weights | selBSS | Δr2022 | Δr2023 | Δr2024 | boot LB5% | p50 | p95 | max\|Δmean\| | verdict | candidate_id |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `champion×0.80 / catboost×0.20` | 775.67 | +1.63 | +2.47 | +7.03 | +2.46 | +3.73 | +5.10 | 0.0002 | accepted | `93ecbe13737b1fcd` |
| 2 | `champion×0.81 / catboost×0.19` | 775.71 | +1.61 | +2.39 | +6.75 | +2.39 | +3.60 | +4.90 | 0.0002 | accepted | `84576302b246322c` |
| 3 | `champion×0.82 / catboost×0.18` | 775.73 | +1.58 | +2.30 | +6.46 | +2.32 | +3.46 | +4.70 | 0.0001 | accepted | `fc47f5de4dcaa93e` |

### Grid B — lgb_mlp_cat family — top-3 by held-out bootstrap LB (5%)

| rank | weights | selBSS | Δr2022 | Δr2023 | Δr2024 | boot LB5% | p50 | p95 | max\|Δmean\| | verdict | candidate_id |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `lgb×0.30 / mlp×0.35 / catboost×0.35` | 774.72 | +1.98 | +3.80 | +9.90 | +3.07 | +5.25 | +7.55 | 0.0001 | accepted | `5890a4c54f502c4e` |
| 2 | `lgb×0.25 / mlp×0.40 / catboost×0.35` | 775.75 | +2.73 | +4.35 | +8.85 | +3.04 | +5.30 | +7.60 | 0.0002 | accepted | `6a65397689834ca1` |
| 3 | `lgb×0.30 / mlp×0.40 / catboost×0.30` | 776.02 | +2.58 | +3.88 | +8.41 | +3.01 | +4.95 | +6.93 | 0.0001 | accepted | `8696654766e1e1b1` |

### Grid C — lgb==mlp — top-3 by held-out bootstrap LB (5%)

| rank | weights | selBSS | Δr2022 | Δr2023 | Δr2024 | boot LB5% | p50 | p95 | max\|Δmean\| | verdict | candidate_id |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `lgb×0.34 / mlp×0.34 / catboost×0.31` | 774.66 | +1.66 | +3.27 | +9.51 | +2.87 | +4.83 | +6.93 | 0.0002 | accepted | `704df31eaa2fead7` |
| 2 | `lgb×0.35 / mlp×0.35 / catboost×0.29` | 774.99 | +1.74 | +3.19 | +9.11 | +2.86 | +4.69 | +6.66 | 0.0002 | accepted | `5745ec8e1c587703` |
| 3 | `lgb×0.34 / mlp×0.34 / catboost×0.33` | 774.28 | +1.56 | +3.35 | +9.89 | +2.85 | +4.95 | +7.18 | 0.0002 | accepted | `241b841604c12db5` |

## Accepted-blend references (Todo 7, fixed step-0.05 weights)

| blend | weights | boot LB5% | Δr2022 | Δr2023 | Δr2024 |
|---|---|---|---|---|---|
| `champ_cat` | champion×0.85 / catboost×0.15 | +2.06 | +1.45 | +2.00 | +5.56 |
| `lgb_mlp_cat` | lgb×0.30 / mlp×0.45 / catboost×0.25 | +2.67 | +2.81 | +3.72 | +6.51 |

## Recommendation

- **Recommended (beats BOTH accepted blends on held-out bootstrap LB AND passes the gate)**: `6a65397689834ca1`, `b69a93c2c4df170e`, `5890a4c54f502c4e`, `8696654766e1e1b1`, `3ff2174d3aa392fd`, `968b7ecb213b37a1`, `a6ca7b11fce30335`, `08e2e43015c82592`, `4921f73ebf8e29d7`, `92c86bf865a8b690`, `f399354cedbae7a9`, `1999fef8d12b936f`, `fd2aa043f471a6ed`, `1507a426836dbe9a`, `237e5721cf0eb14f`, `3ee6126c8f8ecf36`, `91ec87a6b33e2db7`, `dd91d05e7a78349a`, `241b841604c12db5`, `704df31eaa2fead7`, `5745ec8e1c587703`, `adc0d5f0171a52d0`, `9cb8c78d70a1b1ff`, `97b23a028ee0ae38`
- **Best accepted per family**: `grid_a`→`93ecbe13737b1fcd` (champion×0.80 / catboost×0.20, LB +2.46), `grid_b`→`5890a4c54f502c4e` (lgb×0.30 / mlp×0.35 / catboost×0.35, LB +3.07), `grid_c`→`704df31eaa2fead7` (lgb×0.34 / mlp×0.34 / catboost×0.31, LB +2.87)
- **Gate totals**: accepted 69/73 sweep candidates.

### Overall top accepted (held-out bootstrap LB, all families)

| rank | family | weights | selBSS | Δr2022 | Δr2023 | Δr2024 | boot LB5% | p50 | p95 | candidate_id |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | grid_b | `lgb×0.30 / mlp×0.35 / catboost×0.35` | 774.72 | +1.98 | +3.80 | +9.90 | +3.07 | +5.25 | +7.55 | `5890a4c54f502c4e` |
| 2 | grid_b | `lgb×0.25 / mlp×0.40 / catboost×0.35` | 775.75 | +2.73 | +4.35 | +8.85 | +3.04 | +5.30 | +7.60 | `6a65397689834ca1` |
| 3 | grid_b | `lgb×0.30 / mlp×0.40 / catboost×0.30` | 776.02 | +2.58 | +3.88 | +8.41 | +3.01 | +4.95 | +6.93 | `8696654766e1e1b1` |
| 4 | grid_b | `lgb×0.28 / mlp×0.43 / catboost×0.29` | 776.33 | +2.85 | +4.02 | +7.54 | +2.89 | +4.80 | +6.70 | `a6ca7b11fce30335` |
| 5 | grid_b | `lgb×0.27 / mlp×0.43 / catboost×0.30` | 776.29 | +2.88 | +4.12 | +7.63 | +2.89 | +4.88 | +6.85 | `3ee6126c8f8ecf36` |

## Reproduce

```bash
python3 repro_979/blend_weight_sweep.py --smoke    # PASS, coarse 3 pts/family
python3 repro_979/blend_weight_sweep.py            # full sweep, ranked top-3/family
python3 repro_979/blend_weight_sweep.py --report-only  # regenerate REPORT from evidence
```
