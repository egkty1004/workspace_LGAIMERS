# Todo 7 — held-out ensemble selection & transfer gate (compact report)

- **Plan**: `aimers9-top100-score-improvement` Todo 7
- **Runner**: `repro_979/blend_selector.py`
- **Evidence**: `.omo/evidence/aimers9-top100/task-7-blend.{json,log}` (+ `task-7-blend-smoke.*`, `task-7-blend-negative.*`, `task-7-leakage-failure.log`)
- **Inputs (qualified OOF matrices, git-excluded cache)**: champion blend + LGB/MLP components (Task 2),
  xgboost/catboost families (Task 5), 8 feature candidates (Task 4), 4 calibration schedules (Task 6)
- **Run date**: 2026-08-14 | **Selection fold**: `primary` (2024 validation period) | **Held-out**: r2022/r2023/r2024 | ~315 s

## Protocol

- **Selection fold = `primary`** — the 2024 validation period; the same fold where the champion's own
  `w_lgb=0.51` was originally fit (e6c). Weights are fit **only** there; fixed weights are evaluated on the
  held-out R-only folds. Held-out labels never touch fitting (structural guard + failure QA).
- **Weight fit**: numpy simplex grid `w≥0, Σw=1`, step 0.05, argmax selection-fold BSS (scipy-free, deterministic).
- **Deterministic candidate ID**: `sha256(canonical{members, weights, selection_fold, seed=42, step})[:16]`.
- **Transfer gate (all must hold)**: (a) no held-out labels in fit artifact; (b) ΔBSS > +1.0 on **all 3**
  R-only folds; (c) max|Δmean| ≤ 0.005; (d) max member logit corr < 0.99 or residual alignment ≥ 0.02;
  (e) pooled bootstrap (1000, seed 42) lower bound (5%) > 0. Standalone BSS alone never promotes.

## Member manifest (17/17 registered)

| member | task | digest source | note |
|---|---|---|---|
| champion / lgb / mlp | 2 | task-2-control evidence (frozen) | OK |
| xgboost | 5 | **task-5 smoke evidence (2-seed)** — full 10-seed OOF was overwritten by a later `--smoke` run (issues.md Task 5) | registered w/ caveat |
| catboost | 5 | task-5 full evidence (10-seed) | OK |
| task4_{8 candidates} | 4 | per-candidate result.json | OK, all rejected at gate |
| task6_{4 schedules} | 6 | on-disk (no evidence digest recorded) | OK, different training protocol (fixed-109) |

## Candidate results (selection-fold fit → fixed-weight held-out eval; Δ = blend − champion)

| candidate | fitted weights | selBSS | Δr2022 | Δr2023 | Δr2024 | boot LB5% | max\|Δmean\| | verdict |
|---|---|---|---|---|---|---|---|---|
| `lgb_mlp` (recovery) | lgb .50 / mlp .50 | 774.14 | +0.1 | +0.1 | −0.2 | −0.13 | 0.0001 | rejected |
| `champ_xgb` | champ 1.00 | 774.16 | +0.0 | +0.0 | +0.0 | 0.00 | 0.0000 | rejected |
| **`champ_cat`** | **champ .85 / cat .15** | **775.75** | **+1.5** | **+2.0** | **+5.6** | **+2.06** | **0.0001** | **accepted** |
| `champ_xgb_cat` | champ .85 / xgb .00 / cat .15 | 775.75 | +1.5 | +2.0 | +5.6 | +2.06 | 0.0001 | accepted (redundant = champ_cat) |
| `lgb_mlp_xgb` | lgb .45 / mlp .45 / xgb .10 | 774.15 | +0.4 | +0.9 | −1.2 | −0.29 | 0.0001 | rejected |
| **`lgb_mlp_cat`** | **lgb .30 / mlp .45 / cat .25** | **776.46** | **+2.8** | **+3.7** | **+6.5** | **+2.67** | **0.0003** | **accepted** |
| champ_{task4 × 8} | champ 1.00 | 774.16 | +0.0 | +0.0 | +0.0 | 0.00 | 0.0000 | rejected (degenerate) |
| champ_uniform / linear / regime | champ 1.00 | 774.16 | +0.0 | +0.0 | +0.0 | 0.00 | 0.0000 | rejected (degenerate) |
| champ_exp_recent | champ .90 / exp .10 | 774.17 | −0.3 | +0.3 | +0.7 | −0.24 | 0.0002 | rejected |

- **Recovery check**: `lgb_mlp` re-fits to `w_lgb=0.50` (grid step) ≈ champion `0.51` → held-out Δ≈0, corr 1.000.
  The selection protocol correctly reproduces the champion — a protocol sanity check.
- **All Task 4 / most Task 6 members collapse to `w=1.00` on the selection fold** (they add nothing at primary);
  degenerate Δ=0 blends are correctly rejected by gates (b)/(e).
- **Clipping diagnostic**: every candidate had 0.00000 rows hitting 0.30/0.70 bounds pre/post (predictions
  concentrate near 0.49–0.51); C_LOGIT=-0.0404 is a submission-alignment constant, not applied to validation BSS.

## Findings

1. **CatBoost is the only member with real held-out complementarity.** Both accepted blends are the ones
   giving catboost weight (0.15 / 0.25). This matches Task 5's blend-probe observation (catboost blend gains
   +1.6~+12.0) — but Todo 7 validates it honestly: weight fit on `primary` only, then fixed-weight R-only eval.
   Task 5's rejection of catboost *standalone* (no R-only transfer) is not contradicted — catboost alone still
   doesn't transfer; it only helps *blended* with the champion.
2. **Two independent accepted candidates, one redundant.**
   - `champ_cat` (champion×0.85 + catboost×0.15) — candidate_id `691a2947b2b1879c`
   - `lgb_mlp_cat` (lgb×0.30 + mlp×0.45 + catboost×0.25) — candidate_id `7771a019594deedd` (best held-out deltas)
   - `champ_xgb_cat` = duplicate of `champ_cat` (xgboost weight 0.00) — flagged `redundant`, not a real pick.
3. **xgboost contributed nothing** in this run partly because its on-disk OOF is the 2-seed *smoke* artifact
   (full 10-seed OOF was overwritten — known Task 5 issue), and partly because even Task 5's 10-seed xgboost
   was the least complementary family (corr 0.958–0.979, no blend gains).
4. **Deployment caveat for Todo 8**: both accepted blends require **catboost** at inference. Task 5 recorded
   catboost as `install_risk` (not in the evaluation environment's default package list). Todo 8 must verify
   offline `requirements.txt` install + runtime (10-min limit) before materializing; otherwise fall back to the
   champion control.
5. **The protocol is honest**: 15/18 rejected, including all degenerate (Δ=0) and the recovery blend (Δ≈0);
   nothing was promoted on standalone BSS.

## Failure QA (all captured in evidence)

| scenario | command | expected | actual |
|---|---|---|---|
| smoke | `python3 repro_979/blend_selector.py --smoke` | exit 0 PASS | exit 0, PASS (~12 s) |
| leakage | `python3 repro_979/blend_selector.py --leakage-injection` | nonzero (LeakageError) | **exit 2** — fit on r2022 blocked |
| negative transfer | `python3 repro_979/blend_selector.py --force-negative-transfer` | all rejected, nonzero | **exit 1** — 0/18 accepted (evidence isolated in `task-7-blend-negative.*`) |

## Reproduce

```bash
python3 repro_979/blend_selector.py --smoke                     # PASS, ~12 s
python3 repro_979/blend_selector.py                             # full 18-candidate run, ~315 s
python3 repro_979/blend_selector.py --leakage-injection         # exit 2 (leakage guard)
python3 repro_979/blend_selector.py --force-negative-transfer   # exit 1 (gate rejects all)
```

Outputs: `.omo/evidence/aimers9-top100/task-7-blend{,-smoke,-negative}.{json,log}` + `task-7-leakage-failure.log`.

## Todo 8 handoff

- **Selected blend candidates**: `champ_cat` (candidate_id `691a2947b2b1879c`) and `lgb_mlp_cat`
  (`7771a019594deedd`) passed the transfer gate on fixed held-out weights.
- **Todo 8 must** (1) verify offline catboost installability in the eval environment (10-min inference budget,
  245,789 rows); (2) if infeasible, **fall back to the champion control** (`w_lgb=0.51`, C_LOGIT=-0.0404) —
  the local deltas (+1.5~+6.5 BSS) are not worth a deployment failure.
