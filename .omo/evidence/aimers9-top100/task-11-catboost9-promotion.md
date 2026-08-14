# Task 11 — catboost9 promotion decision

## Decision

**REJECTED.** The two-seed screen fails, so the conditional blend-transfer probe was not run and catboost9 must **not** proceed to 10-seed full-data training or Todo 7-8 packaging.

The decisive number is primary BSS: catboost9 scores **617.7551** versus **732.7601** for the frozen 5-cat CatBoost control, a **-115.0050** delta against an allowed floor of -10. Raw logits are decorrelated, but this is not useful residual diversity: prediction-error correlation with champion is **0.999203** and Task-5 residual alignment is only **0.003761**.

## Protocol and alignment

- Pure arithmetic on existing OOF logits; no model training.
- Labels use `common.load_train()` and `qualification_runner.build_folds`.
- BSS uses raw sigmoid; `C_LOGIT` is not applied.
- Catboost9 cache-config digest: `a41e9b220d8b12c6a74138c3118b21c162e7c4c73493b2680be98f513d056ea0`.
- Every catboost9 file SHA256 matches its `.meta.json`; champion/LGB/MLP/5-cat hashes match Task-2/Task-5 evidence.
- Every OOF vector equals the frozen validation-mask size. The packed-mask hashes below freeze the exact masks used for label extraction.

| fold | rows | packed validation-mask SHA256 | catboost9 logits SHA256 | digest |
|---|---:|---|---|---|
| primary | 253,507 | `6b377708ec9d6588f746692112dc39941f5ae78e2c159e591686c57dcb6aa246` | `089107b48169bcacb1bf942c9a22c18afc5ba1beb55adf59ed453b7c0b1e5df4` | MATCH |
| r2022 | 217,024 | `fee07c470d743c45b22d9384f47f9ded85604310a7228a562954a0e2a246447c` | `e4f00e959488f15bece586747bf17fa4daf5deb562e71f96d6535ab5644ebc30` | MATCH |
| r2023 | 219,839 | `5d2308217590f0a48178c8e86d430fa66ec3216e3a496e245e1e49e569626003` | `49125c3dac7362dad8bc353dab63a4d99a52d773adabdd566db92f35f6d60db0` | MATCH |
| r2024 | 223,497 | `c78d82e5cf54fcaf88b13880409c9aa27f68ca49cb56b0948f45096e132751e4` | `8033d91decc34519bc0a52cebb2f5d53bc1290db24c45dcbcea82dc3e00d3bda` | MATCH |

## Standalone BSS

| fold | catboost9 | champion | 5-cat CatBoost | delta 9 - champion | delta 9 - 5-cat |
|---|---:|---:|---:|---:|---:|
| primary | 617.7551 | 774.1586 | 732.7601 | -156.4035 | **-115.0050** |
| r2022 | 508.2144 | 607.2291 | 591.1754 | -99.0147 | -82.9610 |
| r2023 | 429.9082 | 571.8306 | 568.3534 | -141.9224 | -138.4451 |
| r2024 | 573.9698 | 724.1433 | 728.8662 | -150.1735 | -154.8963 |

## Correlation table

All values are Pearson correlations of catboost9 logits against the named OOF logits.

| fold | champion | LGB | MLP | 5-cat CatBoost |
|---|---:|---:|---:|---:|
| primary | 0.872694 | 0.874740 | 0.824114 | 0.910118 |
| r2022 | 0.852369 | 0.834528 | 0.828194 | 0.866429 |
| r2023 | 0.885663 | 0.882807 | 0.859295 | 0.895735 |
| r2024 | 0.882372 | 0.880933 | 0.851594 | 0.884238 |

The `<0.96` raw-logit complementarity condition passes on primary for every family. It does not override the failed quality floor, and residual diagnostics show that the decorrelation is not aligned with champion errors.

## Residual alignment

- Error residual correlation: `corr(y-p_catboost9, y-p_champion)`; the stated strong-diversity condition is `<0.98`.
- Residual alignment: `corr(p_catboost9-p_champion, y-p_champion)`; the inherited Task-5 alternative is `>=0.02`.

| fold | error residual corr | 1 - corr | residual alignment |
|---|---:|---:|---:|
| primary | **0.999203** | 0.000797 | **0.003761** |
| r2022 | 0.999278 | 0.000722 | 0.005928 |
| r2023 | 0.999457 | 0.000543 | -0.005246 |
| r2024 | 0.999292 | 0.000708 | -0.002211 |

## Screening gates

| gate | threshold | actual | result |
|---|---:|---:|---|
| primary BSS delta vs 5-cat | `>= -10` | **-115.004977** | **FAIL** |
| every primary family logit corr | `< 0.96` | max 0.910118 | PASS |
| champion error residual corr | `< 0.98` | **0.999203** | **FAIL** |
| Task-5 residual alignment alternative | `>= 0.02` | **0.003761** | **FAIL** |

**Two-seed screen: REJECT.** Catboost9 misses the quality floor by 105.005 BSS beyond the permitted margin. Neither residual-diversity formulation supplies the alternative signal.

## Blend-transfer probe

Protocol step 3 is conditional on screening PASS. Because screening failed, no catboost9 weights were fitted and no R-only labels were consumed for candidate selection or transfer testing.

| candidate | primary-fit weights | delta vs `champ_cat` on R folds | delta vs `lgb_mlp_cat` on R folds | bootstrap 5% LB | max\|delta mean\| |
|---|---|---|---|---|---|
| champion + catboost9 | N/A — skipped | N/A | N/A | N/A | N/A |
| LGB + MLP + catboost9 | N/A — skipped | N/A | N/A | N/A | N/A |
| LGB + MLP + 5-cat + catboost9 | N/A — skipped | N/A | N/A | N/A | N/A |

Accepted reference baselines from `task-7-blend.json` were left untouched:

| baseline | fixed weights | primary selection BSS | r2022 BSS | r2023 BSS | r2024 BSS | pooled bootstrap LB5 vs champion |
|---|---|---:|---:|---:|---:|---:|
| `champ_cat` | champion 0.85 / 5-cat 0.15 | 775.7468 | 608.6806 | 573.8332 | 729.6994 | +2.0632 |
| `lgb_mlp_cat` | LGB 0.30 / MLP 0.45 / 5-cat 0.25 | 776.4630 | 610.0393 | 575.5507 | 730.6576 | +2.6728 |

## Final action

- Verdict: **REJECTED**
- 10-seed full-data training: **DO NOT RUN**
- Todo 7-8 packaging path: **DO NOT PROCEED**
