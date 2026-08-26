# v93 TrackMan context-prior CatBoost v1

## Purpose and status

This leaderboard-track experiment tests one hypothesis: the seven reviewed,
crosswalk-free TrackMan context-prior features improve a matched v93-style
CatBoost leg and the fixed v93 composite. It is branch-inert and is not part of
the recovery promotion pipeline. This implementation phase performs no model
training, target scoring, GPU work, or submission build.

The technical base is the merge of TrackMan context-prior v2 at
`107c58c2e815abe352977da7546cd46e707cd614`. The reviewed serialized lookup is
pinned by SHA-256
`27747c22e0ff25e86040f5825667f8d9c0b8d7e840037ad9df87072781160515`.
Model and inference code consume that lookup; they never scan raw
`trackman_history.csv`.

## Authorities

The runtime configuration pins and verifies:

- original v93 archive SHA-256
  `8157e144090bcccbf1c44367c75a2d2427e040b8353c8c1e17334b41324ac5fb`;
- original `catboost_prep.pkl` SHA-256
  `0c434e65fd601806134f08e51b5cf5e72c5f43173a1715256ad1eeb1fc2e8653`;
- tracked Task-3 v93 asset evidence and all 51 original model/preprocessor
  member hashes;
- the original bounded B0 cache key, metadata, and r2022/r2023 array hashes;
- the merged TrackMan v2 runner/config hashes and reviewed lookup hash;
- the recovered ordered 49-feature CatBoost contract and five categorical
  features.

The recovered v93 CatBoost recipe is CatBoost 1.2.10, seeds 42 through 51,
Logloss, learning rate 0.05, depth 6, L2 leaf regularization 5, at most 2,000
trees, early-stopping patience 50, `use_best_model=True`, and no post-ES refit.
For each origin, `RandomState(12345).choice` selects exactly
`int(n_rows * 0.05)` ES positions without replacement; positions are sorted and
the complement is the fit set. C0 and C1 share every row, target, split,
parameter, seed, preprocessing, and categorical contract.

## Arms and feature contract

- **B0** is the hash-addressed actual original cached v93 deployed composite.
- **C0** replaces only B0's original CatBoost leg with a newly trained
  reconstructed v93-style matched CatBoost control.
- **C1** is the same matched control plus exactly these seven numeric columns:
  `tm_cf_p_fastball`, `tm_cf_p_breaking`, `tm_cf_p_offspeed`, `tm_cf_p_other`,
  `tm_cf_entropy_norm4`, `tm_cf_support`, and
  `tm_cf_context_vs_global_tv`.

The audit-only `__backoff_level` is forbidden as a model input. The five C0
categorical columns remain `top_bottom`, `game_type`, `base_state`, `platoon`,
and `count_state`; none of the TrackMan features is categorical. The runner
mechanically requires C1 features to equal the ordered C0 49-feature list plus
the seven reviewed columns.

For a main row in season S, the serialized lookup entry for S already embodies
the reviewed strict TrackMan `season < S` construction. Thus 2019 remains
NO_HISTORY and 2025 uses only filtered 2019–2024 TrackMan history. Invalid main
context fails closed. No player/team/game ID, crosswalk, exact row join,
current-pitch measurement, test distribution, or other test row is involved.

## Matched screen and label firewall

The future screen is deliberately separate from active recovery policy. For
each origin it trains on all regular-season rows through the prior season and
evaluates raw CatBoost on the full outer-validation origin:

- r2022: train through 2021, validate 2022;
- r2023: train through 2022, validate 2023.

Only the deployed-composite diagnostic uses the current first 30,000 true
outer-validation positions in source order, because B0 is available under that
active bounded cache contract. Candidate-A validation geometry is not adopted.
Source order is not proven chronology, and the known bounded-panel
representativeness limitation remains explicit.

Feature projection excludes `control_success`. Training labels are read only
for the exact origin training positions. C0 and C1 full-origin outer logits are
both sealed with origin, position, and row-ID hashes before the scoped
outer-validation label reader can run. Primary/r2024/2024 target access is not
implemented.

## B0/C0/C1 diagnostics and gate

The original ten-model v93 CatBoost logits and reconstructed C0 logits are
compared on identical rows. The report records both hashes, means, MAE, RMSE,
correlation, finite/row parity, and raw Brier when origin labels are authorized.
Unless arrays reproduce exactly, the report must call C0 a
**reconstructed v93-style matched control**, not an exact v93 control. A lack of
exact reproduction is diagnostic rather than an automatic experiment failure.

The fixed composite substitution is:

`z_arm = z_B0 + 0.0701066994221915 * (z_arm_cat - z_original_v93_cat)`.

Deployed probabilities preserve the v93 offset
`-0.0461645795229729` and `[0.30, 0.70]` clipping. LGB, MLP, FTT, ArmB, their
weights, and all other v93 behavior remain fixed.

Package worthiness is conjunctive. At both r2022 and r2023 independently:

1. full-origin raw CatBoost Brier(C1) must be lower than Brier(C0);
2. bounded fixed-composite Brier(C1) must be lower than Brier(C0); and
3. bounded fixed-composite Brier(C1) must be lower than Brier(B0 original v93).

No origin average may compensate for a failure. The deployed mean shift of C1
versus B0 is reported but is not a gate. BSS and means are reported alongside
Brier; no bootstrap, recovery gate, terminal-label gate, or calibration search
is part of this experiment.

## Package contract (future, not executed)

The inert builder starts from the pinned original v93 archive. It verifies and
preserves every non-CatBoost model/preprocessor asset byte-identically. Only
the ten CatBoost model files, `catboost_prep.pkl`, the minimum inference logic,
and aggregate TrackMan lookup/provenance are replaced or added.

The original 49-feature frame continues to feed LGB/MLP/FTT/ArmB. Only the
CatBoost path receives the ordered 56-feature frame. The builder refuses
overwrite, checks fixed-asset hashes before and after staging, and creates ZIPs
with fixed metadata/order for deterministic rebuilds. Aggregate provenance may
contain hashes and contracts only—never targets, predictions, row lists,
private absolute paths, or leaderboard information.

Final package construction and full inference validation remain unauthorized
until the matched screen passes and the user explicitly approves them.

## Verification stages

- **CHEAP (this phase):** syntax compilation, synthetic feature/temporal and
  firewall tests, exact contract/hash guards, B0/C0/C1 algebra and conjunctive
  gate tests, package-frame separation/fixed-asset/deterministic-ZIP tests,
  static CLI, external authority hash verification, and `git diff --check`.
- **Later MEDIUM/EXPENSIVE (separate approval):** official matched CatBoost
  training/scoring. Ten seeds across two origins and two arms is expensive and
  is not executed in this phase.
- **Later package stage (separate approval):** final C1 deployment training,
  deterministic package build, offline parity, arbitrary row-count,
  independence, runtime, memory, and size validation.

No active recovery policy, registry, state, fold role, deployed offset,
clipping, or non-CatBoost asset is changed by this branch.
