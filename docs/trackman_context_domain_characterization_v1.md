# TrackMan Context-Domain Characterization v1

This document freezes a target-free characterization of the
`CONTEXT_DOMAIN_NOT_PROVEN` prerequisite failure in PR #12. It is not a
rerun or modification of the crosswalk-free context-prior contract. It does
not construct features, filter rows, use identity information, or authorize
TrackMan modeling.

## Contract

The contract identifier is
`aimers9-trackman-context-domain-characterization-v1`. Both sources are
scanned independently with exactly these approved columns:

```text
season, balls_before, strikes_before, outs_before
```

The target, row IDs, player/team/game/pitch IDs, `pitch_type_group`, physics,
and current-pitch measurements are not projected. Test data, test
distribution, Public/leaderboard information, external data, and models are
outside the audit.

The machine classifier intentionally reproduces PR #12's `_number`,
`_integer`, and `_context_key` semantics:

1. apply `pandas.isna` before conversion;
2. convert with `float(value)` and require a finite result;
3. require `float(value).is_integer()` for integer-like values;
4. apply the frozen domains `balls_before={0,1,2,3}`,
   `strikes_before={0,1,2}`, and `outs_before={0,1,2}`.

Whitespace and lexical forms such as `1` and `1.0` are therefore legal when
the PR #12 semantic conversion considers them legal. A separate raw-token
diagnostic records lexical shape and exact aggregate token counts; it never
changes `v1_compatibility` or the verdict.

No values are rounded, clipped, normalized, filled, filtered, or reinterpreted.
Missing, unparseable, nonfinite, non-integer, and out-of-domain values are
reported separately. Every invalid row is additionally aggregated by
`source × season × invalid-fields × reason/value signature`; no row ID or raw
row is retained.

## Verdict contract

The runner emits only these machine verdicts:

- `SCAN_COMPLETE`: both projections were scanned and the aggregate census is
  complete. This is not a claim that a future filter is safe.
- `NOT_PROVEN`: the scan cannot establish the required contract, season
  stratification is incomplete, or the earlier PR #12 kill is not reproduced
  under its compatible classifier.

The runner never emits `FILTERABLE_RESIDUE` or `DOMAIN_INCOMPATIBLE`.
Those are external-review interpretations of the raw aggregate evidence.

If the combined v1-compatible context-invalid row count is zero while the
frozen prior result was `P_KILL / CONTEXT_DOMAIN_NOT_PROVEN`, the result is
`NOT_PROVEN` with reason `PRIOR_KILL_NOT_REPRODUCED`. An empty invalid set is
not called a filterable residue.

Invalid TrackMan observations may be considered later in a separate
source-local exclusion-contract review. Invalid main observations are not
declared filterable: hidden inference can contain them, so a separate
row-local fallback hypothesis would be required.

## Season diagnostics

The season column is classified with the same integer conversion and is
stratified over 2019--2024. Missing, nonfinite, non-integer, or out-of-range
season values remain visible in an `__INVALID_SEASON__` aggregate and cause
`SEASON_STRATIFICATION_NOT_PROVEN`; they are never silently dropped.

The output reports, for every source and field:

- total, missing, finite/nonfinite, integer-like/non-integer, legal and
  invalid counts;
- finite observed minimum and maximum;
- all invalid reason/value counts and out-of-domain values;
- invalid counts and fractions by season;
- legal-domain histograms by season;
- raw lexical diagnostics, explicitly separate from the v1 classifier.

For every projected context field, the JSON also contains a deterministic
overall legal-domain histogram and a histogram for each expected season (plus
the invalid-season bucket). The report separately aggregates context-invalid
row/observation counts and fractions by source, source × field, and source ×
season. These are descriptive counts only; no concentration pattern is turned
into an automatic filtering or domain-compatibility conclusion. The
source × season × invalid-fields × reason/value row-signature table is retained
alongside these summaries.

## Reproducibility and privacy

The source projection hash is a canonical multiset hash, so aggregate output
does not depend on input row order. The report hash is canonical JSON with
`allow_nan=False`. Outputs are written only to a fresh directory outside the
repository, and existing output directories are rejected.

The report includes the Git, runner, and config hashes; exact projections;
scope/firewall flags; label-access ledger `[]`; machine verdict/reasons; and
the aggregate row-signature census. It excludes target values, row IDs,
entities, raw rows, physics, and test material.

## Scope boundaries

This is a model-free source-characterization phase. It does not rerun PR #12,
implement invalid-row handling, build context-prior lookups, run Level R,
train or score models, access official data during CHEAP verification, or
modify `PROJECT_STATUS.md`/`repro_979/`. An official scan requires a separate
approval after code review. Any future source-local exclusion or main-row
fallback is a new reviewed contract, not an implication of this scan.
