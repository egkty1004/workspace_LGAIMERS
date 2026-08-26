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

## Official characterization evidence

The reviewed implementation at Git SHA
`01440eb9f351dbc698ad42edabf25c9e7422bb87` was executed exactly once against
the approved official projections. The run completed in `2:14.64` with peak
RSS `163,280 KB` and returned machine verdict `SCAN_COMPLETE`, reasons `[]`,
and prior-kill reproduction status `V1_COMPATIBLE_INVALID_OBSERVED`.

The main projection contained `1,475,092` rows and no v1-compatible invalid
context row. Its projected-frame SHA-256 was
`50a1fdeae8c0e1ece5302837e3fb748674f852d436c1931e2001d8a6b7c894e8`.
The TrackMan projection contained `1,793,078` rows, of which `97`
(`5.409692160631049e-05`, approximately `0.0054097%`) were context-invalid.
Its projected-frame SHA-256 was
`934d5757241e78389f0736717c550be8e80c8948f156af0d5abc6e956acdd24c`.
There were no missing, unparseable, nonfinite, or non-integer context values.
The complete TrackMan residue was:

- `balls_before=4`: 1 row;
- `strikes_before=3`: 1 row;
- `outs_before=3`: 83 rows;
- `outs_before=4`: 12 rows.

The invalid rows were confined to 2022 (`86`) and 2023 (`11`); 2019, 2020,
2021, and 2024 each had zero. External review classified this aggregate
evidence as `FILTERABLE_RESIDUE`: the frozen semantic domain remains unchanged,
and the evidence supports a small source-local historical TrackMan residue,
not broad domain incompatibility. This interpretation does not authorize
filtering or modeling. Any future TrackMan exclusion rule requires a separately
reviewed v2 contract; no main-row filtering and no PR #12/context-prior-v1
rerun are authorized.

Aggregate provenance:

- JSON literal SHA-256:
  `5a55510a68a55173ccd38c1563034766f0a4e9b5d9dec5456431c8626958d9e8`;
- Markdown literal SHA-256:
  `6dab9c3038b15476e818298de4936ae8768710cffbb4bcf230abad86d2c84548`;
- canonical report SHA-256:
  `5cc1ee2057b61fa91df3249a3f4a64617a9b48791936f308263d1ab751683e00`;
- runner SHA-256:
  `370b5a09836cf95b7c4499ddea4ef881ee2fd5deb6155aca99c9ec00eb7e5090`;
- config SHA-256:
  `0334bbb8caa7f1915134a7f4dcbb66e5a26ceb08abbe8133a70f247c2643299f`.

The runtime config matched the frozen config. Target, test/test-distribution,
Public/leaderboard, external information, TrackMan entity/physics,
current-pitch measurements, models, and GPU were not accessed; the label-access
ledger was empty and outputs remained aggregate-only.

## Scope boundaries

This is a model-free source-characterization phase. The characterization
execution did not rerun PR #12, implement invalid-row handling, build
context-prior lookups, run Level R, train or score models, or modify
`repro_979/`. Official-data access required separate approval after code
review; the approved official scan above is the only such execution. Any
future source-local exclusion or main-row fallback is a new reviewed contract,
not an implication of this scan.
