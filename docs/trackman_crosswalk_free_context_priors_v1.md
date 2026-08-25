# TrackMan Crosswalk-Free Context Priors v1

This document freezes the model-free, entity-free feature-feasibility audit
for `aimers9-trackman-crosswalk-free-context-priors-v1`. It is a new route
after Crosswalk v1 and repertoire Crosswalk v2 both failed closed; it does not
retune either crosswalk and does not authorize TrackMan modeling.

## Scope and firewall

The main projection is exactly `row_id`, `season`, `balls_before`,
`strikes_before`, and `outs_before`. The TrackMan projection is exactly
`season`, `balls_before`, `strikes_before`, `outs_before`, and
`pitch_type_group`. No player/team/game identity, raw ID, target, test row,
leaderboard, external source, TrackMan physics, or current-pitch measurement
is read. Historical `pitch_type_group` is used only as an aggregate label.
Level R exact pitch-row matching remains out of scope.

## Frozen context and temporal contract

For a main row in season `S`, only TrackMan rows with `season < S` are legal.
The lookup is keyed by the season column; source row order is never treated as
chronology. A 2019 row with no prior history has missing probability summaries,
support `0`, and `NO_HISTORY`. Same-season and future-season rows are never
used.

Required contexts are integer-like and legal only in these domains:

- `balls_before`: `0..3`
- `strikes_before`: `0..2`
- `outs_before`: `0..2`

Missing or out-of-domain context observed in the official projected data is
`CONTEXT_DOMAIN_NOT_PROVEN` and fails closed. A legal but sparse context is
not invalid and follows the frozen `L0 -> L1 -> global` hierarchy:

Nonintegral, out-of-scope, or future season values are likewise
`TEMPORAL_CAUSALITY_NOT_PROVEN`; the run reports that prerequisite as an
aggregate `P_KILL` rather than silently filtering the rows.

1. `L0 = (balls_before, strikes_before, outs_before)` when support is at least
   100;
2. `L1 = (balls_before, strikes_before)` when its support is at least 100;
3. otherwise the strict-prior global history.

The L1 threshold is not an acceptance gate. In particular, L1 support below
100 is allowed to fall back to global, and a synthetic 2025 fixture with 12
count states does not require L1 support of 100. `tm_cf_support` is exactly the
historical TrackMan row count at the selected level: L0 support for L0, L1
support for L1, global support for global, and zero for `NO_HISTORY`. The
selected level is audit-only and is not a first-model feature.

## Taxonomy and feature family

The four TrackMan families are retained directly: `fastball`, `breaking`,
`offspeed`, and `other`. Missing or unexpected groups fail closed; `other` is
not redistributed. At a selected level, the four probabilities are empirical
family counts divided by selected-level support. The same family gives:

- `tm_cf_p_fastball`, `tm_cf_p_breaking`, `tm_cf_p_offspeed`,
  `tm_cf_p_other`;
- normalized four-family entropy
  `-sum(p*log(p))/log(4)`, with `0*log(0)=0`;
- `tm_cf_support`;
- `tm_cf_context_vs_global_tv = 0.5 * sum(abs(p_selected - p_global))`.

No Dirichlet prior, target-derived smoothing, identity mapping, test-derived
fallback, or cross-row adjustment is used. Probabilities are finite and sum to
one within the frozen tolerance whenever history exists; no-history summaries
are explicitly null.

## Structural acceptance

The report provides per-season L0/L1/global/NO_HISTORY counts and fractions,
support, finite/simplex, and missingness diagnostics. The model-free result is:

- `P_KILL` for an invalid required domain, taxonomy failure, temporal/firewall
  failure, or inability to establish the causal contract;
- `P_FAIL` only when legal rows cannot receive valid fallback output, contextual
  L0/L1 usage is globally zero, every contextual prior collapses to its global
  prior, or finite/simplex/determinism invariants fail;
- `P_PASS` when the prerequisites and structural invariants pass.

The official run rebuilds the complete feature output after a fixed reversal of
the TrackMan input rows. The original and rebuilt output hashes are recorded
under `acceptance.determinism`; a mismatch adds
`FINITE_SIMPLEX_OR_DETERMINISM_FAILURE`. Domain and taxonomy prerequisites are
reported as aggregate `P_KILL` results with a deterministic reason and do not
get reinterpreted as ordinary P_FAIL outcomes. Other reader/schema failures
remain execution errors.

Sparse legal contexts are not a failure merely because their L1 support is
below 100. The backoff level is retained for audit diagnostics but is excluded
from the first model feature set.

## Official audit evidence

The official-data model-free audit was executed exactly once from Git SHA
`a4f60fcc3ce7d2bce768905e6027d0465c551db6`. It returned `P_KILL` with the
exact reason `CONTEXT_DOMAIN_NOT_PROVEN`. The result was not tuned or rerun.
Taxonomy compatibility itself passed. The observed TrackMan family counts were
`fastball=931120`, `breaking=512851`, `offspeed=326809`, and `other=22298`;
missing and unexpected taxonomy counts were both `0`.

The projected source counts and frame hashes were:

- main: `1475092` rows,
  `81c374224befe2247e0fad410c88e5549b752f0ddcf27e35a223f6513c4aaa7c`;
- TrackMan: `1793078` rows,
  `3530593d94f320b218b70fc226a045f73218d3b49049be61ea1edc2c217eb96d`.

Because the context-domain prerequisite failed, no lookup or feature
diagnostics were produced and determinism was not run. The preserved external
artifacts were:

- JSON literal SHA-256:
  `654aea608c41f1f8506d0d55c04a12fb2692cc0e3efd6c1f7c042c3a3f0d0790`;
- Markdown literal SHA-256:
  `af0fda779f15ea8565e6ca744777d8b79d34a31a6651138bea059acdb9ec2d59`;
- lookup literal SHA-256:
  `f142a492ecf610a2aa1f5c24954c726a2d9a9a30bc8f7324c134c842ceddbd34`;
- canonical report SHA-256:
  `0d5fae309b4a816215956e28c18111fff08dd700f48e58395145fbca9020c7be`;
- runner SHA-256:
  `2562bd3b21dbe010526eb748add0baccab6b8e11b4ddf2a0c8f73baae4a4a293`;
- config SHA-256:
  `7cb0fd4b26b68fa59d7c726489343c5295af9bb5d2d914a82f27d5958a68d8f7`.

Runtime was `0:48.45` wall time (`47.50s` user, `8.31s` system), with peak
RSS `819080 KiB`. Scope remained aggregate-only: target access, test access,
test-distribution access, Public/leaderboard access, external information,
TrackMan entity access, TrackMan physics/current-pitch measurement access,
model training/scoring, and GPU use were all `false`; the label-access ledger
was `[]` and `branch_inert=true`. The explicit scope fields were
`target_access=false`, `target_column_in_projection=false`, `test_access=false`,
`test_distribution_access=false`, `public_access=false`,
`public_leaderboard_evidence=false`, `external_information_access=false`,
`trackman_entity_access=false`, `trackman_physics_access=false`,
`current_pitch_measurement_access=false`, `model_training_or_scoring=false`,
and `gpu_used=false`. The privacy contract was
`aggregate_only=true`, `row_ids=false`, `entity_ids=false`,
`target_values=false`, `physics=false`, and `raw_rows=false`. This result does not establish that
crosswalk-free TrackMan context priors are unusable. It records only the
failed observed-domain prerequisite, with no guess about which field or source
caused it.

## Outputs and future stages

The future official run writes aggregate-only JSON/Markdown plus a compact
context lookup outside the repository. The lookup is a reproducible aggregate
artifact: for every cutoff season it contains deterministic string-encoded L0
and L1 context keys, global/L0/L1 support, four family counts, probabilities,
entropy, and context-vs-global TV. A synthetic reader can reproduce the
selected feature record from this lookup. It contains no row IDs, entities, or
raw rows. The output rejects directories inside the repository and serializes
with `allow_nan=False`. Reports contain hashes, counts, scope flags, and
support/backoff aggregates, never row-ID lists, targets, entities, physics, or
raw rows.

This implementation is the model-free feasibility phase only. A `P_PASS` does
not authorize a CatBoost feature-lab run; that requires a separate reviewed
Experiment Brief. Any feature-lab stage must preserve row independence and
must not modify recovery policy, validation geometry, registry, or champion
state.
