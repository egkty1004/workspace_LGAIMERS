# TrackMan Crosswalk-Free Context Priors v2

This document freezes the source-quality correction for the model-free,
entity-free TrackMan context-prior audit. It is a new v2 contract after the
v1 official audit was killed by `CONTEXT_DOMAIN_NOT_PROVEN`; it does not rerun
or change v1 and does not authorize modeling.

## Scope and firewall

The main projection is exactly `season`, `balls_before`, `strikes_before`, and
`outs_before`. The TrackMan projection adds only `pitch_type_group`. No target,
test distribution, Public/leaderboard evidence, player/team/game/raw ID,
crosswalk, exact row join, physics/current-pitch measurement, external source,
or model is accessed. Level R exact matching is not implemented.

## PR #13 source authority

Source provenance uses the merged PR #13 characterization implementation and
its exact `canonical-multiset-json-v1` algorithm over the four-column
projection `(season, balls_before, strikes_before, outs_before)`. The v1
`canonical_frame_hash` is not a source authority. Runtime and config pin the
PR #13 helper hashes and the expected official hashes:

- main: `50a1fdeae8c0e1ece5302837e3fb748674f852d436c1931e2001d8a6b7c894e8`;
- TrackMan: `934d5757241e78389f0736717c550be8e80c8948f156af0d5abc6e956acdd24c`.

The hash is a canonical multiset hash, so source row order does not affect it;
a context mutation or row addition/deletion does.

The frozen v1 lookup/probability implementation is also a pinned parent:

- v1 runner SHA-256: `2562bd3b21dbe010526eb748add0baccab6b8e11b4ddf2a0c8f73baae4a4a293`;
- v1 config SHA-256: `7cb0fd4b26b68fa59d7c726489343c5295af9bb5d2d914a82f27d5958a68d8f7`.

After the v2 source-quality guard, bucket construction, probability/entropy/TV
calculation, row application, and lookup serialization delegate to the pinned
v1 implementation. Shared constants and both parent artifact hashes are
checked fail-closed.

## Frozen source-quality correction

Raw TrackMan taxonomy is checked before any filter. After taxonomy and season
prerequisites pass, only TrackMan rows satisfying this predicate are retained
for lookup construction:

```text
balls_before: finite integer-like value in {0,1,2,3}
strikes_before: finite integer-like value in {0,1,2}
outs_before: finite integer-like value in {0,1,2}
```

There is no rounding, normalization, clipping, or hard-coded row-position or
ID exclusion. Main context is never filtered: any missing or out-of-domain
main context is `CONTEXT_DOMAIN_NOT_PROVEN` and yields `P_KILL`.

The official residue contract is fail-closed. It expects 1,793,078 raw
TrackMan rows, 97 excluded rows, and 1,792,981 retained rows, with this
aggregate residue manifest:

| season | field | reason | value | rows |
|---:|---|---|---:|---:|
| 2022 | balls_before | out_of_domain | 4 | 1 |
| 2022 | outs_before | out_of_domain | 3 | 72 |
| 2022 | outs_before | out_of_domain | 4 | 12 |
| 2022 | strikes_before | out_of_domain | 3 | 1 |
| 2023 | outs_before | out_of_domain | 3 | 11 |

Any count, season/field/value/reason manifest, or retained-row drift is
`SOURCE_QUALITY_RESIDUE_DRIFT` and `P_KILL`. No individual row is persisted.
The filter output is explicitly marked; the lookup builder rejects an
unfiltered TrackMan frame, proving excluded rows cannot reach aggregation.

## Temporal lookup and 2025 proof

For every main season `S`, a lookup is constructed only from filtered TrackMan
rows with `season < S`. Same-season and future rows are forbidden. The lookup
uses four independent families (`fastball`, `breaking`, `offspeed`, `other`),
without smoothing. Legal sparse contexts use the frozen hierarchy:

```text
L0 = (balls_before, strikes_before, outs_before)
  -> L1 = (balls_before, strikes_before)
  -> global
```

The support threshold is 100. `tm_cf_support` is the historical row count at
the selected level; the backoff level is audit-only and not a feature. A
no-history lookup emits missing probabilities and support zero.

Before a valid result is reported, the runner constructs a synthetic 2025
probe containing all 36 legal L0 states (`4 × 3 × 3`). Every state must have a
finite four-way simplex and support exactly equal to its selected L0/L1/global
bucket. Sparse L0/L1 states may back off. The official 2025 global support
must equal exactly 1,792,981. This probe reads no test rows or distribution.

## Verdicts

`P_KILL` is reserved for source-hash/residue drift, main-domain failure,
taxonomy/season/firewall failure, or the 2025 legal-state prerequisite.
`P_FAIL` is reserved for legal fallback, contextual-use, context-variation,
finite/simplex, or deterministic output failures after prerequisites pass.
`P_PASS` requires all prerequisites and structural acceptance. A pass is only
model-free feasibility evidence; it does not activate a feature, change
recovery policy, or authorize a CatBoost experiment.

## Provenance and outputs

The external output directory contains only aggregate JSON, Markdown, and an
aggregate lookup. Reports include contract/config/runtime hashes, source
projection hashes, residue counts/manifest, temporal and 2025 proof evidence,
and privacy flags. They contain no row IDs, targets, entities, physics, or raw
rows, and serialize with `allow_nan=False`. Output directories inside the
repository and pre-existing output directories are rejected. Reversing the
filtered TrackMan source must reproduce the same lookup and feature output.
The serialized lookup is replayed against every projected main row and must
match direct feature construction exactly.

The official 2025 lookup, if later authorized, uses filtered 2019–2024
TrackMan history only. No official-data run is implied by this implementation.

## Official model-free MEDIUM evidence

The official v2 audit was executed exactly once from reviewed Git SHA
`0ce000922810f946d6ff3ae651543ffb2fb821ba`. It completed with `P_PASS`,
`feature_output_available=true`, no kill reasons, and no structural-failure
reasons. The audit was not rerun.

The approved projections contained 1,475,092 main rows and 1,793,078 raw
TrackMan rows. The frozen TrackMan-only source-quality predicate excluded
exactly 97 rows and retained 1,792,981 rows. The observed residue matched the
pre-registered manifest exactly: one 2022 `balls_before=4` row, one 2022
`strikes_before=3` row, 72 2022 `outs_before=3` rows, 12 2022
`outs_before=4` rows, and 11 2023 `outs_before=3` rows. No rounding,
normalization, or clipping was applied. The raw four-family taxonomy passed
with 931,120 fastball, 512,851 breaking, 326,809 offspeed, and 22,298 other
rows; missing and unexpected taxonomy counts were zero.

All cutoffs from 2019 through 2025 used only TrackMan seasons strictly before
the main-row season. The 2025 lookup used only filtered 2019–2024 history. Its
36 legal L0 states all produced valid simplexes with exact selected-level
support, and its global support was exactly 1,792,981. Direct construction,
reversed-source reconstruction, serialized-lookup replay, and complete feature
output were deterministic and equal. Serialized replay covered all 1,475,092
projected main rows with zero mismatches.

Official aggregate provenance:

- JSON literal SHA-256: `1a07fdeb0b994ce3533371b7d2a1726b3d08f646a57d6cad346f6403614363ef`;
- Markdown literal SHA-256: `b84279102569bff78cdb7e1f79cbfc8edbfe6bbf8d114f6fa99465c9f429a5c9`;
- serialized lookup literal SHA-256: `27747c22e0ff25e86040f5825667f8d9c0b8d7e840037ad9df87072781160515`;
- canonical report SHA-256: `67327fc36269e14a95006c9775cb9443d07d1ef7d65057b3cfdac1112443f444`;
- runner SHA-256: `db49237ed9c98ffad9cdb0bbaf1664bb12dc02c824bf4c4db7473dd20baf5293`;
- config SHA-256: `5232dc50706df44978f4acfc155dbd5feb43fa5d884699d65b3da55209bd84c3`;
- main projected-frame SHA-256: `50a1fdeae8c0e1ece5302837e3fb748674f852d436c1931e2001d8a6b7c894e8`;
- TrackMan projected-frame SHA-256: `934d5757241e78389f0736717c550be8e80c8948f156af0d5abc6e956acdd24c`.

Execution took 30:06.18 wall time (1,783.58 seconds user and 18.87 seconds
system) with peak RSS 1,695,248 KiB. The audit accessed no target, test or test
distribution, Public/leaderboard, external information, TrackMan entity or
physics fields, current-pitch measurements, model scoring/training, or GPU.
The label-access ledger was empty, and outputs were aggregate-only without row
IDs, entities, targets, or raw rows.

`P_PASS` establishes only model-free structural feasibility for the frozen v2
context-prior lookup. It does not activate the seven features, authorize model
training or packaging, change recovery policy, or establish predictive value.
