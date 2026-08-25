# TrackMan Repertoire-Fingerprint Crosswalk v2

This document freezes the model-free, target-free feasibility contract for
`aimers9-trackman-repertoire-crosswalk-v2`. It is independent of the failed
activity/calendar/team-fingerprint v1 audit and does not reuse v1 thresholds.

## Scope and firewall

The audit reads only the declared non-target projections from official main
`train.csv` and historical `trackman_history.csv`. Historical
`pitch_type_group` is permitted for the repertoire audit; it never reads
`control_success`, test data, Public/leaderboard evidence, external data,
TrackMan physics/current-pitch measurements, or performs model work. It does
not implement Level-R exact pitch-row matching. Reports explicitly declare
`target_2022_access=false`, `target_2023_access=false`,
`target_2024_access=false`, `test_access=false`, `public_access=false`,
`external_access=false`, `source_order_used=false`, `model_training=false`,
and `branch_inert=true`.

Origins are 2022, 2023, and 2024. For origin `Y`, both main and TrackMan
repertoire selection uses seasons `< Y`; season `Y` is reserved for a
method-level, non-repertoire context verifier. The selection map is completely
frozen before that verifier runs. Verifier evidence cannot add, remove, or
change an individual mapping and no verifier-confirmed manifest is emitted.

## Semantic prerequisite

The primary taxonomy mode is fixed before matching:
`exclude_other_renormalize_known`. TrackMan `other` and missing groups are not
arbitrarily redistributed among the three canonical families. The mode is
available only when target-free source-level checks establish that main
fastball/breaking/offspeed rates form a complete three-family denominator and
TrackMan contains the documented known families. Otherwise the audit returns
`SEMANTIC_COMPATIBILITY_NOT_PROVEN` and fails closed.

The runtime-authoritative contract currently records the official-contract
proof as `NOT_PROVEN`: field names and documentation references alone do not
prove that excluding TrackMan `other` and renormalizing the known families uses
the same denominator as main. A separate target-free observational proof may
be `PROVEN` only when the main complete three-family denominator is within the
frozen tolerance, all canonical TrackMan families are supported, no unexpected
groups occur, and both `other_rows` and `missing_rows` are exactly zero. This
proof is independent of candidate matching; a small known-family subset or a
dominant `other` mass cannot make the mode pass.

The repository does not prove that a TrackMan `other` pitch can be assigned to
one of the three main families. Therefore `other` is excluded from the known
family denominator and its support is retained only as an aggregate quality
diagnostic.

## Main representation and chronology

The primary fingerprint is an order-independent annual three-family share
trajectory. Source/DataFrame order is never treated as chronology. For each
main pitcher-season, the maximum-support valid as-of state is used; conflicting
maximum-support states fail closed.

The audit infers either `SEASON_RESET` or `CUMULATIVE` from unordered support
ranges before matching. A transition is cumulative only when the current
season minimum support is at least the previous season maximum (within the
storage tolerance); a genuine backstep is reset evidence. The 0.95 fraction is
applied to these transition classifications, not endpoint maxima alone. In
cumulative mode, annual increments are derived from share-weighted support
with a deterministic storage-precision tolerance. Exact integer reconstruction
of `n * rate` is not required by the primary fingerprint; the tolerance is
measured from numeric storage tokens, not tuned against matching outcomes. If
the mode cannot be established, the result is `MATERIALIZATION_NOT_PROVEN`.

## Selection channel

Selection uses only repertoire evidence:

1. worst common-season total variation of the three-family shares;
2. worst half-L1 trajectory delta over common calendar-adjacent pairs only
   (`season == previous_season + 1`); a 2019/2021 gap supplies no delta.

The two channel thresholds and the second-best-margin threshold are derived
from one familywise best-false-pair statistic per unique calibration
transformation. Evaluation uses disjoint unique transformations and reports
the familywise any-accept rate, one-sided Wilson upper bound, and accepted-count
95th percentile. A candidate pair's margin is valid only when the assigned
right identity is the unique row top-1; a global row margin cannot validate a
non-top assigned pair. A threshold is unavailable when its null space is
insufficient and selection fails closed.

Support is used for validity/quality only, not as an activity similarity score.
Pitcher hand is a hard compatibility filter derived only from seasons `< Y`:
profiles retain `hands_by_season`, and a missing or multi-hand pre-origin
history is incompatible with every counterpart. The same hand-compatible,
finite candidate universe is used by both selection-null calibration and
evaluation; impossible cross-hand null pairs are never counted. Team identity,
numeric ID overlap, physics, and activity/calendar fingerprints are not
selection evidence.

The selection null candidate universe is the finite pre-origin hard-hand,
finite-distance fit matrix itself. The verifier null uses the corresponding
pre-origin hard-hand-compatible universe. Neither null includes impossible
cross-hand assignments. When the graph is not a complete identity matrix,
degree-zero vertices are excluded and each null transformation is a
deterministic maximum-cardinality one-to-one partial matching. The resulting
cardinality `K` is frozen across calibration and evaluation; `K=0` produces no
evidence and fails closed. Verifier context statistics evaluate only the
assigned pairs in each partial null, never silently falling back to the frozen
partner for an unassigned vertex.

Candidate selection is mutual top-1, one-to-one, and ambiguity-safe. Exact ties,
non-mutual top-1, one-to-many/many-to-one conflicts, and missing evidence remain
`AMBIGUOUS` or `UNMATCHED`.

The full hand-compatible distance matrix is ranked before either channel
threshold is applied. Thus a second-best pair just outside a channel threshold
still constrains the observed margin and reverse top-1 decision.

## Independent verification

The held-out season uses only non-repertoire context distributions: month,
day-of-week, inning band, top/bottom, count, outs, and batter hand. These
fields are disjoint from the selection channels. The verifier reports one
aggregate statistic per required channel, with a separately calibrated null
threshold per channel at frozen `q=0.01`, using per-channel median TV; all
required channels must pass conjunctively for method-level PASS.
Every context calibration/evaluation transformation is additionally forbidden
from retaining any actual frozen `selection_map_Y` partner and is restricted
to the same pre-origin hand-compatible universe. Its Wilson result is a
false-accept risk proxy. It reports evidence only and cannot alter
`selection_map_Y` or emit a verifier-confirmed manifest.

Origin-wise OOP verification and partner stability across independently fitted
origin maps are separate evidence types. Stability compares every origin pair,
requires nonempty shared main IDs for every pair, and is false when any pair has
zero overlap. These are not two holdouts of one frozen mapping.

## Within-source diagnostic

Before cross-source matching, main and TrackMan each receive a separate known-ID
diagnostic. Reference data are seasons `<= Y-2`; the query is the single season
`Y-1`. This is explicitly a **next-season repertoire persistence diagnostic**,
not a full multi-year trajectory validation. Its eligibility and null contract
are separate from cross-source matching. It also requires observed correct
acceptance count and coverage to exceed the conservative evaluation-null count
envelope `max(accepted_count_p95, Wilson_UCB * eligible)`. The accepted-count
histogram and p95 are retained as aggregate evidence; a small positive correct
count with zero wrong assignments is insufficient evidence of identifiability.

## Null controls and acceptance

Calibration and evaluation use disjoint deterministic namespaces and unique
identity transformations. Self-identification calibration/evaluation use true
derangements: no eligible query identity maps to the same normalized reference
identity. A repeated transformation never increases the statistical trial
count; finite small spaces are exhausted rather than padded. The trial unit is
one unique origin transformation and the familywise event is “any accepted
pair.” Wilson bounds are reported as a **null false-accept risk proxy**, never
as mapping accuracy. The inherited 1% ceiling is a risk ceiling, not a claim of
99% identity accuracy.

`P_PASS` additionally requires semantic compatibility, both source-level
self-identification diagnostics passing their own null contract, nonempty
ambiguity-safe mappings, held-out context method evidence passing its own null
contract, selection null risk within the ceiling, high-confidence count above
the null accepted-count p95, selection player/main-row coverage, and nonempty
conflict-free cross-origin stability. These are coverage properties of the
frozen selection map, not verifier-confirmed individual mappings. `P_FAIL` means
prerequisites operated but the mapping evidence failed. `P_KILL` means
semantic/materialization/identifiability/null prerequisites failed. A pass does
not authorize TrackMan feature modeling.

## Level R

Exact historical main-pitch to TrackMan-pitch matching is not implemented. The
repository has not proven exact intra-month chronology, so the prerequisite is
`SOURCE_ORDER_PROVEN`; otherwise the report is
`NOT_RUN_SOURCE_ORDER_NOT_PROVEN`.

## Outputs and provenance

Reports are aggregate-only JSON/Markdown written outside the repository. They
contain contract/config/script/Git hashes, runtime-contract/config-match
evidence, source-frame hashes using the `canonical-jsonl-v1` streaming
algorithm, scope flags, taxonomy/materialization decisions, counts, selection
coverage, ambiguity, null proxy, OOP evidence, and cross-origin stability. Raw
mapping IDs, target values, row lists, physics, and predictions are not
serialized; only aggregate selection-map hashes are retained.
