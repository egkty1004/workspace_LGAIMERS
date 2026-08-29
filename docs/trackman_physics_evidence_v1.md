# TrackMan Physics Evidence v1

This document consolidates already-produced, aggregate evidence. It is not a
new model-policy decision and does not activate a TrackMan leg in the v93
ensemble.

## Evidence scope

- The TrackMan physics EDA is a target-free, aggregate-only scan over the
  approved historical TrackMan projection. It does not use target values,
  test rows, Public/leaderboard information, entity crosswalks, or external
  data.
- The copied EDA runner is byte-identical to the reviewed evidence script:
  `c7009bf6de36ff28e981677f50e3ef4f96ae31d7d93e5637df8339566fb1b5ff`.
- The EDA source-frame SHA-256 is
  `f7818f9ee0ccefe7c2cf69fa99efe6e5cb882d8b886dd96d2394bcf3b53f33a9` over
  `1,793,078` rows. Its aggregate report SHA-256 is
  `10f5ce12e1ed0e0e5f2f6bf7ef3cb5b87fca893d1339d8f64914488afda1402b`.
- The EDA projection contains season/calendar, pitch-family and handedness
  metadata plus the eight physical measurements used by that historical
  diagnostic. It is not an inference-time feature contract.

## Aggregate structural observations

The four-family historical taxonomy counts were fastball `931,120`, breaking
`512,851`, offspeed `326,809`, and other `22,298`. The EDA observed no missing
family labels. Physical columns were summarized by season, family, month, and
handedness; no row-level records are retained here.

The EDA is descriptive evidence only. It does not establish a main-row
TrackMan identity join, current-pitch physics availability, or predictive
value.

The reviewed drift-resistant contract derived from this EDA keeps only
strict-past fastball, breaking, and offspeed aggregates. It excludes
`zone_speed`, `rel_side`, signed horizontal break, and the `other` family;
uses `abs(horz_break)`; represents extension and induced vertical break only
relative to same-history global/family baselines; and represents release
height through median/IQR statistics. Support and backoff level are audit
metadata, not model inputs. No clipping or winsorization was introduced.

## Existing Physics17 model evidence

The previously executed Physics17 diagnostic used the pinned C3 parent and a
separate 17-feature historical-physics leg. The immutable authority hashes
were:

- C3 model: `a9b6f0cedb8f13bbb03e5c051596dc77d99538753b4b475860aa06e8e97d4820`
- C3 feature contract: `df80e756fcac17fbfcb0ee01a10ba823b9501ae6dbc19d19db1d97f2976dd246`
- C3 parent package: `7dd7212f3d0a2b2bad4ae853c1d8f3a4a1336f4f952469b0f1c871eaad0b9ad9`
- Physics17 runner: `517f8bcf18ca4a9d801b06d4c155b48799f1d616eacaf5e5afacf7e53b84899f`
- Physics17 config: `8feca976aab3b36be5c12d7e3e6b9c433c57f2ea3354abc7ece83257a21d2f04`
- Physics17 lookup: `333da608fea0c3bd9e314262427d07b1d1385c5314c402d810a07e821efb55e2`
- Physics17 model: `2baa245e2b4892792b591f4fdfc85b129ba750e19be06b4b979276e6eb614f0b`
- Physics17 training-report literal SHA-256:
  `9962f8d0a9919a6b0782e43c901700546e416a0bfd303fe8e50317e735a4af46`
- Physics17 training-report canonical SHA-256:
  `834644449d2d8b1f103ec6481e1b1d82e89c37e1ba131bfcd308f5b93dd1ca77`
- Physics17 package-validation report SHA-256:
  `20b050d01df4614b3c890758ecb3529ac9fb98043c63e7c8ef80032f7918dd08`

The model diagnostic used CatBoost `1.2.10`, seed `42`, the frozen C3
CatBoost contract, and no post-early-stopping refit. It produced a 67-feature
physics diagnostic model with best iteration `1528` and `1529` trees. Its
245,789-row package validation passed row/order preservation, shuffle/chunk/
single-row independence, finite native probabilities, offline inference, and
the no-raw-TrackMan dependency contract. It did not change the v93 package or
active policy.

## Complementarity evidence

The existing four-fit complementarity diagnostic is preserved as aggregate
evidence only:

- report SHA-256: `364595a50ce2f340702426af84b186b917ad6f2ed6d899c1cea497e09b52a06b`
- runner SHA-256: `2fd8570e4b94644f3ddc4369af5d61b57c7e5a85c5a31decd26d7b0570f8bb2e`
- protocol role: `leaderboard_feature_experiment`
- recovery promotion: `false`
- reported diagnostic result: `COMPLEMENTARITY_GO` at shared weight `0.5`

The exact historical-OOS evidence was:

| Origin | C3 Brier | Physics17 Brier | 50:50 Brier | Improvement vs C3 | Analytic optimum |
| --- | ---: | ---: | ---: | ---: | ---: |
| r2022 | 0.249011327609 | 0.248864760064 | 0.248819230178 | 0.000192097431 | 0.6541989641 |
| r2023 | 0.249300022691 | 0.249134955139 | 0.249097400952 | 0.000202621740 | 0.6718194184 |

The shared weight `0.5` was frozen from r2022/r2023 evidence only. The
selection did not access Public, test, primary, or r2024 labels, and Public
results must not be used to retune it. The report literal SHA-256 is
`d6e7be274ad3dad90d83631b6da36560d8aabd235d46f650355426d42c3dccf7`;
the canonical report SHA-256 is the value above. The preserved matched
prediction artifacts have literal SHA-256 values
`78b216af2253f0b9bbb15004b153fcfa46360484afe938736114c1255b8f7164`
for r2022 and
`e8e926bbd74bdc417476a7722c91936ac62cf72c788f44b24621ef7448e6a10f`
for r2023.

## Frozen 50:50 package evidence

The executed package uses the arithmetic probability blend

`p = 0.5 * p_C3 + 0.5 * p_Physics17`.

Its component authorities are the C3 model, Physics17 model, and Physics17
lookup hashes recorded above. The preserved external ZIP is
`/home/2022113165/aimers_final/c3-trackman-physics-blend050-submit.zip`, with
SHA-256
`e266b46cbf84b0683e7e031addf549e942a03200dced22f6e2c38459aab80a45`
and size `1,880,319` bytes. Its aggregate validation report has SHA-256
`766abbe279c8e3331b74764bc433abce0e549c9066a3bc4d7b2ee6bfe11de00a`.
The two deterministic builds had identical ZIP hashes; validation passed for
245,789 rows, exact numeric blend parity, finite/range checks, row ordering,
shuffle/chunk/single-row independence, offline inference, six inference
threads, and no raw TrackMan dependency. Measured full-frame inference was
`3.1827` seconds; the validation process peak RSS was `599,832` KiB.

The following leaderboard results are user-reported evidence only and were
not independently queried by this work:

| Artifact | Public BSS |
| --- | ---: |
| Exact C3 | 821.1201237336 |
| Physics17 standalone | 791.3223702036 |
| Frozen 50:50 blend | 842.8975664653 |

Physics17 was weaker than C3 as a standalone replacement but the frozen blend
was stronger in the user-reported Public result. This is consistent with the
historical-OOS complementarity hypothesis; it does not change v93, activate a
recovery candidate, or authorize Public-based retuning. The executed blend
formula and package evidence are documented here without adding a newly
authored blend builder/config/test as repository authority.

## Policy boundary and next direction

The v93 six-leg baseline and all of its assets remain unchanged. A future
proposal named “v93 + Frozen TrackMan Residual Leg” would test whether the
frozen TrackMan-specific delta `(Physics17 - C3)` adds complementary
historical-OOS signal to the existing v93 prediction. It is not authorized;
it requires a separate Experiment Brief, plan review, and explicit execution
approval. No Public-based alpha selection is allowed, and no calibration,
offset, clipping, or recovery-policy conclusion is inferred here.

The evidence does not establish that TrackMan is globally unusable. It only
records the reviewed physics diagnostics and their provenance. Any future
TrackMan feature work must preserve the existing entity/crosswalk, current
pitch, test-distribution, external-data, and recovery-policy restrictions.
