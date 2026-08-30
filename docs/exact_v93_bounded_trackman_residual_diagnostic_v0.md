# Exact v93 Bounded TrackMan Residual Diagnostic v0

## Status

This is a branch-inert, NumPy-only diagnostic. It uses the preserved exact
v93 first-30k composite-logit caches for `r2022` and `r2023`, plus the reviewed
full-origin matched C3/Physics17 prediction archives. It is not full-origin
v93 evidence and cannot authorize champion, recovery-policy, promotion,
packaging, or submission changes.

No model is fitted by this experiment.

## Official bounded diagnostic result

The single approved official-data diagnostic returned
`BOUNDED_SIGNAL_ABSENT`. This result is first-30k diagnostic evidence only,
not full-origin evidence. Exact full-origin v93 historical-OOS authority was
not available; the preserved bounded caches were not treated as a substitute
for that authority.

The deployed baseline Brier scores were `0.24517497588392245` for `r2022`
and `0.24494018410035600` for `r2023`. The right directional derivatives at
alpha zero were `-2.5462965069777965e-06` and `-3.726602360032416e-05`,
respectively. Both origins failed closed with
`right_derivative_not_improvement_directed`; consequently there was no shared
positive interval and no `alpha_diagnostic`, and no diagnostic Brier or
improvement was evaluated.

Exact authority and row-parity checks passed. The run performed zero model
fits and used no GPU. It took 11.05 seconds wall time (11.20 seconds user,
7.20 seconds system) with peak RSS 592,480 KiB. The preserved aggregate report
is at
`/tmp/aimers9-exact-v93-bounded-trackman-residual-v0-official/exact_v93_bounded_trackman_residual_diagnostic_v0_report.json`;
its literal SHA-256 is
`1181f027a715f6138b18c5ed74b1126f404cb1b771b5c77f32a1e0a9acb3aa3d` and its
canonical report SHA-256 is
`758d1aeebfac17c52916a9e03f90871cf57842cdbdb76946be642ad7e6594ca9`.
The executed runner SHA-256 was
`eb922f71de856c2c89b86a1d074a1c375f11a676d96b8e89aacb372a4c0e7f0c` and the
config SHA-256 was
`8ba3b705033b3b4c7eb45eeca0e21726e15165e8b67fe6f1ae0c69c403434fc4`.

The `(Physics17-C3)` additive residual direction is closed in its current
form. No further alpha or residual-form search, reconstructed-v93 training,
full-origin reconstruction, package, or submission work is authorized from
this result.

## Frozen authorities

The JSON configuration pins the exact v93 metadata and logits:

- r2022 bounded raw logits: SHA-256
  `d3972a5cc7b5c70118e06bb25042b948c399646690ed531e2ddf783ebdda874e`
- r2023 bounded raw logits: SHA-256
  `29d1dec441912b5cea97b1f222c94e34bc604ec7c0aacb361b649fe9991fb858`
- metadata: SHA-256
  `7b8ee3e04eff55e35bcd37374ceafc0db315b6812a288b738465198c5d81bc40`

The reviewed full-origin matched archives are pinned separately:

- r2022 C3/Physics17: SHA-256
  `78b216af2253f0b9bbb15004b153fcfa46360484afe938736114c1255b8f7164`
- r2023 C3/Physics17: SHA-256
  `e8e926bbd74bdc417476a7722c91936ac62cf72c788f44b24621ef7448e6a10f`

The reviewed complementarity report is also pinned. Its literal SHA-256 is
`d6e7be274ad3dad90d83631b6da36560d8aabd235d46f650355426d42c3dccf7`, its
canonical report SHA-256 is
`364595a50ce2f340702426af84b186b917ad6f2ed6d899c1cea497e09b52a06b`, and its
runner SHA-256 is
`2fd8570e4b94644f3ddc4369af5d61b57c7e5a85c5a31decd26d7b0570f8bb2e`.
The report's per-origin NPZ literal hashes, full row counts, position hashes,
row-ID hashes, and post-seal label flags must match the JSON configuration.

The archives contain exactly the `c3` and `physics17` probability arrays and
are independently checked for finite strict `(0,1)` values. The full origin
position and row-identity hashes from the reviewed complementarity evidence
are required before the first 30,000 positions are sliced.

## Row scope and firewall

The source projection reads only:

```text
row_id, season, game_type
```

For each origin, the full validation scope is regular-season rows with the
origin season, in source order. The bounded scope is the first 30,000 true
positions from that full scope. The expected full-origin position and row-ID
hashes are frozen in the configuration. Any count, order, or hash mismatch
fails closed.

Only after the v93 logits, C3 probabilities, Physics17 probabilities, row
positions, and row IDs have been sealed are the corresponding `control_success`
labels read. Only `r2022` and `r2023` are accepted. Primary, r2024, test,
Public, leaderboard, and external sources are not part of this runner.

## Residual and deployed form

The sole residual is computed in logit space:

```text
d_tm = logit(p_Physics17) - logit(p_C3)
z(alpha) = z_v93 + alpha*d_tm
p(alpha) = clip(sigmoid(z(alpha) - 0.0461645795229729), 0.30, 0.70)
```

The alpha domain is frozen to `[0, 1]`. Alpha zero is the preserved v93
deployed form; no v93 leg, weight, offset, or clip is changed.

## Alpha characterization

For each origin:

```text
g(alpha) = Brier(0) - Brier(alpha)
```

The fixed numerical contract is:

- 1,001 equally spaced grid points on `[0,1]`;
- right derivative step `1e-6`;
- derivative improvement tolerance `1e-12`;
- sign tolerance `1e-12`;
- boundary tolerance `1e-12`;
- at most 80 bisection iterations.

The grid is only a deterministic sign/bracketing instrument. The runner first
requires an improvement-directed right derivative and a positive value at the
frozen right-derivative step. It then scans only until the first non-positive
grid point, using that derivative-step value as the left bracket when the
first grid interval already contains the boundary. The first
positive-to-non-positive bracket is refined by bisection, and the component
adjacent to zero is recorded as `(0,U_origin)`. Any later disconnected
positive region is ignored. If improvement remains positive through the
entire frozen domain, the domain endpoint is recorded as `U_origin=1`.
Otherwise, if the component or its first boundary cannot be established,
that origin fails closed.

The shared interval is `(0,min(U_r2022,U_r2023))`; when non-empty, the frozen
diagnostic alpha is its midpoint `U/2`. Direct Brier recomputation must improve
both origins for `BOUNDED_SIGNAL_PRESENT`. Otherwise the result is
`BOUNDED_SIGNAL_ABSENT`. Authority or row-seal failures are contract errors.

This diagnostic has no BSS or promotion gate and does not authorize any later
full-origin reconstruction.

## CHEAP verification

```bash
python -m py_compile \
  scripts/exact_v93_bounded_trackman_residual_diagnostic_v0.py \
  tests/test_exact_v93_bounded_trackman_residual_diagnostic_v0.py
python -m unittest tests.test_exact_v93_bounded_trackman_residual_diagnostic_v0
git diff --check
```

The tests use only synthetic frames and vectors. They cover authority/config
guards, exact row sealing, label ordering, residual algebra, disconnected
alpha regions, derivative failure, boundary refinement, two-origin conjunction,
deterministic report hashing, output-directory safety, and the zero-model-fit
firewall.
