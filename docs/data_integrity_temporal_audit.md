# Data Integrity, Temporal Structure, ABS-Regime, and Trackman Usability Audit

This audit is diagnostic only. It does not train models, score candidates, alter
recovery policy, or establish official submission readiness. The implementation
exposes a reviewed CHEAP guard layer and a MEDIUM aggregate runner. A real official-
data MEDIUM run still requires separate authorization.

## Information boundary

- Structural analysis is limited to official `train.csv` seasons 2019–2023.
- Target-aware exploratory analysis is limited to 2019–2021.
- Target use for the bounded representativeness diagnostic is limited to the
  pre-registered r2022/r2023 origin being checked.
- No test-distribution, Public/leaderboard, web, API, or external-data evidence is
  permitted.
- Trackman is inspected only as officially supplied historical data. Identifier
  overlap is empirical and is not an authoritative player crosswalk.

The isolated 2024 diagnostic is feature-only. Its reader projects
`TRAIN_FEATURE_COLUMNS` with `usecols`, excluding `control_success`. No 2024 target
value may be extracted, converted, aggregated, returned, used for branching, or
reported. Because CSV is row-oriented, this is not a claim that target bytes are
physically unread by the parser.

Any 2024 feature-only finding remains branch-inert: it may describe a structural
break but may not tune a threshold, select a feature, choose a transformation, or
change recovery policy. A later ABS modeling hypothesis requires a separate
Experiment Brief.

## CHEAP commands

Static source and policy guards:

```bash
python scripts/audit_data_integrity_temporal.py static --repo-root .
```

Header-only official-file validation (zero data rows):

```bash
python scripts/audit_data_integrity_temporal.py headers \
  --train-csv /path/to/train.csv \
  --trackman-csv /path/to/trackman_history.csv
```

The semantic source guard examines executable imports, CLI destinations, data-reader
paths, the feature-only target role, and external-information access. It deliberately
does not reject comments or documentation merely for containing words such as
“2024”, “test”, “Public”, or “leaderboard”.

## First-30000 pipeline geometry

The audit does not substitute one generic mask for all consumers. It reproduces and
reports each first-30,000 selection in current DataFrame order independently:

| Pipeline | Origin | Bounded roles recorded |
| --- | --- | --- |
| Recovery CatBoost | r2022, r2023 | inner train, inner validation, outer train, outer validation |
| Baseline reproduction | r2022, r2023 | pretrained outer validation only; no training mask |
| Residual base OOF | validation years 2020–2023 | base OOF train and validation |
| Residual correction | r2022, r2023 | each correction-fit OOF validation year and outer apply |
| Residual C selection | r2023 | 2020 fit and 2021 score masks |
| Calibration | r2022, r2023 | baseline-cache logits, bounded labels, transform apply |
| Calibration fit panel | r2023 | 2022 transform-fit panel |

Every role reports its full and selected sizes, source-position span, truncation
rule, mask hash, and row-ID hash. The future MEDIUM comparison also reports
season/month and game-type composition, categorical total variation, numeric
standardized differences, missing-rate deltas, and player/team coverage. The
r2022/r2023 target-rate delta is a bounded-bias diagnostic only.

All real-data helpers share one scalar missing-value rule covering `None`, the empty
string, NaN, and pandas NA-like values. Missing numeric values are excluded from
mean/std/SMD calculations and included in missing-rate calculations. Missing as-of
rates, including cold starts, are not by themselves domain violations.

## Evidence and verdicts

Findings use `BLOCKER`, `LIKELY_ISSUE`, `BENIGN`, or `UNKNOWN`, together with
`DOCUMENTED_CONTRACT`, `EMPIRICALLY_VERIFIED`, `INFERENCE`, or `NOT_PROVEN`.
Pipeline preprocessing verdicts are kept separate as `PROVEN`, `VIOLATION`,
`NOT_APPLICABLE`, or `NOT_PROVEN`.

Static confirmation that a source path, function, or call exists is empirical source
presence only. It does not prove preprocessing fit scope or dataflow semantics, so
the CHEAP source-presence findings retain a pipeline verdict of `NOT_PROVEN`.

Exact intra-month chronology and exact as-of reconstruction remain `NOT_PROVEN`
without authoritative date/game/pitch-order keys and feature-generation logic. A
decreasing as-of count in CSV order is not automatically a violation. Trackman key
uniqueness and main/Trackman identifier compatibility are empirical unless official
documentation explicitly guarantees them; numeric or string overlap alone never
creates a crosswalk.

Trackman usability reports all A–E levels, including explicit `E_unusable`. If none
of A–D is proven safe, unusability remains `NOT_PROVEN` pending the empirical MEDIUM
coverage/key audit rather than being inferred from missing documentation alone.
Duplicates observed in `(trackman_game_id, pitch_no)` are reported as empirical
`UNKNOWN`: official uniqueness is not guaranteed and the audit does not use that
composite for a join. Such duplicates alone do not make the overall Trackman
structural contract a likely issue.

Historical behavior and current-policy admissibility are separate fields. In
particular, the historical MLP 2024-primary workflow and official baseline 2024-label
workflow can be proven descriptions of old code while remaining inadmissible for
current recovery selection and for this audit.

A `PROJECT_STATUS.md` evidence-base SHA mismatch is `BENIGN` documentary drift unless
a separate policy/state contradiction is found.

## MEDIUM commands and outputs

After a mandatory stop and separate authorization, the main audit is invoked as:

```bash
python scripts/audit_data_integrity_temporal.py medium \
  --train-csv /path/to/official/train.csv \
  --trackman-csv /path/to/official/trackman_history.csv \
  --output-dir /tmp/aimers9-data-integrity-audit \
  --repo-root .
```

It writes only aggregate findings outside Git:

```text
/tmp/aimers9-data-integrity-audit/audit_report.json
/tmp/aimers9-data-integrity-audit/audit_report.md
```

The optional 2024 feature-only diagnostic is a separate invocation and output:

```bash
python scripts/audit_data_integrity_temporal.py abs-2024-features \
  --train-csv /path/to/official/train.csv \
  --output-dir /tmp/aimers9-data-integrity-audit/abs-2024 \
  --repo-root .
```

```text
/tmp/aimers9-data-integrity-audit/abs-2024/abs_feature_report.json
/tmp/aimers9-data-integrity-audit/abs-2024/abs_feature_report.md
```

Outputs must not contain raw rows, individual labels, row-ID lists, model artifacts,
or submissions. The MEDIUM phase covers the 2019–2023 structural scan, per-pipeline
bounded-versus-full comparisons, real-data preprocessing provenance, Trackman
schema/coverage/usability, optional isolated 2024 feature-only diagnostics, and a
deterministic rerun check. No EXPENSIVE work belongs to this audit.

The main train reader first projects only `season` to discover allowed source-row
positions. It then uses the same `skiprows` firewall for the full non-target feature
projection and for a `row_id` + `control_success` projection, so interleaved 2024
feature and target rows are not materialized. Ordered row IDs and their hashes must
match exactly before targets are attached. The main Trackman reader likewise performs
a season-only discovery pass and materializes full Trackman columns only for
2019–2023; it neither materializes nor reports excluded 2024 Trackman rows.

The train source hash covers the scoped feature projection only; target use is
recorded in the role-based label-access ledger. The optional ABS command projects
features only for every year.

Both commands include Git/source hashes and a deterministic canonical report hash
that excludes runtime/timestamp fields. Output directories inside the repository are
rejected. Expected cost is MEDIUM: pandas full-file scans on CPU, with memory roughly
proportional to the projected train and Trackman frames; no model libraries are
instantiated and no BSS is calculated.

**STOP:** Do not run the real-data MEDIUM audit until its implementation and CHEAP
guards have been reviewed and execution is separately authorized.
