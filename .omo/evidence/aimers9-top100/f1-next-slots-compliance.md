# F1 — Plan-Compliance Audit: `aimers9-blend-catboost-next-slots`

- **Audit date**: 2026-08-14
- **Plan**: `.omo/plans/aimers9-blend-catboost-next-slots.md` (Todos 1–9, F1–F4)
- **Audited commit**: `c3252a7a0028f9d916ccfefce05774feee7afe0c`
- **Auditor mode**: read-only (no file modified; only the f1 report written)

## Summary

Completed todos: **1, 2, 3, 4, 5, 6, 9** — all PASS. Todos **7, 8** were correctly
**not executed** (conditional on Todo 6 promotion, which REJECTED) — verified by
absence of all their artifacts. **Verdict: APPROVE.**

## Per-check results

| # | Check | Result | Evidence |
|---|-------|--------|----------|
| 1 | Todo 1 — package rebuild (candidate/weights/hashes) | **PASS** | provenance.json:4,8-12,160-164; script.py:41-43; sha256 comparison of `model/*` (33/33 match) |
| 2 | Todo 2 — validator 10-gate PASS | **PASS** | task-8-package-lgbmlpcat.json:4,121; .log:32-41 |
| 3 | Todo 3 — register + check BLOCK | **PASS** | leaderboard_state.json:28-33; task-9-submission.json:4,13-14,68-70; state:15-17 |
| 4 | Todo 4 — catboost9_screen.py source | **PASS** | commit name-status (`A repro_979/catboost9_screen.py`); catboost9_screen.py:77-80,240-261,393-394; qualification_runner.py:86-89,248-256 |
| 5 | Todo 5 — full 2-seed × 4-fold screen | **PASS** | task-10-catboost9-schema-full.json:5,7-10,174,217,292,367,442 |
| 6 | Todo 6 — promotion gate REJECTED | **PASS** | task-11-catboost9-promotion.json:144-148,78,95,130-142 |
| 7 | Todo 7-8 — MUST NOT have executed | **PASS** | no task-12/13 evidence; `cache/qualification/catboost9/` has only 4 fold `.npy/.meta.json`; no `submit_*catboost9*` dirs |
| 8 | Todo 9 — Notion local-CV + allowlist commit | **PASS** | Notion page `3b55ed6b-...` local-CV table (live API read); commit name-status = exactly 20 allowlist files |
| 9 | Forbidden patterns (R-fold selection / C_LOGIT / 2025) | **PASS** | task-11 JSON:12,17,130-134; C_LOGIT=-0.0404 in all changed sources; leakage guards |

### Check 1 — Todo 1: package rebuild — PASS

- `provenance.json:4` → `"candidate_id": "5890a4c54f502c4e"` ✓
- `provenance.json:8-12` + `:160-164` → weights `{"lgb": 0.3, "mlp": 0.35, "catboost": 0.35}` ✓
- `script.py:41-43` → `W_LGB = 0.30`, `W_MLP = 0.35`, `W_CAT = 0.35` ✓
- 33 model hashes: programmatic `sha256sum` of `model/*` (33 files) vs
  `provenance.json.model_file_sha256` (33 entries) → **ALL 33 MATCH**,
  mismatches `{}`, extra `{}`, missing `{}` ✓
- C_LOGIT preserved (`provenance.json:14` = -0.0404; script.py:44) ✓
- Package path differs from old candidate (`submit_lgb_mlp_cat_20260814-2258` vs `-2006`) ✓

### Check 2 — Todo 2: validator — PASS

- `task-8-package-lgbmlpcat.json:121` → `"result": "PASS"` ✓
- `task-8-package-lgbmlpcat.json:4` → `"submission_dir": "repro_979/submit_lgb_mlp_cat_20260814-2258"` ✓
- Log `.log:32-41` → all **10 gates PASS**:
  `offline_install_sim`, `five_row`, `full_245789` (245,789 rows, 25.9 s < 600 s),
  `parity_script_vs_reference` (max|Δ| = 5.551e-17 < 1e-6),
  `mean_alignment` (0 ≤ 0.005), `mean_alignment_vs_champion_policy` (1.97e-4),
  `champion_component_vs_giho` (5.551e-17), `zip_layout` (33 model files, no stray),
  `provenance_integrity` (33 files, candidate 5890a4c54f502c4e, no problems),
  `migration_audit_strict` (RESULT: PASS) ✓
- Validator diff (commit): only old-candidate constants updated
  (W_MLP 0.45→0.35, W_CAT 0.25→0.35, candidate literal `7771a019594deedd`→`5890a4c54f502c4e`,
  default dir 2006→2258); C_LOGIT/-0.0404 untouched ✓

### Check 3 — Todo 3: register + check — PASS

- `leaderboard_state.json:28` → `qualified_package.candidate_id == "5890a4c54f502c4e"`,
  `qualification_digest` = `e3e047dde80f28b31bb9b98b498e37c9ca95cca80686e23207fe875b50487f72` ✓
- `task-9-submission.json:4` → `"decision": "BLOCK"`, `:68-70` →
  `block_reasons == ["daily_slot_exhausted"]` ✓
- Digest recompute vs registered match (`:13-14` computed == registered) ✓
- `leaderboard_state.json:15-17` → `submissions_by_date` has **only** `{"2026-08-14": 1}`,
  **NO 2026-08-15 row** (no pre-recorded future submission) ✓
- Gates a/b/c/e PASS, only d-daily-slot FAIL (task-9-submission.md:22-26) ✓

### Check 4 — Todo 4: catboost9_screen.py — PASS

- In commit c3252a7 (`A repro_979/catboost9_screen.py`, 528 lines) ✓
- Exactly-9-unique assert: module level `catboost9_screen.py:77-80`
  (`assert len(CAT_SCHEMA) == 9`, `assert len(set(CAT_SCHEMA)) == 9`), plus
  per-fold `:203` (`len(cat_idx) == 9`); schema = `MLP_CATS`
  (qualification_runner.py:86-89 — pitcher/batter/team IDs ×2 + 5 LGB cats, exactly 9) ✓
- Cache-digest stale guard: `_cache_guard` `:240-261` — digest mismatch →
  `RuntimeError` fail-closed (reuse refused); only explicit `--overwrite-cache`
  escape hatch; digest binds cat_schema names+order / dtype policy / features /
  seeds / folds / params (`:134-149`) ✓
- Leakage guard: validation masks checked for season==2025 via
  `_check_leakage` (qualification_runner.py:248-256); training masks additionally
  checked in `catboost9_screen.py:393-394`; any hit → hard failure ✓

### Check 5 — Todo 5: full screen — PASS

- `task-10-catboost9-schema-full.json:5` → `"mode": "full"` ✓
- `:7-10` → `seeds_requested == [42, 47]` ✓
- `:174` → `integrity.gate == "PASS"`; `:180-183` → no 2025 in any validation/
  training mask = true; `:184-201` → all four `expected_n_va_matched == true` ✓
- Per-fold row counts match frozen expectations:
  `primary` 253507 (`:217`), `r2022` 217024 (`:292`), `r2023` 219839 (`:367`),
  `r2024` 223497 (`:442`) ✓
- Eight seed-fold records: 4 folds × per_seed {42, 47} (`:271-282` etc.) ✓
- Full-run log confirms exit 0 with 4-fold PASS lines
  (task-10-catboost9-schema-full.log:3-11) ✓
- Smoke QA (Todo 4 QA): task-10-catboost9-schema.log:12 `[--smoke] PASS`, seeds [42,43] ✓

### Check 6 — Todo 6: promotion gate — PASS

- `task-11-catboost9-promotion.json:144` → `decision.verdict == "REJECTED"` ✓
- `:147` → `proceed_to_todo_7_8 == false` ✓
- Internally consistent rationale:
  - primary BSS catboost9 = 617.755108 (`:78`) vs 5-cat = 732.760084 (`:78`)
  - Δ = 617.7551080470467 − 732.7600845669302 = **−115.0049765198835 ≈ −115.0** ✓
    (matches commit message "Δ-115.0" and `:101` actual_delta_bss)
  - residual corr with champion = **0.9992031089895346** (`:95`) < 0.98 gate FAIL (`:107`) ✓
  - residual alignment 0.003760832887376978 (`:95`) < 0.02 gate FAIL (`:108`) ✓
  - screening gate FAIL (`:109`), rationale (`:110`) — consistent with all numbers ✓
- Blend-transfer probe: `SKIPPED_BY_PROTOCOL` (`:131`), weights all null (`:136-138`),
  `labels_used_for_weight_fitting == []`, `r_only_labels_used == false` (`:133-134`) ✓

### Check 7 — Todos 7-8: MUST NOT have executed — PASS

- No `task-12-catboost9-fulltrain.*` or `task-13-catboost9-package.*` evidence files
  (evidence dir listing, none present) ✓
- `repro_979/cache/qualification/catboost9/` contains **only** the four 2-seed OOF
  outputs (`{primary,r2022,r2023,r2024}.{npy,meta.json}`) — **no full-train manifest** ✓
- No `submit_*catboost9*` package dirs (ls of `repro_979/submit_*` shows none) ✓
- No 10-seed deploy artifacts; commit message explicitly states "Todo 7-8 not executed" ✓

### Check 8 — Todo 9: record + commit — PASS

- **Notion local-CV recorded** — verified live via Notion API (page
  `3b55ed6b-28d5-81a0-80da-fe91eec240c6`, 📊 로컬 CV 결과 table): row
  `2026-08-14 | catboost9 C1 (9-cat MLP_CATS, seeds 42,47) — PROMOTION REJECTED |
  primary+r2022+r2023+r2024 | 617.76 / 508.21 / 429.91 / 573.97 |
  0.248264 / 0.248716 / 0.248916 / 0.248460 | 0.4861 / 0.5037 / 0.5031 / 0.4897 |
  ...promotion REJECTED — primary BSS 617.76 vs 5-cat 732.76 (Δ-115.0), 잔차상관 0.9992,
  residual alignment 0.0038; Todo 7-8 (full train/package) 미실행` ✓
  (Values match task-10-full BSS exactly; notepad corroborates the 3-targeted-edit
  update via `update-page-markdown`.)
- **Commit contains ONLY allowlist files** — `git show --name-status c3252a7` = exactly
  20 files:
  - package source 6: `script.py`, `common.py`, `mlp_model.py`, `requirements.txt`,
    `README.md`, `provenance.json` (under `submit_lgb_mlp_cat_20260814-2258/`) ✓
  - `package_validator_lgb_mlp_cat.py`, `catboost9_screen.py`, `leaderboard_state.json` ✓
  - evidence 11: task-8 ×2, task-9 ×2, task-10 ×2, task-10-full ×2, task-11 ×3 ✓
- **No forbidden paths in commit**: no `.omo.gpu01-backup*`, `베이스라인*`,
  `데이터*`, `*.zip`, `*.npy`, `*.cbm`, `*.pt`, `cache/` (name-status confirms;
  `git status` shows backup/baseline dirs still untracked) ✓
- Commit message `type: 설명` compliant (`feat(repro): ...`) ✓

### Check 9 — Forbidden patterns — PASS

- **No R-fold labels in weight selection**: protocol `selection_fold == "primary"`
  (task-11:12); blend-transfer probe skipped, `r_only_labels_used == false` (task-11:134);
  screening gates evaluated on primary only; baselines reused from prior
  task-7-blend.json evidence (task-11:112-128) ✓
- **No C_LOGIT change**: `-0.0404` everywhere — script.py:44, provenance.json:14/165,
  validator diff (constant untouched), README; catboost9 screen explicitly
  `c_logit_applied: false` (task-10-full:206) with raw-sigmoid BSS policy ✓
- **No 2025/test-row data**: leakage guards `no_2025_in_any_validation_mask`/
  `no_2025_in_any_training_mask == true` (task-10-full:180-183); `no_2025_in_validation`
  (task-11:25); screen source rejects 2025 in any mask (catboost9_screen.py:393-394);
  package models trained 2019-2024 only (README); grep of evidence shows only
  guard assertions, no 2025 data references ✓

## Observations (non-blocking)

1. **Consolidated commit**: the plan's per-todo atomic commit strategy
   (7 commits) was compressed into the single allowlist commit c3252a7. The audit
   scope defines the commit check against this one commit, which passes exactly.
2. **Todo 9 plan evidence naming**: plan referenced `task-14-recording-commit.{json,md}`;
   that file was not produced. Todo 9 acceptance was instead verified directly:
   Notion local-CV row confirmed via live API read, and the commit allowlist verified
   via name-status. Stronger than the referenced artifact would have been; no
   information gap.
3. `task-8-package-lgbmlpcat-audit.json` exists in the evidence dir but is not in the
   commit allowlist — left untracked deliberately (matches explicit-file staging policy).
4. Evidence JSONs record `git_commit: d89cf6cc` (pre-final-commit SHA) — normal, since
   evidence was generated before the closeout commit; content hashes all match.

## Verdict: **APPROVE**

All 9 audit checks PASS with direct, cited evidence. No forbidden artifact, no
R-fold selection, no C_LOGIT drift, no 2025/test-row usage, no unqualified upload
(BLOCK on daily_slot_exhausted is the correct conservative outcome), and Todos 7-8
correctly not executed after the promotion gate REJECTED. The plan is fully compliant
and may proceed to F2 (quality/leakage audit), F3 (hands-on package/decision QA),
and F4 (scope/records audit).
