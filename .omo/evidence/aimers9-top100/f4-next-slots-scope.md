# F4 — Scope and Records Audit: aimers9-blend-catboost-next-slots

**Audit date:** 2026-08-14
**Audited commit:** `c3252a7` (on top of `d89cf6c`; `git log --oneline -3` confirmed)
**Auditor mode:** read-only; only this file written. No commit/push performed.

---

## Check 1 — Scope containment: no unrequested model families / policy changes — **PASS**

Executed scope per plan and commit `c3252a7` (20 files, 2,842 insertions):
T1 package rebuild (`submit_lgb_mlp_cat_20260814-2258/` 6 sources), T2 validator alignment (`package_validator_lgb_mlp_cat.py`), T3 register+check (`leaderboard_state.json`), T4-5 catboost9 screen runner (`catboost9_screen.py` + task-10 evidence), T6 promotion gate (task-11 evidence), T9 record+commit. No other functional code.

Out-of-scope topic sweep (all must be ABSENT as implementations):

| Topic | Evidence |
|---|---|
| DeepFM / DCNv2 / FT-Transformer | `git show c3252a7 | grep -icE 'deepfm|dcnv|ft.?transformer'` → **0** matches; no evidence-file matches. ABSENT |
| Subgroup / isotonic / Platt calibration | `git show c3252a7 | grep -icE 'subgroup|isotonic|platt'` → **0** matches. ABSENT (pre-existing `task-6-calibration*` files are from the previous plan and NOT in this commit) |
| XGBoost rescue | Only occurrence: `assert_gpu()` helper in `submit_lgb_mlp_cat_20260814-2258/common.py:36-44`. Verified **byte-identical** to the 2006 template (`sha256` both `fdf4b7f443650161c17f9eb63837cf117c9c8a5a726985ad5d2398109951353d`; template committed earlier in `c8537b7`, pre-dating this plan). Not imported by `script.py`, absent from `requirements.txt`, dead helper copied verbatim — not a rescue implementation. One stray string `"xgboost": "3.2.0"` in `task-8-package-lgbmlpcat-audit.json`, which is **untracked and not in the commit**. ABSENT as implementation |
| C_LOGIT policy change | `C_LOGIT = -0.0404` preserved everywhere: `script.py` ("챔피언 정렬 상수"), `package_validator_lgb_mlp_cat.py` ("동결"), `catboost9_screen.py` (`"c_logit_applied": False`, raw-sigmoid BSS). Formula `p = clip(sigmoid(0.30*z_lgb + 0.35*z_mlp + 0.35*z_catboost + C_LOGIT), 0.30, 0.70)` — same constant, only blend weights changed (0.30/0.45/0.25 → 0.30/0.35/0.35 per candidate `5890a4c54f502c4e`). NO policy change |

## Check 2 — Conditional-task discipline (T7/T8 skipped after T6 REJECT) — **PASS**

- `ls repro_979/cache/ | grep -i catboost9_full` → no match (exit 1). Only `cache/qualification/catboost9/` (screen OOF, 8 files) exists — the *screen* cache, not a full-train manifest.
- `ls repro_979/ | grep -i 'submit.*catboost9'` → no match (exit 1). Package dirs present: `...-2006`, `...-2258`, `champ_cat`, `plato3b2`, `plato3b2_c0431` — no catboost9 package.
- No `task-12-catboost9-fulltrain*` / `task-13-catboost9-package*` evidence exists (`ls .omo/evidence/aimers9-top100/`).
- Consistent with task-11 verdict: "do **not** run 10-seed full training or Todo 7-8 packaging."

## Check 3 — Notion timing: immediate local-CV record + REJECTED amendment — **PASS**

- **Immediate recording (T5):** notepad "catboost9_screen.py FULL PASS" entry records the local-CV row on **2026-08-14** (same day as run, `recorded_at_utc 2026-08-14T14:25:50` in `task-10-catboost9-schema-full.json`), with Brier `0.248264/0.248716/0.248916/0.248460` and r `0.4861/0.5037/0.5031/0.4897` — recorded right after the screen, not batched.
- **Amendment (T9):** notepad session-close entry documents 3 targeted `update_content` edits appending the REJECTED outcome (모델/BSS/비고 columns); Brier/r untouched.
- **Live verification (read-only Notion API):** page `3b55ed6b-28d5-81a0-80da-fe91eec240c6` 로컬 CV table contains the row: `2026-08-14 | catboost9 C1 (9-cat MLP_CATS, seeds 42,47) — PROMOTION REJECTED | primary+r2022+r2023+r2024 | 617.76 / 508.21 / 429.91 / 573.97 | 0.248264 / ... / 0.248460 | 0.4861 / ... / 0.4897 | ...promotion REJECTED — primary BSS 617.76 vs 5-cat 732.76 (Δ-115.0), 잔차상관 0.9992, residual alignment 0.0038; Todo 7-8 (full train/package) 미실행`. Brier/r match the T5-recorded values; BSS/비고 carry the T9 amendment — exactly as notepad claims. Caveat: precise intra-day wall-clock timestamps are not independently verifiable; same-day (2026-08-14) immediate recording is confirmed by both notepad and live page state.

## Check 4 — Allowlist staging: no forbidden paths in commit — **PASS**

`git show --name-only --format="" c3252a7 | grep -E '\.omo\.gpu01|베이스라인|데이터|\.zip$|\.npy$|\.cbm$|\.pt$|cache/|model/'` → **no matches** (exit 1).
- Committed paths: 11 evidence files (task-8×2, task-9×2, task-10×2, task-10-full×2, task-11×3), `catboost9_screen.py`, `package_validator_lgb_mlp_cat.py`, `leaderboard_state.json`, 6 package sources (`script.py`/`common.py`/`mlp_model.py`/`requirements.txt`/`README.md`/`provenance.json` — `model/` contents correctly gitignored, not force-added).
- Worktree: `git status --short` = exactly 3 untracked paths (`.omo.gpu01-backup-20260814-030428/`, `베이스라인/`, `베이스라인 실험용/`) — matches expectation; **no staged files** (`grep -v '^??'` → empty). Evidence files were force-added individually (`git add -f` per notepad); the pre-commit staged list matched the allowlist exactly.

## Check 5 — Success criteria mapping — **SATISFIED (4/4)**

| # | Criterion | Artifact | Status |
|---|---|---|---|
| 1 | Candidate `5890a4c54f502c4e` uniquely packaged, 10-gate validated, registered, submission-checked with explicit args; upload only on ALLOW | `submit_lgb_mlp_cat_20260814-2258/` (formula/provenance/README all 0.30/0.35/0.35); `task-8-package-lgbmlpcat.{json,log}` (10/10 PASS, parity 5.55e-17); `leaderboard_state.json` qualified_package (candidate `5890a4c54f502c4e`, digest `e3e047dd...`, recompute MATCH, 33/33 model rehash); `task-9-submission.{json,md}` (explicit `--package-dir`/`--task8-evidence`, BLOCK `daily_slot_exhausted`, gates a/b/c/e PASS, cutoff fresh, margin +88.71) — no upload since no ALLOW | ✅ |
| 2 | CatBoost C1 two-seed/four-fold all-9-category OOF evidence, explicit typing, no leakage | `task-10-catboost9-schema-full.{json,log}`: 8 records (seeds 42,47 × 4 folds), exact 9-cat schema, cache digest `a41e9b22...`, val rows 253507/217024/219839/223497, no 2025 rows, no float coercion, file-bytes==meta==evidence hashes | ✅ |
| 3 | Promotion primary-only + R-only transfer evidence; failure rejected with evidence; no full train/package on failure | `task-11-catboost9-promotion.{json,log,md}`: **REJECTED** — primary BSS 617.76 vs 5-cat 732.76 (Δ −115.0, floor −10 FAIL), residual corr 0.999203 (FAIL <0.98), alignment 0.003761 (FAIL ≥0.02), raw-logit corr <0.96 passed but not decisive; blend-transfer probe skipped per protocol; no R-fold labels used; T7-8 explicitly not executed (Check 2) | ✅ |
| 4 | Immediate Notion SSOT records + commit with only intended paths | Live Notion row verified (Check 3); `c3252a7` 20-file allowlist clean (Check 4), `feat(repro):` message per `type: 설명`; no premature leaderboard row (Check 6) | ✅ |

## Check 6 — No premature submission recording — **PASS**

- `repro_979/leaderboard_state.json`: `submissions_by_date = {"2026-08-14": 1}` — **no 2026-08-15 row**; `last_submission` still the real 2026-08-14 champ_cat score (988.3196750794); `qualified_candidate_public_score` not pre-filled.
- Notion 🏆 리더보드 table (read-only): last row = 2026-08-14 `submit_champ_cat.zip` Public 988.3196750794 (real upload). No 2026-08-15 row.
- Notepad confirms: "NO 2026-08-15 row — user adds after real upload."

---

## Final verdict: **APPROVE**

All 6 checks PASS. No scope creep (no DeepFM/DCNv2/FT-Transformer/XGBoost-rescue/subgroup-calibration/C_LOGIT-change implementations; the single XGBoost token is a pre-existing, byte-identical template helper). T7/T8 correctly skipped after T6 REJECT. Notion local-CV recording was same-day immediate and amended with the REJECTED outcome (live page corroborates). Commit `c3252a7` is allowlist-clean with no forbidden paths and no staged leftovers. No premature 2026-08-15 submission recorded anywhere.
