# F3 Approval Gate — Hands-on QA (next slots)

**Package**: `repro_979/submit_lgb_mlp_cat_20260814-2258` (lgb_mlp_cat, `5890a4c54f502c4e`)
**Recorded**: 2026-08-14 (KST), local CPU-only. All checks executed for real — no dry runs.
**Data note**: local `test.csv` is the official 5-row sample; the 245,789-row full run uses the validator's synthetic fixture (train bootstrap, seed 42) — same input as task-8 evidence.

## 1. Package inference — 5 rows ✅ PASS
```
LGA_TEST_PATH=/tmp/qa_f3/fixtures/test5.csv LGA_SAMPLE_PATH=.../sample5.csv LGA_OUT_PATH=.../out5.csv python3 script.py
```
- exit **0**, wall 4.2s (script 0.9s)
- 5 rows, NaN 0, min 0.4231 / max 0.4797, all within **[0.30, 0.70]** ✅

## 2. Package inference — full 245,789 rows ✅ PASS
```
LGA_TEST_PATH=.../cache/t8_install_sim/lgbmlpcat_fixtures/test_245789.csv LGA_SAMPLE_PATH=.../sample_submission_245789.csv LGA_OUT_PATH=/tmp/qa_f3/fixtures/out245789.csv python3 script.py
```
- exit **0**, wall **19.1s** (<< 600s budget)
- **245,789 rows**, NaN 0, mean **0.51415** ∈ [0.40, 0.55]
- frac < 0.30 = 0, frac > 0.70 = 0 (max 0.70 = exact clip bound, not a violation) ✅

## 3. Validator re-run ✅ PASS
```
python3 repro_979/package_validator_lgb_mlp_cat.py --submission-dir repro_979/submit_lgb_mlp_cat_20260814-2258
```
- **RESULT: PASS**, exit **0**, 10/10 gates:
  - offline_install_sim 0.52s (versions exact) · five_row n=5 · full_245789 n=245789, 19.4s
  - parity max|Δ| = **5.6e-17** · mean_alignment 0.0 · vs champion policy 1.97e-04
  - champion_component vs GIHO 5.6e-17 · zip_layout 33 model files, no stray · provenance 33/33 hashes · migration_audit PASS
- ⚠️ Validator **writes task-8 evidence unconditionally** (no CLI override for evidence path) → backed up committed evidence to `/tmp/qa_f3/task-8-package-lgbmlpcat.json.bak` before running, **restored after**; sha256 identical (`adb0c2a9…`), git status clean.
- First attempt failed only due to my cwd (relative `--submission-dir` from `repro_979/`) — operator error, rerun from root as README instructs.

## 4. Decision protocol fixtures ✅ (all via `--state`/`--package-dir`/`--evidence-base` overrides; real `leaderboard_state.json` NEVER modified)

| Fixture | Setup (copy of state) | Exit | Decision | block_reasons | Result |
|---|---|---|---|---|---|
| (a) current state | real state read-only | 1 | BLOCK | `daily_slot_exhausted` (2026-08-14, 1/1) | ✅ |
| (b) stale cutoff | date_captured → 2026-08-12T12:00+09:00 (~50h > 24h TTL) | 1 | BLOCK | `cutoff_stale` (+ daily_slot) | ✅ |
| (c) digest mismatch | `--package-dir` = OLD 2006 pkg + NEW evidence | 1 | BLOCK | `qualification_digest_mismatch` (computed `bcc33f6f…` ≠ registered `e3e047dd…`) | ✅ |
| (d) ALLOW | submissions_by_date=`{"2026-08-15":0}`, fresh date_captured | **0** | **ALLOW** | none (5/5 gates PASS, margin +88.71) | ✅ |

Digest-mismatch detail: 2006 vs 2258 model hashes are identical (33/33), so gate b correctly reaches the digest comparison and fires `qualification_digest_mismatch` — anti-tamper path verified, not `model_files_changed`.

## Not executed
None of the decision fixtures needed code-only fallback (`--state` override exists). The only unavailable input is the real 245,789-row test.csv (not shipped locally) — covered by the documented synthetic fixture (task-8 identical methodology).

## Verdict: **APPROVE**
All 6 executed checks PASS. Inference exit 0 / row counts / bounds / mean / runtime all verified; validator 10/10 PASS with evidence restored; decision protocol verified in both directions (BLOCK×3 with correct reasons, ALLOW×1). No committed files modified.
