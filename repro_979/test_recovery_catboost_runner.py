#!/usr/bin/env python3
"""test_recovery_catboost_runner.py — Todo 4 (aimers9-top100-score-recovery) tests
(pytest-free).

Tests:
  (a) happy --screen-all -> exit 0; writes task-4-catboost.{json,md} with
      label_sources=[r2022,r2023], labels_read=True, immutable config (frozen
      CatBoost params, 5 low-cardinality cats, v93 blend reconciliation, bounded
      subset), per-variant futility + full screen verdicts, and scope-clean
      evidence (no forbidden path tokens / upload markers).
  (b) each of the 4 fixtures (id-as-category, terminal-label-read,
      changed-geometry, nonfinite-logit) exits 2, writes
      task-4-catboost-fixture-<name>.{json,md} with fixture+matched fields, never
      clobbers the main task-4-catboost evidence.
  (c) evidence naming / provenance compliance of all evidence the CLI writes.
  (d) structural firewall: the screen path reads only r2022/r2023 selection labels;
      primary read before freeze raises TerminalFirewallError.
  (e) frozen CatBoost config: C2 Lossguide geometry (max_leaves=63,
      min_data_in_leaf=500), C4 RMSE regression, 5 low-cardinality cats, v93 blend
      reconciliation constants.
  (f) composite formula: z_cand = z_blend + 0.0701*(z_cat - z_cat_base) — replacing
      the CatBoost leg within the v93 6-leg blend.

Usage: python3 repro_979/test_recovery_catboost_runner.py
Exit:  0 = all tests PASS, 1 = any FAIL.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent
ROOT = REPO.parent
CLI = REPO / "recovery_catboost_runner.py"

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))  # repro_979.* 패키지 import 용
import repro_979.recovery_policy as rp  # noqa: E402
import repro_979.recovery_catboost_runner as rcr  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        PASSED.append(name)
        print(f"  PASS {name} {detail}")
    else:
        FAILED.append(name)
        print(f"  FAIL {name} {detail}")


def run_cli(*args: str) -> subprocess.CompletedProcess[Any]:
    return subprocess.run([sys.executable, str(CLI), *args],
                          capture_output=True, text=True, cwd=str(ROOT))


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def scan_clean(rec: dict[str, Any]) -> list[str]:
    return rp.scan_scope(rec, Path("x.json"))


# ── (a) happy --screen-all ────────────────────────────────────────────
def test_screen_all(tmp: Path) -> None:
    print("[test] (a) happy --screen-all -> exit 0, immutable evidence")
    proc = run_cli("--screen-all", "--evidence-dir", str(tmp))
    check("a.exit0", proc.returncode == 0, f"(rc={proc.returncode})")
    ev = tmp / "task-4-catboost.json"
    check("a.evidence_written", ev.is_file(), str(ev))
    if not ev.is_file():
        return
    rec = load(ev)
    check("a.verdict_in", rec.get("verdict") in ("PASS", "REJECT"),
          str(rec.get("verdict")))
    check("a.md_written", (tmp / "task-4-catboost.md").is_file())
    check("a.labels_read", rec.get("labels_read") is True
          and rec.get("label_sources") == list(rp.SELECTION_ORIGINS),
          f"{rec.get('label_sources')}")
    cfg = rec.get("config") or {}
    check("a.cat_features", cfg.get("cat_features") == list(rcr.CATS),
          f"{cfg.get('cat_features')}")
    check("a.futility_seeds", cfg.get("futility_seeds") == [52, 53])
    check("a.full_seeds", cfg.get("full_seeds") == list(range(42, 52)))
    check("a.reconciliation", "0.0701" in str(cfg.get("reconciliation", "")))
    variants = rec.get("variants") or {}
    check("a.variants_present",
          all(cid in variants for cid in rcr.SELECTABLE_CAT_IDS),
          f"{sorted(variants)}")
    for cid in rcr.SELECTABLE_CAT_IDS:
        fut = variants.get(cid, {}).get("futility") or {}
        check(f"a.futility_{cid}", "survives_futility" in fut
              and "mean_delta_bss_vs_c1" in fut)
    check("a.scan_clean", not scan_clean(rec), f"{scan_clean(rec)}")


# ── (b) 4 fixtures ────────────────────────────────────────────────────
def test_fixtures(tmp: Path) -> None:
    print("[test] (b) 4 fixtures exit 2, fixture evidence, main not clobbered")
    main_ev = tmp / "task-4-catboost.json"
    main_sha = main_ev.read_bytes() if main_ev.is_file() else None
    for name in sorted(rcr.FIXTURES):
        proc = run_cli("--fixture", name, "--evidence-dir", str(tmp))
        check(f"b.{name}.exit2", proc.returncode == 2, f"(rc={proc.returncode})")
        fix = tmp / f"task-4-catboost-fixture-{name}.json"
        check(f"b.{name}.artifact", fix.is_file(), str(fix))
        if fix.is_file():
            rec = load(fix)
            check(f"b.{name}.fixture_field", rec.get("fixture") == name)
            check(f"b.{name}.matched", rec.get("matched") is True,
                  f"detail={rec.get('detail')}")
            check(f"b.{name}.labels_unread", rec.get("labels_read") is False
                  and rec.get("label_sources") == [])
            check(f"b.{name}.scan_clean", not scan_clean(rec), f"{scan_clean(rec)}")
    if main_sha is not None:
        check("b.main_not_clobbered", main_ev.read_bytes() == main_sha,
              "(main task-4-catboost.json byte-identical)")


# ── (c) evidence compliance ───────────────────────────────────────────
def test_evidence_compliance(tmp: Path) -> None:
    print("[test] (c) evidence naming / provenance compliance")
    for path in sorted(tmp.glob("*.json")):
        check(f"c.name_{path.name}", bool(rp.EVIDENCE_NAME_RE.match(path.name)), path.name)
        rec = load(path)
        if rec.get("fixture") is None:
            check(f"c.recorded_at_{path.name}",
                  bool(rec.get("recorded_at_utc")) and bool(rec.get("git_head")))
            check(f"c.config_hash_{path.name}", bool(rec.get("config_hash")))
            check(f"c.labels_{path.name}",
                  isinstance(rec.get("label_sources"), list))


# ── (d) structural firewall ───────────────────────────────────────────
def test_firewall(tmp: Path) -> None:
    print("[test] (d) screen reads only selection labels; primary blocked before freeze")
    import pandas as pd  # noqa: PLC0415
    train = pd.DataFrame({"season": [2020, 2021, 2022, 2024],
                          "game_type": ["R", "R", "R", "R"],
                          "control_success": [1, 0, 1, 0]})
    masks = rp.build_origin_masks(train)
    labels = rp.read_selection_labels(train, masks)
    check("d.selection_only", set(labels) == set(rp.SELECTION_ORIGINS),
          f"{sorted(labels)}")
    rp.set_firewall(rp.FIREWALL_UNFROZEN)
    try:
        rp.read_primary_labels(train, masks)
        check("d.primary_blocked", False, "primary read before freeze did NOT raise")
    except rp.TerminalFirewallError:
        check("d.primary_blocked", True, "primary read before freeze raised TerminalFirewallError")
    rp.set_firewall(rp.FIREWALL_UNFROZEN)


# ── (e) frozen CatBoost config ────────────────────────────────────────
def test_frozen_config(tmp: Path) -> None:
    print("[test] (e) frozen CatBoost config / geometry / cats / blend")
    reg = rcr.re.load_registry()
    c2 = (reg.get("candidates") or {}).get("catboost_c2_lossguide", {}).get("config") or {}
    check("e.c2_lossguide", c2.get("grow_policy") == "Lossguide"
          and c2.get("max_leaves") == 63 and c2.get("min_data_in_leaf") == 500)
    c4 = (reg.get("candidates") or {}).get("catboost_c4_rmse", {}).get("config") or {}
    check("e.c4_rmse", c4.get("loss_function") == "RMSE"
          and c4.get("eval_metric") == "RMSE")
    check("e.cats_5", len(rcr.CATS) == 5 and len(set(rcr.CATS)) == 5,
          f"{rcr.CATS}")
    check("e.no_id_cats", not any(c in rcr.CATS for c in
                                  ("pitcher_id", "batter_id", "pitcher_team_id",
                                   "batter_team_id")))
    check("e.features_49", len(rcr.FEATURES) == 49)
    check("e.lam_cat", abs(rcr.V93_LAM_CAT - 0.0701066994221915) < 1e-12)
    # C2 params from registry config
    params = rcr._catboost_params("catboost_c2_lossguide", 52)
    check("e.c2_params", params.get("grow_policy") == "Lossguide"
          and params.get("max_leaves") == 63
          and params.get("min_data_in_leaf") == 500
          and params.get("random_seed") == 52)
    check("e.c4_is_rmse", rcr._is_rmse("catboost_c4_rmse")
          and not rcr._is_rmse("catboost_c1_control"))


# ── (f) composite formula ─────────────────────────────────────────────
def test_composite(tmp: Path) -> None:
    print("[test] (f) composite replaces CatBoost leg within v93 blend")
    rng = np.random.default_rng(0)
    n = 100
    z_blend = rng.normal(0, 1, n)
    z_cat_base = rng.normal(0, 1, n)
    z_cat = rng.normal(0, 1, n)
    comp = rcr._composite(z_cat, z_blend, z_cat_base)
    expected = z_blend + rcr.V93_LAM_CAT * (z_cat - z_cat_base)
    check("f.composite_formula", np.allclose(comp, expected))
    # identity: if candidate cat == baseline cat, composite == baseline blend
    comp_id = rcr._composite(z_cat_base, z_blend, z_cat_base)
    check("f.identity", np.allclose(comp_id, z_blend))


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="t4cb_test_") as td:
        tmp = Path(td)
        test_screen_all(tmp / "a")
        test_fixtures(tmp / "a")
        test_evidence_compliance(tmp / "a")
        test_firewall(tmp / "d")
        test_frozen_config(tmp / "e")
        test_composite(tmp / "f")

    n_pass, n_fail = len(PASSED), len(FAILED)
    print(f"\n[test] {n_pass} PASS / {n_fail} FAIL")
    for f in FAILED:
        print(f"  FAILED: {f}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
