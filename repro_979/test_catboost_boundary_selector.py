#!/usr/bin/env python3
"""test_catboost_boundary_selector.py — Todo 2 tests-after (agent-executable, pytest 무의존).

계획 QA 시나리오 (task-2):
  (a) smoke grid emits feasibility and one frozen candidate
  (b) monkeypatch a higher R-only metric for a lower-primary row → selected ID does NOT change
  (c) inject an R-field sort key → hard failure (exit 2)

실행: python3 repro_979/test_catboost_boundary_selector.py  (exit 0 = 전부 PASS)

모든 테스트는 선택 로직의 순수 함수(그리드/랭킹/가드)를 사용 — 실데이터 로드 없이
빠르게 실행되고, (c)는 subprocess 로 CLI fixture 를 검증한다.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import repro_979.catboost_boundary_selector as sel  # noqa: E402
from repro_979.next_round_policy import load_json, validate_policy  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLI = str(Path(__file__).resolve().parent / "catboost_boundary_selector.py")


def _synthetic():
    rng = np.random.default_rng(20260816)
    n = 8000
    s = rng.standard_normal(n)
    y = (s > 0.0).astype(np.float64)  # 라벨과 상관된 신호 → BSS 가 0 으로 클램프되지 않음
    z = np.stack([s + 0.3 * rng.standard_normal(n),
                  s + 0.6 * rng.standard_normal(n),
                  s + 1.0 * rng.standard_normal(n)], axis=1)
    rows = sel._grid_rows()
    weights_mat = np.asarray([r["weights"] for r in rows], dtype=np.float64)
    bss = np.asarray([sel._primary_bss(z @ w, y) for w in weights_mat], dtype=np.float64)
    assert len(set(bss.tolist())) == 78, "합성 BSS 가 78행에서 전부 구분되어야 함"
    return rows, z, y, bss


# ── (a) 그리드 78행 + 정확히 하나의 동결 후보 ──
def test_a_grid_and_single_frozen() -> None:
    rows = sel._grid_rows()
    assert len(rows) == 78, f"feasible rows = {len(rows)} (필요 78)"
    ids = [sel._row_candidate_id(r) for r in rows]
    assert len(set(ids)) == 78, "candidate_id 중복 (결정적 해시 위반)"

    _, _, _, bss = _synthetic()
    ranked = sel._rank_rows(rows, bss)
    frozen = sel._freeze(ranked)
    assert frozen["n_frozen"] == 1, "동결 후보는 정확히 1개"
    assert ranked[0]["primary_bss"] == max(bss), "rank 1 = 최대 primary_bss"


def test_a_tie_rule_ascending_ticks() -> None:
    rows = sel._grid_rows()
    bss_tie = np.zeros(78)
    bss_tie[0] = bss_tie[1] = 10.0
    ranked = sel._rank_rows(rows, bss_tie)
    assert ranked[0]["primary_bss"] == ranked[1]["primary_bss"] == 10.0
    k0 = (ranked[0]["cat_tick"], ranked[0]["lgb_tick"], ranked[0]["mlp_tick"])
    k1 = (ranked[1]["cat_tick"], ranked[1]["lgb_tick"], ranked[1]["mlp_tick"])
    assert k0 < k1, f"동점은 tick 오름차순이어야 함: {k0} vs {k1}"


# ── (b) monkeypatch: lower-primary 행에 높은 R-only 메트릭 주입 → 선택 ID 불변 ──
def test_b_r_metric_monkeypatch_does_not_change_selection() -> None:
    rows, _, _, bss = _synthetic()
    ranked = sel._rank_rows(rows, bss)
    frozen_a = sel._freeze(ranked)["candidate_id"]

    lower_primary_row = ranked[-1]  # primary_bss 최하위 행
    assert lower_primary_row["primary_bss"] < ranked[0]["primary_bss"]
    assert lower_primary_row["candidate_id"] != frozen_a

    # monkeypatch: 최하위 primary 행에 R-only 메트릭(Δr2022 / boot LB5)을 극단값 주입 —
    # 선택 경로가 이 값을 절대 읽지 않으므로 동결 ID 가 바뀌면 안 된다.
    saved = sel.R_ONLY_METRICS
    patched = {lower_primary_row["candidate_id"]: {
        "delta_r2022": +999.0, "boot_lb5": +999.0}}
    sel.R_ONLY_METRICS = patched
    try:
        ranked_b = sel._rank_rows(rows, bss)
        frozen_b = sel._freeze(ranked_b)["candidate_id"]
        assert frozen_b == frozen_a, (
            f"R-only 메트릭 주입에도 선택 ID 는 불변: {frozen_a} vs {frozen_b}")
    finally:
        sel.R_ONLY_METRICS = saved

    # 만약 누수 구현이 R-only 값을 정렬에 썼다면 선택이 달라졌을 것 (가드의 필요성 증명)
    leaky = sorted(ranked, key=lambda r: -patched.get(
        r["candidate_id"], {}).get("delta_r2022", 0.0))
    assert leaky[0]["candidate_id"] != frozen_a, "누수 정렬이면 다른 행이 뽑혀야 함 (증명용)"

    # 가드 자체 검증: primary_bss 단일 키는 통과, R 키는 PolicyViolation
    sel._assert_primary_only_sort_keys(("primary_bss",))
    try:
        sel._assert_primary_only_sort_keys(("primary_bss", "delta_r2022"))
    except sel.PolicyViolation:
        pass
    else:  # pragma: no cover
        raise AssertionError("R-only 정렬 키가 가드에 차단되지 않음")


def test_b_no_rfold_labels_in_selection_path() -> None:
    sel._assert_primary_only_labels({"primary": np.zeros(3)})
    try:
        sel._assert_primary_only_labels({"primary": np.zeros(3), "r2022": np.zeros(3)})
    except sel.LeakageError:
        pass
    else:  # pragma: no cover
        raise AssertionError("R-only 라벨이 선택 경로에 허용됨")
    try:
        sel._read_label_fold({"primary": np.zeros(3)}, "r2022")
    except sel.LeakageError:
        pass
    else:  # pragma: no cover
        raise AssertionError("r2022 라벨 읽기가 차단되지 않음")


# ── (c) R-field sort key 주입 → CLI 하드 실패 (exit 2) ──
def test_c_fixtures_exit_2() -> None:
    for fx in ("r-sort-key", "attempted-r-fold-read"):
        proc = subprocess.run([sys.executable, CLI, "--fixture", fx],
                              cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300)
        assert proc.returncode == 2, f"--fixture {fx}: exit {proc.returncode} (필요 2)\n{proc.stdout}"


# ── 정책 불변성 교차 검증 (이 태스크가 정책을 건드리지 않았음을 증명) ──
def test_d_policy_intact() -> None:
    policy = load_json(Path(__file__).resolve().parent / "next_round_policy.json")
    assert policy is not None
    violations = validate_policy(policy)
    assert not violations, f"동결 정책 위반: {violations}"
    assert policy["selection"]["label_source"] == "primary"
    assert policy["selection"]["sort_keys"] == ["primary_bss"]


def main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"[PASS] {name}")
    print(f"\ntest_catboost_boundary_selector: {len(tests) - failed}/{len(tests)} PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
