#!/usr/bin/env python3
"""test_subgroup_calibration_runner.py — Todo 8 tests-after (agent-executable, pytest 무의존).

계획 QA 시나리오 (task-8):
  (a) SKIPPED_NO_BASE 경로를 실제 증거로 end-to-end 검증 (exit 0, 라벨 미로드)
  (b) fixtures same-year-residual / nonunit-slope / cap-violation /
      future-year-coefficient / attempted-r-fold-read 각각 exit 2
  (c) 단위 테스트: zero-sum 제약 / 수축 순서 / 클립 순서 / lexicographic tie-break
      (+ F 가드, 미관측 키 0, LBFGS 그래디언트, 부트스트랩 결정성/신호)

실행: python3 repro_979/test_subgroup_calibration_runner.py  (exit 0 = 전부 PASS)

모든 테스트는 순수 함수(합성 데이터) + subprocess CLI 검증 — 실데이터 로드 없음.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import repro_979.subgroup_calibration_runner as sc  # noqa: E402
import repro_979.common as common  # noqa: E402
from repro_979.next_round_policy import load_json, validate_policy  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLI = str(Path(__file__).resolve().parent / "subgroup_calibration_runner.py")
EVIDENCE_DIR = PROJECT_ROOT / ".omo" / "evidence" / "aimers9-next-round"


# ── 합성 데이터 ──────────────────────────────────────────────────────
def _synthetic_panel(n_per_year: int = 6000, seed: int = 20260816) -> dict[str, Any]:
    """2021-2024 합성 인과 패널 — F(2023+)/count '11' 잔차 신호 포함."""
    rng = np.random.default_rng(seed)
    parts = {k: [] for k in ("year", "row_ids", "z", "y", "game_type", "count_state",
                             "cluster_id")}
    for y in sc.PANEL_YEARS:
        n = n_per_year
        z = rng.standard_normal(n)
        gt = rng.choice(np.array(["R", "F"]), n, p=[0.88, 0.12])
        cs = rng.integers(0, 12, n).astype(str)
        bias = (0.4 * ((gt == "F") & (np.full(n, y) >= sc.F_MIN_SEASON)).astype(float)
                + 0.25 * (cs == "11").astype(float))
        p = common.sigmoid(z + bias)
        yy = (rng.random(n) < p).astype(float)
        game = np.asarray([f"{y}-{g}" for g in rng.integers(0, 60, n)])
        parts["year"].append(np.full(n, y, dtype=np.int64))
        parts["row_ids"].append(np.arange(n, dtype=np.int64) + 10_000 * y)
        parts["z"].append(z)
        parts["y"].append(yy)
        parts["game_type"].append(gt)
        parts["count_state"].append(cs)
        parts["cluster_id"].append(game)
    return {k: np.concatenate(v) for k, v in parts.items()}


def _scored_arrays(panel: dict[str, Any], year: int = sc.SCORED_YEAR):
    m = panel["year"] == year
    return {
        "z": panel["z"][m], "y": panel["y"][m],
        "game_type": panel["game_type"][m], "count_state": panel["count_state"][m],
        "cluster_id": panel["cluster_id"][m],
    }


# ── (c1) zero-sum 제약 ───────────────────────────────────────────────
def test_c1_zero_sum_constraint() -> None:
    panel = _synthetic_panel()
    m = panel["year"] < sc.SCORED_YEAR
    z, y = panel["z"][m], panel["y"][m]
    keys = panel["count_state"][m]
    for lam in sc.LAMBDA_GRID:
        for k in sc.K_GRID:
            _, diag = sc.fit_term(z, y, keys, lam, k)
            # raw beta 는 가중 zero-sum: |sum(n_g * beta_raw)| <= 1e-8
            n_g = np.asarray(diag["n_g"], dtype=np.float64)
            raw = np.asarray(diag["beta_raw"], dtype=np.float64)
            assert abs(float(np.dot(n_g, raw))) <= 1e-8, (
                f"zero-sum 위반 lam={lam} k={k}: {np.dot(n_g, raw):.3e}")
            # 수축/클립 후에도 그룹 수 = 12 (count_state 0..11)
            assert diag["n_groups"] == 12, diag["n_groups"]


def test_c1_single_group_edge() -> None:
    """G==1 → 제약이 beta=0 을 강제 (n_g>0) — zero-sum 유지."""
    rng = np.random.default_rng(7)
    z = rng.standard_normal(500)
    y = (rng.random(500) < common.sigmoid(z)).astype(float)
    keys = np.full(500, "R")
    delta_map, diag = sc.fit_term(z, y, keys, 0.01, 5000)
    assert delta_map == {"R": 0.0}, delta_map
    assert diag["beta_raw"] == [0.0]


# ── (c2) 수축 순서 ───────────────────────────────────────────────────
def test_c2_shrink_order() -> None:
    panel = _synthetic_panel()
    m = panel["year"] < sc.SCORED_YEAR
    z, y = panel["z"][m], panel["y"][m]
    keys = panel["game_type"][m]
    lam, k = 0.01, 5000
    _, diag = sc.fit_term(z, y, keys, lam, k)
    n_g = np.asarray(diag["n_g"], dtype=np.float64)
    raw = np.asarray(diag["beta_raw"], dtype=np.float64)
    shrunk = np.asarray(diag["beta_shrunk"], dtype=np.float64)
    # 순서 고정: raw(zero-sum) → n/(n+k) 수축 → 클립
    expect_shrunk = raw * n_g / (n_g + float(k))
    assert np.allclose(shrunk, expect_shrunk, atol=1e-12), "수축 공식 불일치"
    # 수축은 zero-sum 을 깨는 것이 문서화된 순서 (raw 에만 제약)
    assert abs(float(np.dot(n_g, shrunk))) > 1e-12 or abs(float(np.dot(n_g, raw))) <= 1e-8
    # 클립은 수축 후
    final = np.asarray(diag["beta_final"], dtype=np.float64)
    assert np.allclose(final, np.clip(shrunk, -sc.TERM_CAP, sc.TERM_CAP), atol=1e-12)


# ── (c3) 클립 순서 (term 먼저, 합산 후 total) ───────────────────────
def test_c3_clip_order() -> None:
    # term 클립이 먼저: 개별 term 은 캡 내, 합산은 캡 초과 → total 클립 적용
    z = np.zeros(6)
    t1 = {"a": 0.04, "b": 0.04}   # 각 term 캡 내
    t2 = {"x": 0.04, "y": 0.04}
    keys1 = np.asarray(["a", "a", "a", "b", "b", "b"])
    keys2 = np.asarray(["x", "x", "x", "y", "y", "y"])
    delta = sc.apply_delta(z, [t1, t2], [keys1, keys2])
    assert np.all(delta == sc.TOTAL_CAP), f"total 클립 필요: {delta}"
    assert np.allclose(delta, np.clip(0.08, -sc.TOTAL_CAP, sc.TOTAL_CAP))
    # 미관측 키 → 0 (unseen key mapping)
    delta2 = sc.apply_delta(z, [t1], [np.asarray(["zz", "zz", "zz", "zz", "zz", "zz"])])
    assert np.all(delta2 == 0.0), "미관측 키는 delta 0 이어야 함"
    # term 자체는 캡 내 클립됨을 fit_term 이 보장 (사후 조건 가드)
    rng = np.random.default_rng(3)
    zz = rng.standard_normal(3000)
    yy = (rng.random(3000) < common.sigmoid(zz + 0.8)).astype(float)
    keys = rng.choice(np.array(["R", "F"]), 3000, p=[0.5, 0.5])
    delta_map, _ = sc.fit_term(zz, yy, keys, 0.0001, 5000)
    assert all(abs(v) <= sc.TERM_CAP + 1e-12 for v in delta_map.values())


# ── (c4) lexicographic tie-break ─────────────────────────────────────
def test_c4_lexicographic_tie_break() -> None:
    recs = [
        {"li": 0, "ki": 0, "ci": 0, "bss": 100.0},   # lowest lexicographic among max
        {"li": 0, "ki": 0, "ci": 1, "bss": 100.0},
        {"li": 0, "ki": 1, "ci": 0, "bss": 100.0},
        {"li": 1, "ki": 0, "ci": 0, "bss": 100.0},
        {"li": 2, "ki": 2, "ci": 2, "bss": 101.0},   # higher BSS wins
        {"li": 0, "ki": 0, "ci": 0, "bss": 99.0},
    ]
    best = sc._select_best(recs)
    assert best["bss"] == 101.0, "BSS 최대 우선"
    recs_max = [r for r in recs if r["bss"] == 100.0]
    best_tie = sc._select_best(recs_max)
    assert best_tie["li"] == best_tie["ki"] == best_tie["ci"] == 0, \
        "동점은 lexicographic 최저 (lambda, k, configuration)"
    # ci 순서: game_type < count_state < game_type+count_state (CONFIGURATIONS 순서)
    assert sc.CONFIGURATIONS == ("game_type", "count_state", "game_type+count_state")


# ── F 가드 (game-type F 는 2023+ OOF 만) ────────────────────────────
def test_f_guard() -> None:
    panel = _synthetic_panel()
    # outer year 2023: 패널 <2023 = {2021, 2022} — 2023+ F 행 없음 → F delta 0
    terms_2023, diags_2023 = sc.fit_configuration_for_year(panel, 2023, "game_type", 0.01, 5000)
    assert "F" not in terms_2023[0], f"2023 에서 F 가 피팅됨: {terms_2023[0]}"
    assert diags_2023[0]["groups"] == ["R"], diags_2023[0]
    # outer year 2024: 패널 {2021,2022,2023} — F 는 2023 행만 (n_g_F == 2023 F 수)
    terms_2024, diags_2024 = sc.fit_configuration_for_year(panel, 2024, "game_type", 0.01, 5000)
    assert "F" in terms_2024[0], f"2024 에서 F 미피팅: {terms_2024[0]}"
    m2023f = (panel["year"] == 2023) & (panel["game_type"] == "F")
    gF = diags_2024[0]["groups"].index("F")
    assert diags_2024[0]["n_g"][gF] == int(m2023f.sum()), \
        f"F n_g={diags_2024[0]['n_g'][gF]} != 2023 F 행 수 {int(m2023f.sum())}"
    # R 그룹 = (<2024 전체) − (모든 F 행: pre-2023 F 는 제외, 2023 F 는 자체 그룹으로)
    m_f_all = (panel["year"] < 2024) & (panel["game_type"] == "F")
    assert diags_2024[0]["n_g"][diags_2024[0]["groups"].index("R")] == \
        int((panel["year"] < 2024).sum()) - int(m_f_all.sum())
    # additivie 구성에서도 game_type term 은 같은 F 가드 적용
    terms_add, _ = sc.fit_configuration_for_year(panel, 2024, "game_type+count_state", 0.01, 5000)
    assert len(terms_add) == 2
    assert "F" in terms_add[0] and set(terms_add[1]) == {str(i) for i in range(12)}


# ── 구조 가드 ────────────────────────────────────────────────────────
def test_guards() -> None:
    try:
        sc._assert_no_future_fit([2021, 2024], outer_year=2024)
    except sc.PolicyViolation:
        pass
    else:  # pragma: no cover
        raise AssertionError("future-year 가드 미동작")
    try:
        sc._assert_no_same_year_residual(np.array([1, 2, 3]), np.array([3, 4]))
    except sc.LeakageError:
        pass
    else:  # pragma: no cover
        raise AssertionError("same-year 가드 미동작")
    try:
        sc._assert_cap_respected([0.051], cap=0.05)
    except sc.PolicyViolation:
        pass
    else:  # pragma: no cover
        raise AssertionError("캡 가드 미동작")
    try:
        sc._assert_unit_slope(0.99)
    except sc.PolicyViolation:
        pass
    else:  # pragma: no cover
        raise AssertionError("slope 가드 미동작")
    try:
        sc._read_label_fold({"primary": np.zeros(3)}, "r2022")
    except sc.LeakageError:
        pass
    else:  # pragma: no cover
        raise AssertionError("R-fold 가드 미동작")
    try:
        sc._assert_primary_only_labels({"primary": np.zeros(3), "r2023": np.zeros(3)})
    except sc.LeakageError:
        pass
    else:  # pragma: no cover
        raise AssertionError("primary-only 라벨 가드 미동작")
    sc._assert_unit_slope(1.0)          # 동결 slope 통과
    sc._assert_no_future_fit([2021, 2022, 2023], outer_year=2024)  # 정상 통과
    sc._assert_no_same_year_residual(np.array([1, 2]), np.array([3, 4]))  # 정상 통과
    sc._assert_cap_respected([0.05, -0.05, 0.0])  # 경계 포함 통과


# ── LBFGS 그래디언트 수치 검증 + 부트스트랩 ─────────────────────────
def test_lbfgs_gradient_check() -> None:
    rng = np.random.default_rng(11)
    z = rng.standard_normal(2000)
    y = (rng.random(2000) < common.sigmoid(z)).astype(float)
    keys = rng.integers(0, 6, 2000)
    n_g = np.bincount(keys).astype(np.float64)
    N = sc._null_space_basis(n_g)
    q = rng.standard_normal(N.shape[1]) * 0.1
    eps = 1e-6
    obj0, g0 = sc._obj_grad_q(q, z, y, keys, n_g, 0.01, N)
    for j in range(N.shape[1]):
        qp = q.copy()
        qp[j] += eps
        objp, _ = sc._obj_grad_q(qp, z, y, keys, n_g, 0.01, N)
        num = (objp - obj0) / eps
        assert abs(num - g0[j]) < 1e-4, f"grad[{j}] {num} vs {g0[j]}"


def test_bootstrap_deterministic_and_signal() -> None:
    rng = np.random.default_rng(5)
    z = rng.standard_normal(20000)
    y = (z + 0.5 > 0).astype(float)
    delta = np.full(len(z), 0.5)  # 정확한 보정
    cluster = rng.integers(0, 200, len(z)).astype(str)
    lb_a = sc.paired_bootstrap_lb5(z, delta, y, cluster)
    lb_b = sc.paired_bootstrap_lb5(z, delta, y, cluster)
    assert lb_a == lb_b, "부트스트랩 비결정적 (seed 고정)"
    assert lb_a > 0.0, f"완벽 보정인데 LB5={lb_a} <= 0"
    # 보정이 나쁜 경우 (반대 방향 delta) LB5 < 0
    lb_bad = sc.paired_bootstrap_lb5(z, -delta, y, cluster)
    assert lb_bad < 0.0, f"반대 보정인데 LB5={lb_bad} >= 0"


# ── cold-start (2021) ────────────────────────────────────────────────
def test_cold_start_2021() -> None:
    panel = _synthetic_panel()
    terms, diags = sc.fit_configuration_for_year(panel, 2021, "game_type", 0.01, 5000)
    assert terms == [] and "cold-start" in diags[0]["note"], "2021 은 delta 0 (cold-start)"


# ── 전체 흐름 (합성 패널 → 그리드 → 선택 → 게이트) ──────────────────
def test_full_flow_synthetic() -> None:
    panel = _synthetic_panel()
    scored = _scored_arrays(panel)
    base_bss, base_p_mean = sc.score_base(scored["z"], scored["y"])
    best, records = sc.evaluate_calibration_grid(
        panel, scored["z"], scored["y"],
        {"game_type": scored["game_type"], "count_state": scored["count_state"]})
    assert len(records) == len(sc.LAMBDA_GRID) * len(sc.K_GRID) * len(sc.CONFIGURATIONS) == 36
    assert best["bss"] >= max(r["bss"] for r in records), "best = max BSS"
    # 동점 시 lexicographic 최저 — 모든 레코드 재검토
    for r in records:
        if r["bss"] == best["bss"]:
            assert sc._lexico_key(r) >= sc._lexico_key(best)
    assert 0.0 <= best["max_abs_delta"] <= sc.TOTAL_CAP + 1e-12
    key_arrays = sc._key_arrays_for(
        {"game_type": scored["game_type"], "count_state": scored["count_state"]},
        best["configuration"])
    delta = sc.apply_delta(scored["z"], best["delta_terms"], key_arrays)
    lb5 = sc.paired_bootstrap_lb5(scored["z"], delta, scored["y"], scored["cluster_id"])
    verdict, gate = sc.gate_verdict(best, base_bss, base_p_mean, lb5)
    assert verdict in ("PASS", "REJECT")
    assert gate["delta_bss"] == best["bss"] - base_bss
    assert gate["mean_shift"] >= 0.0
    # 캘리브레이션이 잔차 신호를 잡았으면 ΔBSS > 0 (합성 신호 존재)
    assert gate["delta_bss"] > 0.0, f"합성 신호 캘리브레이션 ΔBSS={gate['delta_bss']}"


# ── (b) fixtures 각각 exit 2 ─────────────────────────────────────────
def test_b_fixtures_exit_2() -> None:
    for fx in ("same-year-residual", "nonunit-slope", "cap-violation",
               "future-year-coefficient", "attempted-r-fold-read"):
        proc = subprocess.run([sys.executable, CLI, "--fixture", fx],
                              cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300)
        assert proc.returncode == 2, (
            f"--fixture {fx}: exit {proc.returncode} (필요 2)\n{proc.stdout[-1500:]}")
        log = EVIDENCE_DIR / f"task-8-calibration-fixture-{fx}.log"
        assert log.is_file(), f"fixture 로그 부재: {log}"


# ── (a) SKIPPED_NO_BASE end-to-end (실제 증거 기반) ─────────────────
def test_a_skipped_no_base_e2e() -> None:
    proc = subprocess.run([sys.executable, CLI, "--primary-causal"],
                          cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, (
        f"--primary-causal: exit {proc.returncode} (필요 0)\n{proc.stdout[-2000:]}")
    ev = EVIDENCE_DIR / "task-8-calibration.json"
    assert ev.is_file(), f"증거 부재: {ev}"
    rec = json.loads(ev.read_text(encoding="utf-8"))
    assert rec["verdict"] == "SKIPPED_NO_BASE", rec["verdict"]
    assert rec["exit_code"] == 0
    assert rec["label_sources"] == [], f"label_sources={rec['label_sources']} (필요 [])"
    assert rec["labels_read"] is False
    assert rec["config_pre_registered"]["written_before_labels_read"] is True
    assert rec["config_hash"] and rec["policy_config_hash"]
    assert rec["base_eligibility"]["eligible"] is False
    assert rec["base_eligibility"]["task3_verdict"] == "PRIMARY_REJECT"
    assert rec["base_eligibility"]["task7_verdict"] == "SKIPPED"
    assert "SKIPPED_NO_BASE" in proc.stdout


# ── 정책 불변성 ──────────────────────────────────────────────────────
def test_policy_intact() -> None:
    policy = load_json(Path(__file__).resolve().parent / "next_round_policy.json")
    assert policy is not None
    violations = validate_policy(policy)
    assert not violations, f"동결 정책 위반: {violations}"
    assert policy["selection"]["label_source"] == "primary"


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
    print(f"\ntest_subgroup_calibration_runner: {len(tests) - failed}/{len(tests)} PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
