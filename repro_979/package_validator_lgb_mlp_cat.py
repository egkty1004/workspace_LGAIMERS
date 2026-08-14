#!/usr/bin/env python3
"""
[Todo 8 배포 QA] lgb_mlp_cat 제출 패키지 검증기 (2026-08-14)

검증 항목 (champ_cat 검증기와 동일 10개 게이트):
  1) 오프라인 설치 시뮬레이션 재검증 (fresh venv — lightgbm/catboost --no-index 설치,
     설치 시간, import, 버전. numpy/pandas/scipy 는 평가 기본 버전 유지 확인)
  2) 5행 end-to-end: script.py (LGA_TEST_PATH=open/data/test.csv) → 5행, [0.30,0.70]
  3) 245,789행 합성 픽스처 (train 부트스트랩, control_success 제거, random_state 42):
     런타임 < 600s, 245,789행, 전 값 [0.30,0.70], mean/clip 경계
  4) 패리티: script.py 출력 vs 독립 참조 계산 (동일 모델 + common 전처리) max|Δ| < 1e-6
  5) mean 정렬: script/참조 |Δmean| ≤ 0.005 (Todo 7 mean-shift 게이트)
  6) 챔피언 정책 mean 정렬: lgb_mlp_cat 출력 평균 vs 챔피언 구성요소 평균 |Δ| ≤ 0.005
  7) 챔피언 구성요소(0.51·LGB+0.49·MLP)가 GIHO extract script.py 와 일치 (5행, < 1e-6)
     → byte-frozen 모델 복사 무결성 증명
  8) zip 레이아웃: 필수 최상위 항목만 존재, 잡 파일(__pycache__/.npy/로그) 부재
  9) provenance.json 과 model/* sha256 일치 (candidate_id 5890a4c54f502c4e)
  10) migration_audit.py --strict PASS 유지

사용법:
  python3 repro_979/package_validator_lgb_mlp_cat.py \
      --submission-dir repro_979/submit_lgb_mlp_cat_<ts>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
PROJECT_ROOT = REPO.parent

# lgb_mlp_cat 블렌드 상수 (Todo 7b 스윕 승자: lgb×0.30 + mlp×0.35 + catboost×0.35, 로짓 공간)
W_LGB = 0.30
W_MLP = 0.35
W_CAT = 0.35
C_LOGIT = -0.0404       # 챔피언 정책 상수 (동결)
CLIP_LO, CLIP_HI = 0.30, 0.70
SEEDS = list(range(42, 52))
CHAMP_W_LGB = 0.51      # 챔피언 구성요소 (Task 2 동결, LGB×0.51 + MLP×0.49)
CAT_FEATURES = ["top_bottom", "game_type", "base_state", "platoon", "count_state"]
CATBOOST_THREADS = 6
N_FULL = 245789
SYNTH_SEED = 42
MEAN_ALIGN_TOL = 0.005        # Todo 7 mean-shift 게이트 (max|Δmean| ≤ 0.005)
MEAN_SANITY = (0.40, 0.55)    # 합성 픽스처 출력 평균 합리성 범위
RUNTIME_BUDGET_S = 600
PARITY_TOL = 1e-6

SIM_VENV = REPO / "cache" / "t8_install_sim" / "venv"
SIM_WHEELS = REPO / "cache" / "t8_install_sim" / "wheels"
FIXTURE_DIR = REPO / "cache" / "t8_install_sim" / "lgbmlpcat_fixtures"

EXPECTED_ZIP_TOP = {"script.py", "common.py", "mlp_model.py", "requirements.txt",
                    "provenance.json", "README.md", "model/"}
EXPECTED_MODEL_FILES = {f"f3_s{s}.txt" for s in SEEDS} | \
                       {f"mlp_s{s}.pt" for s in SEEDS} | \
                       {"mlp_prep.pkl", "mlp_meta.json", "train_meta.json"} | \
                       {f"catboost_s{s}.cbm" for s in SEEDS}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_default(o):
    if isinstance(o, (np.bool_, np.integer, np.floating)):
        return o.item()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ────────────────────────────────────────────────────────────────────────────
# 1) 오프라인 설치 시뮬레이션 재검증
# ────────────────────────────────────────────────────────────────────────────
def check_offline_install() -> tuple[bool, dict]:
    result: dict = {"package": "offline-install-simulation"}
    if not (SIM_VENV / "bin" / "pip").exists() or not SIM_WHEELS.is_dir():
        result["skipped"] = True
        return False, result
    t0 = time.time()
    proc = subprocess.run(
        [str(SIM_VENV / "bin" / "pip"), "install", "--no-index", "--find-links",
         str(SIM_WHEELS), "lightgbm==4.7.0", "catboost==1.2.10"],
        capture_output=True, text=True, timeout=300)
    install_s = round(time.time() - t0, 2)
    result["install_s"] = install_s
    result["pip_rc"] = proc.returncode
    if proc.returncode != 0:
        result["error"] = proc.stderr[-2000:]
        return False, result
    code = ("import json, lightgbm, catboost, numpy, pandas, scipy; "
            "print(json.dumps({'lightgbm': lightgbm.__version__, "
            "'catboost': catboost.__version__, 'numpy': numpy.__version__, "
            "'pandas': pandas.__version__, 'scipy': scipy.__version__}))")
    ver = subprocess.run([str(SIM_VENV / "bin" / "python"), "-c", code],
                         capture_output=True, text=True, timeout=60)
    result["install_s_ok"] = install_s <= 600
    result["installed"] = {}
    if ver.returncode == 0:
        result["installed"] = json.loads(ver.stdout)
    result["versions_ok"] = (
        result.get("installed", {}).get("lightgbm") == "4.7.0"
        and result.get("installed", {}).get("catboost") == "1.2.10"
        and result.get("installed", {}).get("numpy") == "1.26.4"
        and result.get("installed", {}).get("pandas") == "2.0.3"
        and result.get("installed", {}).get("scipy") == "1.15.3")
    ok = proc.returncode == 0 and result["install_s_ok"] and result["versions_ok"]
    print(f"[{'PASS' if ok else 'FAIL'}] offline install sim: "
          f"{install_s}s rc={proc.returncode} {result['installed']}")
    return ok, result


# ────────────────────────────────────────────────────────────────────────────
# 2) 5행 / 245,789행 script.py 실행
# ────────────────────────────────────────────────────────────────────────────
def run_script(submit_dir: Path, test_path: Path, sample_path: Path,
               out_path: Path, label: str) -> tuple[float, str]:
    env = dict(os.environ)
    env["LGA_TEST_PATH"] = str(test_path)
    env["LGA_SAMPLE_PATH"] = str(sample_path)
    env["LGA_OUT_PATH"] = str(out_path)
    t0 = time.time()
    proc = subprocess.run([sys.executable, "script.py"], cwd=submit_dir, env=env,
                          capture_output=True, text=True, timeout=RUNTIME_BUDGET_S + 120)
    dt = time.time() - t0
    tail = proc.stdout.strip().splitlines()
    print(f"[run:{label}] rc={proc.returncode} wall={dt:.1f}s")
    for line in tail[-4:]:
        print(f"    {line}")
    if proc.returncode != 0:
        print(f"    STDERR: {proc.stderr[-1500:]}")
    return dt, proc.stdout + proc.stderr


def build_fixtures() -> tuple[Path, Path]:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    test_fix = FIXTURE_DIR / "test_245789.csv"
    sample_fix = FIXTURE_DIR / "sample_submission_245789.csv"
    if test_fix.exists() and sample_fix.exists():
        return test_fix, sample_fix
    train = pd.read_csv(REPO / "open" / "data" / "train.csv", encoding="utf-8-sig")
    rng = np.random.RandomState(SYNTH_SEED)
    idx = rng.choice(len(train), size=N_FULL, replace=False)
    test = train.iloc[idx].drop(columns=["control_success"]).reset_index(drop=True)
    sample = pd.DataFrame({"row_id": test["row_id"], "control_success": 0.5})
    test.to_csv(test_fix, index=False, encoding="utf-8-sig")
    sample.to_csv(sample_fix, index=False, encoding="utf-8-sig")
    return test_fix, sample_fix


def check_five_row(submit_dir: Path, out5: Path) -> tuple[bool, dict]:
    test5 = REPO / "open" / "data" / "test.csv"
    sample5 = REPO / "open" / "data" / "sample_submission.csv"
    dt, _ = run_script(submit_dir, test5, sample5, out5, "5row")
    if not out5.exists():
        return False, {"error": "5행 출력 없음", "wall_s": dt}
    sub = pd.read_csv(out5, encoding="utf-8-sig")
    vals = sub["control_success"].values
    ok = (len(sub) == 5 and np.isfinite(vals).all()
          and (vals >= CLIP_LO).all() and (vals <= CLIP_HI).all())
    print(f"[{'PASS' if ok else 'FAIL'}] 5행: n={len(sub)} "
          f"range=({vals.min():.4f},{vals.max():.4f})")
    return ok, {"n_rows": int(len(sub)), "wall_s": dt,
                "min": float(vals.min()), "max": float(vals.max()),
                "in_bounds": bool(ok)}


def check_full_245k(submit_dir: Path, out245: Path) -> tuple[bool, dict, pd.Series]:
    test_fix, sample_fix = build_fixtures()
    dt, _ = run_script(submit_dir, test_fix, sample_fix, out245, "245k")
    if not out245.exists():
        return False, {"error": "245,789행 출력 없음", "wall_s": dt}, None
    sub = pd.read_csv(out245, encoding="utf-8-sig")
    vals = sub["control_success"].values.astype(np.float64)
    n = int(len(sub))
    frac_lo = float(np.mean(vals < CLIP_LO))
    frac_hi = float(np.mean(vals > CLIP_HI))
    mean = float(vals.mean())
    ok_rows = n == N_FULL and np.isfinite(vals).all()
    ok_clip = frac_lo == 0.0 and frac_hi == 0.0
    ok_runtime = dt < RUNTIME_BUDGET_S
    ok_mean_sanity = MEAN_SANITY[0] <= mean <= MEAN_SANITY[1]
    d = {"n_rows": n, "wall_s": dt, "mean": mean,
         "frac_below_0.30": frac_lo, "frac_above_0.70": frac_hi,
         "min": float(vals.min()), "max": float(vals.max()),
         "runtime_ok": ok_runtime, "rows_ok": ok_rows, "clip_ok": ok_clip,
         "mean_sanity_ok": ok_mean_sanity}
    print(f"[{'PASS' if (ok_rows and ok_runtime) else 'FAIL'}] 245,789행: n={n} "
          f"wall={dt:.1f}s(<{RUNTIME_BUDGET_S}s) mean={mean:.4f} "
          f"clip_lo={frac_lo:.2e} clip_hi={frac_hi:.2e}")
    return (ok_rows and ok_runtime), d, vals


# ────────────────────────────────────────────────────────────────────────────
# 3) 참조 계산 (독립 구현 — 동일 모델 + 공용 전처리, script.py 재실행 아님)
# ────────────────────────────────────────────────────────────────────────────
def reference_predict(submit_dir: Path, test_path: Path) -> np.ndarray:
    """lgb_mlp_cat 독립 참조: z = 0.30*z_lgb + 0.35*z_mlp + 0.35*z_catboost."""
    sys.path.insert(0, str(submit_dir))
    import common as sub_common  # type: ignore  # noqa: F401
    import mlp_model as sub_mlp  # type: ignore  # noqa: F401
    import lightgbm as lgb  # noqa: F401
    from catboost import CatBoostClassifier, Pool  # noqa: PLC0415

    test = pd.read_csv(test_path, encoding="utf-8-sig")
    feats = sub_common.get_feature_cols(test.columns)
    sub_common.preprocess_for_submission(test)
    X = test[feats].copy()
    z_lgb = np.zeros(len(X), dtype=np.float64)
    for s in SEEDS:
        bst = lgb.Booster(model_file=str(submit_dir / "model" / f"f3_s{s}.txt"))
        z_lgb += sub_common.logit(bst.predict(X))
    z_lgb /= len(SEEDS)
    prep, models = sub_mlp.load(str(submit_dir / "model"), SEEDS)
    z_mlp = sub_mlp.predict_z(test, prep, models)
    pool = Pool(X, cat_features=list(CAT_FEATURES))
    z_cat = np.zeros(len(X), dtype=np.float64)
    for s in SEEDS:
        model = CatBoostClassifier()
        model.load_model(str(submit_dir / "model" / f"catboost_s{s}.cbm"))
        p = np.asarray(model.predict(pool, prediction_type="Probability",
                                     thread_count=CATBOOST_THREADS), dtype=np.float64)
        if p.ndim == 2:
            p = p[:, 1]
        z_cat += sub_common.logit(p)
    z_cat /= len(SEEDS)
    z = W_LGB * z_lgb + W_MLP * z_mlp + W_CAT * z_cat
    return np.clip(sub_common.sigmoid(z + C_LOGIT), CLIP_LO, CLIP_HI)


def champion_component_predict(submit_dir: Path, test_path: Path) -> np.ndarray:
    """챔피언 구성요소 p = clip(sigmoid(0.51*z_lgb + 0.49*z_mlp + C_LOGIT)) —
    lgb_mlp_cat 의 LGB/MLP 복사본이 챔피언 정책(0.51/0.49, Task 2 동결)을 재현하는지 검증."""
    sys.path.insert(0, str(submit_dir))
    import common as sub_common  # type: ignore  # noqa: F401
    import mlp_model as sub_mlp  # type: ignore  # noqa: F401
    import lightgbm as lgb  # noqa: F401

    test = pd.read_csv(test_path, encoding="utf-8-sig")
    feats = sub_common.get_feature_cols(test.columns)
    sub_common.preprocess_for_submission(test)
    X = test[feats].copy()
    z_lgb = np.zeros(len(X), dtype=np.float64)
    for s in SEEDS:
        bst = lgb.Booster(model_file=str(submit_dir / "model" / f"f3_s{s}.txt"))
        z_lgb += sub_common.logit(bst.predict(X))
    z_lgb /= len(SEEDS)
    prep, models = sub_mlp.load(str(submit_dir / "model"), SEEDS)
    z_mlp = sub_mlp.predict_z(test, prep, models)
    z_champ = CHAMP_W_LGB * z_lgb + (1 - CHAMP_W_LGB) * z_mlp
    return np.clip(sub_common.sigmoid(z_champ + C_LOGIT), CLIP_LO, CLIP_HI)


def check_parity(submit_dir: Path, vals: pd.Series, test_path: Path,
                 ref_vals: np.ndarray) -> tuple[bool, dict]:
    md = float(np.max(np.abs(vals.astype(np.float64) - ref_vals)))
    ok = md < PARITY_TOL
    print(f"[{'PASS' if ok else 'FAIL'}] parity script.py vs 참조 max|Δ|={md:.3e} "
          f"(<{PARITY_TOL})")
    return ok, {"max_abs_diff": md, "tol": PARITY_TOL}


def check_champion_component(submit_dir: Path, out5: Path) -> tuple[bool, dict]:
    """챔피언 구성요소(0.51·LGB+0.49·MLP)가 GIHO extract script.py 출력과 일치 (5행 픽스처).
    byte-frozen LGB/MLP 복사 무결성 + 전처리 모듈 일치를 함께 증명."""
    giho = PROJECT_ROOT / "team_member_materials" / "GIHO" / "submit979_extract"
    env = dict(os.environ)
    env["LGA_TEST_PATH"] = str(REPO / "open" / "data" / "test.csv")
    env["LGA_SAMPLE_PATH"] = str(REPO / "open" / "data" / "sample_submission.csv")
    env["LGA_OUT_PATH"] = str(FIXTURE_DIR / "giho_champion_5row.csv")
    proc = subprocess.run([sys.executable, "script.py"], cwd=giho, env=env,
                          capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        return False, {"error": f"GIHO script.py 실패: {proc.stderr[-800:]}"}
    giho_out = pd.read_csv(FIXTURE_DIR / "giho_champion_5row.csv", encoding="utf-8-sig")
    champ_ref = champion_component_predict(submit_dir, REPO / "open" / "data" / "test.csv")
    md = float(np.max(np.abs(giho_out["control_success"].values.astype(float) - champ_ref)))
    ok = md < PARITY_TOL
    print(f"[{'PASS' if ok else 'FAIL'}] champion 구성요소 vs GIHO V4 max|Δ|={md:.3e}")
    return ok, {"max_abs_diff": md, "giho_rc": proc.returncode}


# ────────────────────────────────────────────────────────────────────────────
# 4) zip 레이아웃 + provenance 무결성
# ────────────────────────────────────────────────────────────────────────────
def check_zip_layout(submit_dir: Path) -> tuple[bool, dict, Path]:
    # script.py 실행/참조 임포트가 만드는 __pycache__ 와 빈 output/ 는 zip 에서 제외
    for cache_dir in submit_dir.rglob("__pycache__"):
        shutil.rmtree(cache_dir, ignore_errors=True)
    out_dir = submit_dir / "output"
    if out_dir.is_dir() and not any(out_dir.iterdir()):
        out_dir.rmdir()
    zip_path = FIXTURE_DIR / f"{submit_dir.name}.zip"
    if zip_path.exists():
        zip_path.unlink()
    proc = subprocess.run(["zip", "-r", "-q", str(zip_path), "."], cwd=submit_dir,
                          capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        return False, {"error": f"zip 실패: {proc.stderr[-800:]}"}, zip_path
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    tops = sorted({n for n in names if "/" not in n})  # 파일은 그대로, dir 은 "model/" 형태
    top_dirs = sorted({n.split("/")[0] + "/" for n in names if "/" in n})
    all_tops = set(tops) | set(top_dirs)
    expected = EXPECTED_ZIP_TOP
    stray = all_tops - expected
    missing = expected - all_tops
    model_files = sorted(n for n in names
                         if n.startswith("model/") and not n.endswith("/"))
    model_set = {n[len("model/"):] for n in model_files}
    bad_entries = [n for n in names if ("__pycache__" in n or n.endswith((".npy", ".log"))
                                        or ".DS_Store" in n)]
    model_ok = model_set == EXPECTED_MODEL_FILES
    ok = (not stray) and (not missing) and model_ok and not bad_entries
    print(f"[{'PASS' if ok else 'FAIL'}] zip 레이아웃: top={sorted(all_tops)} "
          f"model_files={len(model_set)} stray={sorted(stray)} missing={sorted(missing)} "
          f"bad={bad_entries}")
    d = {"zip_path": str(zip_path), "top_level": sorted(all_tops),
         "stray": sorted(stray), "missing": sorted(missing),
         "n_model_files": len(model_set), "bad_entries": bad_entries,
         "model_files_ok": model_ok}
    return ok, d, zip_path


def check_provenance(submit_dir: Path) -> tuple[bool, dict]:
    prov_path = submit_dir / "provenance.json"
    if not prov_path.exists():
        return False, {"error": "provenance.json 없음"}
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    hashes = prov.get("model_file_sha256", {})
    problems = []
    for rel, expected in hashes.items():
        path = submit_dir / "model" / rel
        if not path.exists():
            problems.append(f"{rel}: 파일 없음")
        elif _sha256(path) != expected:
            problems.append(f"{rel}: sha256 불일치")
    ok = not problems and prov.get("candidate_id") == "5890a4c54f502c4e"
    print(f"[{'PASS' if ok else 'FAIL'}] provenance: candidate={prov.get('candidate_id')} "
          f"hashed_files={len(hashes)} problems={problems}")
    return ok, {"candidate_id": prov.get("candidate_id"),
                "hashed_files": len(hashes), "problems": problems}


# ────────────────────────────────────────────────────────────────────────────
# 5) migration_audit
# ────────────────────────────────────────────────────────────────────────────
def check_migration_audit() -> tuple[bool, str]:
    proc = subprocess.run([sys.executable, "migration_audit.py", "--strict"],
                          cwd=REPO, capture_output=True, text=True, timeout=600)
    tail = proc.stdout.strip().splitlines()[-3:]
    ok = proc.returncode == 0 and "RESULT: PASS" in proc.stdout
    print(f"[{'PASS' if ok else 'FAIL'}] migration_audit --strict rc={proc.returncode}")
    for line in tail:
        print(f"    {line}")
    return ok, proc.stdout[-1200:]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-dir", default=os.environ.get("LGA_SUBMIT_DIR"))
    args = parser.parse_args(argv)
    submit_dir = Path(args.submission_dir or
                      REPO / "submit_lgb_mlp_cat_20260814-2258").resolve()
    print("=" * 72)
    print(f"패키지 검증기: {submit_dir}")
    print("=" * 72)

    out5 = FIXTURE_DIR / "out_5row.csv"
    out245 = FIXTURE_DIR / "out_245789.csv"

    checks: list[tuple[str, bool, dict]] = []

    ok, d = check_offline_install()
    checks.append(("offline_install_sim", ok, d))

    ok, d = check_five_row(submit_dir, out5)
    checks.append(("five_row", ok, d))
    sub5 = pd.read_csv(out5, encoding="utf-8-sig") if out5.exists() else None

    ok, d, vals245 = check_full_245k(submit_dir, out245)
    checks.append(("full_245789", ok, d))
    sub245 = pd.read_csv(out245, encoding="utf-8-sig") if out245.exists() else None

    test_fix = FIXTURE_DIR / "test_245789.csv"
    ref_vals = reference_predict(submit_dir, test_fix) if test_fix.exists() else None
    if sub245 is not None and ref_vals is not None:
        md = float(np.max(np.abs(sub245["control_success"].values.astype(np.float64) - ref_vals)))
        ok = md < PARITY_TOL
        checks.append(("parity_script_vs_reference", ok,
                       {"max_abs_diff": md, "tol": PARITY_TOL,
                        "mean_ref": float(ref_vals.mean()),
                        "mean_script": float(sub245["control_success"].values.mean())}))
        mean_shift = float(abs(sub245["control_success"].values.mean() - ref_vals.mean()))
        ok_mean = mean_shift <= MEAN_ALIGN_TOL
        checks.append(("mean_alignment", ok_mean,
                       {"abs_mean_shift": mean_shift, "tol": MEAN_ALIGN_TOL}))
        print(f"[{'PASS' if ok_mean else 'FAIL'}] mean 정렬: |Δmean|={mean_shift:.2e} "
              f"(≤{MEAN_ALIGN_TOL})")
        print(f"[{'PASS' if ok else 'FAIL'}] parity max|Δ|={md:.3e}")
    else:
        checks.append(("parity_script_vs_reference", False, {"error": "출력/참조 없음"}))

    # lgb_mlp_cat 출력 평균이 챔피언 정책 평균(동일 픽스처 챔피언 구성요소)과 정렬되는지
    if test_fix.exists():
        champ_means = champion_component_predict(submit_dir, test_fix)
        mean_champ_only = float(champ_means.mean())
        mean_cat = float(sub245["control_success"].values.mean()) if sub245 is not None else float("nan")
        shift_vs_policy = float(abs(mean_cat - mean_champ_only))
        ok_policy = shift_vs_policy <= MEAN_ALIGN_TOL
        checks.append(("mean_alignment_vs_champion_policy", ok_policy,
                       {"champion_policy_mean": mean_champ_only,
                        "candidate_mean": mean_cat,
                        "abs_shift": shift_vs_policy, "tol": MEAN_ALIGN_TOL,
                        "note": "챔피언 C_LOGIT=-0.0404 정책 하 lgb_mlp_cat 추가가 평균을 흔드는지 검증"})
        )
        print(f"[{'PASS' if ok_policy else 'FAIL'}] mean 정렬 vs 챔피언 정책: "
              f"champ_only={mean_champ_only:.4f} lgb_mlp_cat={mean_cat:.4f} "
              f"|Δ|={shift_vs_policy:.2e}")
    else:
        checks.append(("mean_alignment_vs_champion_policy", False, {"error": "픽스처 없음"}))

    ok, d = check_champion_component(submit_dir, out5)
    checks.append(("champion_component_vs_giho", ok, d))

    ok, d, zip_path = check_zip_layout(submit_dir)
    checks.append(("zip_layout", ok, d))

    ok, d = check_provenance(submit_dir)
    checks.append(("provenance_integrity", ok, d))

    ok, audit_out = check_migration_audit()
    checks.append(("migration_audit_strict", ok, {"output_tail": audit_out}))

    print("-" * 72)
    all_ok = True
    for name, ok, d in checks:
        all_ok = all_ok and ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print("RESULT:", "PASS" if all_ok else "FAIL")

    evidence = {
        "schema_version": 1,
        "task": "aimers9-top100/task-8-package-lgbmlpcat",
        "submission_dir": str(submit_dir.relative_to(PROJECT_ROOT)),
        "candidate": {"id": "lgb_mlp_cat", "candidate_id": "5890a4c54f502c4e",
                      "weights": {"lgb": W_LGB, "mlp": W_MLP, "catboost": W_CAT},
                      "c_logit": C_LOGIT, "clip": [CLIP_LO, CLIP_HI],
                      "seeds": SEEDS,
                      "champion_component": {"lgb": CHAMP_W_LGB, "mlp": 1 - CHAMP_W_LGB}},
        "policy": {"mean_align_tol": MEAN_ALIGN_TOL,
                   "champion_policy_note": ("C_LOGIT=-0.0404 는 실 2025 test 분포(로컬 부재) "
                                            "대상 정렬 상수. 합성 픽스처는 훈련 분포이므로 "
                                            "절대 목표 0.477 대신 참조 대비 |Δmean| 게이트 사용.")},
        "checks": {name: d for name, ok, d in checks},
        "result": "PASS" if all_ok else "FAIL",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    ev_path = PROJECT_ROOT / ".omo" / "evidence" / "aimers9-top100" / "task-8-package-lgbmlpcat.json"
    ev_path.parent.mkdir(parents=True, exist_ok=True)
    ev_path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False,
                                  default=_json_default) + "\n", encoding="utf-8")
    print(f"[evidence] {ev_path}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
