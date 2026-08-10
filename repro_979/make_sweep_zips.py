#!/usr/bin/env python3
"""
[repro_979] 챔피언 ±δ 평균 스윕 제출 ZIP 생성기 (2026-08-10)

TRUE 979.31 챔피언 파이프라인(team_member_materials/GIHO/submit979_extract/)에서
C_LOGIT 상수만 확률 공간 δ만큼 이동시킨 변형 제출을 생성한다.
(순수 평균 이동 — Oracle 전략. C_LOGIT = logit(0.477+δ) - logit(0.477) 을 정확히 계산)

인터페이스:
  python make_sweep_zips.py [--deltas D1 D2 ...] [--no-zip] [--champion-dir PATH]

  --deltas      확률 공간 δ 목록 (float). 기본: -0.010 -0.006 -0.003 0 0.003
                Day2 재실행: --deltas 0.006 0.010 (코드 수정 없음)
  --no-zip      빌드 + 스모크 + 매니페스트만 수행, ZIP 생성 생략
  --champion-dir  챔피언 소스 디렉터리 (기본: 워크스페이스 루트 기준
                team_member_materials/GIHO/submit979_extract/)

출력 구조 (staging root: repro_979/submit_sweep/, gitignore 대상):
  submit_sweep/variant_delta_<d>/          각 변형 디렉터리
  submit_sweep/zips/submit_delta_<d>.zip   제출 ZIP
  submit_sweep/manifest.json               변형별 메타데이터
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np
import pandas as pd

# ── 상수 ──────────────────────────────────────────────────────────────
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAMPION_DIR_DEFAULT = os.path.join(
    WORKSPACE_ROOT, "team_member_materials", "GIHO", "submit979_extract"
)
STAGING_ROOT = os.path.join(WORKSPACE_ROOT, "repro_979", "submit_sweep")
ZIPS_DIR = os.path.join(STAGING_ROOT, "zips")
MANIFEST_PATH = os.path.join(STAGING_ROOT, "manifest.json")

# 챔피언 script.py 상수 (검증된 값)
CHAMP_C_LOGIT = -0.0404        # = logit(0.477) - logit(0.4871), 2025 base rate 추정
CHAMP_BASE_RATE = 0.477        # 2025 베이스레이트 추정치 (스윕 기준점)
C_LOGIT_LINE = "C_LOGIT = -0.0404"  # script.py 내 정확히 1회 등장해야 함

# 스모크 픽스처
SMOKE_TEST_PATH = os.path.join(WORKSPACE_ROOT, "repro_979", "open", "data", "test.csv")
SMOKE_SAMPLE_PATH = os.path.join(
    WORKSPACE_ROOT, "repro_979", "open", "data", "sample_submission.csv"
)
INTERPRETER = "/home/gpu_01/.conda/envs/aimers9/bin/python"
SMOKE_TIMEOUT_SEC = 600

# 출력 포맷 / 검증
CLIP_LO, CLIP_HI = 0.30, 0.70
IDENTITY_TOL = 1e-6          # 로짓 차이 항등식 허용 오차
UNCLIPPED_LO, UNCLIPPED_HI = 0.30, 0.70  # 미클리핑 판정 기준 (strict inside)
TOP_LEVEL_FILES = ["script.py", "common.py", "mlp_model.py", "requirements.txt"]
DEFAULT_DELTAS = [-0.010, -0.006, -0.003, 0.0, 0.003]


# ── 유틸 ──────────────────────────────────────────────────────────────
def logit(p):
    """확률 → 로짓 (정확한 변환, 근사 아님)."""
    return np.log(np.asarray(p, dtype=float) / (1.0 - np.asarray(p, dtype=float)))


def compute_c_logit(delta):
    """δ(확률 공간)에 대한 C_LOGIT 변형값. 반올림 6자리.

    C_LOGIT_variant = -0.0404 + (logit(0.477+δ) − logit(0.477))
    """
    return round(
        CHAMP_C_LOGIT + (logit(CHAMP_BASE_RATE + delta) - logit(CHAMP_BASE_RATE)),
        6,
    )


def delta_label(delta):
    """디렉터리/ZIP 라벨. 소수점 3자리 고정, 음수는 부호 유지, 0="0", 양수는 + 생략.
    예: delta_-0.010, delta_0, delta_0.003, delta_0.010"""
    s = f"{delta:.3f}"
    if s in ("0.000", "-0.000"):
        return "delta_0"
    return f"delta_{s}"


def fmt_delta(delta):
    """라벨과 동일한 δ 문자열 (표/매니페스트용)."""
    s = f"{delta:.3f}"
    if s in ("0.000", "-0.000"):
        return "0"
    return s


# ── 빌드 ──────────────────────────────────────────────────────────────
def build_variant_dir(delta, c_logit, champion_dir, variant_dir):
    """챔피언에서 모델/공통 모듈을 byte-identical 복사하고,
    script.py의 C_LOGIT 상수만 교체한 변형 디렉터리를 만든다."""
    os.makedirs(variant_dir, exist_ok=True)

    # model/* (23 파일) byte-identical 복사
    champ_model = os.path.join(champion_dir, "model")
    variant_model = os.path.join(variant_dir, "model")
    os.makedirs(variant_model, exist_ok=True)
    model_files = sorted(os.listdir(champ_model))
    for f in model_files:
        shutil.copy2(os.path.join(champ_model, f), os.path.join(variant_model, f))

    # 공통 모듈 + requirements byte-identical 복사
    for f in ["common.py", "mlp_model.py", "requirements.txt"]:
        shutil.copy2(os.path.join(champion_dir, f), os.path.join(variant_dir, f))

    # script.py 복사 후 C_LOGIT 상수 1회 등장 확인 + 교체
    champ_script = os.path.join(champion_dir, "script.py")
    with open(champ_script, "r", encoding="utf-8") as fh:
        content = fh.read()
    count = content.count(C_LOGIT_LINE)
    if count != 1:
        raise RuntimeError(
            f"챔피언 script.py에서 '{C_LOGIT_LINE}' 등장 횟수 = {count} (기대: 1)"
        )
    variant_value = repr(c_logit)
    new_content = content.replace(C_LOGIT_LINE, f"C_LOGIT = {variant_value}", 1)
    out_script = os.path.join(variant_dir, "script.py")
    with open(out_script, "w", encoding="utf-8") as fh:
        fh.write(new_content)
    return model_files


# ── 스모크 테스트 (TDD 게이트) ─────────────────────────────────────────
def run_smoke(variant_dir, out_path):
    """aimers9 python으로 변형 script.py를 실행한다.
    env 오버라이드: LGA_TEST_PATH / LGA_SAMPLE_PATH / LGA_OUT_PATH.
    반환: (returncode, stdout, stderr, runtime_sec)"""
    env = dict(os.environ)
    env["LGA_TEST_PATH"] = SMOKE_TEST_PATH
    env["LGA_SAMPLE_PATH"] = SMOKE_SAMPLE_PATH
    env["LGA_OUT_PATH"] = out_path
    t0 = time.time()
    proc = subprocess.run(
        [INTERPRETER, "script.py"],
        cwd=variant_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=SMOKE_TIMEOUT_SEC,
    )
    return proc.returncode, proc.stdout, proc.stderr, time.time() - t0


def validate_output(out_path, sample_path):
    """출력 CSV 포맷 검증.
    - row_id, control_success 컬럼
    - 행 수 == 샘플 행 수
    - utf-8-sig BOM (EF BB BF)
    - NaN 없음
    - 모든 값 [0.30, 0.70] 범위
    반환: (pred Series, mean, std, unclipped_rows)"""
    with open(out_path, "rb") as fh:
        head = fh.read(3)
    if head != b"\xef\xbb\xbf":
        raise AssertionError(f"출력 파일 BOM 없음 (첫 3바이트: {head!r})")

    df = pd.read_csv(out_path, encoding="utf-8-sig")
    missing = set(["row_id", "control_success"]) - set(df.columns)
    if missing:
        raise AssertionError(f"출력 컬럼 누락: {sorted(missing)}")
    sample = pd.read_csv(sample_path, encoding="utf-8-sig")
    if len(df) != len(sample):
        raise AssertionError(
            f"출력 행 수 {len(df)} != 샘플 행 수 {len(sample)}"
        )
    if df["control_success"].isna().any():
        raise AssertionError("출력에 NaN 존재")

    p = df["control_success"].astype(float).values
    if (p < CLIP_LO - 1e-12).any() or (p > CLIP_HI + 1e-12).any():
        bad = np.where((p < CLIP_LO - 1e-12) | (p > CLIP_HI + 1e-12))[0]
        raise AssertionError(
            f"출력 값 범위 위반 [{CLIP_LO},{CLIP_HI}]: {len(bad)}행 (예: {p[bad[:3]]})"
        )
    mean = float(p.mean())
    std = float(p.std())
    unclipped = int(((p > UNCLIPPED_LO) & (p < UNCLIPPED_HI)).sum())
    return pd.Series(p, index=df["row_id"]), mean, std, unclipped


def check_logit_identity(base_pred, var_pred, delta_logit):
    """δ=0 베이스 대비 각 변형의 행별 로짓 차이 항등식 검증.

    미클리핑 행(베이스 예측이 (0.30, 0.70) 엄밀 내부)에 대해
      |logit(p_var) − logit(p_base) − Δlogit| < 1e-6
    이 성립해야 C_LOGIT만 바뀌었음을 증명한다. (δ=0 자신과는 비교 생략)
    """
    if len(base_pred) != len(var_pred):
        raise AssertionError("베이스/변형 행 수 불일치")
    mask = (base_pred > UNCLIPPED_LO) & (base_pred < UNCLIPPED_HI)
    n_unclipped = int(mask.sum())
    if n_unclipped == 0:
        return 0, 0.0  # 검증 가능한 행 없음 — 통과로 간주
    diff = logit(var_pred[mask]) - logit(base_pred[mask]) - delta_logit
    max_err = float(np.abs(diff).max())
    if max_err >= IDENTITY_TOL:
        idx = int(np.argmax(np.abs(diff)))
        raise AssertionError(
            f"로짓 항등식 위반: max|logit(p_var)−logit(p_base)−Δlogit| = "
            f"{max_err:.3e} >= {IDENTITY_TOL} (행 {idx}: p_base="
            f"{base_pred[mask][idx]:.6f}, p_var={var_pred[mask][idx]:.6f})"
        )
    return n_unclipped, max_err


# ── ZIP ───────────────────────────────────────────────────────────────
def make_zip(variant_dir, zip_path):
    """variant_dir 내부 파일들을 ZIP 루트에 직접 기록 (래퍼 디렉토리 없이).
    엔트리: model/ 디렉터리 + 모델 파일 + 4개 최상위 파일 (28 엔트리).
    python -m zipfile -c 는 variant_dir 자체를 래퍼로 포함하므로 사용하지 않는다.
    반환: zip 파일 크기(bytes)"""
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    tmp_zip = zip_path + ".tmp"
    if os.path.exists(tmp_zip):
        os.remove(tmp_zip)
    # __pycache__ 제거 (스모크 실행 부산물) — ZIP 엔트리 오염 방지
    pycache = os.path.join(variant_dir, "__pycache__")
    if os.path.isdir(pycache):
        shutil.rmtree(pycache)
    model_dir = os.path.join(variant_dir, "model")
    model_names = sorted(os.listdir(model_dir))
    with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("model/", "")
        for name in model_names:
            zf.write(os.path.join(model_dir, name), arcname=f"model/{name}")
        for top in TOP_LEVEL_FILES:
            zf.write(os.path.join(variant_dir, top), arcname=top)
    os.replace(tmp_zip, zip_path)
    return os.path.getsize(zip_path)


def verify_zip(zip_path, n_model_files):
    """ZIP 구조 검증: 최상위 이름 5개 정확 일치 + 총 엔트리 수.
    총 엔트리 = 1 (model/) + n_model_files + 4 (최상위 파일)"""
    expected_top = {"model/"} | set(TOP_LEVEL_FILES)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        # 최상위 이름: 루트 파일(TOP_LEVEL_FILES)은 그대로, 그 외는 첫 컴포넌트를 디렉터리로 정규화
        top = set()
        for n in names:
            first = n.split("/")[0]
            top.add(first if first in TOP_LEVEL_FILES else first + "/")
        if top != expected_top:
            raise AssertionError(
                f"ZIP 최상위 이름 불일치: {sorted(top)} != {sorted(expected_top)}"
            )
        expected_count = 1 + n_model_files + len(TOP_LEVEL_FILES)
        if len(names) != expected_count:
            raise AssertionError(
                f"ZIP 엔트리 수 {len(names)} != 기대 {expected_count}"
            )
    return expected_count


# ── 매니페스트 ────────────────────────────────────────────────────────
def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"champion_dir": CHAMPION_DIR_DEFAULT, "variants": []}


def save_manifest(manifest):
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── 메인 ──────────────────────────────────────────────────────────────
def build_one(delta, champion_dir, do_zip, manifest, model_files_expected):
    """단일 δ 변형을 빌드 + 스모크 + ZIP + 매니페스트 갱신.
    반환: (entry_dict, ok)"""
    label = delta_label(delta)
    c_logit = compute_c_logit(delta)
    variant_dir = os.path.join(STAGING_ROOT, label)
    out_path = os.path.join(variant_dir, "output", "submission.csv")
    entry = {
        "delta": delta,
        "c_logit": c_logit,
        "mean": None,
        "std": None,
        "unclipped_rows": None,
        "runtime_sec": None,
        "status": "FAILED",
        "zip_sha256": None,
    }

    try:
        if os.path.isdir(variant_dir):
            shutil.rmtree(variant_dir)
        model_files = build_variant_dir(delta, c_logit, champion_dir, variant_dir)
        if model_files_expected is not None and len(model_files) != model_files_expected:
            raise RuntimeError(
                f"모델 파일 수 {len(model_files)} != 기대 {model_files_expected}"
            )

        rc, stdout, stderr, runtime = run_smoke(variant_dir, out_path)
        entry["runtime_sec"] = round(runtime, 2)
        if rc != 0:
            raise AssertionError(
                f"스모크 실행 실패 (exit={rc})\n--- stdout tail ---\n"
                + "\n".join(stdout.splitlines()[-20:])
                + "\n--- stderr tail ---\n"
                + "\n".join(stderr.splitlines()[-20:])
            )
        if "test: (5," not in stdout:
            raise AssertionError(
                f"stdout에 'test: (5,' 없음\n{stdout[-500:]}"
            )
        if "submission:" not in stdout:
            raise AssertionError(
                f"stdout에 'submission:' 없음\n{stdout[-500:]}"
            )

        pred, mean, std, unclipped = validate_output(out_path, SMOKE_SAMPLE_PATH)
        entry.update(mean=mean, std=std, unclipped_rows=unclipped)

        if do_zip:
            zip_path = os.path.join(ZIPS_DIR, f"submit_{label}.zip")
            size = make_zip(variant_dir, zip_path)
            verify_zip(zip_path, len(model_files))
            entry["zip_sha256"] = sha256_file(zip_path)
            entry["zip_size"] = size

        entry["status"] = "OK"
        # 로짓 항등식은 전체 루프 후 베이스(δ=0)와 비교하므로 여기선 결과만 보관
        entry["_pred_path"] = out_path
        return entry, True, out_path
    except Exception as exc:  # noqa: BLE001 — 변형별 실패는 개별 처리
        entry["error"] = str(exc)
        entry["_pred_path"] = out_path
        return entry, False, out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description="챔피언 ±δ 평균 스윕 ZIP 생성기")
    ap.add_argument(
        "--deltas", type=float, nargs="+", default=DEFAULT_DELTAS,
        help=f"확률 공간 δ 목록 (기본: {DEFAULT_DELTAS})",
    )
    ap.add_argument(
        "--no-zip", action="store_true",
        help="ZIP 생성 생략 (빌드 + 스모크 + 매니페스트만)",
    )
    ap.add_argument(
        "--champion-dir", default=CHAMPION_DIR_DEFAULT,
        help="챔피언 소스 디렉터리",
    )
    args = ap.parse_args(argv)

    champion_dir = os.path.abspath(args.champion_dir)
    if not os.path.isdir(os.path.join(champion_dir, "model")):
        ap.error(f"챔피언 디렉터리에 model/ 없음: {champion_dir}")
    if not os.path.isfile(os.path.join(champion_dir, "script.py")):
        ap.error(f"챔피언 디렉터리에 script.py 없음: {champion_dir}")

    deltas = args.deltas
    # 0 제거 후 정렬 (스윕 순서 결정) + 0을 베이스로 먼저 빌드
    base_included = 0.0 in deltas
    deltas = sorted(set(deltas))
    if not base_included:
        deltas = [0.0] + deltas  # 베이스(δ=0)는 항상 빌드 → 항등식 기준

    manifest = load_manifest()
    manifest.setdefault("variants", [])
    entries = {}
    failures = []
    model_files_expected = len(os.listdir(os.path.join(champion_dir, "model")))

    # 1) 전 변형 빌드 + 스모크
    for delta in deltas:
        entry, ok, _ = build_one(delta, champion_dir, not args.no_zip, manifest,
                                 model_files_expected)
        entries[delta] = entry
        if not ok:
            failures.append(delta)

    # 2) 로짓 항등식: δ≠0 변형 ↔ δ=0 베이스
    base_entry = entries.get(0.0)
    for delta in deltas:
        if delta == 0.0:
            continue
        entry = entries[delta]
        if entry["status"] != "OK" or base_entry is None:
            continue
        try:
            base_pred = pd.read_csv(base_entry["_pred_path"], encoding="utf-8-sig")[
                "control_success"].astype(float)
            var_pred = pd.read_csv(entry["_pred_path"], encoding="utf-8-sig")[
                "control_success"].astype(float)
            delta_logit = entry["c_logit"] - CHAMP_C_LOGIT
            n_unclipped, max_err = check_logit_identity(base_pred, var_pred, delta_logit)
            entry["identity_checked_rows"] = n_unclipped
            entry["identity_max_err"] = max_err
        except Exception as exc:  # noqa: BLE001 — 항등식 실패는 개별 FAILED
            entry["status"] = "FAILED"
            entry["error"] = f"항등식 검증 실패: {exc}"
            failures.append(delta)

    # 3) 단조성: δ 오름차순 → 평균 오름차순 (성공 변형만)
    ok_entries = [(d, e) for d, e in entries.items()
                  if e["status"] == "OK" and e["mean"] is not None and d != 0.0]
    ok_entries.sort()
    means = [e["mean"] for _, e in ok_entries]
    if len(means) >= 2 and any(a > b for a, b in zip(means, means[1:])):
        # δ=0 베이스도 단조성에 포함 (베이스가 성공했을 때)
        if base_entry is not None and base_entry["status"] == "OK":
            all_ok = sorted(
                [(d, e) for d, e in entries.items()
                 if e["status"] == "OK" and e["mean"] is not None]
            )
            all_means = [e["mean"] for _, e in all_ok]
            if any(a > b for a, b in zip(all_means, all_means[1:])):
                failures.append("monotonic")
                print("[FAIL] 단조성 위반: δ 증가에 따라 평균이 감소함", file=sys.stderr)

    # 4) 매니페스트 저장 — 베이스(0)가 요청 목록에 없으면 제외
    requested = set(args.deltas)
    new_entries = []
    for d in sorted(entries):
        if d == 0.0 and d not in requested:
            continue  # 내부 베이스는 매니페스트에 기록하지 않음
        e = dict(entries[d])
        e.pop("_pred_path", None)
        e["delta"] = d  # 수치 δ 유지 (라벨은 디렉터리명에 사용)
        new_entries.append(e)
    manifest["variants"] = new_entries
    manifest["no_zip"] = args.no_zip
    save_manifest(manifest)

    # 5) 요약 테이블
    print(f"\n=== 스윕 요약 (챔피언: {champion_dir}) ===")
    print(f"{'δ':>8} | {'C_LOGIT':>10} | {'mean':>8} | {'status':>8} | {'zip size':>10}")
    print("-" * 60)
    for d in sorted(entries):
        e = entries[d]
        if d == 0.0 and d not in requested:
            continue
        size = e.get("zip_size", "-")
        size_s = f"{size}" if isinstance(size, int) else "-"
        mean_s = f"{e['mean']:.4f}" if e["mean"] is not None else "-"
        print(f"{fmt_delta(d):>8} | {e['c_logit']:>10} | "
              f"{mean_s:>8} | "
              f"{e['status']:>8} | {size_s:>10}")

    if failures:
        print(f"\n[FAIL] 실패 변형: {sorted(set(failures))}", file=sys.stderr)
        for d in sorted(set(failures)):
            if d == "monotonic":
                continue
            e = entries.get(d, {})
            if e.get("error"):
                print(f"  δ={d}: {e['error']}", file=sys.stderr)
        return 1
    print("\n[PASS] 모든 변형 성공")
    return 0


if __name__ == "__main__":
    sys.exit(main())
