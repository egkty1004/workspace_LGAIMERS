#!/usr/bin/env python3
"""챔피언 출처(provenance) 검증 게이트.

1) 진짜 979.31 챔피언이 GIHO extract(submit979_extract)임을 검증(버려진 build 아님).
2) aimers9 env(lgb 4.7.0 / pandas 2.0.3 / torch) 검증.
3) 챔피언 소스 전체 sha256 매니페스트 스냅샷.

사용법: <aimers9 python> check_champion_provenance.py [--dir <경로>]
--dir 또는 CHAMPION_DIR env 로 검증 대상을 오버라이드(음성 대조군: repro_979/submit).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHAMPION_DIR = ROOT / "team_member_materials" / "GIHO" / "submit979_extract"
ZIP_PATH = (
    ROOT / "team_member_materials" / "GIHO" / "리더보드 979.31"
    / "submit_F3MLP_cl0404.zip"
)
MANIFEST_PATH = ROOT / "repro_979" / "submit_sweep" / "champion_manifest.json"
OPEN_DATA_DIR = ROOT / "repro_979" / "open" / "data"
AIMERS9_PYTHON = "/home/gpu_01/.conda/envs/aimers9/bin/python"

# 챔피언 핵심 파일 md5 지문 (불일치 시 FAIL)
CHAMPION_MD5 = {
    "script.py": "6845d6d8a9ab9a98dae36440b195348f",
    "model/f3_s42.txt": "22a12c381c939004d86b72b065126d5d",
    "model/mlp_prep.pkl": "dfccfb24ea9e47a9af9e67012a54ec77",
}
EXPECTED_LGB_VERSION = "4.7.0"
EXPECTED_PANDAS_VERSION = "2.0.3"
EXPECTED_FEATURE_COUNT = 49
EXPECTED_CATS_COUNT = 9
FORBIDDEN_FEATURES = ("asof_n_bucket", "score_diff_binary")
C_LOGIT_FRAGMENT = "C_LOGIT = -0.0404"


def file_hash(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _header(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig") as f:
        return f.readline().strip().split(",")


def check_md5_fingerprints(d: Path) -> tuple[bool, str]:
    """핵심 파일 md5 지문이 챔피언과 일치하는지."""
    problems: list[str] = []
    for rel, expected in CHAMPION_MD5.items():
        path = d / rel
        if not path.exists():
            problems.append(f"{rel}: 파일 없음")
        else:
            actual = file_hash(path, "md5")
            if actual != expected:
                problems.append(f"{rel}: 기대 {expected} / 실제 {actual}")
    if problems:
        return False, "; ".join(problems)
    return True, "md5 지문 일치 (script.py / f3_s42.txt / mlp_prep.pkl)"


def check_meta(d: Path) -> tuple[bool, str]:
    """train_meta(49피처, 금지 피처 없음) / mlp_meta(cats 9) 확인."""
    model_dir = d / "model"
    problems: list[str] = []

    train_meta_path = model_dir / "train_meta.json"
    if not train_meta_path.exists():
        problems.append("model/train_meta.json 없음")
    else:
        meta = json.loads(train_meta_path.read_text(encoding="utf-8"))
        n_feat = len(meta.get("features", []))
        if n_feat != EXPECTED_FEATURE_COUNT:
            problems.append(f"feature 수 {n_feat} != 기대 {EXPECTED_FEATURE_COUNT}")
        text = json.dumps(meta, ensure_ascii=False)
        problems += [
            f"train_meta에 금지 피처 포함: {x}"
            for x in FORBIDDEN_FEATURES if x in text
        ]

    mlp_meta_path = model_dir / "mlp_meta.json"
    if not mlp_meta_path.exists():
        problems.append("model/mlp_meta.json 없음")
    else:
        meta = json.loads(mlp_meta_path.read_text(encoding="utf-8"))
        cats = meta.get("cats", [])
        if len(cats) != EXPECTED_CATS_COUNT:
            problems.append(f"cats len {len(cats)} != 기대 {EXPECTED_CATS_COUNT}")
        emb_dim = meta.get("emb_dim")
        if emb_dim is not None and len(emb_dim) != EXPECTED_CATS_COUNT:
            problems.append(f"emb_dim len {len(emb_dim)} != 기대 {EXPECTED_CATS_COUNT}")

    if problems:
        return False, "; ".join(problems)
    return True, f"meta 일치 (features {EXPECTED_FEATURE_COUNT}, cats {EXPECTED_CATS_COUNT})"


def check_script_v4(d: Path) -> tuple[bool, str]:
    """script.py V4 구성(C_LOGIT 1회, MODEL_DIR, env override) 확인."""
    script_path = d / "script.py"
    if not script_path.exists():
        return False, "script.py 없음"
    text = script_path.read_text(encoding="utf-8")
    problems: list[str] = []
    if text.count(C_LOGIT_FRAGMENT) != 1:
        problems.append(f"'{C_LOGIT_FRAGMENT}' 등장 {text.count(C_LOGIT_FRAGMENT)}회 != 1회")
    if 'MODEL_DIR = "model"' not in text:
        problems.append('MODEL_DIR = "model" 없음')
    problems += [f"env override {v} 없음" for v in
                 ("LGA_TEST_PATH", "LGA_SAMPLE_PATH", "LGA_OUT_PATH") if v not in text]
    if problems:
        return False, "; ".join(problems)
    return True, "script.py V4 구성 일치 (C_LOGIT 1회, MODEL_DIR, env override 3종)"


def check_smoke_fixtures() -> tuple[bool, str]:
    """스모크 픽스처(test.csv / sample_submission.csv)가 5행인지."""
    problems: list[str] = []
    test_path = OPEN_DATA_DIR / "test.csv"
    if not test_path.exists():
        problems.append(f"픽스처 없음: {test_path}")
    else:
        lines = test_path.read_text(encoding="utf-8-sig").strip().splitlines()
        if len(lines) != 6:
            problems.append(f"test.csv 행 수 {len(lines) - 1} != 5")
        if _header(test_path)[0] != "row_id":
            problems.append("test.csv 첫 컬럼이 row_id 아님")

    sample_path = OPEN_DATA_DIR / "sample_submission.csv"
    if not sample_path.exists():
        problems.append(f"픽스처 없음: {sample_path}")
    else:
        lines = sample_path.read_text(encoding="utf-8-sig").strip().splitlines()
        if len(lines) != 6:
            problems.append(f"sample_submission.csv 행 수 {len(lines) - 1} != 5")
        if _header(sample_path)[:2] != ["row_id", "control_success"]:
            problems.append("sample_submission.csv 컬럼이 row_id/control_success 아님")

    if problems:
        return False, "; ".join(problems)
    return True, "스모크 픽스처 일치 (test.csv / sample_submission.csv, 각 5행)"


def check_env() -> tuple[bool, str]:
    """aimers9 env 가 lgb 4.7.0 / pandas 2.0.3 / torch 인지."""
    if not os.path.exists(AIMERS9_PYTHON):
        return False, f"aimers9 파이썬 없음: {AIMERS9_PYTHON}"
    code = "import lightgbm, pandas, torch; print(lightgbm.__version__, pandas.__version__)"
    try:
        proc = subprocess.run([AIMERS9_PYTHON, "-c", code],
                              capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"env 실행 실패: {exc}"
    if proc.returncode != 0:
        return False, f"env import 실패: {proc.stderr.strip() or proc.stdout.strip()}"
    parts = proc.stdout.strip().split()
    if len(parts) != 2:
        return False, f"env 출력 형식 이상: {proc.stdout.strip()!r}"
    lgb_ver, pd_ver = parts
    problems = []
    if lgb_ver != EXPECTED_LGB_VERSION:
        problems.append(f"lightgbm {lgb_ver} != 기대 {EXPECTED_LGB_VERSION}")
    if pd_ver != EXPECTED_PANDAS_VERSION:
        problems.append(f"pandas {pd_ver} != 기대 {EXPECTED_PANDAS_VERSION}")
    if problems:
        return False, "; ".join(problems)
    return True, f"aimers9 env 일치 (lightgbm {lgb_ver}, pandas {pd_ver}, torch OK)"


def check_zip_identity(d: Path) -> tuple[bool, str]:
    """제출 zip 내 핵심 3파일이 extract 와 byte 동일한지 (zip 없으면 WARNING)."""
    if not ZIP_PATH.exists():
        return True, f"WARNING: zip 없음 — 스킵 ({ZIP_PATH})"
    problems: list[str] = []
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for rel in ("script.py", "model/f3_s42.txt", "model/mlp_prep.pkl"):
            src = d / rel
            if not src.exists():
                problems.append(f"{rel}: extract 파일 없음")
            elif rel not in zf.namelist():
                problems.append(f"{rel}: zip 내 없음")
            else:
                zip_md5 = hashlib.md5(zf.read(rel)).hexdigest()
                extract_md5 = file_hash(src, "md5")
                if zip_md5 != extract_md5:
                    problems.append(f"{rel}: zip {zip_md5} != extract {extract_md5}")
    if problems:
        return False, "; ".join(problems)
    return True, "zip ↔ extract byte 동일 (script.py / f3_s42.txt / mlp_prep.pkl)"


def write_manifest(d: Path) -> None:
    """챔피언 소스 전체 sha256 매니페스트 스냅샷 (__pycache__ 제외)."""
    files = {}
    for path in sorted(d.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            files[str(path.relative_to(d))] = file_hash(path, "sha256")

    manifest = {
        "champion_dir": str(d),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "env": {"python": AIMERS9_PYTHON, "lightgbm": None, "pandas": None},
        "sha256": files,
    }
    code = "import lightgbm, pandas; print(lightgbm.__version__, pandas.__version__)"
    try:
        proc = subprocess.run([AIMERS9_PYTHON, "-c", code],
                              capture_output=True, text=True, timeout=120)
        if proc.returncode == 0:
            parts = proc.stdout.strip().split()
            manifest["env"]["lightgbm"] = parts[0] if parts else None
            manifest["env"]["pandas"] = parts[1] if len(parts) > 1 else None
    except (OSError, subprocess.TimeoutExpired):
        pass

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main(argv: list[str]) -> int:
    champion_dir = Path(os.environ.get("CHAMPION_DIR", DEFAULT_CHAMPION_DIR))
    if "--dir" in argv and argv.index("--dir") + 1 < len(argv):
        champion_dir = Path(argv[argv.index("--dir") + 1])

    is_default = champion_dir == DEFAULT_CHAMPION_DIR
    if not is_default:
        print(f"[INFO] CHAMPION_DIR 오버라이드: {champion_dir}")

    checks = [
        ("md5 지문", check_md5_fingerprints(champion_dir)),
        ("meta 구성", check_meta(champion_dir)),
        ("script.py V4", check_script_v4(champion_dir)),
        ("스모크 픽스처", check_smoke_fixtures()),
        ("aimers9 env", check_env()),
        ("zip identity", check_zip_identity(champion_dir)),
    ]

    print("=" * 64)
    print("챔피언 출처 검증 게이트")
    print(f"검증 대상: {champion_dir}")
    print("-" * 64)

    all_ok = True
    for name, (ok, detail) in checks:
        if not ok:
            all_ok = False
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    print("-" * 64)
    if all_ok and is_default:
        write_manifest(champion_dir)
        print(f"[OK] 매니페스트 작성: {MANIFEST_PATH}")
    elif all_ok and not is_default:
        print("[WARN] 오버라이드 모드 — 매니페스트 작성 생략")

    if all_ok:
        print("RESULT: PASS — 검증 대상은 진짜 979.31 챔피언 소스입니다.")
        return 0
    print("RESULT: FAIL — 검증 대상은 챔피언이 아닙니다 (위 FAIL 항목 확인).")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
