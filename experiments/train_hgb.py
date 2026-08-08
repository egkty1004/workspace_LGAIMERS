"""HistGradientBoosting 학습/검증 스크립트 (RF 베이스라인 교체 1차 개선).

train_baseline.py(RF)의 데이터 로드 방식 / 검증 분할 / BSS 공식을 그대로 따르되,
분류기를 HGB로 교체하고 다음을 추가한다:
- 게이트 앵커(C1): --anchor 인자 또는 backup/rf_anchor_bss.txt (없거나 비면 exit 2)
- 모드 플래그: --validate(기본, 게이트 판정 후 종료) / --full(전체 재학습 + dump)
- cross-process pickle 왕복 QA(N1): 본 검증 fit 전에 소형 모델을 dump 후
  별도 새 프로세스에서 로드해 1행 predict_proba 검증 (실패 시 exit 2)
- _to_category는 bss_preprocess 모듈에서 import (람다/__main__ 정의 금지)

실행(CWD = experiments/):
    python train_hgb.py               # --validate (기본)
    python train_hgb.py --validate
    python train_hgb.py --full
    python train_hgb.py --validate --anchor 415.57
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from bss_preprocess import CAT_COLS, _to_category, InteractionAdder

DATA_DIR = "./data"
ID = "row_id"
TARGET = "control_success"
ANCHOR_FILE = "backup/rf_anchor_bss.txt"
MODEL_PATH = "./model/rf.pkl"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# HGB 파라미터 (나머지는 기본값)
HGB_KWARGS = dict(
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42,
)

QA_ROWS = 10000      # cross-process QA용 소형 fit 행 수
QA_MAX_ITER = 10     # cross-process QA용 소형 fit 반복 수


def build_pipeline(**clf_kwargs):
    """HGB 파이프라인 (3단계).

    1. add_interact: base_state_li / count_cat / runner_risk 상호작용 피처 3종 추가
       (InteractionAdder — 게이트 +55.36 검증, ID 인코딩 FAIL로 enc는 제외)
    2. to_cat     : 7개 범주형 컬럼을 pandas category dtype으로 (bss_preprocess._to_category)
    3. clf        : HistGradientBoostingClassifier(**clf_kwargs)
    """
    return Pipeline([
        ("add_interact", InteractionAdder()),
        ("to_cat", FunctionTransformer(_to_category)),
        ("clf", HistGradientBoostingClassifier(**clf_kwargs)),
    ])


def load_data():
    """test.csv 컬럼 기준으로 피처를 결정하고 train.csv를 로드 (train_baseline.py 방식)."""
    test_cols = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),
                            encoding="utf-8-sig", nrows=0).columns
    features = [c for c in test_cols if c != ID]
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                        encoding="utf-8-sig", usecols=features + [TARGET])

    num_cols = [c for c in features if c not in CAT_COLS]
    print("train:", train.shape, "| 피처:", len(features),
          f"(범주형 {len(CAT_COLS)}, 수치형 {len(num_cols)})")
    print("시즌:", train["season"].min(), "~", train["season"].max())
    print(f"제구 성공률: {train[TARGET].mean():.4f}")
    return train, features


def compute_bss(y_true, proba):
    """BSS 공식 (train_baseline.py와 동일): max(0, 100000 * (1 - brier/(r*(1-r))))."""
    r = y_true.mean()
    brier = ((proba - y_true) ** 2).mean()
    score = max(0.0, 100000.0 * (1.0 - brier / (r * (1.0 - r))))
    return score, r, brier


def load_anchor(anchor_arg):
    """게이트 앵커 주입(C1): --anchor 인자 우선, 없으면 파일. 없거나 비면 exit 2."""
    if anchor_arg is not None:
        print(f"앵커 BSS(--anchor): {anchor_arg:.2f}")
        return anchor_arg
    if not os.path.exists(ANCHOR_FILE):
        print(f"[FATAL] 앵커 파일 없음: {ANCHOR_FILE} (--anchor 지정 또는 파일 생성 필요)",
              file=sys.stderr)
        sys.exit(2)
    with open(ANCHOR_FILE, encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        print(f"[FATAL] 앵커 파일 비어있음: {ANCHOR_FILE}", file=sys.stderr)
        sys.exit(2)
    try:
        anchor = float(raw)
    except ValueError:
        print(f"[FATAL] 앵커 파일 파싱 실패: {ANCHOR_FILE} -> {raw!r}", file=sys.stderr)
        sys.exit(2)
    print(f"앵커 BSS(파일 {ANCHOR_FILE}): {anchor:.2f}")
    return anchor


def _cleanup(paths):
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


def _roundtrip_subprocess(tmp_pkl, tmp_csv):
    """별도 새 프로세스에서 모델 로드 + 1행 predict_proba 검증 (in-process 검증 금지)."""
    code = (
        "import sys\n"
        "sys.path.insert(0, %r)\n"
        "import joblib\n"
        "import pandas as pd\n"
        "m = joblib.load(%r)\n"
        "row = pd.read_csv(%r)\n"
        "p = m.predict_proba(row)\n"
        "assert p.shape == (1, 2), p.shape\n"
        "print('PICKLE_ROUNDTRIP_OK', round(float(p[0, 1]), 6))\n"
    ) % (SCRIPT_DIR, tmp_pkl, tmp_csv)
    try:
        proc = subprocess.run([sys.executable, "-c", code],
                              capture_output=True, text=True,
                              timeout=300, cwd=SCRIPT_DIR)
    except subprocess.TimeoutExpired:
        print("[FATAL] cross-process pickle QA 타임아웃", file=sys.stderr)
        sys.exit(2)
    if proc.returncode != 0 or "PICKLE_ROUNDTRIP_OK" not in proc.stdout:
        print("[FATAL] cross-process pickle QA 실패 "
              "(by-reference 함수 직렬화 문제 가능)", file=sys.stderr)
        print("--- stdout ---", proc.stdout, file=sys.stderr)
        print("--- stderr ---", proc.stderr, file=sys.stderr)
        sys.exit(2)
    print(proc.stdout.strip())


def cross_process_pickle_qa(model, row_df, label):
    """소형 모델(또는 dump 경로)에 대한 cross-process pickle 왕복 QA (N1)."""
    tmp_paths = []
    try:
        if isinstance(model, str):
            tmp_pkl = model  # 이미 dump된 경로(예: --full의 실제 모델 파일)
        else:
            fd, tmp_pkl = tempfile.mkstemp(suffix=".pkl", prefix="hgb_qa_")
            os.close(fd)
            tmp_paths.append(tmp_pkl)
            joblib.dump(model, tmp_pkl, compress=3)
        fd, tmp_csv = tempfile.mkstemp(suffix=".csv", prefix="hgb_qa_")
        os.close(fd)
        tmp_paths.append(tmp_csv)
        row_df.to_csv(tmp_csv, index=False)
        print(f"[QA:{label}] cross-process pickle 왕복 시작")
        _roundtrip_subprocess(tmp_pkl, tmp_csv)
    finally:
        _cleanup(tmp_paths)


def run_validate(train, features, anchor_arg):
    """--validate(기본): QA -> 검증 fit(2019~2023) -> BSS 계산 -> 게이트 판정 -> exit 0/1."""
    anchor = load_anchor(anchor_arg)

    is_val = train["season"] == 2024
    X_train, y_train = train.loc[~is_val, features], train.loc[~is_val, TARGET]
    X_val, y_val = train.loc[is_val, features], train.loc[is_val, TARGET]
    print("train(2019~2023):", len(X_train), "| val(2024):", len(X_val))

    # ---- N1: cross-process pickle 왕복 QA (본 검증 fit 이전) ----
    qa_df = train.head(QA_ROWS)
    qa_model = build_pipeline(max_iter=QA_MAX_ITER, early_stopping=False,
                              random_state=42)
    t = time.time()
    qa_model.fit(qa_df[features], qa_df[TARGET])
    qa_n_iter = getattr(qa_model.named_steps["clf"], "n_iter_", None)
    print(f"QA 소형 fit({QA_ROWS}행, max_iter={QA_MAX_ITER}) :: "
          f"{time.time() - t:.1f}s | n_iter_: {qa_n_iter}")
    cross_process_pickle_qa(qa_model, qa_df[features].head(1), "qa")

    # ---- 검증 fit (2019~2023) ----
    model = build_pipeline(**HGB_KWARGS)
    t = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t
    n_iter = model.named_steps["clf"].n_iter_
    print(f"검증 fit(2019~2023) :: {fit_time:.1f}s | n_iter_: {n_iter}")

    # ---- 2024 BSS ----
    val_pred = model.predict_proba(X_val)[:, 1]
    bss_2024, r_2024, brier_2024 = compute_bss(y_val, val_pred)
    print(f"Brier(2024): {brier_2024:.6f} | r(2024): {r_2024:.4f} (≈0.4861 기대)")
    print(f"2024 BSS: {bss_2024:.2f}")

    # ---- 2023 BSS (2019~2023 fit 모델로 2023 행 예측, in-sample) ----
    is_2023 = train["season"] == 2023
    pred_2023 = model.predict_proba(train.loc[is_2023, features])[:, 1]
    bss_2023, r_2023, brier_2023 = compute_bss(train.loc[is_2023, TARGET], pred_2023)
    print(f"Brier(2023, in-sample): {brier_2023:.6f} | r(2023): {r_2023:.4f}")
    print(f"2023 BSS (in-sample): {bss_2023:.2f}")

    # ---- 게이트 판정 ----
    ok = bss_2024 > anchor
    print(f"게이트: HGB 2024 BSS {bss_2024:.2f} vs RF 앵커 BSS {anchor:.2f}")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def run_full(train, features):
    """--full: 게이트 통과 전제 하에 전체(2019~2024) 재학습 + dump + 스모크 검증."""
    model = build_pipeline(**HGB_KWARGS)
    t = time.time()
    model.fit(train[features], train[TARGET])
    print(f"전체 재학습(2019~2024) :: {time.time() - t:.1f}s | "
          f"n_iter_: {model.named_steps['clf'].n_iter_}")

    os.makedirs("./model", exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    print(f"저장 완료: {MODEL_PATH}")

    # ---- 스모크 predict(1행) 검증 + 실제 모델 파일 cross-process 확인 ----
    row = train[features].head(1)
    loaded = joblib.load(MODEL_PATH)
    p = loaded.predict_proba(row)
    if p.shape != (1, 2):
        print(f"[FATAL] 스모크 predict 실패: shape={p.shape}", file=sys.stderr)
        sys.exit(2)
    print(f"스모크 predict OK: {p[0].round(4).tolist()}")
    cross_process_pickle_qa(MODEL_PATH, row, "model")


def main():
    parser = argparse.ArgumentParser(
        description="HGB 학습/검증 (게이트: HGB 2024 BSS > RF 앵커 BSS)")
    parser.add_argument("--anchor", type=float, default=None,
                        help="게이트 앵커 BSS (미지정 시 backup/rf_anchor_bss.txt 읽음)")
    parser.add_argument("--validate", action="store_true",
                        help="검증 모드 (기본): QA -> 검증 fit -> BSS -> PASS/FAIL 후 종료")
    parser.add_argument("--full", action="store_true",
                        help="전체 재학습 + ./model/rf.pkl dump + 스모크 검증")
    args = parser.parse_args()

    train, features = load_data()

    if args.full:
        run_full(train, features)
    else:
        run_validate(train, features, args.anchor)


if __name__ == "__main__":
    main()
