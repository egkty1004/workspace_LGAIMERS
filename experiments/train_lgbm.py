"""LightGBM 학습/검증 스크립트 (HGB 대체 2차 개선).

train_hgb.py의 데이터 로드 / 검증 분할 / BSS 공식 / QA 구조를 그대로 따르되,
분류기를 LightGBM으로 교체한다 (2026-08-08 선택 이유):
- HGB는 시드 노이즈 ±100 (335~532)로 검증 신뢰성 저하
- LGBM은 시드 노이즈 ±12 (445~470), 중앙값 452.17 > HGB 439.00 (+13)
- 평가 서버 설치 리스크: lightgbm==4.7.0 wheel은 glibc 2.35(Ubuntu 22.04) 호환,
  의존성(numpy/scipy/narwhals) 전부 평가 서버 기본 패키지로 충족.
  설치 오류는 제출 횟수 미반영 → 실질 리스크 없음.

파이프라인: base 47컬럼 (interact 제외 — Public -103 실증).
early stopping: train_test_split(내부 검증 10%, random_state=시드).

실행(CWD = experiments/):
    python train_lgbm.py --validate [--anchor 439.00] [--seeds 42,7,123]
    python train_lgbm.py --full [--seed 42]
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from bss_preprocess import CAT_COLS, _to_category
from feat_eng import HandMatch

DATA_DIR = "./data"
ID = "row_id"
TARGET = "control_success"
ANCHOR_FILE = "backup/rf_anchor_bss.txt"
MODEL_PATH = "./model/rf.pkl"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

LGBM_KWARGS = dict(
    n_estimators=1000,
    learning_rate=0.1,
    num_leaves=31,
    early_stopping_rounds=10,
    verbosity=-1,
)

QA_ROWS = 10000
QA_N_ESTIMATORS = 10


def build_pipeline(**clf_kwargs):
    """LGBM 파이프라인 (3단계: add_hand → to_cat → clf).

    - add_hand: E2 hand_match (좌우 상성) — LGBM 이중 게이트 채택 (중앙값 +48.1, 3시드 일관)
    - base 47컬럼 + hand_match 1컬럼, interact/enc 제외
    """
    return Pipeline([
        ("add_hand", HandMatch()),
        ("to_cat", FunctionTransformer(_to_category)),
        ("clf", lgb.LGBMClassifier(**clf_kwargs)),
    ])


def load_data():
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
    r = y_true.mean()
    brier = ((proba - y_true) ** 2).mean()
    score = max(0.0, 100000.0 * (1.0 - brier / (r * (1.0 - r))))
    return score, r, brier


def load_anchor(anchor_arg):
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
        print(f"[FATAL] 앵커 파싱 실패: {raw!r}", file=sys.stderr)
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
    tmp_paths = []
    try:
        if isinstance(model, str):
            tmp_pkl = model
        else:
            fd, tmp_pkl = tempfile.mkstemp(suffix=".pkl", prefix="lgbm_qa_")
            os.close(fd)
            tmp_paths.append(tmp_pkl)
            joblib.dump(model, tmp_pkl, compress=3)
        fd, tmp_csv = tempfile.mkstemp(suffix=".csv", prefix="lgbm_qa_")
        os.close(fd)
        tmp_paths.append(tmp_csv)
        row_df.to_csv(tmp_csv, index=False)
        print(f"[QA:{label}] cross-process pickle 왕복 시작")
        _roundtrip_subprocess(tmp_pkl, tmp_csv)
    finally:
        _cleanup(tmp_paths)


def fit_with_early_stop(X, y, seed):
    """전처리(add_hand+to_cat)를 먼저 적용 → train_test_split (카테고리 집합 공유) →
    LGBM early stopping fit."""
    pipe = build_pipeline(**LGBM_KWARGS, random_state=seed)
    X_pre = pipe[:-1].fit_transform(X)
    Xt, Xv, yt, yv = train_test_split(X_pre, y, test_size=0.1, random_state=seed)
    t0 = time.time()
    pipe.fit(Xt, yt, clf__eval_set=[(Xv, yv)],
             clf__eval_metric="binary_logloss")
    ft = time.time() - t0
    n_est = pipe.named_steps["clf"].best_iteration_ or pipe.named_steps["clf"].n_estimators
    return pipe, ft, n_est


def run_validate(train, features, anchor_arg, seeds):
    anchor = load_anchor(anchor_arg)
    is_val = train["season"] == 2024
    X_tr, y_tr = train.loc[~is_val, features], train.loc[~is_val, TARGET]
    X_val, y_val = train.loc[is_val, features], train.loc[is_val, TARGET]
    print("train(2019~2023):", len(X_tr), "| val(2024):", len(X_val))

    # ---- N1 QA (소형) ----
    qa_df = train.head(QA_ROWS)
    qa_pipe = build_pipeline(n_estimators=QA_N_ESTIMATORS, verbosity=-1, random_state=42)
    t = time.time()
    qa_pipe.fit(qa_df[features], qa_df[TARGET])
    print(f"QA 소형 fit({QA_ROWS}행, n_estimators={QA_N_ESTIMATORS}) :: {time.time()-t:.1f}s")
    cross_process_pickle_qa(qa_pipe, qa_df[features].head(1), "qa")

    # ---- 다중 시드 검증 ----
    results = []
    for seed in seeds:
        pipe, ft, n_est = fit_with_early_stop(X_tr, y_tr, seed)
        pred = pipe.predict_proba(X_val)[:, 1]
        bss, r, brier = compute_bss(y_val, pred)
        results.append(dict(seed=seed, n_est=n_est, brier=brier, bss=bss, fit=ft))
        print(f"seed={seed:>3}: n_est={n_est:>3} fit={ft:.1f}s | "
              f"Brier24={brier:.6f} BSS24={bss:.2f}")
    med_bss = float(np.median([r["bss"] for r in results]))
    print(f"\n중앙값 BSS24: {med_bss:.2f} (시드 {seeds})")

    ok = med_bss > anchor
    print(f"게이트: LGBM 중앙값 {med_bss:.2f} vs 앵커 {anchor:.2f}")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def run_full(train, features, seed):
    pipe, ft, n_est = fit_with_early_stop(train[features], train[TARGET], seed)
    print(f"전체 재학습(2019~2024) :: {ft:.1f}s | best_iteration: {n_est}")

    os.makedirs("./model", exist_ok=True)
    joblib.dump(pipe, MODEL_PATH, compress=3)
    print(f"저장 완료: {MODEL_PATH}")

    row = train[features].head(1)
    loaded = joblib.load(MODEL_PATH)
    p = loaded.predict_proba(row)
    if p.shape != (1, 2):
        print(f"[FATAL] 스모크 predict 실패: shape={p.shape}", file=sys.stderr)
        sys.exit(2)
    print(f"스모크 predict OK: {p[0].round(4).tolist()}")
    cross_process_pickle_qa(MODEL_PATH, row, "model")


def main():
    parser = argparse.ArgumentParser(description="LightGBM 학습/검증")
    parser.add_argument("--anchor", type=float, default=None)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--seeds", default="42,7,123")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train, features = load_data()
    seeds = [int(s) for s in args.seeds.split(",")]

    if args.full:
        run_full(train, features, args.seed)
    else:
        run_validate(train, features, args.anchor, seeds)


if __name__ == "__main__":
    main()
