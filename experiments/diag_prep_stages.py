"""diag_prep_stages.py — Wave B 전처리 단계별 분리 검증 (진단 전용, 제출물 아님).

Wave B 게이트 FAIL(2024 BSS 0.00)의 원인을 pinpoint 하기 위해,
전처리 단계(enc_ids / add_missing / add_interact)를 조합별로 on/off 하며
Wave A와 동일한 2024 홀드아웃 게이트 프로토콜로 BSS를 추적한다.

- best_params overlay 없음 (기본 HGB_KWARGS — Wave A와 동일 조건)
- QA(소형 fit + pickle 왕복)는 진단과 무관하므로 생략 → 실행 시간 단축
- 각 조합: 2019~2023 학습 → 2024 BSS, 2023 in-sample BSS, n_iter_ 기록
- 파이프라인 transform 후 train/val 컬럼명·순서 일치 어서션 (정렬 버그 탐지)

실행 (CWD = 베이스라인 실험용/):
    python diag_prep_stages.py
"""
import time

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from bss_preprocess import _to_category, IDTargetEncoder, MissingIndicatorAdder, InteractionAdder
from train_hgb import HGB_KWARGS, compute_bss, load_data


def make_pipeline(stages):
    """전처리 단계 조합('enc','missing','interact' 문자열 집합)으로 파이프라인 구성."""
    steps = []
    if "enc" in stages:
        steps.append(("enc_ids", IDTargetEncoder()))
    if "missing" in stages:
        steps.append(("add_missing", MissingIndicatorAdder()))
    if "interact" in stages:
        steps.append(("add_interact", InteractionAdder()))
    steps.append(("to_cat", FunctionTransformer(_to_category)))
    steps.append(("clf", HistGradientBoostingClassifier(**HGB_KWARGS)))
    return Pipeline(steps)


def stage_label(stages):
    return "base" if not stages else "+".join(sorted(stages))


VARIANTS = [
    ("base", set()),
    ("enc", {"enc"}),
    ("missing", {"missing"}),
    ("interact", {"interact"}),
    ("enc+missing", {"enc", "missing"}),
    ("enc+interact", {"enc", "interact"}),
    ("missing+interact", {"missing", "interact"}),
    ("all", {"enc", "missing", "interact"}),
]


def check_alignment(model, X_train, X_val):
    """fit 완료된 파이프라인의 전처리 transform 출력이 train/val 동일 컬럼인지 검증."""
    pre = model[:-1]
    tr = pre.transform(X_train)
    va = pre.transform(X_val)
    assert list(tr.columns) == list(va.columns), (
        f"컬럼 불일치!\ntrain: {list(tr.columns)}\nval  : {list(va.columns)}"
    )
    assert not tr.columns.duplicated().any(), "중복 컬럼 존재"
    return tr, va


def main():
    train, features = load_data()
    is_val = train["season"] == 2024
    is_2023 = train["season"] == 2023
    X_train, y_train = train.loc[~is_val, features], train.loc[~is_val, "control_success"]
    X_val, y_val = train.loc[is_val, features], train.loc[is_val, "control_success"]
    X_2023, y_2023 = train.loc[is_2023, features], train.loc[is_2023, "control_success"]
    print("train(2019~2023):", len(X_train), "| val(2024):", len(X_val))
    print("=" * 100)

    results = []
    for name, stages in VARIANTS:
        model = make_pipeline(stages)
        t0 = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - t0

        # 전처리 출력 정렬 검증 (fit 후)
        tr, va = check_alignment(model, X_train, X_val)
        n_cols = tr.shape[1]

        pred_val = model.predict_proba(X_val)[:, 1]
        bss24, r24, brier24 = compute_bss(y_val, pred_val)
        pred_23 = model.predict_proba(X_2023)[:, 1]
        bss23, r23, brier23 = compute_bss(y_2023, pred_23)
        n_iter = model.named_steps["clf"].n_iter_

        results.append(
            dict(name=name, n_cols=n_cols, n_iter=n_iter, fit_time=fit_time,
                 brier24=brier24, bss24=bss24, bss23=bss23)
        )
        print(f"[{name:>15}] cols={n_cols:2d} n_iter={n_iter:4d} "
              f"fit={fit_time:5.1f}s | Brier24={brier24:.6f} "
              f"BSS24={bss24:8.2f} | BSS23(in)= {bss23:8.2f}")

    # 요약 테이블
    print("=" * 100)
    print("요약 (기본 HGB_KWARGS, best_params 미적용):")
    print(f"{'조합':>15} | {'cols':>4} | {'n_iter':>5} | {'Brier24':>8} | "
          f"{'BSS24':>8} | {'BSS23(in)':>9} | {'fit_s':>6}")
    for r in results:
        print(f"{r['name']:>15} | {r['n_cols']:>4} | {r['n_iter']:>5} | "
              f"{r['brier24']:>8.6f} | {r['bss24']:>8.2f} | {r['bss23']:>9.2f} | {r['fit_time']:>6.1f}")

    pd.DataFrame(results).to_csv("backup/diag_prep_stages.csv", index=False)
    print("저장: backup/diag_prep_stages.csv")


if __name__ == "__main__":
    main()
