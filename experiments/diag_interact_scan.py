"""diag_interact_scan.py — 상호작용 피처 후보 이론 상한 스캔 (2026-08-08)

GIHO §3-2 방법론: "주효과 0, 상호작용만 존재" 구조가 GBDT에 유리.
이론 상한 = dRes / (r(1-r)) * 1e5 (검증셋 셀 평균을 완벽히 알 때의 BSS 상한).

스캔 목적:
  1. 후보 조합별 이론 상한 계산 (2024 검증 — R-only와 primary 둘 다)
  2. 각 단일 변수 주효과 상한과 비교 → 상호작용 증분(interaction gain) 계산
  3. 셀당 최소 표본 >= 3000 필터 (GIHO §9 과적합 방지)

산출: 상한 높은 순 정렬. 상위 후보를 diag_ronly_protocol.py로 실측 검증.
"""
import itertools
import time

import numpy as np
import pandas as pd

DATA_DIR = "data"
TARGET = "control_success"

# 스캔할 변수 정의: (이름, 변환 함수) — 원본 컬럼 또는 파생 구간
def _ident(s): return s


def scan():
    t0 = time.time()
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
    features = [c for c in test_cols if c != "row_id"]
    train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                        usecols=features + [TARGET])
    print(f"train: {train.shape} | 로드 {time.time()-t0:.1f}s", flush=True)

    # 파생 변수 (구간화된 연속형)
    train["inning_bin"] = pd.cut(train["inning"], bins=[0, 3, 6, 9, 99], labels=False)
    train["li_bin"] = pd.cut(train["li"], bins=[0, 0.5, 1.0, 1.5, 2.0, 99],
                             labels=False).fillna(0)
    train["score_diff_bin"] = pd.cut(train["score_diff_pitcher_team"],
                                     bins=[-99, -3, -1, 1, 3, 99], labels=False)
    train["cond_bin"] = pd.cut(train["asof_pitcher_success_rate"],
                               bins=[0, 0.45, 0.475, 0.50, 0.525, 0.55, 1.0],
                               labels=False).fillna(0)
    train["middle_bin"] = pd.cut(train["asof_pitcher_middle_rate"],
                                 bins=[0, 0.25, 0.30, 0.35, 0.40, 1.0],
                                 labels=False).fillna(0)
    train["month_bin"] = train["game_month"] % 4  # 3~11월을 분기로
    train["runner_cnt_bin"] = train["num_runners_on"]
    train["outs_bin"] = train["outs_before"]
    train["count_sum"] = (train["balls_before"] + train["strikes_before"]).astype(int)

    # 후보 변수 풀 (각각 단일 변수로도 상한 계산)
    cand_cols = [
        "inning_bin", "li_bin", "score_diff_bin", "cond_bin", "middle_bin",
        "month_bin", "runner_cnt_bin", "outs_bin", "count_sum",
        "top_bottom", "game_type", "balls_before", "strikes_before",
        "outs_before", "num_runners_on", "base_state",
        "pitcher_hand", "batter_hand", "inning",
    ]
    # 이미 사용 중 (GIHO 구성) — 스캔에서 제외하되 baseline으로 표기
    used_cols = {"pitcher_hand", "batter_hand", "balls_before", "strikes_before"}

    for vy, ronly in [(2024, False), (2024, True)]:
        tag = "primary(2024)" if not ronly else "r2024(R-only)"
        va = train[train["season"] == vy]
        if ronly:
            va = va[va["game_type"] == "R"]
        y = va[TARGET].values
        r = y.mean()
        n = len(va)
        print(f"\n{'='*88}\n[검증: {tag}] n={n:,} r={r:.4f}\n{'='*88}", flush=True)

        # 1) 단일 변수 상한
        single = {}
        for c in cand_cols:
            g = va.groupby(c)[TARGET].agg(["mean", "size"])
            w = g["size"] / n
            d = float((w * (g["mean"] - r) ** 2).sum())
            single[c] = d / (r * (1 - r)) * 1e5
        single_sorted = sorted(single.items(), key=lambda x: -x[1])

        # 2) 쌍 조합 상한 + 상호작용 증분
        print(f"{'조합':<34s} {'상한':>7s} {'주효과합':>8s} {'상호증분':>8s} {'최소셀':>6s}  판정")
        rows = []
        pairs = list(itertools.combinations(cand_cols, 2))
        for a, b in pairs:
            key = f"{a} x {b}"
            # 새로 만든 파생 변수 중복 조합 스킵 (정보 중복)
            g = va.groupby([a, b])[TARGET].agg(["mean", "size"])
            min_cell = int(g["size"].min())
            w = g["size"] / n
            d = float((w * (g["mean"] - r) ** 2).sum())
            upper = d / (r * (1 - r)) * 1e5
            main_sum = single[a] + single[b]
            gain = upper - main_sum
            # 판정: 상호작용 증분이 의미 있고(>10), 셀 표본 충분(>=3000)
            verdict = ""
            if gain > 10 and min_cell >= 3000:
                verdict = "CANDIDATE"
            elif min_cell < 3000:
                verdict = "cells<3000"
            rows.append((key, upper, main_sum, gain, min_cell, verdict))

        rows.sort(key=lambda x: -x[3])  # 상호작용 증분 내림차순
        shown = 0
        for key, upper, main_sum, gain, min_cell, verdict in rows:
            if verdict == "CANDIDATE" and shown < 20:
                print(f"{key:<34s} {upper:7.1f} {main_sum:8.1f} {gain:8.1f} "
                      f"{min_cell:6,d}  {verdict}", flush=True)
                shown += 1
        if shown == 0:
            print("  (조건 충족 후보 없음)")
        print(f"  총 {len(rows)} 조합 스캔 | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    scan()
