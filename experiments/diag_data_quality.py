"""diag_data_quality.py — 학습 데이터 품질 실측 진단 (단일 실행, 1회 로드).

DACON Aimers9 투구 제구 성공 확률 예측 — train.csv(1,475,092×49) 품질 진단.
목적: REPORT_data_quality.md 작성을 위한 실측 수치 확보.

섹션:
  A. 결측치: 컬럼별 결측률 / 패턴 그룹 / 결측↔target 연관 / 시즌별 추이
  B. 이상치: 수치형 극단 분포 / 극단 행 target 영향 / prev1결측×asof_n 복귀 탐지
  C. 전처리: 메모리(다운캐스팅) / 범주 카디널리티 / 스케일링 불필요 근거 / fallback 전략
  D. 카디널리티·cold-start: ID 고유값 / season별 신인 비율 / asof_pitcher_n 분포
  E. 신규 EDA 시그널: prev1-asof 최신성 / 복귀(공백) 투수 / 중복행 / 매치업 카디널리티

실행: conda 활성화 후 `python diag_data_quality.py` (train 1회 로드, ~10초)
"""
import time
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

TARGET = "control_success"
DATA = "/home/gpu_01/workspace_LGAIMERS/데이터/open/data/train.csv"

# ---------- 컬럼 정의 ----------
INT_COLS = [
    "season", "game_month", "game_dayofweek", "inning",
    "balls_before", "strikes_before", "outs_before",
    "run_top_before", "run_bot_before", "run_total_before",
    "score_diff_home", "score_diff_pitcher_team",
    "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on",
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n",
]
FLOAT_COLS = [
    "home_win_expectancy", "away_win_expectancy", "li",
    "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate", "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]
CAT_COLS = ["top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand"]

DTYPES = {c: "int32" for c in INT_COLS}
DTYPES.update({c: "float32" for c in FLOAT_COLS})
DTYPES.update({c: "category" for c in CAT_COLS})
DTYPES[TARGET] = "int8"
USECOLS = INT_COLS + FLOAT_COLS + CAT_COLS + [TARGET]

t0 = time.time()
df = pd.read_csv(DATA, usecols=USECOLS, dtype=DTYPES, encoding="utf-8-sig")
N = len(df)
print(f"[load] {df.shape} | {time.time()-t0:.1f}s | 메모리 {df.memory_usage(deep=True).sum()/2**20:.0f}MB")


def sec(t):
    print("\n" + "=" * 88)
    print(t)
    print("=" * 88)


# ============================================================
sec("A. 결측치 실측 진단")
# ============================================================
miss = df.isna().sum()
miss = miss[miss > 0].sort_values(ascending=False)
print(f"[A1] 결측 있는 컬럼 {len(miss)}개 / {df.shape[1]}")
for c, n in miss.items():
    print(f"  {c:<42} {n:>9,} ({n/N*100:.3f}%)")

# A2 패턴 그룹: 결측 벡터 군집
sec("[A2] 결측 패턴 그룹")
miss_cols = list(miss.index)
pat = df[miss_cols].isna().agg(tuple, axis=1)
pc = pat.value_counts()
for p, n in pc.head(10).items():
    cols = [c for c, m in zip(miss_cols, p) if m]
    print(f"  n={n:>8,} ({n/N*100:.3f}%) | 결측: {', '.join(cols[:6])}{'...' if len(cols)>6 else ''}")

# A3 결측 ↔ target
sec("[A3] 결측 ↔ target 연관 (missing vs present mean 차이)")
base = df[TARGET].mean()
print(f"  baseline target mean: {base:.4f}")
for c in miss_cols:
    m = df[c].isna()
    if m.sum() < 50:
        continue
    ym, yp = df.loc[m, TARGET].mean(), df.loc[~m, TARGET].mean()
    print(f"  {c:<42} miss={ym:.4f} present={yp:.4f} Δ={ym-yp:+.4f} | miss n={int(m.sum()):>7,}")

# A4 시즌별 결측률 (대표: prev_game 6종 동일 / asof_n)
sec("[A4] 시즌별 결측률 (cold-start 추이)")
g = df.groupby("season", observed=True)
for c in ["asof_pitcher_n", "asof_batter_n", "asof_pitcher_prev1_game_success_rate", "asof_pitcher_pitchmix_n"]:
    r = g[c].apply(lambda s: s.isna().mean())
    print(f"  {c:<40} " + " ".join(f"{k}:{v*100:.2f}%" for k, v in r.items()))

# ============================================================
sec("B. 이상치 실측 진단")
# ============================================================
sec("[B1] 수치형 극단 분포 + target (p0/p1/p99/p100 & 극단 행 target)")
base = df[TARGET].mean()
num_cols = INT_COLS + FLOAT_COLS
for c in num_cols:
    s = df[c].dropna()
    if s.nunique() <= 3:
        continue
    p1, p99 = s.quantile(0.01), s.quantile(0.99)
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    hi = q3 + 3.0 * iqr
    n_hi = int((s > hi).sum())
    # 극단 행 target (p99 초과)
    mask = df[c].notna() & (df[c] > p99) if s.max() > p99 else None
    if mask is not None and mask.sum() >= 30:
        y_ext = df.loc[mask, TARGET].mean()
        tag = f"| p99초과({mask.sum():,}행) target={y_ext:.4f} (Δ{y_ext-base:+.4f})"
    else:
        tag = ""
    print(f"  {c:<40} min={s.min():>10.2f} p1={p1:>8.2f} p99={p99:>8.2f} max={s.max():>10.2f} IQR3x초과={n_hi:>7,}{tag}")

# B2 도메인 극단 행 target
sec("[B2] 도메인 극단 행 target 영향")
checks = [
    ("li", lambda s: s >= 5, "li>=5"),
    ("run_total_before", lambda s: s >= 20, "run_total>=20"),
    ("score_diff_pitcher_team", lambda s: s.abs() >= 15, "점수차|>=15"),
    ("asof_pitcher_success_rate", lambda s: (s == 0) | (s == 1), "asof_success_rate 0/1"),
    ("asof_pitcher_prev1_game_success_rate", lambda s: (s == 0) | (s == 1), "prev1_success_rate 0/1"),
    ("asof_pitcher_n", lambda s: (s == 0) | (s == 1), "asof_pitcher_n<=1"),
]
for c, cond, name in checks:
    m = df[c].notna() & cond(df[c])
    n = int(m.sum())
    if n == 0:
        print(f"  {name:<38} 0건")
        continue
    y = df.loc[m, TARGET].mean()
    # 소표본 여부 (n<=10 비중)
    small = "-"
    if "asof_pitcher_n" in c or c == "asof_pitcher_n":
        small = f"| n<=10 비중 {df.loc[m,'asof_pitcher_n'].le(10).mean()*100:.1f}%"
    print(f"  {name:<38} n={n:>8,} ({n/N*100:.2f}%) | target={y:.4f} (Δ{y-base:+.4f}) {small}")

# B3 prev1 결측 × asof_n 큼 = 복귀(공백) 투수, rookies와 분리
sec("[B3] prev1_game 결측의 이중성 — 신인 vs 복귀(공백)")
m_prev = df["asof_pitcher_prev1_game_success_rate"].isna()
m_rookie = df["asof_pitcher_n"].isna()  # 이력 자체 없음
m_gap = m_prev & ~m_rookie
print(f"  전체 prev1 결측: {m_prev.sum():,} ({m_prev.mean()*100:.3f}%)")
print(f"    ├─ 이력 자체 없음(asof_n 결측, 진짜 신인): {m_rookie.sum():,} | target={df.loc[m_rookie, TARGET].mean():.4f}")
print(f"    └─ 이력 있음 but prev1 결측 (복귀/공백):    {m_gap.sum():,} | target={df.loc[m_gap, TARGET].mean():.4f}")
# 복귀 그룹을 asof_n 크기로 세분
if m_gap.sum() >= 100:
    bins = [0, 10, 50, 100, 1000, 10**6]
    cut = pd.cut(df.loc[m_gap, "asof_pitcher_n"], bins=bins, right=False)
    tab = df.loc[m_gap].groupby(cut, observed=True)[TARGET].agg(["size", "mean"])
    print(tab.round(4).to_string())

# ============================================================
sec("C. 전처리 상태 점검")
# ============================================================
sec("[C1] 메모리 — 현재 로드 vs float64 전부 (다운캐스팅 효과)")
mem_now = df.memory_usage(deep=True).sum()
mem_f64 = N * (len(INT_COLS) + len(FLOAT_COLS) + 1) * 8  # target int8->float64 가정
mem_obj = N * 4 * 26  # category 5종: category 자체 압축, 근사치만
print(f"  현재(다운캐스트 적용): {mem_now/2**20:.0f} MB")
print(f"  전부 float64 가정:     {(mem_f64+mem_obj)/2**20:.0f} MB (근사)")
print(f"  절감: {(1 - mem_now/mem_f64)*100:.0f}%")
print("  → 10시드×다중 피처 학습 시 로드×n_splits 반복 → 다운캐스팅이 학습 속도/메모리 직접 개선")

sec("[C2] 범주형 카디널리티 (LGBM category / HGB 255 제한)")
for c in CAT_COLS:
    print(f"  {c:<18} nunique={df[c].nunique(dropna=True):>5,} | 결측 {df[c].isna().sum():>6,}")

sec("[C3] 스케일링 필요성 — 트리 모델 관점")
print(f"  트리(분할 기반)는 단조변환·스케일에 불변 → 스케일링 불필요 (LGBM/HGB 공통).")
print(f"  li/win_expectancy/rate가 상이한 스케일이어도 무해. 단, rate 특성상 수렴 분포(0.4~0.6).")

sec("[C4] 결측 fallback 전략 — 현재 방식 평가")
print("  현재: SimpleImputer 사용 안 함. LGBM NaN 네이티브 + category 결측은 범주로 처리.")
print(f"  결측 컬럼은 전부 수치형({len(miss_cols)}개) + asof 계열 → NaN을 그대로 split 신호로 사용 가능.")
print("  위험: prev1/3/5 결측(NaN)을 0.5나 -1로 대체하면 '복귀/신인 구분' 신호 소실 — 대체 금지.")

# ============================================================
sec("D. 카디널리티 · cold-start")
# ============================================================
sec("[D1] pitcher/batter ID")
for c in ["pitcher_id", "batter_id"]:
    s = df[c]
    print(f"  {c:<10} 고유값 {s.nunique():>6,} | 행당 등장 p50={s.value_counts().quantile(0.5):.0f} p99={s.value_counts().quantile(0.99):.0f}")

sec("[D2] season별 신인(데뷔 시즌) 비율")
debut = df.groupby("pitcher_id")["season"].min().rename("debut")
m = df.merge(debut, on="pitcher_id")
rook = m["season"] == m["debut"]
tab = m.groupby("season", observed=True)[TARGET].agg(["size", "mean"])
tab["rookie_rows"] = rook.groupby(m["season"], observed=True).sum()
tab["rookie_rows%"] = (rook.groupby(m["season"], observed=True).sum() / tab["size"] * 100).round(2)
tab["rookie_pitchers"] = debut.value_counts().sort_index()
print(tab.to_string())

sec("[D3] asof_pitcher_n 분포 (소표본 비중)")
s = df["asof_pitcher_n"]
q = s.quantile([0, 0.25, 0.5, 0.75, 0.9, 0.99]).to_dict()
print(f"  quantiles: " + ", ".join(f"p{k}={v:,.0f}" for k, v in q.items()))
print(f"  결측(cold-start): {s.isna().sum():,} ({s.isna().mean()*100:.2f}%)")
bins = [0, 1, 10, 30, 100, 300, 10**6]
cut = pd.cut(s, bins=bins, right=False)
print("  소표본 비중:")
print(df.groupby(cut, observed=True)[TARGET].agg(["size", "mean"]).assign(
    pct=lambda t: (t["size"] / N * 100).round(2)).round(4).to_string())
print(f"  n==0 (0구 방문이력): {(s==0).sum():,} ({(s==0).mean()*100:.2f}%) | target={df.loc[s==0, TARGET].mean():.4f}")

sec("[D4] 2025 샘플 대비 cold-start 리스크 (test 5행만)")
print("  test 샘플: TEST_005332 pitcher asof_n=0 (데뷔), TEST_000017 asof_n=86지만 prev1~5 전부 결측(복귀).")
print("  → 2025도 신인+복귀 혼재. 학습에 B3의 이중성 분리 신호가 그대로 필요.")

# ============================================================
sec("E. 신규 EDA 시그널 (기존 EDA와 비중복)")
# ============================================================
sec("[E1] 최신성: prev1(직전경기) vs asof(누적) — target 상관 비교")
mok = df["asof_pitcher_prev1_game_success_rate"].notna()
for c in ["asof_pitcher_success_rate", "asof_pitcher_prev1_game_success_rate",
          "asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev5_game_success_rate",
          "asof_pitcher_middle_rate"]:
    r = df.loc[mok, c].corr(df.loc[mok, TARGET])
    print(f"  {c:<42} corr={r:+.4f}")
# gap = prev1 - asof
gap = df.loc[mok, "asof_pitcher_prev1_game_success_rate"] - df.loc[mok, "asof_pitcher_success_rate"]
print(f"  gap(prev1-asof): p1={gap.quantile(0.01):+.3f} p50={gap.quantile(0.5):+.3f} p99={gap.quantile(0.99):+.3f} | corr(target)={gap.corr(df.loc[mok, TARGET]):+.4f}")
q_gap = gap.quantile([0.25, 0.5, 0.75])
for lab, g in [("하위25%", gap <= q_gap[0.25]), ("상위25%", gap >= q_gap[0.75])]:
    print(f"  {lab}: n={g.sum():>8,} | target={df.loc[mok, TARGET].loc[g].mean():.4f}")

sec("[E2] 중복행 / ID 무결성")
print(f"  전체 행: {N:,} | 완전 중복 행: {df.drop_duplicates().shape[0]:,} → 중복 {N - df.drop_duplicates().shape[0]:,}")
print(f"  row_id 대신 컬럼 조합으로 판단 (row_id 미로드). pitcher_team_id 고유값: {df['pitcher_team_id'].nunique()}, batter_team_id: {df['batter_team_id'].nunique()}")

sec("[E3] 매치업 카디널리티 — 범주 처리 관점")
print(f"  (pitcher_team_id, batter_team_id) 조합: {df.groupby(['pitcher_team_id','batter_team_id'], observed=True).ngroups:,}")
print(f"  (pitcher_hand, batter_hand) 조합(platoon 4셀): {df.groupby(['pitcher_hand','batter_hand'], observed=True).ngroups}")

sec("[E4] 컬럼 간 중복 정보 (fallback 시그널 후보)")
# home/away win_expectancy 합
he = df["home_win_expectancy"] + df["away_win_expectancy"]
print(f"  home_win_exp + away_win_exp: min={he.min():.1f} max={he.max():.1f} | 100 아닌 행 {(he != 100).sum():,} ({(he!=100).mean()*100:.2f}%)")
print(f"  asof_pitcher_fastball+breaking+offspeed 합: min={ (df['asof_pitcher_fastball_rate']+df['asof_pitcher_breaking_rate']+df['asof_pitcher_offspeed_rate']).dropna().min():.3f} "
      f"max={ (df['asof_pitcher_fastball_rate']+df['asof_pitcher_breaking_rate']+df['asof_pitcher_offspeed_rate']).dropna().max():.3f}")

print(f"\n[done] {time.time()-t0:.1f}s")
