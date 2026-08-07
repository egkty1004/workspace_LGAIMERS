# Wave B 데이터 기술통계 & 시계열 분석 리포트

> **목적**: train.csv 수치형 컬럼의 기술통계(평균/분산/최빈값/중앙값/왜도/첨도)와
> 시즌별 시계열 추세·동조를 종합 분석한다. 분포 유형 판정으로 모델 입력 설계 근거를 확보한다.
> **검증일**: 2026-08-08 | **인터프리터**: aimers9 (sklearn 1.8.0, pandas 2.0.3)
> **산출 스크립트**: `eda_descriptive_ts.py` (베이스라인 실험용/)
> **로그**: `backup/eda_descriptive_ts.log`, `backup/desc_stats.csv`, `backup/ts_season_trend.csv`, `backup/ts_target_corr.csv`

---

## 1. 수치형 컬럼 기술통계 총람

47개 수치형 컬럼(season/ID/team 제외 43개 + target)의 mean/median/var/skew/kurt/mode 산출.
전체 표는 `backup/desc_stats.csv`. 핵심만 요약:

### 1.1 분포 유형 5그룹 분류

| 그룹 | 컬럼 | 근거 |
|---|---|---|
| **종 모양 (정규 근사)** | `asof_*_success_rate`, `prev{1,3,5}_game_success_rate`, `home/away_win_expectancy`, `score_diff_*` | skew≈0, median≈mean |
| **강한 우측 꼬리** (log정규/지수적) | `li` (skew 2.23, kurt 8.97), `run_{top,bot}_before` (skew 1.6), `asof_pitcher/batter_n` (skew 1.6), `asof_pitcher_middle_rate` (kurt 55.6), `asof_batter_middle_rate` (**kurt 102.4**) | 극단값 우측 집중 |
| **우측 꼬리** (약함) | `run_total_before`, `fastball/offspeed_rate`, `batter_success_rate`, `prev1_game_middle_rate` | skew 0.5~1.4 |
| **저차원 이산** | `runner_on_{1,2,3}b`, `num_runners_on`, `balls/strikes/outs_before` | mode=0이 34~89% |
| **이산 순서형** | `inning` (1~13), `game_month` (3~10), `game_dayofweek` (0~6) | 균등 분포 (kurt<0) |

### 1.2 극단 분포 주의 컬럼

| 컬럼 | skew | kurt | 의미 |
|---|---|---|---|
| `asof_batter_middle_rate` | 3.23 | **102.42** | 대부분 0.14 근처, 소수 극단 — 뾰족+꼬리 |
| `asof_pitcher_middle_rate` | 2.23 | **55.58** | 동일 |
| `asof_pitcher_ball_rate` | 1.90 | 30.24 | |
| `asof_pitcher_strike_rate` | -0.14 | 30.29 | 좌우 비대칭 없는 뾰족 |
| `li` | 2.23 | 8.97 | 전형적 우측 꼬리 |
| `runner_on_3b` | 2.47 | 4.08 | 88.8%가 0 |

### 1.3 최빈값(mode) 요약

- `balls_before` mode=0 (44.2%), `strikes_before` mode=0 (41.2%) — "0-0 카운트"가 최빈
- `runner_on_3b` mode=0 (88.8%) — 3루 주자는 드묾
- `li` mode=0.87 (3.0%만) — 최빈값이 아닌 연속형
- rate 계열 mode≈0.5 (종 모양 정점)

### 1.4 모델 입력 관점

- **강한 우측 꼬리 컬럼(li, run_*, asof_n)은 HGB에 자연 적합** — 트리 기반은 스케일링 불필요.
- **kurt 100+ 컬럼은 이상치 제거 대상이 아님** (Oracle 검증 §2.2와 일치) — 그대로 두는 게 정답.
- `asof_pitcher_n` = `asof_pitcher_pitchmix_n` **완전 중복** (mean/var/std 동일) — 사용 시 1개만.

---

## 2. 시계열 분석 — 시즌별 추세

### 2.1 단조 증가 (mono_frac=1.0, 2019→2024)

| 컬럼 | 2019 | 2024 | Δ | 상대 변화 |
|---|---|---|---|---|
| `asof_batter_n` | 720.8 | 5,402.5 | +4,681.8 | **+651%** |
| `asof_pitcher_n` / `pitchmix_n` | 768.7 | 3,928.2 | +3,159.5 | **+411%** |
| `prev{1,3,5}_game_middle_rate` | 0.127 | 0.172~0.176 | +0.046~0.048 | **+35%** 🔴 |
| `asof_pitcher_reverse_rate` | 0.182 | 0.237 | +0.055 | +30% 🔴 |
| `asof_pitcher_middle_rate` | 0.125 | 0.156 | +0.031 | +25% 🔴 |
| `asof_batter_middle_rate` | 0.124 | 0.152 | +0.028 | +23% 🔴 |

### 2.2 단조 감소 (target과 동반 하락)

| 컬럼 | 2019 | 2024 | Δ | 상대 변화 |
|---|---|---|---|---|
| **target** | 0.5647 | 0.4861 | -0.0786 | **-14%** |
| `prev1_game_success_rate` | 0.5613 | 0.4884 | -0.0728 | -13% |
| `prev3_game_success_rate` | 0.5592 | 0.4895 | -0.0697 | -12.5% |
| `prev5_game_success_rate` | 0.5587 | 0.4909 | -0.0679 | -12.2% |
| `asof_pitcher_success_rate` | 0.5598 | 0.5114 | -0.0484 | -8.7% |
| `asof_batter_success_rate` | 0.5615 | 0.5175 | -0.0441 | -7.9% |

### 2.3 🔴 핵심 발견 — 피처가 target 드리프트와 완전 동조

시즌별 평균 기준 target 평균과의 상관 (corr of season-means):

| 컬럼 | corr |
|---|---|
| `prev1_game_success_rate` | **+0.9966** |
| `prev3_game_success_rate` | +0.9912 |
| `prev5_game_success_rate` | +0.9876 |
| `asof_batter_success_rate` | +0.9862 |
| `asof_pitcher_success_rate` | +0.9689 |
| `asof_pitcher_middle_rate` | **-0.9667** |
| `season` | -0.9540 |
| `asof_batter_n` | -0.9483 |
| `asof_pitcher_n` | -0.9299 |

**해석**: 시즌 드리프트는 무작위가 아니라 **리그 자체의 변화**다.
middle 코스 투구 비율이 6년간 +35% 늘었고, 그 결과 제구 성공률이 -14% 하락.
**asof 피처들이 이 드리프트를 이미 내장**하고 있다 → 모델이 season을 피처로 안 써도
`prev*_success_rate`를 통해 드리프트를 간접 학습한다.

**모델링 함의**: 2025(test) 예측 시 이 추세가 이어질 경우, 2019~2023 학습 모델은
2024에서도 드리프트를 뒤쫓는 구조. 게이트(2024 홀드아웃)가 드리프트 일반화의 현실적 검증이다.

### 2.4 월별 패턴 (시즌 내 주기성)

| game_month | target 평균 | 행 비율 |
|---|---|---|
| 3 | 0.5375 | 1.8% |
| 4 | 0.5260 | 14.0% |
| 5 | 0.5323 | 17.3% |
| 6 | 0.5270 | 17.2% |
| 7 | 0.5211 | 12.0% |
| 8 | 0.5190 | 15.0% |
| 9 | 0.5206 | 15.5% |
| 10 | 0.5092 | 7.3% |

- **시즌 진행에 따라 target 하락**: 3월 0.538 → 9월 0.521 → 10월 0.509.
- 2024만 보면 3월 0.503 → 9월 0.476, 10월 0.505 (포스트시즌 반등).
- 10월 반등은 `game_type=F`(포스트시즌, +0.0795) 효과와 겹침.

---

## 3. 결론

1. **분포**: 수치형은 5그룹(종모양/강꼬리/약꼬리/이산/순서형) — HGB에 모두 자연 적합, 전처리 불필요.
2. **드리프트의 본질**: 리그 변화(middle_rate +35%)가 원인이며, asof 피처가 이미 동조(+0.99).
3. **2025 예측 리스크**: 추세 지속 여부가 관건. 게이트(2024 홀드아웃)가 드리프트 일반화 검증의 현실적 기준.
4. **중복 컬럼**: `asof_pitcher_n` = `asof_pitcher_pitchmix_n` — 1개만 사용 권장.
5. **월 주기성**: 시즌 후반 성공률 하락 + 10월 포스트시즌 반등 — `game_type=F` 신호와 연결.

---

*본 문서는 데이터 분석 전용 — 모델 성능(BSS) 판정은 게이트 검증에서 별도 수행.*
