# Wave B 종합 EDA 리포트 — 데이터 프로파일 + FAIL 원인 진단 + 튜닝 사후분석

> **목적**: (1) 전체 데이터(train/test/trackman/sample_submission)의 특성·분포·결측·이상치 프로파일,
> (2) Wave B 게이트 FAIL(2024 BSS 0.00)의 원인 pinpoint,
> (3) 하이퍼파라미터 튜닝 파손 사후분석.
> **검증일**: 2026-08-08 | **인터프리터**: aimers9 (sklearn 1.8.0, pandas 2.0.3, numpy 1.26.4)
> **산출 스크립트**: `eda_full_profile.py` / `eda_fail_diag.py` / `diag_prep_stages.py` / `diag_stage_ablation.py` / `diag_enc_split.py` (베이스라인 실험용/)
> **로그**: `backup/eda_full_profile.log` / `backup/eda_fail_diag.log` / `backup/profile_{train,trackman}.csv` / `backup/eda_{pitcher,batter}_enc_drift.csv` / `backup/eda_missing_signal.csv` / `backup/diag_stage_ablation.log` / `backup/diag_enc_split.log` / `backup/diag_debut_ablation.log`
> **방법론 검토**: Oracle 에이전트 재검증 완료 (2026-08-08) — §2.1/§2.2/§5.2/§5.3 반영. 결론 방향 유지, 근거 3건 정정 (이진 IQR 아티팩트, rate 0/1 컬럼별 해석, 결측=데뷔 경기).
> **FAIL 원인 확정**: 단계별 ablation (2026-08-08) — IDTargetEncoder(ID 인코딩) 전체가 유일한 FAIL 범인 (pitcher_enc/batter_enc 단독 각각 0.00). missing 지시자 8종 무죄(439.00 유지), interact 3종 +55.36 개선(494.36). 기존 "지시자 6종 + batter_enc" 결론 정정.

---

## 1. 데이터 구성

| 파일 | 행 수 | 컬럼 | 크기 | 용도 |
|---|---|---|---|---|
| `train.csv` | 1,475,092 | 49 (47피처 + row_id + control_success) | 368MB | 학습·검증 (2019~2024) |
| `test.csv` | 5 (형식 샘플) | 48 | 1.9KB | 형식 확인 전용, **season=2025** |
| `trackman_history.csv` | 1,793,078 | 30 | 353MB | 과거 로그 (활용 불가 확정 — §6) |
| `sample_submission.csv` | 5 | 2 | 112B | 제출 양식 (row_id, control_success) |

---

## 2. train.csv 컬럼별 프로파일 요약

### 2.1 결측 (핵심: 결측은 asof_* 16개에만 존재)

| 그룹 | 컬럼 | 결측 행 | 비율 |
|---|---|---|---|
| G1: prev_game 이력 | `asof_pitcher_prev{1,3,5}_game_{success,middle}_rate` (6종) | 29,185 | 1.98% |
| G2: 투수 cold-start | `asof_pitcher_*_rate` (8종) | 792 | 0.05% (792명 × 1행) |
| G3: 타자 cold-start | `asof_batter_*_rate` (2종) | 830 | 0.06% (830명 × 1행) |

- **그 외 31개 컬럼은 결측 0건** — 결측은 전적으로 "이력 부족"에서만 발생.
- **결측 메커니즘 (Oracle 재검증 2026-08-08)**: G1 결측 29,185행 = 792개의 (시즌×투수) 블록(블록 중앙값 23투구, 최대 151) = **각 투수의 데이터셋 내 첫 경기 전체 투구**. G2(792행, 전부 `asof_pitcher_n=0`) = **각 투수의 첫 투구 행**, G2⊂G1(765+27=792)은 "첫 경기 첫 투구"의 자연스러운 교집합.
- ⚠️ **"시즌 첫 등장"은 오해**: 2020년 이후 G1 결측 행 중 해당 투수가 이전 시즌 데이터를 가진 경우 **0건/16,907행** → 복귀 투수의 시즌 개막전은 결측이 아님. 즉 결측 = **데이터셋 데뷔(첫 경기)**. "792=고유 투수 수" 동치는 이 구조에서 자명한 동어반복.
- 결측은 게임 이력의 **결정적 함수(MAR에 가까움)** — "6개 결측 프로필"도 실은 3개 메커니즘(G1 데뷔 경기 / G2 첫 투구 / G3 타자 데뷔)의 결정적 교집합.
- test(2025) 샘플 5행에서도 동일 NaN 패턴 확인 → 2025에도 데뷔 경기 결측 예상.

### 2.2 이상치 (IQR 기준)

| 컬럼군 | IQR 이상치 | 판정 |
|---|---|---|
| `score_diff_home` (±1.5×IQR 밖 138,418건) | 많음 | ✅ **자연 극단값** — 점수 차는 야구에서 정상 범위 (-27~+27). 제거 금지 |
| `run_total_before` (29,750건) | 많음 | ✅ 자연 극단값 (최대 37점) |
| `li` (67,303건, 최대 10.83) | 많음 | ✅ 연속형 중요도 지표의 오른쪽 꼬리 — 제거 금지 |
| `asof_pitcher_*_rate` (rate 0 또는 1) | 컬럼별 상이 | ⚠️ **컬럼별 해석 필요 — 일괄 "소표본 노이즈"는 오류** (아래 상세) |
| `pitcher_id` low (959건) | 있음 | ✅ ID가 밀집 구간 밖일 뿐 — 의미 없음 |
| `runner_on_{1,2,3}b`, `num_runners_on` | **분석 제외** | ⚠️ **이진/0-3 저차원 변수에 IQR은 산술 아티팩트** (아래 상세) |

**ⓐ 이진·저차원 변수의 IQR은 산술 아티팩트 (Oracle 재검증):**
- `runner_on_2b`: 78%가 값 0 → Q1=Q3=0, IQR=0 → fence 상한=0 → **값 1인 행 전부(319,768건=21.68%, = P(1)과 정확히 일치)**가 "이상치"로 플래그됨. `runner_on_3b`(164,662건) 동일.
- `num_runners_on`: fence=2.5 → **만루(값 3) 전부 48,408건**이 플래그됨 — `base_state=123` 행 수와 정확히 일치.
- → 이들은 **변수 분포의 수학적 결과이지 이상치 발견이 아님**. 보고서에서 이진/0-3 변수의 "이상치 비율" 수치는 **의미 없음** (판정 "자연 극단"이 맞은 건 우연).

**ⓑ rate 0/1 극단값 — 컬럼별로 의미가 다름 (Oracle 재검증):**

| 컬럼군 | 0/1 극단값 중 표본≤10 비율 | 판정 |
|---|---|---|
| 커리어 `success`/`ball`/`strike_rate` | **99.6%** | ✅ 소표본 노이즈 (원래 주장 성립) |
| `prev{1,3,5}_game_*_rate` | **0.4% / 0.13%** | ❌ **구조적 0/1** — 1경기 윈도우 투구수가 실제 표본이라 커리어 대표본이어도 0/1 자연 발생 |
| `offspeed_rate` | 85%가 대표본 | ❌ **행동 신호** — 구종 미사용/전량 사용은 노이즈가 아니라 정보 |
| `reverse_rate` | 34%가 대표본 | ❌ 위와 동일 |

→ "rate 0/1 극단값의 99.6%가 소표본" 주장은 **커리어 success/ball/strike 컬럼에만** 성립. 원인: 초기 분석이 prev_game의 표본수로 커리어 `asof_pitcher_n`을 쓴 도구 오류 (prev_game의 실제 표본 = 직전 경기 윈도우 투구수). `offspeed_rate=0`(구종 미사용)과 같은 **대표본 0/1은 행동 신호**로 재해석해야 함.

**결론: 이상치는 전부 스포츠 데이터의 자연 극단값 — 제거/클리핑 금지.** 이상치 처리는 "없다"가 정답. 단, 이진/저차원 변수는 IQR 판정에서 제외하고, rate 0/1은 컬럼별(커리어/윈도우/행동)로 해석한다.

### 2.3 분포 요약 (수치형 핵심)

| 컬럼 | min | p1 | median | p99 | max | mean | std |
|---|---|---|---|---|---|---|---|
| `li` | 0 | 0 | 0.80 | 4.37 | 10.83 | 0.98 | 0.89 |
| `home_win_expectancy` | 0 | 0.1 | 52.2 | 99.9 | 100 | 50.27 | 30.27 |
| `asof_pitcher_success_rate` | 0 | 0.395 | 0.532 | 0.704 | 1 | 0.535 | 0.062 |
| `asof_pitcher_reverse_rate` | 0 | 0.065 | 0.218 | 0.363 | 1 | 0.215 | 0.061 |
| `asof_pitcher_n` | 0 | 18 | 1,776 | 11,870 | 15,450 | 2,661 | 2,690 |
| `asof_batter_n` | 0 | 18 | 2,263 | 11,830 | 13,928 | 3,271 | 3,049 |

- `asof_pitcher_n` = `asof_pitcher_pitchmix_n` **완전 동일** (고유값 수 15,450 동일) — 중복 컬럼.
- `run_total_before` = `run_top_before + run_bot_before` 관계 확인 (파생 가능).

### 2.4 target(control_success)과의 관계

**수치형 spearman 상관 (절대값 상위):**

| 컬럼 | spearman |
|---|---|
| `asof_pitcher_success_rate` | **+0.0845** (최강) |
| `asof_pitcher_reverse_rate` | **-0.0791** |
| `asof_pitcher_prev5_game_success_rate` | +0.0784 |
| `asof_pitcher_prev3_game_success_rate` | +0.0747 |
| `asof_pitcher_prev1_game_success_rate` | +0.0630 |
| `asof_batter_success_rate` | +0.0529 |
| `asof_batter_n` | -0.0405 |
| `asof_batter_middle_rate` | -0.0389 |
| `asof_pitcher_middle_rate` | -0.0388 |
| `asof_pitcher_n` / `pitchmix_n` | -0.0157 |
| `season` | -0.0483 |

- **최근 컨디션(asof_pitcher_success_rate)이 가장 강한 안정 신호** (시즌 드리프트에도 견고 — §4.6).
- 역률(reverse_rate)은 음의 방향으로 강한 신호.
- `season` 음의 상관 = 시즌 드리프트와 일치.

**범주형 target 편차 (흥미 발견):**

| 컬럼 | 범주 | target 평균 | 편차 | 행 수 |
|---|---|---|---|---|
| `game_type` | **F (포스트시즌)** | 0.6033 | **+0.0795** | 161,004 |
| `game_type` | R (정규) | 0.5140 | -0.0097 | 1,314,088 |
| `base_state` | `__3` (2사 3루) | 0.5361 | +0.0123 | 35,513 |
| `base_state` | `123` (만루) | 0.5161 | -0.0076 | 48,408 |
| `top_bottom` | T/B | 0.5231/0.5244 | ±0.001 | — |

- **`game_type=F`는 +0.0795로 범주형 중 최강 신호** — 포스트시즌 투구가 제구 성공률 유의하게 높음. 모델이 활용해야 할 신호.
- base_state는 ±0.012 수준의 약신호 — 범주형으로 충분히 처리 가능.

---

## 3. 🔴 FAIL 원인 진단 (Wave B 게이트 2024 BSS 0.00)

### 3.1 게이트 결과 재확인 (동일 프로토콜 비교)

| 조건 | 컬럼 | n_iter_ | Brier(2024) | 2024 BSS |
|---|---|---|---|---|
| Wave A (47컬럼, 전처리 X) | 47 | 247 (early stop) | 0.2487 | **439.00** |
| Wave B (60컬럼, l2=1.0 overlay) | 60 | 999 | 0.2719 | 0.00 |
| Wave B (60컬럼, 기본 파라미터) | 60 | 1000 | 0.2740 | 0.00 |

- **핵심 단서**: Brier 0.274 > 기준선 r(1-r)=0.2498 → **상수 예측보다 나쁨**. 단순 과적합이 아니라 "피처가 2024에서 능동적으로 틀린 방향 예측"을 유도.

### 3.2 재검증: prev_game missing 지시자 — **무죄 판정** (2026-08-08 단계별 ablation)

**배경**: 기존에 "지시자 신호 역전(+0.027→-0.024)이 FAIL 원인"으로 의심했으나, **단계별 분리 검증으로 무죄 확정**.

`asof_pitcher_prev1_game_success_rate` 결측 행의 target 평균 vs 비결측 (참고 데이터 — 역전 자체는 실재하지만 FAIL 원인이 아님):

| season | missing 행 target | present 행 target | **delta** |
|---|---|---|---|
| 2019 | 0.5869 | 0.5635 | +0.0234 |
| 2020 | 0.5542 | 0.5324 | +0.0218 |
| 2021 | 0.5563 | 0.5324 | +0.0239 |
| 2022 | 0.5553 | 0.5286 | +0.0266 |
| 2023 | 0.4897 | 0.5001 | **-0.0104** |
| 2024 | 0.4621 | 0.4865 | **-0.0244** |

- **게이트 실증 (결정적)**: `MissingIndicatorAdder` 8종만 추가하면 2024 BSS **439.00 유지** (n_iter 247 동일) — 지시자 역전 신호가 존재해도 HGB 게이트 성능을 해치지 않음. enc와 결합될 때만 FAIL (enc가 원인).
- 역전은 실재하지만(결측 인구 target 0.587→0.462, 전체 0.564→0.487보다 가파름) **모델이 이를 활용하지 않거나 무해하게 처리**함.
- Oracle 재검증: G1 결측 = "데이터셋 데뷔 경기"이며, 이는 "신인 인구의 시즌별 성적 변화"가 원인.
- `asof_n` 기반 대체 ablation: pitcher는 asof_n>10인 결측이 70.9%라 **대체 불가** (asof_n≤10 지시자는 오히려 -21.82 해로움), batter는 asof_n==0과 1:1이라 대체 가능.

### 3.3 원인: IDTargetEncoder(enc) 전체 — pitcher_enc/batter_enc 단독 모두 FAIL (2026-08-08 enc 분리 검증)

**단계별 검증에서 enc가 유일한 FAIL 범인으로 확정된 후, pitcher/batter를 분리 검증**:

| 조합 | n_iter | Brier24 | BSS24 |
|---|---|---|---|
| base | 247 | 0.2487 | **439.00** |
| **pitcher_enc 단독** | 1000 | 0.2563 | **0.00** 🔴 |
| **batter_enc 단독** | 283 | 0.2543 | **0.00** 🔴 |
| both_enc | 1000 | 0.2739 | 0.00 🔴 |
| pitcher_enc+interact | 823 | 0.2567 | 0.00 🔴 |
| batter_enc+interact | 810 | 0.2660 | 0.00 🔴 |

- **pitcher_enc 단독으로도 FAIL** (Brier 0.2563 > 기준선 0.2498, n_iter 1000) — 기존 "pitcher는 +0.026 유지라 무죄" 판정 **정정**. ID 인코딩 자체가 시간 일반화 실패.
- **batter_enc 단독도 FAIL** (Brier 0.2543, n_iter 283).
- 배경 상관 데이터 (이유 설명용): 타자 과거 성공률은 2023부터 역상관(2023 -0.0078, 2024 -0.0035), pitcher는 +0.026 약신호 유지 — 하지만 **이런 약신호도 enc로 만들면 FAIL**.
- **메커니즘**: ID별 target 평균(spearman 1.0)이 in-sample 최강 신호 → HGB가 강하게 신뢰 + 내부 검증(랜덤 10%)이 같은 분포라 early stop 미발동(n_iter 1000) → 2024에서 신호 드리프트/약화 → 상수 예측보다 나쁜 Brier.
- interact와 결합해도 enc가 개선 효과를 흡수(0.00).

### 3.4 단계별 검증 요약 — 범인과 순기능 (2026-08-08)

**8조합 게이트 ablation 결과** (base 439.00 대비):

| 조합 | BSS24 | 판정 |
|---|---|---|
| base (47) | 439.00 | 기준 |
| +enc (pitcher_enc/batter_enc) | **0.00** | 🔴 **FAIL — 단독 범인** |
| +missing (8 지시자) | 439.00 | ✅ 무죄 |
| **+interact** (base_state_li/count_cat/runner_risk) | **494.36** | 🟢 **+55.36 개선** |
| enc+missing | 0.00 | 🔴 enc 때문에 FAIL |
| enc+interact | 0.00 | 🔴 enc 때문에 FAIL |
| missing+interact | 494.36 | 🟢 개선 유지 |
| all (60) | 0.00 | 🔴 enc 때문에 FAIL |

**2024에서도 살아있는 안정 신호** (데이터 레벨):

| 신호 | 2024 상태 |
|---|---|
| `count_cat` (count 단조: 3-2=0.457 < 0-0=0.486) | ✅ 2024에서도 단조 유지 |
| `asof_pitcher_success_rate` (최근 컨디션) | ✅ 5분위 단조 (2024: 0.452→0.538) |
| `asof_pitcher_reverse_rate` | ✅ (음의 방향 강신호) |
| base_state/runner 상황 | ✅ 약하지만 안정 |

### 3.5 FAIL 원인 요약 (최종)

> **게이트 FAIL의 직접 원인 = IDTargetEncoder(ID 인코딩) 전부** (2026-08-08 ablation으로 확정):
> - pitcher_enc / batter_enc **단독 각각** 2024 BSS 0.00 (Brier 0.256/0.254 > 기준선 0.2498)
> - enc가 포함된 모든 조합이 FAIL (n_iter 1000 early stop 미발동)
> - **missing 지시자 8종: 무죄** (단독 439.00 유지 — 기존 의심 정정)
> - **interact 3종: +55.36 순기능** (439.00 → 494.36) — 유일하게 통과한 전처리
>
> 파라미터 무죄(기본값도 동일 FAIL)는 유지. 다음 단계: **enc 제거(또는 시간 안정적 재설계) + interact 유지** → 게이트 재검증.

---

## 4. 하이퍼파라미터 튜닝 사후분석 (파손 확인)

### 4.1 `hgb_best_params.json`(l2=1.0, leaf=31, lr=0.1)의 실제 출처

세션 기록(`ses_0234c8296ffe982FvAHNdd8TK8`) 및 `/tmp/tuning/` 로그로 확인:

| 단계 | 실제 일어난 일 |
|---|---|
| RS 시도 1 (`shuffle=False`, cv=5) | 1,155초 완료 → **best_score_ 0.0** — 시즌 정렬 데이터에서 KFold 경계 라벨 편향으로 전 fold BSS 붕괴 (쓸 수 없는 결과) |
| RS 시도 2 (`StratifiedKFold shuffle=True`) | 100 fits 미완료, 밤새 54코어 점유 → **강제 종료** |
| "빠른 검증" fit | RS 포기 후 **사람이 l2=1.0/leaf=31/lr=0.1을 수기 입력** ("Wave A 로컬 439 vs Public 770 격차 → 1.0 상향" 정성 근거) |

→ **현재 JSON은 튜닝 결과가 아니라 수기값**. `load_best_params()`가 게이트에 자동 주입됨.

### 4.2 튜닝의 구조적 문제 4가지

1. **피처 공간 불일치**: Wave A(47컬럼, 전처리 X)의 문제를 고치려고 만든 파라미터를 Wave B(60컬럼, enc 포함)에 적용 — 피처가 바뀌면 최적 파라미터도 이동하는데 그 순서를 어김.
2. **튜닝 CV 오용**: 2019~2023 내부를 5-fold 랜덤 셔플로 분할 — 게이트는 2024 시간 홀드아웃. in-sample 랜덤 CV 점수는 시간 일반화를 보장하지 않음.
3. **BSS `max(0,·)` 클램프**: fold BSS가 0으로 붕괴하면 튜너에게 신호가 없음 (attempt 1의 best_score 0.0 = 이 증상).
4. **게이트 판정 오염**: overlay가 기본 파라미터 실행보다 먼저 끼어들어 원인 규명을 1회 지연.

### 4.3 올바른 순서

> **① 피처/전처리 확정 → ② 그 피처 위에서 튜닝(시간 순서 CV) → ③ 게이트 검증 → ④ 제출**

---

## 5. EDA 관점 데이터 활용 전략

### 5.1 즉시 적용 가능 (신규 피처 후보)

| 피처 | 근거 | 상태 |
|---|---|---|
| `game_type` F vs R 분리 또는 F 플래그 | +0.0795 편차, 161,004행 | ✅ 기존 CAT_COLS에 포함 — 활용 확인 필요 |
| `run_diff` = run_total_before 파생 | score_diff_* 상관 낮음 | 후보 |
| 최근 컨디션 × ID 상호작용 | asof_pitcher_success_rate 최강 신호 | 후보 |
| 역률·middle rate 결합 | reverse_rate -0.079 강신호 | 후보 |

### 5.2 전처리 수정 방향 (FAIL 대응 — 2026-08-08 ablation 확정 기준)

| 단계 | 조치 | 근거 |
|---|---|---|
| `IDTargetEncoder` | **enc 전체 제거** (pitcher_enc·batter_enc 모두) 또는 **시간 안정적 재설계** (시즌 가중·최근 시즌만·smooth 상향 등, 재검증 필수) | 🔴 pitcher_enc/batter_enc 단독 각각 FAIL 확정 (§3.3). ID 인코딩 자체가 시간 일반화 실패 |
| `MissingIndicatorAdder` | **사용 가능 (무죄)** — 단독 439.00 유지. `asof_n` 기반 대체는 pitcher 불가(70.9%가 asof_n>10) | ✅ §3.2/§3.4 |
| `InteractionAdder` | **유지** — +55.36 개선 (494.36). base_state_li 비안정성 리스크만 확인 | 🟢 §3.4 |
| `asof_*` 원본 | 유지 + **최근 시즌 게이트 검증 필수** | ⚠️ "HGB NaN 네이티브 = 안전"이 아니라 "드리프트 신호이므로 검증 필수"가 정확한 근거 (Oracle 재검증) |

**⚠️ HGB NaN 네이티브 처리의 한계 (Oracle 재검증)**: HGB는 NaN 브랜치를 내부적으로 학습한다 — Wave B의 명시적 지시자 FAIL과 **동일 메커니즘**. NaN 네이티브는 "안전해서"가 아니라 "드리프트 신호를 최근 시즌 게이트로 관리"해야 한다는 점에서 채택하는 것. 2025 추론 시점의 NaN 비율 불확실성도 존재. 단, missing 지시자 8종은 게이트에서 무해로 실증됨(§3.2) — NaN 브랜치 자체가 문제라기보다 enc의 과도한 신뢰가 문제였다는 점에 유의.

### 5.3 리스크 노트

1. **2025 드리프트**: test season=2025, target 평균 2019 0.5647 → 2024 0.4861 단조 하락. 2025도 하락 가정 시 base rate 표류 — season 보정 고려.
2. **head-50k 샘플 편향**: head(n)은 2019 개막 초반 구간(결측 13.94% vs 전체 1.98%) — 샘플링 시 주의.
3. **ID 인코딩은 폐기 확정** (enc 무수축·시간 일반화 실패 — §3.3). ID 정보를 쓰려면 enc 대신 시즌 가중·최근 시즌 기반 통계 등 시간 안정적 설계 필요.
4. **결측(데뷔) 신호의 시즌 반전** (Oracle 재검증): 결측 인구 target +0.024(2019-22 평균) → −0.0174(2023-24 평균). 단, missing 지시자 게이트 무죄 실증(§3.2) — 반전이 있어도 모델 성능엔 영향 없음.
5. **2025 NaN 비율 불확실성**: 2025 test의 데뷔 경기(결측) 비율이 학습과 다를 수 있음 — NaN 브랜치 의존도 낮추는 설계 권장.

---

## 6. trackman_history.csv — 활용 불가 확정 (재검증 완료)

| 매핑 시도 | 결과 |
|---|---|
| `pitcher_trackman_id`(906) ∩ `pitcher_id`(792) | **0** |
| `batter_trackman_id`(913) ∩ `batter_id`(830) | **0** |
| (season, pitcher_hand, pitcher_team) 조합 (258 ∩ 134) | **0** |
| 날짜 기반 조인 | 불가 — train에 `game_date` 컬럼 없음 (game_month만 존재) |
| 팀 ID 체계 | trackman: 문자열(`DOO_BEA` 등 26종) vs train: 숫자(12~25) — 완전 이질 |

- trackman 데이터 자체는 깨끗함 (구속 0.43%·회전수 0.70% 결측, pitch_type_group 4분류, rel_speed median 137.4km/h).
- **하지만 train/test와 연결 키가 0개** → 어떤 형태로도 피처 결합 불가. 기존 OUT 결정 유지.

---

## 7. 결론 및 다음 단계

1. **원인 확정 (2026-08-08 ablation)**: FAIL = **IDTargetEncoder(ID 인코딩) 전체** — pitcher_enc/batter_enc 단독 각각 2024 BSS 0.00 (n_iter 1000 early stop 미발동, Brier > 기준선). 기존 의심이던 "prev_game missing 지시자 6종 + batter_enc"는 **정정**: missing 지시자 8종은 무죄(단독 439.00 유지), **interact 3종은 +55.36 개선**(494.36).
2. **Oracle 재검증 반영 (2026-08-08)**: 결론 방향(이상치 제거 금지, imputation 금지)은 유지. 단, ① 이진/저차원 변수의 IQR "이상치 수치"는 산술 아티팩트로 삭제, ② "rate 0/1 = 소표본 노이즈 99.6%"는 커리어 success/ball/strike에만 성립(prev_game 0.4%, offspeed 85% 대표본 = 행동 신호), ③ 결측 메커니즘은 "시즌 첫 등장"이 아닌 **데이터셋 데뷔 경기**(복귀 투수 결측 0건/16,907행).
3. **다음 단계 (제안)**:
   - ① **enc 폐기 + interact 유지 파이프라인**으로 게이트 재검증: base(47) + MissingIndicator(선택) + InteractionAdder(3종) → 494.36(+55.36) 달성 확인
   - ② 통과 시: 시간 순서 CV 기반 튜닝 → 전체 재학습 → zip → 제출
   - ③ (선택) ID 정보를 쓰려면 enc 대신 시간 안정적 설계(시즌 가중·최근 시즌만) 탐색
4. **커밋**: 본 리포트 + EDA 스크립트 + ablation 로그는 gitignore 대상(`**/backup/`) 확인 후 커밋.

---

*본 문서는 데이터 분석 전용 — 모델 성능(BSS) 판정은 게이트 검증에서 별도 수행.*
