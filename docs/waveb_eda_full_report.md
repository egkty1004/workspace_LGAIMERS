# Wave B 종합 EDA 리포트 — 데이터 프로파일 + FAIL 원인 진단 + 튜닝 사후분석

> **목적**: (1) 전체 데이터(train/test/trackman/sample_submission)의 특성·분포·결측·이상치 프로파일,
> (2) Wave B 게이트 FAIL(2024 BSS 0.00)의 원인 pinpoint,
> (3) 하이퍼파라미터 튜닝 파손 사후분석.
> **검증일**: 2026-08-08 | **인터프리터**: aimers9 (sklearn 1.8.0, pandas 2.0.3, numpy 1.26.4)
> **산출 스크립트**: `eda_full_profile.py` / `eda_fail_diag.py` / `diag_prep_stages.py` (베이스라인 실험용/)
> **로그**: `backup/eda_full_profile.log` / `backup/eda_fail_diag.log` / `backup/profile_{train,trackman}.csv` / `backup/eda_{pitcher,batter}_enc_drift.csv` / `backup/eda_missing_signal.csv`

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

- **그 외 31개 컬럼은 결측 0건** — 결측은 전적으로 "이력 부족"(시즌 초·경기수 5 미만·첫 등장)에서만 발생.
- G2/G3의 792/830 = pitcher_id/batter_id 고유값 수와 **일치** → "시즌 첫 등장"이 결측의 유일 원인.
- test(2025) 샘플 5행에서도 동일 NaN 패턴 확인 → 2025에도 cold-start 결측 예상.

### 2.2 이상치 (IQR 기준)

| 컬럼군 | IQR 이상치 | 판정 |
|---|---|---|
| `score_diff_home` (±1.5×IQR 밖 138,418건) | 많음 | ✅ **자연 극단값** — 점수 차는 야구에서 정상 범위 (-27~+27). 제거 금지 |
| `run_total_before` (29,750건) | 많음 | ✅ 자연 극단값 (최대 37점) |
| `li` (67,303건, 최대 10.83) | 많음 | ✅ 연속형 중요도 지표의 오른쪽 꼬리 — 제거 금지 |
| `asof_pitcher_*_rate` (rate 0 또는 1) | 많음 | ✅ rate 0/1은 소표본에서 자연 발생 — 제거 금지 |
| `pitcher_id` low (959건) | 있음 | ✅ ID가 밀집 구간 밖일 뿐 — 의미 없음 |
| `runner_on_{1,2,3}b`, 카운트/카운트·outs | 0 | ✅ 이산 변수, 정상 |

**결론: 이상치는 전부 스포츠 데이터의 자연 극단값 — 제거/클리핑 금지.** 이상치 처리는 "없다"가 정답.

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

### 3.2 원인 1: prev_game missing 지시자 신호 역전 (결정적)

`asof_pitcher_prev1_game_success_rate` 결측 행의 target 평균 vs 비결측:

| season | missing 행 target | present 행 target | **delta** |
|---|---|---|---|
| 2019 | 0.5869 | 0.5635 | +0.0234 |
| 2020 | 0.5542 | 0.5324 | +0.0218 |
| 2021 | 0.5563 | 0.5324 | +0.0239 |
| 2022 | 0.5553 | 0.5286 | +0.0266 |
| 2023 | 0.4897 | 0.5001 | **-0.0104** |
| 2024 | 0.4621 | 0.4865 | **-0.0244** |

- 학습 데이터(2019~2023)의 5시즌 중 4개가 "+신호"(missing = 성공률 높음) → 모델이 "missing → 높은 성공률" 학습.
- **2024에서는 정반대(-0.0244)** → missing 행(3,638행)에 체계적 과대 예측 → 상수보다 나쁜 Brier.
- 원인은 전처리 버그가 아니라 **데이터의 시즌 드리프트**: 2020년대 중반 이후 "이력 없는 투수(신인/복귀)"의 성공률이 오히려 낮아짐.
- `_missing` 지시자 6종(prev1/3/5_game success/middle)은 모두 동일 패턴 → 전부 같은 역전 신호.

### 3.3 원인 2: batter_enc 역전

"이전 시즌(<s) batter_id 성공률"과 당해 시즌 target의 상관:

| season | corr |
|---|---|
| 2022 | +0.0639 |
| 2023 | **-0.0078** |
| 2024 | **-0.0035** |

- 타자의 과거 성공률은 2023부터 예측력을 잃고 **역상관**으로 전환.
- 2023→2024 연속 등장 batter 295명의 재현성 corr도 **+0.0817** (거의 무신호).
- pitcher_enc는 2024에서도 +0.026 (약신호 유지, 재현성 corr +0.3553) — pitcher는 유지 가능, **batter_enc는 제거 대상**.

### 3.4 유지 가능한 신호 (2024에서도 살아있음)

| 신호 | 2024 상태 |
|---|---|
| `count_cat` (count 단조: 3-2=0.457 < 0-0=0.486) | ✅ 2024에서도 단조 유지 |
| `asof_pitcher_success_rate` (최근 컨디션) | ✅ 5분위 단조 (2024: 0.452→0.538) |
| `asof_pitcher_reverse_rate` | ✅ (음의 방향 강신호) |
| `base_state`/runner 상황 | ✅ 약하지만 안정 |

### 3.5 FAIL 원인 요약

> **게이트 FAIL의 직접 원인은 하이퍼파라미터가 아니라 전처리 피처 2종의 2024 역전**:
> ① prev_game missing 지시자 6종 (+0.027 → -0.024, 학습 중 4/5시즌이 반대 방향)
> ② batter_enc (2023부터 역상관)
>
> 기본 파라미터에서도 동일 FAIL(0.00)이므로 파라미터 무죄 확정. Interaction(count_cat/runner_risk/base_state_li)은 2024에서도 신호 유지 → 무죄.

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

### 5.2 전처리 수정 방향 (FAIL 대응)

| 단계 | 조치 | 근거 |
|---|---|---|
| `MissingIndicatorAdder` | **prev_game missing 지시자 제거** (또는 최근 2시즌만 fit) | 2024 역전 확정 (§3.2) |
| `IDTargetEncoder` | **batter_enc 제거**, pitcher_enc만 유지 | batter 역전 확정 (§3.3) |
| `InteractionAdder` | base_state_li 비안정성 리스크 있으나 신호 유지 — 유지 또는 base_state_li만 제거 | §3.4 |
| `asof_*` 원본 | 유지 (HGB NaN 네이티브) | Wave A에서 정상 |

### 5.3 리스크 노트

1. **2025 드리프트**: test season=2025, target 평균 2019 0.5647 → 2024 0.4861 단조 하락. 2025도 하락 가정 시 base rate 표류 — season 보정 고려.
2. **head-50k 샘플 편향**: head(n)은 2019 개막 초반 구간(결측 13.94% vs 전체 1.98%) — 샘플링 시 주의.
3. **enc 무수축**: smooth="auto"가 이 데이터 규모에서 사실상 무수축 → cv=5 cross-fitting이 leakage 방어의 유일 버팀목. enc 유지 시 cross-fitting 필수.

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

1. **원인 확정**: FAIL = prev_game missing 지시자 6종 + batter_enc의 2024 신호 역전 (파라미터 무죄).
2. **다음 단계 (제안)**:
   - ① `MissingIndicatorAdder`에서 prev_game 지시자 제거, `IDTargetEncoder`에서 batter_enc 제거 → `diag_prep_stages.py`로 재검증
   - ② 수정 파이프라인으로 게이트 재검증 (기본 파라미터, Wave A 439.00 vs 비교)
   - ③ 통과 시: 시간 순서 CV 기반 튜닝 → 전체 재학습 → zip → 제출
3. **커밋**: 본 리포트 + EDA 스크립트 3종 + 백업 로그는 gitignore 대상(`**/backup/`) 확인 후 커밋.

---

*본 문서는 데이터 분석 전용 — 모델 성능(BSS) 판정은 게이트 검증에서 별도 수행.*
