# Wave B 전처리 데이터 유효성 검증 & EDA 리포트

> **목적**: Wave B 전처리 파이프라인(`bss_preprocess.py` — IDTargetEncoder / MissingIndicatorAdder / InteractionAdder)의 **데이터적 유효성** 검증 + EDA 인사이트. 모델 fit 없이 데이터 신호만 검토한다.
>
> **범위**: train 1,475,092행 × 47피처 (2019~2024), transform 검증은 50,000행 샘플
> **검증일**: 2026-08-08 | **인터프리터**: aimers9 (sklearn 1.8.0)
> **참조**: [exp_hgb_validation.md](./exp_hgb_validation.md) — Wave B 파이프라인 구현 검증 문서

---

## 1. 전처리 60컬럼 변환 검증

`build_pipeline(max_iter=10, early_stopping=False)` → `pipe[:-1]`(전처리 4단계만)를 50,000행 샘플에 fit→transform.

### 1.1 실행 stdout

```
train: (1475092, 48) | 피처: 47 (범주형 7, 수치형 40)
시즌: 2019 ~ 2024 | 제구 성공률: 0.5238

TRANSFORM_COLS (50000, 60)
NaN_TOTAL 43712
ENC_RANGE 0.20098463572703518 0.8044966999290495 0.24772783246316227 0.7850674208564029
ENC_NaN 0 0
MISSING_RATES {'asof_pitcher_prev1_game_success_rate_missing': 0.1394,
               'asof_pitcher_prev3_game_success_rate_missing': 0.1394,
               'asof_pitcher_prev5_game_success_rate_missing': 0.1394,
               'asof_pitcher_prev1_game_middle_rate_missing': 0.1394,
               'asof_pitcher_prev3_game_middle_rate_missing': 0.1394,
               'asof_pitcher_prev5_game_middle_rate_missing': 0.1394,
               'asof_pitcher_success_rate_missing': 0.0037,
               'asof_batter_success_rate_missing': 0.0041}
INTERACT {'base_state_li': (1, 4), 'count_cat': (0, 11), 'runner_risk': (0, 11)}
INDICATOR_MATCH_OK   # _missing 지시자 = 원본 isna와 100% 일치 (3개 컬럼 spot check)
NEW_COLS_NaN 0        # 신규 13컬럼 (2 enc + 8 missing + 3 interact) NaN 0건
ORIG_COLS_NaN 43712   # NaN 43,712건은 전부 원본 asof_* 컬럼 (imputation 금지 설계)
```

### 1.2 검증 결과 해석

| 검증 항목 | 기대 | 실제 | 판정 |
|---|---|---|---|
| 출력 컬럼 수 | 60 (=47+2+8+3) | **60** | ✅ |
| enc fillna (pitcher_enc/batter_enc NaN) | 0건 | **0 / 0** | ✅ |
| pitcher_enc / batter_enc 값 범위 | global_mean(0.5238) 근처 | **0.201~0.805 / 0.248~0.785** | ✅ |
| `*_missing` 8컬럼 결측 비율 | 배경 수치와 유사 | 샘플 편향 존재 (아래) | ⚠️ |
| base_state_li 범위 | 0~4 (ordinal) | **1~4 (실제 {1,3,4})** | ⚠️ |
| count_cat 범위 | 0~11 | **0~11 (전 범위)** | ✅ |
| runner_risk 범위 | 0~11 | **0~11 (전 범위)** | ✅ |
| unseen base_state → -1 | -1 할당 | train 8종 전부 매핑 (50k fit 기준) | ✅ (안전망) |

### 1.3 ⚠️ 주의: head-50k 샘플의 결측 편향 (중요 발견)

50,000행 head 샘플의 `*_missing` 비율(13.94%)이 **전체 데이터(1.98%)보다 약 7배 높다**. head 50k는 2019 시즌 **개막 직후 구간**이라 prev_game 이력이 없는 투수가 과대표집된다.

| 결측 비율 | head 50k | 전체 |
|---|---|---|
| prev_game 6종 | 13.94% | 1.98% |
| asof_pitcher_success_rate | 0.37% | 0.05% |
| asof_batter_success_rate | 0.41% | 0.06% |

→ `head(50000)`은 시즌 전체를 대표하지 않는다. transform 형태(60컬럼, NaN 패턴) 자체는 정상이나, **수치 해석(예: 결측 비율)은 전체 데이터 기준으로** 해야 한다.

---

## 2. EDA 핵심 인사이트

### 2.1 시즌 드리프트 (target 평균 단조 하락)

| season | 행 수 | target 평균 |
|---|---|---|
| 2019 | 237,413 | **0.5647** |
| 2020 | 244,087 | 0.5327 |
| 2021 | 247,088 | 0.5328 |
| 2022 | 247,472 | 0.5289 |
| 2023 | 245,525 | 0.5000 |
| 2024 | 253,507 | **0.4861** |

- 2019→2024 누적 하락 **-0.0786**, 시즌당 ≈ -0.016. 단조 감소 추세 (2020~2021은 거의 평탄).
- ⚠️ **test.csv는 season 2025** (공식 test 5행 기준) — 제구 성공률 하락 추세가 2025까지 이어질 경우, 훈련 분포(0.5238 평균)보다 **낮은 base rate** 구간을 예측하게 된다. 시즌을 특성으로 쓰지 않으면 표류(이동평균 기반 asof 피처의 계절 lag)가 있을 수 있음.

### 2.2 pitcher_id / batter_id 프로파일

| 항목 | pitcher_id | batter_id |
|---|---|---|
| 고유값 수 | **792** | **830** |
| rows/ID 중앙값 | 856 (p25 174 / p75 2,569 / max 15,450) | 473 (p25 87 / p75 2,011 / max 13,928) |
| HGB 범주형 제한(255) 초과 | ✅ (792 > 255 → 수치 인코딩 정당) | ✅ (830 > 255) |

**신규 ID 행 비율 (이전 시즌에 없던 ID가 차지하는 행 비율):**

| season | pitcher 신규ID 행 비율 | batter 신규ID 행 비율 |
|---|---|---|
| 2019 | 1.000 (355개) | 1.000 (400개) |
| 2020 | 0.224 (110개) | 0.116 (95개) |
| 2021 | 0.206 (95개) | 0.088 (88개) |
| 2022 | 0.157 (88개) | 0.110 (88개) |
| 2023 | 0.138 (63개) | 0.094 (71개) |
| 2024 | 0.199 (81개) | 0.093 (88개) |

- 시즌마다 신규 투수/타자가 10~20% 행을 차지 → **unseen ID 처리가 실제 평가에서 필수** (2025 test에서도 신규 ID 존재 가능성 높음).
- 투수 로스터 회전(batter보다 신규 비율 높음) — 투수 ID가 타자 ID보다 시간 표류가 크다.

### 2.3 asof_* 결측 패턴 — 3개 그룹으로 수렴

전체 데이터 기준 16개 컬럼에 NaN이 존재하나, 결측 패턴은 **정확히 3개 그룹**이다:

| 그룹 | 컬럼 | 결측 행 | 비율 | 의미 |
|---|---|---|---|---|
| G1: prev_game 이력 | `asof_pitcher_prev{1,3,5}_game_{success,middle}_rate` (6종) | 29,185 | 1.98% | 해당 투수에게 최근 N경기 데이터 부족 (시즌 초·경기수 5 미만) |
| G2: 투수 시즌 스탯 | `asof_pitcher_{success,reverse,middle,ball,strike,fastball,breaking,offspeed}_rate` (8종) | 792 | 0.05% | **cold-start 투수 792명 × 정확히 1행** (시즌 첫 등판) |
| G3: 타자 시즌 스탯 | `asof_batter_{success,middle}_rate` (2종) | 830 | 0.06% | cold-start 타자 830명 × 1행 |

- 792 = pitcher_id 고유값 수, 830 = batter_id 고유값 수와 **일치** → "시즌 시작 전 등장"이 결측의 유일 원인.
- **`_missing` 지시자 8개는 16개 결측 컬럼을 그룹 단위로 대표**한다 (G1의 6개가 완전 동일 패턴 → 3개만 표시해도 정보 동일, G2/G3의 대표 1개씩). 지시자 8개 = 결측 정보량의 완전 커버. 정보 중복은 있으나 손실 없음.
- G1 결측의 시즌 분포: 2019년 5.17% → 2023년 1.11%로 감소 후 2024년 1.44%. **head-50k 편향의 원인** (2019 시즌 초반 몰림: 2019의 12,278행이 G1 전체의 42%).

### 2.4 li / base_state / count 상호작용

**li 분포**: median 0.80, p90 2.02, p95 2.63, p99 4.37, max 10.83, **li==0이 1.86%** (주자 없는 상황의 다수 tie → qcut 구간 감소의 원인).

**base_state별 target 평균:**

| base_state | target 평균 | 행 수 |
|---|---|---|
| __3 (2사 3루) | **0.5361** (최고) | 35,513 |
| 1_3 | 0.5279 | 48,839 |
| 1__ | 0.5253 | 293,724 |
| _2_ / _23 | 0.5249 | 113,627 / 31,902 |
| ___ (무사 무루) | 0.5230 | 777,248 (52.7%) |
| 12_ | 0.5213 | 125,831 |
| 123 (만루) | **0.5161** (최저) | 48,408 |

→ 아웃카운트·주자 상황별 편차는 **±0.01 수준**으로 작지만 존재. 만루·2사 3루처럼 "득점 압박이 큰" 상황에서 제구 성공률이 낮은 경향 (123 최저 0.5161, __3이 유일하게 높음).

**count(balls×3+strikes)별 target 평균 — 단조 패턴:**

| count | 0-0 | 0-1 | 0-2 | 1-0 | 1-1 | 1-2 | 2-0 | 2-1 | 2-2 | 3-0 | 3-1 | 3-2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| target | 0.5266 | **0.5341** | 0.5185 | 0.5249 | 0.5285 | 0.5244 | 0.5189 | 0.5218 | 0.5210 | 0.5073 | 0.5044 | **0.4996** |
| 행 수 | 380,996 | 181,775 | 89,281 | 154,878 | 150,304 | 139,598 | 54,187 | 80,348 | 119,702 | 18,060 | 35,425 | 70,538 |

→ **풀카운트(3-2) 0.4996 < 기준(0-0) 0.5266**: 볼카운트가 몰릴수록 제구 성공률 하락, 스트라이크 우위(0-1)에서 최고. count는 **단조성 있는 실질 신호** — count_cat 인코딩의 타당 근거.

**num_runners_on × li 3구간:**

| | li<1 | 1≤li<2 | li≥2 |
|---|---|---|---|
| 0루 | 0.5231 | 0.5225 | 0.5249 |
| 1루 | 0.5214 | 0.5326 | 0.5210 |
| 2루 | 0.5133 | 0.5313 | 0.5239 |
| 3루 | **0.5048** | 0.5164 | 0.5209 |

→ 주자 3명 + li<1 (득점권 압박)에서 최저 0.5048. 단, 0루에서는 li 구간별 차이 거의 없음 (0.522~0.525) → **주자 유무 없이 li 단독으로는 신호가 약하다** (li 단독보다 runner_risk 형태의 결합이 데이터에 부합).

### 2.5 전처리가 데이터에서 드러내는 인사이트

| 전처리 | 데이터 검증 | 결론 |
|---|---|---|
| **prev_game missing = target +0.028?** | missing 행 target 0.5508 vs present 0.5232 → **delta +0.0276** | ✅ 실질 신호. "이력 없는 투수"가 오히려 제구 성공률 높음 (샘플 편향이 아니라 실제 시즌 초반 효과 — 2019 전체에서도 동일한 방향, 시즌 초반 투수/타자 매치업의 진입 편향) |
| **pitcher_enc가 ID별 target 반영?** | ID별 target 평균과 pearson **0.9999** / spearman **1.0000** (300k 샘플) | ✅ 완벽 반영. 희귀 ID(≤10행)도 |enc−global| 0.181로 **global로 수축하지 않음** — `smooth="auto"`(empirical Bayes)는 이 데이터 규모에서 사실상 무수축 → **cv=5 cross-fitting이 leakage 방어의 유일한 버팀목** |
| enc 범위 | pitcher_enc 0.201~0.805, batter_enc 0.248~0.785 (global 0.5238) | ✅ target 평균의 자연스러운 범위. ID별 target 평균 분포와 일치 |
| base_state_li | 실제 bin {1,3,4}만 존재 (bin 0·2 공백) | ⚠️ docstring의 "0~4"와 다름. li==0 tie(1.86%)로 qcut `duplicates="drop"`이 구간을 병합 → **fit 데이터에 따라 bin 구성이 달라지는 비안정성** (ordinal 연속성 깨짐 — 트리 모델에는 무해) |
| count_cat | 0~11 전 범위 존재, target 단조 관계 확인 | ✅ 타당 |
| runner_risk | 0~11 전 범위, 0(주자없음+li<1)이 43.5%로 최다, li 단독보다 결합이 데이터에 부합 | ✅ 타당 |
| unseen ID / base_state | 신규 ID 매 시즌 10~20% / base_state 8종 모두 train 커버, test도 {1__, ___}만 | ✅ -1·global_mean fallback은 실전(2025)에서 필요 |

---

## 3. 전처리 타당성 요약 (데이터 신호 관점)

| 단계 | 타당성 판정 | 데이터 근거 |
|---|---|---|
| **enc_ids** (IDTargetEncoder) | ✅ **필수 + 정확** | 카디널리티 792/830이 HGB 범주형 제한(255) 초과 → 수치 인코딩 불가피. enc가 ID별 target 평균을 spearman 1.0으로 반영. 무수축(auto-smooth) 때문에 **cross-fitting 없이는 leakage** — 현재 구현(cv=5)이 데이터 규모에서 필수 |
| **add_missing** (MissingIndicatorAdder) | ✅ **타당 (그룹 대표)** | 16개 결측 컬럼 = 3개 패턴 그룹, 지시자 8개가 전부 커버. prev_game missing이 target +0.0276로 실질 신호. 단, 지시자 6개(prev_game)는 완전 중복 정보 — 유지해도 무해 |
| **add_interact** | ⚠️ **count_cat·runner_risk 타당 / base_state_li 비안정** | count 단조 신호 확인, runner×li 결합이 li 단독보다 데이터에 부합. base_state_li는 bin 공백({1,3,4})·fit 의존성 — "base_state별 li median의 구간화"가 8개 범주에선 과설계일 수 있음 (base_state 자체가 이미 범주형으로 존재) |
| **to_cat** (7종 category) | ✅ 타당 | 7개 모두 카디널리티 ≤255 (base_state 8, hand 2 등) — HGB 네이티브 처리 정당 |

### 데이터 관점 리스크 노트 (모델 성능 무관)

1. **시즌 표류**: target 평균이 연 0.016씩 하락, 2025 예측 대상. asof 피처가 "이동 평균" 기반이면 시즌 간 lag 존재.
2. **head-샘플 편향**: `head(n)` 결측률/분포 해석 시 주의 (13.94% vs 1.98%).
3. **enc 무수축**: 희귀 ID의 enc가 극단값 유지 → cv=5가 없으면 target leakage. 향후 실험에서 smooth 고정값(예: 10~50) 비교 가능.
4. **base_state_li bin 비안정성**: fit 데이터 변경 시 bin 경계·코드가 달라짐 (추론 안정성 관점에서 고정 bins 옵션 고려).

---

*본 문서는 데이터 검증 전용 — 모델 성능(BSS) 비교는 제출 전 별도로 수행한다.*
