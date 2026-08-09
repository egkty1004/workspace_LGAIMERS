# REPORT — EDA 종합 리포트 (recent_gap · missing · cross · regime)

분석일: 2026-08-09 | 데이터: train.csv 1,475,092행 (2019~2024, R/F) | 목표: control_success 예측

이 리포트는 4개 EDA 결과(`eda_recent_gap.json`, `eda_missing.json`, `eda_cross.json`, `eda_regime.json`)를 취합해 유도 피처 후보의 우선순위표와 BSS 기대치를 정리한다. 모든 수치는 각 JSON에서 그대로 인용했다. 이 리포트는 Wave C 인코딩의 입력으로 사용된다.

---

## 1. EDA 4종 개요 (요약)

| EDA | JSON | n | 핵심 발견 | BSS 관측 (bss_bucket_calib) |
|---|---|---|---|---|
| recent_gap | `eda_recent_gap.json` | 1,475,092 | 직전 1경기 성공률 gap(prev1−asof): spread **+3.5pp**, corr 0.028. **음수 gap(급락)만 신호**. middle_rate는 비신호(−0.8pp) | success bucket 16.7~37.0 |
| missing | `eda_missing.json` | 1,475,092 | G1(복귀/공백) 29,185행 · G2(투수 데뷔) 792행 · G3(타자 데뷔) 830행. **G1 Δ는 game_type 구성 효과**, 단독 플래그 BSS≈0, 2024에서 Δ 역전 | 0.0 (oracle ≤3.5) |
| cross | `eda_cross.json` | 1,475,092 | **count_platoon 3-2+동손 Δ−8.4pp 최강**. score_diff +1.6~+2.4pp, li_risp +1.4~+2.3pp, outs×count +1.1~+2.5pp | pair_calib 최대 39.3 (count_platoon) |
| regime | `eda_regime.json` | 1,475,092 | F 레짐 2019-22 +15.3pp → 2023-24 −3.0pp. ABS(2024) R 전체 −1.3pp, 카운트별 비균등(1-1 −2.1pp 최대). **test F 존재 미확정** | 구조적 드리프트 (피처 신호 아님) |

> fold 정의: `primary` = train season≤2023 / val 2024 (R+F 혼합). `r2022/r2023/r2024` = R-only (val 각 2022/2023/2024). Δ는 percentage point.

---

## 2. EDA별 핵심 수치

### 2.1 recent_gap (`eda_recent_gap.json`)

- 전체: corr_success **0.028**, spread_2525_success **+3.51pp** (q25 −0.079 → q75 0.056, lo 0.505 → hi 0.540). corr_middle −0.008, spread_2525_middle **−0.81pp** (비신호).
- fold별 success spread/Δ(음수 구간 `<-0.1`)/BSS:

| fold | val 연도 | corr | spread_pp | `<-0.1` Δ(pp) | `<-0.1` n | bss_bucket_calib | bss_oracle |
|---|---|---|---|---|---|---|---|
| primary | 2024 (R+F) | 0.014 | 2.07 | **−1.81** | 58,389 | 16.71 | 39.47 |
| r2022 | 2022 (R) | 0.015 | 2.08 | **−1.66** | 44,563 | 29.22 | 31.75 |
| r2023 | 2023 (R) | 0.020 | 2.66 | **−1.80** | 46,242 | 37.02 | 44.94 |
| r2024 | 2024 (R) | 0.014 | 1.87 | **−1.86** | 48,644 | 35.59 | 39.23 |

- **음수 gap 구간(`<-0.1`, 전체의 21~23%)이 모든 fold에서 −1.7~−1.9pp로 유일한 신호**. 양수 구간(`0~0.05`, `>0.05`)은 0~+1.2pp로 약하고 fold 간 불안정. → 이진/단조 변환(급락 플래그)으로 써야 한다.
- 시즌별 gap 평균은 2019 +0.0027 → 2024 −0.0234로 단조 하락. label 드리프트(§2.4)와 겹치므로 fold별 검증 필수.

### 2.2 missing (`eda_missing.json`)

| 플래그 | 정의 | n (전체) | target | Δ(pp) | 비고 |
|---|---|---|---|---|---|
| g1_flag | prev1/3/5_game 6종 전부 결측 (복귀/공백) | 29,185 (1.98%) | 0.5508 | +2.71 | **시즌별 Δ 역전**: 2019-22 +2.1~+2.6 → 2023 −1.0 → 2024 **−2.4** |
| g2_flag | asof_pitcher 8종 결측 & n==0 (투수 데뷔) | 792 | 0.5518 | +2.80 | g2 ⊂ g1 |
| g3_flag | asof_batter 2종 결측 & n==0 (타자 데뷔) | 830 | 0.6024 | +7.86 | 소표본, fold 불안정 |
| return_gap_flag | prev1 결측 & asof_pitcher_n<100 | 29,086 | — | — | G1 중 소이력 구간만 분리 |

- **G1 Δ는 game_type 구성 효과로 밝혀짐**: F의 G1 prevalence 5.60% vs R 1.53% (F가 G1에 과대 대표). F 내부 Δ +1.02pp, R 내부 Δ +0.88pp로 희석. 2019-22 G1 Δ의 원인은 해당 시즌 F 레짐(높은 레이블)의 혼입.
- g1_asof_n 곡선: [0,10) +2.12pp · [10,50) +3.26pp · [50,100) +2.26pp · [100,1000) **−1.87pp** (n=99) → n≥100이면 기준치로 회귀.
- fold별 단독 BSS: primary/r2022/r2024 전부 **bss_bucket_calib 0.0** (oracle 최대 3.51). r2023만 0.17~0.19. → **플래그 단독은 BSS 기여 없음** (이미 asof 16종 NaN 네이티브로 흡수됨).

### 2.3 cross (`eda_cross.json`)

| 신호 | 정의 | 전체/최근 Δ(pp) | fold별 Δ(pp) | bss_flag_calib | bss_pair_calib |
|---|---|---|---|---|---|
| count_platoon_3b2_same | 3-2 & 동손 (vs 0-1 & 이손) | **−8.39** (최근 23-24, pos 11,211) | −6.73 ~ **−8.58** | 18.8~29.3 | **25.7~39.3** |
| score_diff_binary | \|diff\|≤1 (vs ≥6) | +1.62 (전체, pos 699,647) | +1.66 ~ +2.43 | 6.1~14.6 | 11.4~16.8 |
| li_risp_flag | RISP & LI≥1.0 (vs LI<0.5) | +1.93 (전체, pos 264,180) | +1.41 ~ +2.31 | 0.0~3.3 (r2023=0.0) | 1.8~9.5 |
| outs_count_3b2_2out | 3-2 & 2아웃 (vs 0아웃) | +1.69 (전체, pos 24,070) | +1.07 ~ +2.50 | 1.4~3.9 | 5.9~19.5 |

- `bss_flag` = pos vs 나머지 이진, `bss_pair` = pos/ref/기타 3셀. 둘 다 base-rate(logit shift) 보정 후 예상 기여.
- game_type×count: primary R−F Δ는 0-0 +3.19pp · 0-1 +6.65pp · 1-0 +3.05pp (나머지 카운트는 표본 부족으로 생략). F는 post-2023 카운트 전 구간(0-0~2-2)이 0.46~0.48로 평평하고 3볼 셀(F 3-2 n=2,570)은 표본 부족 → **F 카운트 구조가 R과 다름**.

### 2.4 regime (`eda_regime.json`)

- F 레짐: pre-2023(F 2019-22, n=105,308) 0.676 vs R 0.523 → **f−r +15.31pp**. 2023 −3.02pp, 2024 −3.04pp로 역전. "관대한 존(2019-22) → 표준 존(2023~)" 전환.
- ABS(2024) 도입: R 전체 −1.34pp. 카운트별 비균등: 1-1 **−2.10pp** 최대, 2-0 −2.02, 2-1 −1.98, 3-2 −0.55pp 최소.
- r32_longterm: R 3-2 pre-2023 0.5001 (n=41,886) → post-2023 0.4619 (n=21,097), **−3.82pp** 추가 하락.
- **test F 미확정**: 로컬 test.csv는 5행 형식 샘플(season 2025, 전부 R, has_F=false). 실제 평가는 245,789행이며 2025 F 존재 여부는 확인 불가. F가 있으면 post-2023 레짐(≈0.46)으로 매핑해야 하고, game_type을 상수 피처로 쓰면 안 된다.

---

## 3. 유도 피처 우선순위표

| 순위 | 피처명 | 정의 | fold별 Δ(pp) | n | BSS 잠재력 (bss_bucket_calib) | 피처화 난이도 | 예상 BSS gain | 소스 |
|---|---|---|---|---|---|---|---|---|
| 1 | `count_platoon_3b2_same` | 3-2 & 동손 (vs 0-1 & 이손) | **−6.7 ~ −8.6** | pos 4,688~5,453/val | flag 18.8~29.3 / **pair 25.7~39.3** | 중 (조합) | 높음 | cross |
| 2 | `recent_gap_success` | prev1_success − asof_success, 음수(급락) 구간 | `<-0.1` **−1.7 ~ −1.9** | 44.6~58.4k/val | bucket **16.7~37.0** | 낮음 (기존 col diff) | 높음 | recent_gap |
| 3 | `score_diff_binary` | \|score_diff_pitcher_team\|≤1 close | +1.7 ~ +2.4 | pos 100~114k/val | flag 6.1~14.6 / pair 11.4~16.8 | 매우 낮음 (이진화) | 중간 | cross |
| 4 | `outs_count_3b2_2out` | 3-2 & 2아웃 (vs 0아웃) | +1.1 ~ +2.5 | pos 3.5~4.1k/val | flag 1.4~3.9 / pair 5.9~19.5 | 낮음 | 중간 | cross |
| 5 | `li_risp_flag` | RISP & LI≥1.0 (vs LI<0.5) | +1.4 ~ +2.3 | pos 38~45k/val | flag 0.0~3.3 / pair 1.8~9.5 | 낮음 | 낮음~중간 | cross |
| 6 | `return_gap_flag` | prev1 결측 & asof_n<100 | −2.5 ~ +0.8 (시즌 역전) | 29,086 (전체) | **0.0** (oracle ≤3.5) | 낮음 | 낮음 (스크리닝 판단) | missing |
| 7 | `debut_flag` (투수/타자) | asof_pitcher_n==0 / asof_batter_n==0 | −7.9 ~ +7.9 (fold 불안정) | 792 / 830 | **0.0** | 낮음 | 낮음 | missing |
| 8 | `asof_n_bucket` | asof_pitcher_n 구간화 (기존 피처) | 곡선 +2.1~+3.3 (G1 내) | 전 구간 | — (기존 연속 피처) | 낮음 | 낮음 | missing |

> BSS 잠재력은 **단변량 버킷 피처의 예상 기여**이며, 기존 모델(이미 platoon/count_state/asof 포함) 위 증분 gain은 이보다 낮다. `bss_oracle`(검증 그룹 평균 적용 상한)은 참고용 상한선. 실제 판정은 스크리닝(2024 홀드아웃, R-only)에서.

---

## 4. 피처별 상세 증거

### F1. `count_platoon_3b2_same` (1순위)

- 정의: `(balls==3 & strikes==2) & (pitcher_hand==batter_hand)`, 기준 = 0-1 & 이손.
- 전체(2023-24) 실측: pos 0.4365 (n=11,211) vs ref 0.5205 (n=30,879) → **Δ −8.39pp**.

| fold | val 연도 | rate_pos | rate_ref | Δ(vs ref, pp) | bss_flag | bss_pair | oracle(pair) |
|---|---|---|---|---|---|---|---|
| primary | 2024 (R+F) | 0.4355 | 0.5163 | −8.08 | 21.32 | 36.75 | 45.13 |
| r2022 | 2022 (R) | 0.4508 | 0.5181 | −6.73 | 21.25 | 25.73 | 30.70 |
| r2023 | 2023 (R) | 0.4418 | 0.5276 | −8.58 | 29.31 | 39.32 | 49.80 |
| r2024 | 2024 (R) | 0.4424 | 0.5216 | −7.92 | 18.78 | 36.30 | 45.44 |

- 4개 fold 전부 Δ −6.7pp 이상, bss_pair_calib 25.7~39.3으로 EDA 전체에서 가장 안정적이고 강한 신호. 풀카운트 모서리 승부 + 동손 유인구가 겹치는 "제구 붕괴 극단 셀".
- 난이도: 중. platoon(동손/이손)과 count_state는 이미 존재하므로 상호작용 셀을 명시 피처로 추가. **주의**: F 레짐에서는 카운트 구조가 다르므로 R/F 분리 학습 또는 F 셀 제외 옵션 검토.

### F2. `recent_gap_success` (2순위)

- 정의: `asof_pitcher_prev1_game_success_rate − asof_pitcher_success_rate`, 급락(음수) 구간을 신호로.
- 전체 corr 0.028, spread +3.51pp. fold별 spread 1.87~2.66pp, bucket_calib 16.71~37.02 (oracle 31.8~44.9).
- **음수 구간(`<-0.1`)만 신호**: 4개 fold에서 Δ −1.66~−1.86pp (n 44,563~58,389, 전 행의 21~23%). 양수 구간은 0~+1.2pp로 약하고 불안정 → 급락 플래그/단조 변환 권장.
- 난이도: 낮음. 기존 asof 컬럼 2개의 차이 1줄. gap 결측(prev1 결측, 1.98%)은 0으로 채우고 prev1_missing 플래그 병기(기존 처리와 동일).
- 참고: `recent_gap_middle`은 **비신호**(spread −0.8pp 전체, −1.1pp primary) → 후보에서 제외 (아래 §5).

### F3. `score_diff_binary` (3순위)

- 정의: `abs(score_diff_pitcher_team)<=1` (close), 기준 = `abs(...)>=6` (blowout).
- 전체 실측: close 0.5283 (n=699,647) vs blowout 0.5121 (n=169,508) → +1.62pp.

| fold | Δ(pp) | bss_flag | bss_pair |
|---|---|---|---|
| primary | +1.66 | 10.32 | 12.23 |
| r2022 | +2.43 | 6.14 | 16.76 |
| r2023 | +1.70 | 8.94 | 11.40 |
| r2024 | +1.88 | 14.57 | 16.79 |

- 4개 fold 전부 양의 Δ, pair_calib 11.4~16.8. 이미 score_diff_pitcher_team 수치가 존재하므로 이진화만으로 "접전 집중" 비선형성을 명시화. 난이도 매우 낮음.

### F4. `outs_count_3b2_2out` (4순위)

- 정의: `(balls==3 & strikes==2) & outs_before==2` (vs outs_before==0).
- 전체 실측: 2아웃 0.5087 (n=24,070) vs 0아웃 0.4918 (n=23,441) → +1.69pp.

| fold | Δ(pp) | bss_flag | bss_pair |
|---|---|---|---|
| primary | +1.07 | 2.34 | 8.18 |
| r2022 | +1.18 | 1.45 | 5.94 |
| r2023 | +2.50 | 3.86 | 19.45 |
| r2024 | +1.32 | 2.73 | 10.13 |

- Δ는 4개 fold 모두 양수지만 flag 단독 BSS는 1.4~3.9로 약함. pair_calib 5.9~19.5 (r2023에서 19.45). "2아웃 풀카운트는 볼넷 부담이 줄어 한 박자 여유" 해석. 난이도 낮음.

### F5. `li_risp_flag` (5순위)

- 정의: `(runner_on_2b==1 | runner_on_3b==1) & li>=1.0` (vs `li<0.5`).
- 전체 실측: pos 0.5289 (n=264,180) vs ref 0.5095 (n=85,159) → +1.93pp.

| fold | Δ(pp) | bss_flag | bss_pair |
|---|---|---|---|
| primary | +2.15 | 3.03 | 8.43 |
| r2022 | +2.13 | 3.06 | 6.78 |
| r2023 | +1.41 | **0.0** | 1.76 |
| r2024 | +2.31 | 3.33 | 9.52 |

- Δ는 4개 fold 모두 양수(+1.4~+2.3pp)지만 **flag 단독 BSS가 0.0~3.3으로 약하고 r2023에서 0.0**. 기존 li/주자 피처가 대부분 흡수하는 것으로 보임. pair_calib 1.8~9.5. 낮은~중간 우선순위.

### F6. `return_gap_flag` (6순위, 낮음 — 스크리닝에서 최종 판단)

- 정의: prev1_game 결측 & `asof_pitcher_n < 100`. n=29,086 (전체).
- **G1 단독 BSS≈0**: fold별 bss_bucket_calib 전부 0.0 (primary/r2022/r2024), r2023만 0.19. oracle 상한 3.51.
- **Δ는 game_type 구성 효과**: G1의 F prevalence 5.60% vs R 1.53% (F 과대 대표). F 내부 +1.02pp, R 내부 +0.88pp로 희석. 2019-22의 +2.1~+2.6pp는 해당 시즌 F 레짐(높은 레이블) 혼입이 원인.
- **2024에서 Δ 역전**: g1 Δ 2019-22 +2.1~+2.6 → 2023 −1.02 → 2024 **−2.40**. return_gap primary Δ −2.47, r2024 Δ −1.20.
- 결론: 단독 신호로 쓰지 말고, 기존 asof 16종 NaN 네이티브(이미 흡수 중)에 더해 스크리닝에서 n≥100 회귀 곡선과 함께 최종 판단. 낮은 우선순위로 표기.

### F7. `debut_flag` (7순위, 낮음)

- 정의: `asof_pitcher_n==0` (투수 데뷔, g2, n=792) / `asof_batter_n==0` (타자 데뷔, g3, n=830).
- 전체 Δ: 투수 +2.80pp, 타자 +7.86pp로 보이나 **fold 불안정**: g3 primary +1.39 → r2022 −3.49 → r2023 −0.31 → r2024 −6.87. g2도 primary −7.87, r2023 −7.45.
- 소표본(0.05%) + 방향 불안정 + 단독 BSS 0.0 → 낮은 우선순위. 다만 시즌 첫 등판/첫 타석은 기존 NaN 처리로 이미 자연 흡수됨.

### F8. `asof_n_bucket` (8순위, 기본)

- 정의: asof_pitcher_n 구간화 (예: [0,10)/[10,50)/[50,100)/100+). 기존 연속 피처의 bucket화.
- G1 내 곡선: [0,10) +2.12 · [10,50) +3.26 · [50,100) +2.26 · [100,1000) −1.87pp (n=99) → n<100 구간에서만 +2~+3pp.
- 기존 asof_n(연속)이 이미 있으므로 증분 정보는 제한적. 단순 bucket화는 트리가 비선형성을 스스로 학습할 가능성이 높아 우선순위 최하.

---

## 5. 제외 / 비권장 항목

| 항목 | 근거 | 처리 |
|---|---|---|
| **middle_rate gap** (`recent_gap_middle`) | spread −0.8pp (전체), −1.1pp (primary), corr −0.008. 4개 fold 모두 음의 spread (−0.1~−1.3pp). **비신호** | 후보에서 제외 |
| **game_type × count** | R은 3볼 페널티 뚜렷(0-1 vs 3-2 Δ5.0pp)하지만 F는 0-0~2-2 전 구간 0.46~0.48 평평, 3볼 셀 표본 부족(F 3-2 n=2,570). F 카운트 구조가 R과 다름 → 독립 피처로 쓰면 구식 F 레짐을 평균에 새김 | **경계 해소(regime resolution)에서 판정** |
| **score_diff × LI** | 개별 신호(score_diff +1.6~+2.4pp, li_risp +1.4~+2.3pp)는 있으나 교차 조합은 EDA에서 단독 신호로 확인되지 않음 | 단독 피처로 채택하지 않음 |
| G1 플래그 단독 (`g1_flag`) | 단독 BSS≈0, Δ game_type 구성 효과, 2024 역전 | return_gap_flag로 세분화 후에도 낮은 우선순위 |

---

## 6. 리스크

1. **2025 test F 존재 불확실성** (`eda_regime.json` test_structure/test_inference): 로컬 test 5행은 전부 R(season 2025). 실제 평가 245,789행의 F 포함 여부 확인 불가.
   - F가 있으면 post-2023 레짐(≈0.46)으로 매핑해야 하고, game_type을 상수 피처로 쓰면 안 된다.
   - F 레짐 의존 피처(F 카운트 계수, G1 플래그의 F 구성 효과)는 **레짐 리스크를 그대로 받는다**. count_platoon 등 상황 피처도 F 행에서 계수가 다를 수 있으므로 R-only fold(r2022~r2024)와 primary를 함께 보고한다.
2. **label 드리프트**: R 성공률 2019 0.5495 → 2024 0.4897 (−6.0pp), F는 0.59~0.71 → 0.459 (레짐 브레이크). 시즌별로 Δ가 역전하는 피처(return_gap, debut)는 2024 홀드아웃 기준으로만 판단해야 한다.
3. **단변량 BSS의 과대 추정**: 우선순위표의 bss_bucket_calib는 단일 버킷 피처의 예상 기여. 기존 모델(platoon/count_state/asof 16종 포함) 위 실제 증분은 낮다. oracle 값은 상한선 참고용.
4. **ABS(2024) 이후 계수 변화**: 카운트별 Δ가 비균등(1-1 −2.1pp vs 3-2 −0.5pp). 2023 이전 가중이 커지면 2024 예측이 왜곡될 수 있다.

---

## 7. 권고 (Wave C 인코딩 입력)

- **필수 (1~3순위)**: `count_platoon_3b2_same` (상호작용 셀), `recent_gap_success` (음수 구간), `score_diff_binary` (close 플래그).
- **선택 (4~5순위)**: `outs_count_3b2_2out`, `li_risp_flag`. 기존 피처와 중복이 커 스크리닝에서 증분 확인 후 채택.
- **스크리닝 전용 (6~7순위)**: `return_gap_flag`, `debut_flag`. 단독 BSS≈0이므로 2024 홀드아웃 R-only 게이트에서만 판정. `asof_n_bucket`은 기본값으로 최하위.
- **인코딩 원칙**: gap/score_diff는 NaN 결측(prev1 결측 1.98%)을 0으로 채우고 결측 플래그 병기. game_type은 상수 피처 금지, F는 post-2023 레짐 분리 학습 권장. 모든 Δ 판정은 primary(2024, R+F)와 r2024(R-only)를 함께 확인.
- BSS 기대치 합계(단변량 상한 합): count_platoon pair 25.7~39.3 + recent_gap bucket 16.7~37.0 + score_diff pair 11.4~16.8 등이 실질 스크리닝 후보. 실제 수료 기준(Public ≥ 549.51) 대비 증분은 다음 단계(스크리닝 → 모델 학습)에서 확정한다.

---

*수치 출처: `repro_979/experiments/eda_recent_gap.json`, `eda_missing.json`, `eda_cross.json`, `eda_regime.json` (2026-08-09). 참고 리포트: `experiments/REPORT_kbo_insights.md`, `experiments/REPORT_data_quality.md` §5.*
