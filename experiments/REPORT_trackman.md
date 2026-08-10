# Trackman 활용 실측 분석 요약 (DACON Aimers9)

분석일: 2026-08-08 | 데이터: trackman_history.csv(1,793,078행×30), train.csv(1,475,092행), test.csv(2025, TM 데이터 없음)

## 1. 1:1 조인키 존재 여부 — **없음 (직접 조인 0%)**
- train/test 컬럼: row_id, season, game_month/dayofweek, inning, top_bottom, balls/strikes/outs, pitcher_id/batter_id(익명화), 팀ID 등 — **trackman_game_id, pitch_no, game_date, pitcher_trackman_id 없음**
- trackman 컬럼: trackman_game_id, pitch_no, game_date, pitcher_trackman_id/batter_trackman_id(50008~71775155) — train ID(20700~24633)와 완전히 다른 체계
- 공통 컬럼은 season/month/dayofweek/inning/count/outs/핸드/팀뿐 → **pitch 1:1 직접 조인 불가**

## 2. 확률적/지문 매칭 실측
| 방법 | 매칭 결과 |
|---|---|
| 팀+핸드+시즌 (순수 확률) | 셀당 평균 17.3명 투수, 고유셀 1.5% → **사실상 불가** |
| **게임 카운트-워크 지문 매칭** (이번 분석에서 발굴) | train 게임경계(팀쌍 변화) 복원 → TM 게임과 (이닝,카운트,아웃,핸드) 시퀀스 완전일치 |
| 완전 일치 | **2,418/4,605 게임, 행 49.3%** (2024: 55.5%) — 충돌 0 |
| prefix≥0.9 / ≥0.5 | 행 54.0% / 69.1% (2024: 61.1% / 76.9%) |
| **pitcher_id ↔ pitcher_trackman_id** | **653명 1:1 매핑** (신뢰도≥0.976, 시즌 불일치 94/727,732=0.01%), train 행 99.1%·2024 행 99.4% 커버 |

- 매칭은 **동일 실제 경기**임을 검증 (팀/날짜/길이 일치 + 핸드시퀀스 일치). 일부 게임(웍스 갈림)은 train과 TM의 투구별 결과 기록이 서로 다른 제3자 데이터라는 증거.
- 중복/누수: 매칭 게임당 1:1, 위치 정렬로 현재 투구의 TM 값만 부착 → 누수 없음.

## 3. 피처 후보 + 2024 검증셋 이론상한 (BSS, r=0.486, upper=dRes/(r(1-r))×1e5)
### (a) 피치-레벨 (매칭 2024 행 140,813, 검증셋 셀평균 oracle)
| 피처 | 상한 | | 피처 | 상한 |
|---|---|---|---|---|
| induced_vert_break | **268.0** | | extension | 69.9 |
| rel_speed | **224.3** | | rel_side | 74.7 |
| pitch_type_group | **198.5** | | horz_break | 63.7 |
| spin_rate | 87.5 | | rel_height | 57.0 |
### (b) 피치-레벨 2023→2024 정직 전이 (학습셋 셀평균을 2024에 적용)
- **ivb +179.0, rel_speed +118.1**, 나머지(spin/hb/ext/rel_height/rel_side)는 음수
- 풀링(2019-23) 학습 시 전부 음수 → **연도 드리프트 존재**, 최근년(2023)만 유효
### (c) asof 투수 시즌 프로필 (테스트에 적용 가능한 유일한 형태)
- 2023프로필→2024 전이: **전 피처 음수 (-9.3~-163.9)**, 2024 quintile 성공률 평평(노이즈)
- LightGBM 실측(2019-23 학습, 2024 검증): baseline 512.68 → +TM프로필 491.66 (**-21.0**), 단일 피처 최대 +10(spin_rate)

## 4. 결론: **NO — 현실적으로 활용 불가**
1. **직접 조인키 없음** (0%), 순수 확률적 매칭도 불가 (셀당 17명).
2. 게임 지문 매칭은 기술적으로 성공(행 49~69%, 2024 55~77%)하나 **2025 test에 Trackman 데이터가 전혀 없어** 피치-레벨 피처를 추론에 사용 불가.
3. 사용 가능한 유일한 형태인 **asof 투수 프로필은 2024 검증에서 예측력 없음** (전이 BSS 전부 음수, 모델 BSS -21), 게다가 **test 투수 4명 중 1명만 매핑**됨(신인 2명은 train 자체에 없음) → 적용 커버리지도 부족.
4. **가장 유망한 피처**: 만약 2025 TM 데이터가 있었다면 **induced_vert_break(+179)와 rel_speed(+118)** (per-pitch, 최근년 전이 유효) — 그러나 데이터가 없어 원천 차단.
- 권고: Trackman 분석에 시간을 쓰기보다 기존 GIHO 파이프라인(platoon/count_state/10시드/delta)의 피처·앙상블 개선에 집중. 유일한 후속 여지: 매칭 게임의 per-pitch TM 값을 teacher-label로 활용하는 증류(distillation) 기법(복잡·불확실).
