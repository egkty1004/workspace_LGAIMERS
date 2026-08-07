# exp: HGB 2024 검증 결과 (Wave A)

- 날짜: 2026-08-07
- 모델: sklearn HistGradientBoostingClassifier (max_iter=1000, early_stopping=True, n_iter_no_change=10, random_state=42)
- 범주형: 7개 (pitcher_id/batter_id는 sklearn 카디널리티 255 제한으로 수치형 처리 — 사용자 승인 옵션 A)
- 검증 프로토콜: 2019~2023 학습 → 2024 홀드아웃, BSS = max(0, 100000*(1 - brier/(r(1-r)))), r = y_val.mean()
- 게이트 앵커: RF 검증 BSS 415.57 (backup/rf_anchor_bss.txt)
- **HGB 2024 검증 BSS: 439.00 > 415.57 → PASS** (Δ +23.43)
- 상세: Brier(2024) 0.248710, r(2024) 0.4861, 2023 BSS(in-sample) 2503.05, n_iter_ 247, 검증 fit 12.3s
- 전처리 QA: cross-process pickle 왕복 OK (PICKLE_ROUNDTRIP_OK)
- 참고(G3): 로컬 검증 BSS(4xx)는 Public 549.51과 다른 r/프로토콜 — 직접 비교 불가

재현: `python train_hgb.py 2>&1 | tee backup/hgb_validate.log` (베이스라인 실험용/)

## Wave B 게이트 검증 (2026-08-07)

- **결과: FAIL** — Wave B(60컬럼) HGB 2024 검증 BSS **0.00** ≤ Wave A 앵커 439.00
- 원인: 60컬럼 전처리가 2024 일반화 실패 (Brier 0.2740 > 기준선 0.2498, n_iter 1000 early stop 미발동)
- 진단: l2 파라미터와 무관 (기본 파라미터에서도 동일 FAIL) → 전처리(특히 TargetEncoder 무수축 과적합)가 원인 후보
- 다음 단계: 전처리 단계별 분리 검증으로 원인 pinpoint (enc/missing/interact 각각 2024 BSS 추적)
- 로그: backup/hgb_validate_wb.log, backup/hgb_validate_wb_base.log

## 결측(데뷔) 신호 대체 ablation (2026-08-08)

**질문**: prev_game missing 지시자(2024 역전)를 `asof_n` 소표본 신호로 대체 가능한가? (Oracle [High] 권고)

**게이트 결과 (기본 HGB_KWARGS, 2019~2023 → 2024)**:

| 조합 | 피처 | n_iter | Brier24 | BSS24 | BSS23(in) |
|---|---|---|---|---|---|
| base (47) | 47 | 247 | 0.248710 | **439.00** | 2503.05 |
| +prev 지시자 1종 | 48 | 247 | 0.248710 | **439.00** | 2503.05 |
| +n0 (asof_n==0) | 49 | 247 | 0.248710 | **439.00** | 2503.05 |
| +nle10 (asof_n≤10) | 49 | 200 | 0.248765 | **417.18** | 2213.74 |
| +all (prev+n0+nle10) | 52 | 200 | 0.248765 | **417.18** | 2213.74 |

**결론**:
1. **prev 지시자 1종은 무해** (BSS 439.00 동일, n_iter 247 — HGB가 무시). → Wave B FAIL의 범인은 지시자 6종이 아니라 **다른 단계**(enc 또는 60컬럼 조합)일 가능성 — 기존 §3.2 결론 재검증 필요.
2. **asof_n≤10 지시자는 오히려 해로움** (-21.82): 데이터 레벨에서 pitcher 결측 행의 70.9%가 asof_n>10 (데뷔 경기 20~100투구에서 asof_n 이미 큼) → 소표본 신호로 pitcher 결측 대체 **불가능**. batter는 asof_n==0과 완벽 1:1 (max 0) → 대체 가능.
3. 로그: backup/diag_debut_ablation.log, backup/diag_debut_ablation.csv

## 단계별 분리 검증 (2026-08-08) — FAIL 범인 pinpoint

**질문**: Wave B 60컬럼 FAIL의 범인이 정확히 어떤 전처리 단계인가? (enc / missing / interact 분리)

**게이트 결과 (기본 HGB_KWARGS)**:

| 조합 | 피처 | n_iter | Brier24 | BSS24 | BSS23(in) |
|---|---|---|---|---|---|
| base (47) | 47 | 247 | 0.248710 | **439.00** | 2503.05 |
| **+enc** (pitcher_enc/batter_enc) | 49 | 1000 | 0.273938 | **0.00** 🔴 | 135.80 |
| +missing (8 지시자) | 55 | 247 | 0.248710 | **439.00** ✅ | 2503.05 |
| **+interact** (3종) | 50 | 177 | 0.248572 | **494.36** 🟢 | 2014.78 |
| enc+missing | 57 | 1000 | 0.273938 | 0.00 🔴 | 135.80 |
| enc+interact | 52 | 1000 | 0.274034 | 0.00 🔴 | 483.75 |
| missing+interact | 58 | 177 | 0.248572 | **494.36** 🟢 | 2014.78 |
| all (60) | 60 | 1000 | 0.274034 | 0.00 🔴 | 483.75 |

**결론**:
1. **FAIL의 진짜 범인 = IDTargetEncoder(enc) 단독** — enc가 들어가면 무조건 0.00 (n_iter 1000 폭증, early stop 미발동).
2. **missing 지시자 8종은 완전 무죄** (단독 439.00, n_iter 247 동일) — 기존 §3.2 "지시자 6종 범인" 결론 **정정**.
3. **interact 3종은 +55.36 개선** (439.00 → 494.36) — 전처리 중 유일한 순기능. missing과 결합해도 개선 유지.
4. 로그: backup/diag_stage_ablation.log, backup/diag_stage_ablation.csv

## enc 분리 검증 (2026-08-08) — pitcher_enc/batter_enc 개별 판정

**질문**: enc 단독이 범인인데, pitcher_enc와 batter_enc 중 어느 쪽(또는 둘 다)인가?

| 조합 | 피처 | n_iter | Brier24 | BSS24 | BSS23(in) |
|---|---|---|---|---|---|
| base | 47 | 247 | 0.248710 | **439.00** | 2503.05 |
| **pitcher_enc 단독** | 48 | 1000 | 0.256317 | **0.00** 🔴 | 1590.80 |
| **batter_enc 단독** | 48 | 283 | 0.254273 | **0.00** 🔴 | 772.07 |
| both_enc | 49 | 1000 | 0.273938 | 0.00 🔴 | 135.80 |
| pitcher_enc+interact | 51 | 823 | 0.256708 | 0.00 🔴 | 1487.73 |
| batter_enc+interact | 51 | 810 | 0.266029 | 0.00 🔴 | 2057.89 |

**결론**:
1. **pitcher_enc 단독으로도 FAIL** (Brier 0.2563 > 기준선 0.2498, n_iter 1000) — 이전 EDA의 "pitcher는 +0.026 유지라 무죄" 판정 **정정**. ID 인코딩 자체가 시간 일반화 실패.
2. **batter_enc 단독도 FAIL** (Brier 0.2543).
3. **범인 = TargetEncoder(ID 인코딩) 전체**: pitcher든 batter든 단독으로 게이트를 망침. 메커니즘: ID별 target 평균(spearman 1.0)이 in-sample 최강 신호 → HGB가 강하게 신뢰, 내부 검증(랜덤 10%)이 같은 분포라 early stop 미발동(n_iter 1000) → 2024에서 신호 드리프트/약화 → 상수 예측보다 나쁜 Brier.
4. **interact는 enc와 결합하면 흡수됨** (개선 효과 사라짐).
5. 로그: backup/diag_enc_split.log, backup/diag_enc_split.csv

**최종 범인 요약**: Wave B FAIL = **IDTargetEncoder(ID 인코딩) 전부** (missing 지시자 무죄, interact는 +55.36 순기능). 다음 단계: enc 제거(또는 시간 안정적 설계) + interact 유지 → 게이트 재검증.

## Plan A 게이트 검증 (2026-08-08) — base(47)+interact(3) 정식 파이프라인 PASS

**변경**: `bss_preprocess.py`에 InteractionAdder(모듈 레벨) 추가, `train_hgb.py` build_pipeline을 3단계(add_interact→to_cat→clf)로 확장. enc/missing 미사용 (ID 인코딩 FAIL 확정 + missing은 최소 변경 원칙). best_params overlay 비활성 유지.

**게이트 결과 (기본 HGB_KWARGS, 2019~2023 → 2024)**:

| 조건 | 컬럼 | n_iter_ | Brier(2024) | 2024 BSS |
|---|---|---|---|---|
| Wave A (base 47) | 47 | 247 | 0.248710 | **439.00** |
| **Plan A (base+interact 50)** | 50 | **177** | **0.248572** | **494.36 (+55.36)** |

- **PASS** (494.36 > 앵커 439.00) — `diag_stage_ablation.py`의 "+interact" 결과와 **정확히 일치** (정식 구현 검증 완료).
- cross-process pickle 왕복 QA 통과 (PICKLE_ROUNDTRIP_OK 0.545342).
- n_iter 247→177 감소 = 상호작용 피처가 수렴 촉진 (과적합 신호 완화).
- 로그: backup/hgb_validate_interact.log
- **다음 단계**: (1) --full 전체 재학습 + zip 재구성 + 평가 서버 시뮬레이션 → 제출, 또는 (2) 대회 후반에 튜닝과 함께 일괄 진행 (사용자 결정 대기).

## Plan A 전체 재학습 + zip 재구성 (2026-08-08)

- `--full` 전체(2019~2024) 재학습: 50컬럼 파이프라인, n_iter 381, 20.6s → `model/rf.pkl` 갱신
- zip 재구성: model/ + script.py + **bss_preprocess.py**(InteractionAdder import 필요) + requirements.txt (5항목, 695KB)
- 평가 서버 시뮬레이션: FORMAT_OK, row_id 순서 일치, 확률 [0.44, 0.55], 결측 0
- 예측 평균 0.4917 — 2025 base rate 추정(0.47~0.49)과 부합
- 추론 시간: 50k행 0.25s → 전체 245,789행 환산 1.23s (10분 예산 대비 여유 599초)
- 로그: backup/hgb_full_interact.log | 시뮬레이션: submit_sim_planA/
- **제출 대기**: baseline_submit.zip (베이스라인 실험용/) — 사용자가 DACON 업로드
