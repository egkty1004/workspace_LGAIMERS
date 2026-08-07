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
