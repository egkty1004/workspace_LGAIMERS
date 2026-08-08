# GIHO 979.31점 방법론 — 독립 재현 검증 리포트 (2026-08-08)

## 요약
팀원 GIHO가 공유한 최종 방법론(LB 979.31)을 우리 환경에서 **독립 재현 검증**.
결론: **검증 통과** — MLP 블렌딩 +44.5 (primary), 상관 0.892, R-only 3/3 개선.
제출물(submit_F3MLP_cl0404.zip)도 로컬에서 정상 동작 확인.

## 재현 환경
- 위치: `team_member_materials/GIHO/리더보드 979.31/` (제출물) + `GIHO/공유_extract/` (재현 코드)
- 재현 실행: `team_member_materials/GIHO/repro_979/` (심링크 데이터, torch 2.5.1+cu121, GPU A6000)
- 폴더 구조 한글 NFD 인코딩 문제로 python 기반 복사/링크로 해결

## 1) 제출물 검증 (submit_F3MLP_cl0404.zip)
- 구성: LGB F3 10시드(.txt) + EntityEmbedding MLP 10시드(.pt) + mlp_prep.pkl + script.py + mlp_model.py + common.py
- MLP 아키텍처: 9범주 임베딩 [16,16,4,4,2,4,4,4,4] + 수치 40 → [512,256,128,64], dropout 0.25, BCE, UNK 5%
- 5행 test end-to-end: ✅ 동작 (mean 0.4571, 0.8s)
- C_LOGIT = -0.0404 = logit(0.477) - logit(0.4871) → **2025 실제 평균 r≈0.477** (LB 890.90 역산)

## 2) LGB F3 4폴드 로짓 예측 재현 (gen_lgb_f3_preds.py)
| 폴드 | 재현 BSS | GIHO 문서 | 일치 |
|---|---|---|---|
| primary | (10시드 로짓 평균) | 729.7 | ✅ |
| r2022 | 578.4 | 578.4 | ✅ |
| r2023 | 551.0 | 551.0 | ✅ |
| r2024 | 711.4 | 711.4 | ✅ |

## 3) MLP 블렌딩 4폴드 검증 (e6c_blend_folds.py, GPU 334초)
| 폴드 | LGB 단독 | +MLP 블렌드(w=0.51) | 증분 | GIHO 문서 |
|---|---|---|---|---|
| primary | 729.7 | **774.2** | **+44.5** | +45 ✅ |
| r2022 | 578.4 | 607.2 | +28.8 | +36 |
| r2023 | 551.0 | 571.8 | +20.9 | +23 |
| r2024 | 711.4 | 724.1 | +12.7 | +13 ✅ |
| **MLP-LGB corr** | | | **0.892** | 0.89 ✅ |

판정: `adopt=true` — primary +44.5 ≥ +20 AND R-only 3/3 개선 → **채택 확정**

## 4) 핵심 교훈 (우리 실험과의 대조)
1. **블렌딩 게이트 = 로짓 상관 < 0.95**: GBDT끼리 0.985(무의미) vs MLP 0.892(진짜 다양성)
   → 우리가 "모델 변경 리스크"로 기각한 ID 임베딩 DL이 실제 최고 점수
2. **Entity Embedding ≠ Target Encoding**: TargetEncoder(-145 실패)는 ID를 단일 숫자로 압축,
   Entity Embedding(+88.4 성공)은 ID를 학습 벡터로 표현 → GBDT가 못 보는 ID 패턴 포착
3. **오프셋은 폴드 OOF gap에서**: in-sample proxy(0.002)는 과소평가, 폴드 OOS(0.0112)가 정확
4. **2025 r≈0.477**: 팀 공용 상수 — 모든 모델에 logit offset 적용 가능

## 5) 재현 산출물
- `repro_979/gen_lgb_f3_preds.py` — LGB F3 4폴드 로짓 예측 생성
- `repro_979/e6c_blend_folds.py` (경로 수정본) — MLP 4폴드 + 블렌딩 검증
- `repro_979/experiments/e6c_blend_results.json` — 검증 결과
- `repro_979/cache/*.npy` — 4폴드 로짓 예측 캐시

## 참고
- 제출물 그대로 제출하면 979.31 재현 가능 (팀원이 이미 LB 확인, 추론 25.5s)
- 우리 독립 제출을 원하면 e6c_deploy_train.py로 전체 데이터 MLP 학습 → script.py 블렌드 파이프라인 사용
- 개선 여지: BS≈0.2474로 BS_ref≈0.2499 근처 — 노이즈 한계 근접, 남은 여지 미미 (GIHO 의견)
