# F2 — 품질/누수 감사: aimers9-blend-catboost-next-slots (commit c3252a7)

- **일자**: 2026-08-14
- **감사자**: Sisyphus (read-only code + evidence audit; 훈련/검증기 실행 안 함)
- **범위**: `repro_979/catboost9_screen.py`(신규 528행), `package_validator_lgb_mlp_cat.py`(diff),
  `submit_lgb_mlp_cat_20260814-2258/{script.py, provenance.json, common.py, mlp_model.py, requirements.txt}`,
  참조 동결 `qualification_runner.py` / `model_family_runner.py` / `common.py`.
- **판정**: **APPROVE** (8/8 PASS; 결함 3건은 전부 비차단·저위험 관찰)

---

## Check 1 — 범주 강제(coercion) — **PASS**

- `_validate_cat_schema` (catboost9_screen.py:152-182): 각 범주 컬럼에 대해
  `is_int = is_integer_dtype`, `is_cat = isinstance(dtype, CategoricalDtype)` 검사 —
  둘 다 아니면 `problems.append("category/int 만 허용 (무성 coercion 금지)")` (:168-170),
  `n_nulls > 0` 이면 reject (:171-173). `problems` 존재 시 `RuntimeError` (fail-closed, :179-181).
  float/nulls → 무성 수치 변환 경로 없음.
- CAT_SCHEMA = `tuple(MLP_CATS)` (:76) + 모듈 레벨 assert `len==9` (:77-78) + `len(set)==9` unique (:79-80).
  MLP_CATS (qualification_runner.py:86-89) = pitcher_id, batter_id, pitcher_team_id, batter_team_id,
  top_bottom, game_type, base_state, platoon, count_state — **정확히 요구된 9종**.
- 증거: task-10 evidence `dtype_report` — 9종 전부 `int32`(ID 4종)/`category`(5종), `n_nulls_train=0` 전부.
  feature contract "MLP cats 9" 동결 아티팩트 READ 대조 PASS.
- 참고(정보): 9종 어서션은 모듈 assert + `_train_catboost9_fold` cat_idx assert(:203) — `python -O` 에서
  제거되나, MLP_CATS 를 조작하면 qualification_runner 의 구성 다이제스트 게이트(SystemExit, qr:164-173)가
  import 시 차단 → 구조적 방어 유지.

## Check 2 — 폴드 마스크 / 누수 — **PASS**

- `build_folds` = qualification_runner import 재사용 (catboost9_screen.py:62, 호출 :388;
  qr:237-245 — primary/r2022/r2023/r2024 시계열 정의 동일).
- `_check_leakage(folds, train)` 호출 (:389; qr:248-256) — 검증 마스크 2025 금지.
- **추가**: 학습 마스크 2025 명시 검사 (:393-394) — "학습 마스크에 2025 시즌 포함 — 누수!".
- 검증 행 수 동결 대조 (:395-397): `EXPECTED_N_VA = {primary:253507, r2022:217024, r2023:219839, r2024:223497}`
  (:89) — 불일치 시 `problems` → return 1 (fail-closed, :398-401).
- 증거: task-10 evidence `expected_n_va_matched` 4폴드 전부 `[True, 동결값]`,
  `leakage_guard.no_2025_in_any_validation_mask=True` 및 `no_2025_in_any_training_mask=True` 모두 실측 기록.
  task-11 에서 packed-mask sha256 로 동일 마스크 재증명 (`same_validation_masks=true`).

## Check 3 — 캐시 정체성 — **PASS** (관찰 1건)

- `_cache_digest` (catboost9_screen.py:134-149) 포함: `cat_schema`(list — 이름+순서 보존),
  `dtype_policy_version`, `features`(CHAMPION_FEATURES), `seeds`(sorted), `folds` + `fold_definitions`,
  `params_digest`(→ `_params_digest` :121-131: params_base/iterations/esr/od_type/use_best_model),
  `bss_policy`. — 감사 요구 6요소 전부.
- `_cache_guard` (:240-261): npy 존재+meta 부재 → RuntimeError (:247-249, stale);
  digest 불일치+`--overwrite-cache` 미지정 → RuntimeError (:258-261, fail-closed);
  digest 일치 → 재사용 허용. meta+npy 동일 런에 항상 재생성 → stale 재사용 구조적으로 불가.
- 실증: smoke(digest b22fc8a1…)와 full(digest a41e9b22…, seeds 42,47) 로그에서 시드 변경 → digest
  불일치 → `--overwrite-cache` 경고 경로 실제 발동. 캐시 meta 파일 digest a41e9b22… 일치 확인.
- **관찰 1(비차단·저위험)**: digest 일치 시 "캐시 재사용" 메시지(:252)가 출력되지만 호출부
  (:415-426)는 훈련을 건너뛰지 않고 항상 재학습 후 덮어씀 — 메시지가 사실과 다르고 계산 낭비.
  fail-closed 속성엔 영향 없음. **수정 제안**: digest 일치 시 훈련 스킵 + npy 재로드, 또는 메시지를
  "digest 일치 — 캐시 재생성" 으로 정정.

## Check 4 — 학습/배포 패리티 — **PASS**

- `_catboost_params` (:111-118) = `FAMILIES["catboost"]["params_base"]` (model_family_runner.py:134-146:
  Logloss/Logloss, lr 0.05, depth 7, l2_leaf_reg 3.0, Bernoulli 0.8, random_strength 1.0, thread 32)
  + `iterations = rounds = 4000` + `od_type="Iter"` + `od_wait = early_stopping_rounds = 100` —
  Task 5 `_train_catboost_fold` (model_family_runner.py:337-341)와 **동일 프로토콜**.
  `model.fit(..., use_best_model=True)` (:225) — Task 5 와 동일 (:343).
- 검증 BSS = raw sigmoid: `_evaluate_fold` `p = common.sigmoid(z)` (:273) → `common.score(p, yv)` (:280),
  per_seed 도 동일 (:234). **C_LOGIT 미적용** — evidence `bss_policy.c_logit_applied=False` 실측.

## Check 5 — 패키지 추론 — **PASS**

- script.py: `z = W_LGB*z_lgb + W_MLP*z_mlp + W_CAT*z_cat` (:125, 가중치 :41-43 = 0.30/0.35/0.35)
  → `+C_LOGIT` → `sigmoid` → `clip(0.30, 0.70)` (:128).
- CatBoost: `Pool(X, cat_features=CAT_FEATURES)` — CAT_FEATURES = **5종** (:48:
  top_bottom/game_type/base_state/platoon/count_state = LGB_CATS) → 동결 **5-cat 모델**과 일치
  (provenance.json `config.cat_features` 5종, `catboost_iterations: 300`); 기각된 9-cat 사용 안 함.
  열 이름 기반 cat_features 전달 — Task 8 훈련(Pool name 기반)과 동일 의미론.
- SEEDS = 42..51 10시드 (:46), `thread_count=CATBOOST_THREADS=6` (:49,:78 — 평가 6 vCPU 대응).
  실패 시 명시적 예외 + 챔피언 롤백 안내 (:65-68, :73-74) — 조용한 폴백 없음.
- 실증: task-8 evidence — 5행(open/data/test.csv 5행 픽스처 확인) + 245,789행 합성 픽스처 런타임
  25.9s(<600s), script vs 독립 참조 `max|Δ| = 5.6e-17` (<1e-6), clip 경계 준수, champion 구성요소 vs
  GIHO V4 동일 5.6e-17. 패키지 common.py/mlp_model.py 는 GIHO extract 와 byte-동일(diff rc=0).

## Check 6 — 검증기 무결성 — **PASS** (관찰 2건)

- 상수 (package_validator_lgb_mlp_cat.py:45-49): W_LGB 0.30 / W_MLP 0.35 / W_CAT 0.35 / C_LOGIT -0.0404
  / clip 0.30-0.70 — 패키지와 정합. diff(2006→2258)는 W_MLP/W_CAT 0.45/0.25→0.35/0.35, candidate_id,
  참조 공식 docstring, 기본 dir 만 변경 — **임계값 완화 없음** (MEAN_ALIGN_TOL 0.005, PARITY_TOL 1e-6,
  RUNTIME_BUDGET_S 600, MEAN_SANITY 유지).
- 참조 계산: `z = W_LGB*z_lgb + W_MLP*z_mlp + W_CAT*z_cat` (:246) → `sigmoid(z + C_LOGIT)` → clip (:247)
  — script.py 와 수식 동일, `thread_count=CATBOOST_THREADS` 일치 → 패리티 게이트가 참조 경로에서만
  통과하는 불일치 불가.
- provenance 게이트 (:344-361): `candidate_id == "5890a4c54f502c4e"` (:357) + hashes 전부 실해시 대조.
  증거: `hashed_files: 33, problems: []` — provenance.json `model_file_sha256` 33건 실측
  (catboost 10 + f3 10 + mlp 10 + mlp_meta/mlp_prep/train_meta), LGB/MLP 23건은 동결
  CHAMPION_FILES_SHA256(qr:94-119)과 **값 일치** (byte-copy 증명).
- **관찰 2(비차단·저위험)**: `model_file_sha256` 키가 없거나 빈 dict 면 hashes 루프가 0건 → candidate_id
  만 일치하면 PASS 가능 (count 미검증). zip_layout 게이트(:332, EXPECTED_MODEL_FILES 33개 정확 일치,
  증거 n_model_files=33)와 script.py 실행 게이트가 실방어 → 실질 위험 낮음. **수정 제안**:
  `assert len(hashes) == len(EXPECTED_MODEL_FILES)` 추가.

## Check 7 — 누수 / C_LOGIT 튜닝 부재 — **PASS**

- C_LOGIT: 세 파일 전부 상수 -0.0404 (validator:48, script.py:44, catboost9_screen docstring "무적용").
  튜닝/스윕 코드 없음 (grep: C_LOGIT 등장 7건 전부 상수/설명; `sweep` 은 주석 언급뿐).
  mean_alignment_vs_champion_policy 게이트(:424-438)는 C_LOGIT 변경 없이 |Δmean| ≤ 0.005 확인
  (증거 1.97e-4).
- 2025/rolling/expanding/frequency/test 행: 두 변경 .py 파일에서 `2025` 는 **누수 가드 검사**(catboost9
  :393-394)와 증거 필드(:483-484)뿐; rolling/expanding/frequency/test 행 사용 0건. catboost9_screen 은
  `common.load_train()` 만 로드 (test 미접촉). validator 의 합성 픽스처는 train 부트스트랩 +
  `control_success` 제거(:163) — 추론 런타임/패리티 검증 전용, 가중치 피드백 없음 (스코어 기반 의사결정에
  미사용; MEAN_SANITY 는 합리성 범위).
- task-11 승격 평가: "pure cached-OOF logit arithmetic; no training", blend transfer 는 screen 실패로
  `SKIPPED_BY_PROTOCOL` — R-only 라벨이 선택에 사용되지 않음.

## Check 8 — 스코프 크립 — **PASS**

- `git show c3252a7` 변경 파일 20건 = 증거 11건(task-8×2, task-9×2, task-10×4, task-11×3) +
  `catboost9_screen.py`(신규 러너) + `package_validator_lgb_mlp_cat.py`(상수 갱신) +
  `leaderboard_state.json`(qualified_package 등록: 5890a4c54f502c4e / digest e3e047dd… — 계획상
  등록 단계) + 제출 패키지 6건(script/common/mlp_model/requirements/README/provenance; model/ 는
  gitignore). 그 외 파일 변경 없음.

---

## 결함 요약 (전부 비차단)

| # | 위치 | 결함 | 심각도 | 제안 |
|---|---|---|---|---|
| 1 | catboost9_screen.py:252 vs :415-426 | digest 일치 시 "캐시 재사용" 메시지에도 불구 훈련 스킵 안 함(항상 재학습·덮어씀) — 오해 유발 로그 + 계산 낭비 | Low | digest 일치 시 훈련 스킵 + 캐시 재로드, 또는 메시지 정정 |
| 2 | package_validator:344-361 | hashes dict 가 비어 있으면 candidate_id 만으로 PASS 가능 (개수 미단언) | Low | `assert len(hashes) == 33` 추가 |
| 3 | catboost9_screen:77-80,203 | 9종/unique 어서션이 module assert — `python -O` 시 무효 | Info | qr 구성 다이제스트 게이트가 실방어 (조작 시 import 차단) |

## 검증된 핵심 근거 (실행 기록)

- task-8: 10/10 게이트 PASS (offline install sim, 5행, 245,789행 25.9s, parity 5.6e-17, mean 정렬,
  champion 구성요소 5.6e-17, zip 레이아웃 33파일, provenance 33건, migration_audit PASS).
- task-9: 제출 **BLOCK** (daily_slot_exhausted, unused_daily_slots 0) — 승인 게이트 아님.
- task-10 smoke(42,43) + full(42,47): integrity PASS, 4폴드 행 수 동결 일치, 누수 가드(train+val) 실측,
  dtype 계약 9종 int32/category·null 0.
- task-11: **REJECTED** — primary Δ-115.005 vs 5-cat(617.7551−732.7601), 잔차상관 0.999203,
  alignment 0.0038, `proceed_to_todo_7_8=false`; 기각 모델이 제출 패키지에 미사용(패키지는 5-cat .cbm).

## 최종 판정: **APPROVE**

- 범주 무성 coercion: 없음 (fail-closed 검증). 폴드/누수/행 수: 코드+증거 이중 강제.
- 캐시/패리티/검증기: stale 재사용 구조적 차단, 참조-스크립트 수식 동일, 임계값 미완화.
- C_LOGIT 동결 유지, 2025/test/rolling/expanding/frequency 사용 0건, 스코프 크립 없음.
- 관찰 3건은 비차단(저위험) — 승격 결정에 영향 없음.
