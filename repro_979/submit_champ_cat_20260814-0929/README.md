# champ_cat 제출 패키지 — Todo 8 (2026-08-14)

DACON Aimers 9기 투구 제구 성공 확률 예측 (평가지표: Brier Skill Score).
Todo 7 수락 블렌드 **`champ_cat`** (candidate_id `691a2947b2b1879c`)의 배포 패키지.

## 구성

```
submit_champ_cat_<ts>/
├── script.py          # 추론 스크립트 (V5 champ_cat, env 오버라이드 지원)
├── common.py          # 공용 전처리/유틸 (챔피언 GIHO extract와 동일, byte 동일)
├── mlp_model.py       # MLP 추론 모듈 (챔피언 GIHO extract와 동일, byte 동일)
├── requirements.txt   # lightgbm==4.7.0, catboost==1.2.10
├── provenance.json    # 후보/가중치/모델 sha256/롤백 참조
├── README.md          # 본 파일
├── model/             # 추론 아티팩트 (git 제외, zip 포함)
│   ├── f3_s42..s51.txt          # 챔피언 LGB 10시드 (byte-frozen GIHO)
│   ├── mlp_s42..s51.pt          # 챔피언 MLP 10시드 (byte-frozen GIHO)
│   ├── mlp_prep.pkl, mlp_meta.json, train_meta.json
│   └── catboost_s42..s51.cbm    # CatBoost full-data 10시드 (Task 8 재학습)
└── output/submission.csv # 실행 시 생성
```

## 실행

평가 서버 구조(`data/test.csv`, `data/sample_submission.csv` 읽기전용, `output/`에
`submission.csv` 생성)에 맞춰 설계:

```bash
python3 script.py
# env 오버라이드 (로컬 검증용)
LGA_TEST_PATH=<test.csv> LGA_SAMPLE_PATH=<sample.csv> LGA_OUT_PATH=<out.csv> python3 script.py
```

파이프라인: `utf-8-sig` 로드 → `common.preprocess_for_submission`
→ LGB/MLP/CatBoost 10시드 로짓 평균 → 로짓 공간 블렌드
`z = 0.85*z_champion + 0.15*z_catboost` (`z_champion = 0.51*z_lgb + 0.49*z_mlp`)
→ `C_LOGIT=-0.0404` → `clip(0.30, 0.70)` → `output/submission.csv`.

## ⚠️ 챔피언 롤백 지시 (중요)

catboost 설치 실패/평가 오류 시 즉시 롤백:

```
경로: team_member_materials/GIHO/submit979_extract/  (byte-frozen, 979.31 실측)
실행: script.py (V4, LGB×0.51 + MLP×0.49, C_LOGIT=-0.0404, clip 0.30/0.70)
requirements.txt: lightgbm==4.7.0 (catboost 불요)
```

롤백 참조 해시는 `provenance.json.champion_rollback`에 기록됨. 이 패키지는
GIHO extract를 **절대 수정하지 않고** 모델 파일을 복사(copy)해서만 사용한다.

## 패키지 결정 근거 (요약)

- **오프라인 설치 시뮬레이션**: catboost 1.2.10 / lightgbm 4.7.0 휠을
  `pip install --no-index --find-links <local-wheel-dir>` (fresh venv)로 설치 성공.
  설치 시간 9.8s (한도 600s). numpy 1.26.4 / pandas 2.0.3 / scipy 1.15.3 (평가 기본) 유지 확인.
  import/버전/학습/predict/모델 reload 전부 통과 → **catboost 배포 등급 viable** (Task 5의 install_risk 해소).
- **승자**: `champ_cat` (Todo 7 수락, R-only ΔBSS +1.45/+2.00/+5.56, bootstrap 5% 하한 +2.06).
- **catboost full-data 모델**: 시즌 2019-2024 전체(1,475,092행) 학습, 10시드,
  Task 5 동일 파라미터(depth 7, lr 0.05, Bernoulli 0.8, thread 32), 고정 300 iter.
  2025 라벨/test 행 미사용 (누수 가드 통과).
- **패리티/런타임 검증**: `repro_979/package_validator_champ_cat.py` — 5행 + 245,789행
  (합성 픽스처, train 부트스트랩, control_success 제거) 런타임 < 600s, 참조 대비 max|Δ| < 1e-6,
  mean/clip 경계, zip 레이아웃. 상세: `.omo/evidence/aimers9-top100/task-8-package.{json,log}`.

## 검증 명령

```bash
python3 repro_979/migration_audit.py --strict          # 경로/데이터/챔피언 감사 (PASS 유지)
python3 repro_979/package_validator_champ_cat.py --full  # 5행 + 245,789행 + 패리티 + zip
```
