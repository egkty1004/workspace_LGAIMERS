# Todo 9 — Top-100 제출 의사결정 프로토콜 (compact report)

- **Plan**: `aimers9-top100-score-improvement` Todo 9
- **Runner**: `repro_979/submission_decision.py` (기본 `--check` / `--register-qualified`)
- **State**: `repro_979/leaderboard_state.json` (커밋 대상 — 사용자가 live 리더보드 확인 후 갱신)
- **Evidence**: `.omo/evidence/aimers9-top100/task-9-submission.{json,md}` (gitignore 대상)
- **Run date**: 2026-08-14 | **Candidate**: `champ_cat` (`691a2947b2b1879c`)
- **Package**: `repro_979/submit_champ_cat_20260814-0929/` (Todo 8, 배포 QA 10/10 PASS)

## 프로토콜 (제출 행동 전 반드시 통과)

```
1. 사용자가 live DACON 리더보드를 확인 → leaderboard_state.json 의
   rank_100_cutoff / date_captured / current_best_public_score 갱신
   (컷오프는 사용자 확인 값. 플랜 TL;DR 의 1066.47756 은 역사 참조 — 하드코딩 금지)
2. Todo 8 PASS 후 1회:  python3 repro_979/submission_decision.py --register-qualified
   → 자격 다이제스트를 상태에 등록 (게이트 b의 입력)
3. 제출 행동 전:        python3 repro_979/submission_decision.py
   → ALLOW(exit 0) 일 때만 업로드 가능. BLOCK(exit 1)이면 사유 확인 후 조치
4. 실제 점수 확보 후:   사용자가 Notion SSOT 행을 append (아래 페이로드 템플릿)
   + submissions_by_date[today]=1, last_submission 기록
```

### 5개 게이트 (모두 충족 시에만 ALLOW)

| gate | 조건 | BLOCK 사유 |
|---|---|---|
| a-package-pass | Todo 8 검증기 `result == "PASS"` (+상태 `package_validated != false`) | `task8_evidence_missing` / `package_not_pass` / `package_validated_false` |
| b-qualification-digest | 등록 다이제스트 == 현재 패키지 재계산 + `model/*` 33건 sha256 재대조 | `qualification_digest_not_registered` / `qualification_digest_mismatch` / `model_files_changed` |
| c-cutoff-freshness | `date_captured` 가 `CUTOFF_TTL_HOURS`(=24h) 이내 | `cutoff_unknown`(BLOCKING) / `cutoff_no_date` / `cutoff_stale` |
| d-daily-slot | `submissions_by_date[오늘] == 0` (하루 1회) | `daily_slot_exhausted` |
| e-target-margin | `rank_100_cutoff − current_best_public_score > 0` 또는 `allow_below_cutoff: true` | `margin_unknown`(BLOCKING) / `margin_not_positive` |

**자격 다이제스트** (결정적, `431a8c93…`):
`sha256(canonical JSON {candidate_id, candidate, blend weights, c_logit, clip, seeds,
model_file_sha256(sorted), package_validator_result, package_recorded_at})` —
`qualification_runner.py` 의 정규화 JSON 컨벤션 재사용. task-8 증거를 재생성하면
날짜가 달라져 다이제스트 불일치 → 재자격 필요 (의도된 안티탬퍼).

**하루 단위 키**: Asia/Seoul 달력일 (`zoneinfo` 불가 시 시스템 로컬, 실패 시 UTC 폴백).
**exit code**: 0 = ALLOW, 1 = BLOCK(정책), 2 = 치명적 입력 오류.

## 현재 의사결정 상태 (2026-08-14)

- **champ_cat qualified**: Todo 8 검증기 PASS, 자격 다이제스트 `431a8c93f6c3cf1d…`
  등록됨 (상태 파일 `qualified_package`), 모델 33건 해시 일치 확인.
- **BLOCK 유지 중 (exit 1)**: `cutoff_unknown` — live rank-100 컷오프를 사용자가
  리더보드에서 확인해 `rank_100_cutoff`/`date_captured` 를 채우기 전까지는
  사용자 제출 행동을 허용하지 않는다. 그 후 `--check` 재실행으로 ALLOW 확인.
- **하루 1회 정책**: 오늘(2026-08-14, Asia/Seoul) 제출 0회 — 슬롯 사용 가능.
- ⚠️ 자동 업로드 없음. 본 프로토콜은 결정 기록만 생성하며, DACON 업로드는
  사용자만 수행한다. Notion 행 append 역시 사용자 행동 (아래 템플릿 제공).

## Notion SSOT 행 페이로드 템플릿 (사용자 append용)

페이지 `3b55ed6b-28d5-81a0-80da-fe91eec240c6` (실험 기록 SSOT). 실제 제출 점수가
나온 뒤 아래 JSON 을 각 테이블에 행으로 append 한다. `Private` 는 대회 종료 후 갱신.

### 1) 📊 로컬 CV 테이블 (columns: 날짜·모델·검증 세트·BSS·Brier·r·비고)

```json
{
  "date": "2026-08-14",
  "model": "champ_cat (691a2947b2b1879c) — champion×0.85 + catboost×0.15",
  "validation_set": "primary(선택 폴드 775.75) / R-only r2022,r2023,r2024",
  "bss": "primary 775.75 / R-only Δ +1.5,+2.0,+5.6",
  "brier": "(옵션) 폴드별 mean((p-y)^2) 있는 경우",
  "r": "(옵션) 검증 세트 평균 제구 성공률",
  "note": "Todo 7 수락 블렌드; bootstrap LB5% +2.06; C_LOGIT=-0.0404; clip 0.30/0.70"
}
```

### 2) 🏆 리더보드 테이블 (columns: 제출일·파일/모델·Public·Private·비고, δ/C_LOGIT 포함)

```json
{
  "date": "2026-08-14",
  "file_model": "submit_champ_cat_20260814-0929 (champ_cat, 691a2947b2b1879c)",
  "public": "<실제 Public score>",
  "private": "<대회 종료 후 갱신>",
  "note": "δ=<Public − 979.31(챔피언)> / C_LOGIT=-0.0404 / clip 0.30,0.70 / "
          "runtime=<script.py 소요초>s / verdict=<수락·기각>"
}
```

> 제출 후 상태 파일도 함께 갱신: `submissions_by_date[오늘] = 1`,
> `last_submission = {date, file_model, public_score, verdict}`, 판정 후
> `qualified_candidate_public_score` / `current_best_public_score` 갱신.

## leaderboard_state.json 사용법 (live 리더보드 확인 후)

```bash
# 1) live 리더보드(https://dacon.io/competitions/official/236743/leaderboard)에서
#    rank-100 컷오프 + 내 최고 Public 점수를 확인
# 2) repro_979/leaderboard_state.json 수정:
#    - rank_100_cutoff: <확인한 값>        (예: 1070.0)
#    - date_captured:   <확인 시각 ISO>     (예: 2026-08-14T01:00:00+09:00)
#    - current_best_public_score: <내 최고 Public>
# 3) 허용 확인:
python3 repro_979/submission_decision.py          # ALLOW(exit 0) 시에만 업로드
# 4) 추가 게이트: Todo 8 PASS + 자격 다이제스트 등록은 이미 완료(상태 파일에 기록됨)
```

- 컷오프 미확인 = `cutoff_unknown` **BLOCK** (경고가 아님).
- TTL: `date_captured` 부터 24시간(`CUTOFF_TTL_HOURS`) — 그 이후엔 재확인 필요.
- 마진 ≤ 0 인데 꼭 제출해야 하면 상태 파일에 `allow_below_cutoff: true` (명시적 오버라이드).

## Failure QA (검증 기록)

| 시나리오 | 실행 | 기대 | 실제 |
|---|---|---|---|
| Happy (신선 컷오프 + 등록됨 + 오늘 0회) | `--check` (test state) | ALLOW exit 0 | **exit 0** |
| (1) 패키지 PASS 없음 (증거 result=FAIL) | `--task8-evidence <FAIL>` | BLOCK | **exit 1** (`package_not_pass`) |
| (1) 패키지 PASS 없음 (증거 부재) | `--task8-evidence <없음>` | BLOCK | **exit 1** (`task8_evidence_missing`) |
| (1) 상태 `package_validated=false` | `--check` | BLOCK | **exit 1** (`package_validated_false`) |
| (2) 컷오프 낡음 (captured 2일 전) | `--check` | BLOCK | **exit 1** (`cutoff_stale`) |
| (3) 오늘 1회 제출됨 | `--check` (submissions[today]=1) | BLOCK | **exit 1** (`daily_slot_exhausted`) |
| (4) 모델 파일 변조 | `--check` (m1.bin 변조) | BLOCK | **exit 1** (`model_files_changed`) |
| 기본 상태 (컷오프 미확인) | `--check` (leaderboard_state.json) | BLOCK | **exit 1** (`cutoff_unknown`, `margin_unknown`) |

## 제약 준수 (MUST NOT)

- 자동 업로드 없음 / 하루 1회 초과 시 BLOCK (게이트 d) / 기각 후보 챔피언화 없음.
- Notion API 호출 없음 — 리포트는 페이로드 템플릿만 제공, append 는 사용자 행동.
- 실시간 컷오프 미인지 상태를 값으로 위장하지 않음 (unknown = BLOCK).
- 기존 `repro_979/` 파일 미수정 — 신규 파일만 (`submission_decision.py`,
  `leaderboard_state.json`, 본 리포트).
