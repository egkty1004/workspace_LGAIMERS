# Aimers 9기 : 투구 제구 성공 확률 예측 AI 온라인 해커톤 (Phase 2)

> **대회 링크**: https://dacon.io/competitions/official/236743/overview/description
> **문서 정리일**: 2026-08-07 (수집 기준)

---

## 1. 대회 개요

| 항목 | 내용 |
|---|---|
| 대회명 | Aimers 9기 : 투구 제구 성공 확률 예측 AI 온라인 해커톤 |
| 주최 | LG AI 연구원 |
| 주관 | 데이콘 (DACON) |
| 참여 | 한경닷컴 |
| 유형 | 알고리즘 · 코드제출 · 정형 · 스포츠(야구) |
| 참가 자격 | LG Aimers 9기 교육생 누구나 (Phase 1 이수 여부 무관) |
| 참가 인원 | 1,491명 (확인 시점 기준) |

### 대회 배경
- 야구에서 투수의 제구력은 실점 억제, 볼카운트 운영, 타자 대응 전략에 직접 영향을 주는 핵심 요소
- 기존에는 경기 후 집계 지표(평균자책점, 볼넷 수, 스트라이크 비율)로 평가했지만, 실제 경기에서는 매 투구 직전의 **볼카운트, 주자 상황, 타자·투수 특성, 과거 투구 이력** 등이 복합적으로 작용
- 투구가 이루어지기 **전까지 확인 가능한 정보만**으로 제구 성공 가능성을 예측하는 AI 모델링이 목표
- 트랙맨(Trackman) 데이터(2019~2024년 과거 투구 특성)는 보조 데이터로 제공

### 주제
**투구 단위의 제구 성공 확률 예측 AI 모델 개발**

### 문제 정의
- 테스트 데이터의 각 투구에 대해 `control_success`의 **확률값**을 예측
- 학습 데이터의 `control_success`: 제구 성공 = `1`, 제구 실패 = `0` (이진 레이블)
- **투구 이전 시점에서 활용 가능한 정보만**을 바탕으로 예측 모델 설계

### 제구 성공/실패 정의 (Phase 2 기준, 공의 위치 기반)
다음 3가지 경우는 **제구 실패**:
1. 스트라이크존 가운데 부근으로 들어간 공
2. 스트라이크존에서 크게 벗어난 공
3. 포수의 요구 방향과 반대로 들어간 공

그 외 유효한 투구는 **제구 성공**.

---

## 2. 대회 구조 (Phase 1/2/3)

| Phase | 내용 |
|---|---|
| **Phase 1** | 온라인 AI 교육 (이수 여부와 관계없이 Phase 2 참가 가능) |
| **Phase 2** | 온라인 해커톤 (본 문서) — 문제 해결 능력 검증, Phase 3 진출자(약 100명) 선발 |
| **Phase 3** | 오프라인 해커톤 — 1박 2일, 9/19~9/20 예정, 야구 데이터 기반 AI 문제 |

> ⚠️ **Phase 3 참가 조건**: Phase 1과 Phase 2를 **모두 이수한 교육생만** 참가 가능.
> Phase 1 미이수 시 Phase 2에서 우수한 성적을 거두어도 Phase 3 참가 불가.

---

## 3. 대회 일정 (2026)

| 날짜 | 일정 |
|---|---|
| 2026-08-05(수) 10:00 | 대회 시작 |
| 2026-08-26(수) 23:59 | 팀 병합 마감 |
| 2026-09-01(화) 10:00 | 리더보드 제출 마감 (대회 종료 1일 전) |
| 2026-09-02(수) 10:00 | 대회 종료 |
| 2026-09-02(수) 12:00 ~ 09-07(월) 10:00 | 코드 및 PPT 제출 |
| 2026-09-07(월) 10:00 ~ 09-11(금) | 코드 검증 |
| 2026-09-14(월) 10:00 | 오프라인 해커톤(Phase 3) 진출자 발표 |
| 2026-09-19 ~ 09-20 | 오프라인 해커톤(Phase 3) 개최 (예정) |

> ※ 세부 일정은 대회 운영 상황에 따라 변동될 수 있음.

---

## 4. 데이터 소개

**배포용 데이터 구조 (`open.zip`):**

```
open.zip
├── baseline_submit.zip      # 베이스라인 코드 기반 리더보드 제출 파일(zip, 참고용)
├── data/
│   ├── train.csv            # 학습 입력 및 정답 (1,475,092행 × 49컬럼)
│   ├── test.csv             # 평가 입력 데이터 (형식 확인용 5건 샘플, 48컬럼)
│   ├── sample_submission.csv # 제출 양식 (형식 확인용 5행 × 2컬럼)
│   └── trackman_history.csv  # 2019~2024년 Trackman 과거 로그 (1,793,078행 × 30컬럼)
└── data_description.md      # 데이터 설명서 (반드시 확인)
```

**⚠️ 중요 사항:**
- 배포용 `test.csv`는 **형식 확인용 샘플 5건만** 포함. 실제 평가 데이터는 **245,789행**으로 비공개
- 평가 서버에서 동일한 경로·컬럼 구조의 실제 평가 데이터로 교체되어 처리됨
- 실제 평가 시 `test.csv` 245,789행 수와 동일한 제출 양식으로 `submission.csv` 생성 필요
- 데이터는 본 대회 참여 목적으로만 사용 가능

---

## 5. 평가 방식

### 리더보드 평가지표: **Brier Skill Score (BSS)**

확률 예측 과제 — 추론 확률이 실제 정답에 가까울수록 높은 점수.

```
Brier Score = mean((p_i - y_i)^2)
r = mean(y_i)                     # 전체 평가 데이터의 평균 제구 성공률 (비공개)
평균 제구율 Brier Score = r × (1 - r)

Score = max(0, 100000 × (1 - Brier Score / 평균 제구율 Brier Score))
```

- `p_i`: i번째 샘플의 제구 성공 예측 확률
- `y_i`: i번째 샘플의 실제 정답 (0, 1)
- **Public Score** = 전체 테스트 데이터 100%
- **Private Score** = 대회 종료 시점의 Public Score

### LG Aimers 수료 조건
- Phase 1 이수 + Phase 2 Public Score **549.51 이상** (베이스라인 추론 코드 점수 기준)

### 1차 평가
- 리더보드 **Private Score 100%** (동점자는 기존 리더보드 순위 산정 방식 따름)

### 2차 평가 (Phase 3 진출)
- Private 리더보드 상위팀(약 100명)은 **코드 및 PPT 필수 제출** 대상
- 코드 제출 + 코드 검증 모두 통과한 상위팀(약 100명)이 오프라인 해커톤(Phase 3) 진출

---

## 6. 코드 제출 대회 (submit.zip)

`submit.zip` 업로드 방식. **디렉토리/파일 명 반드시 일치 필수.**

### 제출 파일 구조
```
submit.zip
├── model/              # 모델 가중치 파일 (예: model.pt 등)
├── script.py           # 실제 추론 실행 코드 (평가 서버에서 자동 실행)
└── requirements.txt    # 필요한 패키지 및 버전 (pip install -r requirements.txt 가능해야 함)
```

### 평가 서버에서 자동 추가되는 항목
```
submit.zip
├── model/              # 참가자 구성
├── script.py           # 참가자 구성
├── requirements.txt    # 참가자 구성
├── data/               # 실제 평가 데이터 (읽기전용, 수정 불가)
└── output/submission.csv  # 예측 결과 저장 경로 (script.py가 반드시 생성)
```

### 제약 조건 (중요!)
| 항목 | 제한 |
|---|---|
| 추론 실행 시간 | ≤ 10분 (245,789개 샘플) |
| 패키지 설치 시간 | ≤ 10분 |
| 제출 파일 용량 | ≤ 10GB (압축 해제 후 ≤ 32GB) |
| 인터넷 | ❌ 패키지 설치 외 외부 연결/다운로드 불가 |
| 평가 환경 | 6 vCPU, 28GB RAM, L4 GPU(VRAM 22.4GiB) |

### 평가 서버 사양
- OS: Ubuntu 22.04.5 LTS
- GPU: NVIDIA L4 (VRAM 22.4GiB), CUDA 12.8
- CPU: 6 vCPU, RAM 28GB
- Python: 3.11.15
- 인터넷 접속: 비활성화

### 기본 설치 패키지 (requirements.txt에 포함하지 말 것 권장)
```
torch==2.7.1+cu128
pandas==2.0.3
numpy==1.26.4
scipy==1.15.3
scikit-learn==1.8.0
joblib==1.5.3
threadpoolctl==3.6.0
narwhals==2.21.2
transformers==4.46.3
accelerate==1.9.0
sentencepiece==0.1.99
regex==2023.12.25
tqdm==4.66.4
loguru==0.7.2
pyyaml==6.0.1
rich==13.7.1
```

> 버전이 명시된 기본 설치 패키지의 다른 버전을 요구하면 설치 에러 발생 가능. **기본 설치 패키지를 활용하고 requirements.txt에는 포함하지 않을 것 권장.**

### 시스템 패키지
git, build-essential, python3.11(+dev/venv), python3-pip, libffi-dev, libblas3, liblapack3, libomp-dev, tzdata, unzip, p7zip-full, gfortran, libatlas-base-dev, default-jre-headless, cmake, pkg-config, ninja-build, libgl1, libglib2.0-0

### 오류 유형 (일일 제출 횟수 반영 기준 — 반드시 숙지)
| 유형 | 내용 | 일일 제출 횟수 반영 |
|---|---|---|
| **설치 오류** | zip 구조 불일치, 패키지 설치 오류 | ❌ 반영 안 됨 |
| **제출 오류** | script.py 실행 후 발생하는 모든 오류 | ✅ 반영됨 |

- `script.py`에서 `data/` 로드, `output/submission.csv`로 예측 결과 저장 필수
- 인터넷 불가 → 패키지 설치 이후 외부 다운로드가 필요한 코드/모델은 동작하지 않음

---

## 7. 대회 규칙

### 참여 규칙
- 개인 또는 팀(최대 **5명**)으로 참가 가능
- 동일인이 개인 또는 복수 팀에 중복 등록 불가
- 팀 병합 마감: 2026-08-26 23:59

### 핵심 제한
1. **사전학습모델**: 공개 + 비상업적 이용 허용 라이선스(MIT, Apache 2.0 등) 모델만 사용 가능
2. **외부 API 금지**: OpenAI API, Gemini API 등 원격 서버 기반 API 모델 사용 불가. 모든 작업은 로컬에서 재현 가능해야 함
3. **외부 데이터 금지**: 제공 공식 데이터 외 외부 데이터 사용 불가
4. **추론 원칙**: test.csv 각 행은 독립적 예측 대상. 다른 행/전체 데이터 분포를 이용한 보정은 인정되지 않음

### 코드 및 PPT 제출 규칙 (Phase 3 진출 희망 팀)
- 기한 내 `dacon@dacon.io`로 제출
- 코드 확장자: .py, .ipynb / 인코딩: UTF-8
- 필수 제출물:
  - (필수) 학습 코드 개발 환경(OS) 및 라이브러리 버전
  - (필수) Private Score 재현용 학습 코드 (추론 코드는 리더보드 제출 코드로 대체)
  - (필수) 자유 형식 솔루션 PPT
  - (필수) 팀원들의 오프라인 해커톤(Phase 3) 참가 여부 기재

### 유의 사항
- 1일 최대 제출 횟수: **5회**
- 사용 언어: Python
- Private 리더보드 랭킹은 최종 순위가 아니며, **코드 검증 후 Phase 3 진출자 결정**
- 평가 데이터 유출 등 규칙 위반 시 실격
- 제출 시 평가 서버에서 측정·기록된 점수만 공식 평가 결과로 인정

---

## 8. 상금

### 오프라인 해커톤(Phase 3) 총 상금 1,000만 원
| 순위 | 상금 | 상명 |
|---|---|---|
| 🥇 1위 | 500만 원 | 고용노동부장관상 |
| 🥈 2위 | 300만 원 | LG AI연구원장상 |
| 🥉 3위 | 200만 원 | 한경닷컴 사장상 |

> - 온라인 해커톤(Phase 2) 자체에는 **별도 상금 없음** (Phase 3 진출 선발 과정)
> - Phase 3 상위 3팀에게 상금 수여
> - 제세공과금은 개인 부담

---

## 9. 코드 공유 현황 (2026-08-07 확인 기준)

코드 공유 페이지: https://dacon.io/competitions/official/236743/codeshare

현재 공유된 게시물은 **2건 (공식 데이콘 베이스라인)** — 모두 데이콘 공식 계정(DACON.KMS)이 게시:

| 게시물 | 게시일 | 조회수 | 링크 |
|---|---|---|---|
| [Baseline/Train] RandomForest를 활용한 모델 학습 및 피쳐엔지니어링 (학습) | 2026-08-04 | 1,597 | https://dacon.io/competitions/official/236743/codeshare/14147 |
| [Baseline/Inference] RandomForest를 활용한 모델 학습 및 피쳐엔지니어링 (추론) | 2026-08-04 | 1,391 | https://dacon.io/competitions/official/236743/codeshare/14146 |

### 내용 요약
- **모델**: RandomForest (scikit-learn)
- **구성**: 학습(Train) 코드와 추론(Inference) 코드 2개로 분리
  - Train: 피쳐엔지니어링 + 모델 학습 (배포용 `open.zip`의 `baseline_submit.zip`과 동일 계열)
  - Inference: 학습된 모델 로드 → `data/`에서 테스트 데이터 로드 → `output/submission.csv` 생성
- **목적**: 모든 참가자의 '제출'을 목표로 하는 공식 베이스라인
- **참고**: 게시물의 실제 코드는 페이지에서 다운로드 가능 (로그인 필요). 배포용 데이터의 `baseline_submit.zip`과 동일한 베이스라인 코드로 보임

> 💡 **활용 팁**: 이 베이스라인 점수(LB 549.51)가 LG Aimers 수료 기준이므로, 최소한 이 코드로 첫 제출을 완료하고 이후 성능 개선을 진행하는 것을 권장.

---

## 10. 참고 링크

| 항목 | 링크 |
|---|---|
| 대회 페이지 | https://dacon.io/competitions/official/236743/overview/description |
| 평가 탭 | https://dacon.io/competitions/official/236743/overview/evaluation |
| 규칙 탭 | https://dacon.io/competitions/official/236743/overview/rules |
| 데이터 | https://dacon.io/competitions/official/236743/data |
| 코드 공유 | https://dacon.io/competitions/official/236743/codeshare |
| 코드 제출 가이드 | https://cfiles.dacon.co.kr/competitions/236564/guide.html |
| 토크(질문) | https://dacon.io/competitions/official/236743/talkboard |
| 리더보드 | https://dacon.io/competitions/official/236743/leaderboard |
| 문의 메일 | dacon@dacon.io |

---

## 11. 참가자 체크리스트 (핵심 요약)

- [ ] 참여 버튼 클릭 후 성명 + LG Aimers 가입 메일 기재 (데이콘 계정 메일과 달라도 됨)
- [ ] Phase 3 진출 희망 시 → **Phase 1 이수 필수** (Phase 2 성적과 무관)
- [ ] `data_description.md` 반드시 확인
- [ ] 1일 5회 제출 한도 관리, 리더보드 제출 마감(09/01 10:00) 준수
- [ ] submit.zip 구조(`model/`, `script.py`, `requirements.txt`) 정확히 일치
- [ ] 추론 10분 / 설치 10분 / 10GB 제한 내 확인
- [ ] 기본 설치 패키지 버전 준수 (requirements.txt 불필요한 중복 금지)
- [ ] 인터넷 미사용(오프라인) 환경에서 동작 검증
- [ ] 수료 기준: Public Score ≥ 549.51 (베이스라인 이상)
- [ ] 외부 데이터/API/비공개 모델 사용 금지
