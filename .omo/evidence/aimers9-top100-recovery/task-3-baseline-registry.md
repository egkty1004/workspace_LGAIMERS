# Todo 3 — reproduce reconciled v93 6-leg baseline — PASS (exit 0)

- **recorded_at_utc**: 2026-08-18T00:53:40+00:00
- **git_head**: fb2f49c14f34e9995306842d490b4718bb6964cc
- **config_hash**: `158be42401fc5ce3fec78af960900a4d7c040fa08d960a4849113608eeb4b420`
- **label_sources**: ['r2022', 'r2023'] (labels_read=True)

## Checks

- **[PASS]** baseline_r2022: BSS=1929.7791 Brier=0.245175 r=0.5008 pred_mean=0.4985 n=30000
- **[PASS]** baseline_r2023: BSS=1884.7226 Brier=0.244940 r=0.4812 pred_mean=0.4863 n=30000

## Findings

- **[PASS]** hash_addressed_cache: baseline 로짓은 해시 주소 캐시에만 저장 (cache_key=ee8ea9579969deaa…)
- **[PASS]** selection_labels_only: r2022/r2023 선택 라벨만 읽음 — primary 미로드 (구조적 파이어월)

## Verdict: **PASS** (exit 0)
