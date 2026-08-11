#!/usr/bin/env python3
"""
e6c_blend_platoon3b2.py — count_platoon_3b2_same 단일 피처 LGB 10시드 + MLP 5시드 블렌딩 4폴드 검증.

배경: screen_all_10seed.py 10시드 강화 게이트에서 유일한 생존 후보 count_platoon_3b2_same
(R-only 3/3 & 전 폴드 양성, primary +4.9). F3 전체(base) 대비 해당 피처 추가(cand)의
10시드 LGB 재검증 + cand 구성 EntityMLP 5시드 블렌딩으로 최종 채택 여부를 판정한다.

- LGB: base = common.get_feature_cols(test_cols) (F3_EXTRA 10종 포함 57피처),
       cand = base + count_platoon_3b2_same. 10시드(42..51) 로짓 평균.
       cats = CAT_COLS + [platoon, count_state] (신규 int8은 numeric).
       캐시: cache/plato3b2_{base,cand}_{fold}_lgb.npy
- MLP: cand 구성만 5시드(42..46) EntityMLP. cats = e6c_blend_adopted.py CATS 11종
       (count_platoon_3b2_same은 numeric이라 cats 미포함). 캐시: cache/plato3b2_cand_{fold}_mlp.npy
- blend: z = w·z_lgb + (1−w)·z_mlp (로짓 공간), primary에서 w∈[0.5,1.0] 51점 탐색 →
         R-only 3폴드 고정 w 적용(blend_fixed) + per-fold 최적(잠재력) 병기.

강화 게이트 (10시드 전략):
  1. LGB gain gate: primary gain ≥ +15 AND R-only 3/3 (실패 서브조건 보고)
  2. corr gate: corr(lgb_cand, mlp_cand) < 0.95 전 폴드
  3. Δmean gate: max|Δmean| ≤ 0.005 전 폴드 (cand sigmoid mean − base sigmoid mean)
  4. Blend acceptance: primary blend 증분 > 0 AND R-only 3/3
  최종: 1~4 모두 충족 → 채택 | 그 외 → 기각 (실패 사유 포함)

--smoke: LGB 1시드(42) × primary 1폴드 + MLP 1시드 — 코드 경로/피처 추가 검증.
출력: experiments/e6c_blend_platoon3b2.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightgbm as lgb

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)
sys.path.insert(0, REPO)
import common  # noqa: E402

# ── MLP 구성 (e6c_blend_adopted.py 동일) ──
CATS = ["pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
        "top_bottom", "game_type", "base_state", "platoon", "count_state",
        "asof_n_bucket", "score_diff_binary"]
EMB_DIM = [16, 16, 4, 4, 2, 4, 4, 4, 4, 4, 2]
UNK_P = 0.05
HIDDEN = [512, 256, 128, 64]
DROPOUT = 0.25
BATCH = 8192
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 30
PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

# ── 실험 구성 ──
FEATURE = "count_platoon_3b2_same"
LGB_CATS = common.CAT_COLS + ["platoon", "count_state"]  # 신규 int8은 numeric
LGB_SEEDS = list(range(42, 52))  # 10시드
MLP_SEEDS = list(range(42, 47))  # 5시드
FOLDS = ["primary", "r2022", "r2023", "r2024"]
R_FOLDS = ["r2022", "r2023", "r2024"]
C_LOGIT = -0.0404  # 제출 정렬용 (검증 BSS는 적용 전 측정)

# 강화 게이트
GATE_PRIMARY = 15.0
GATE_R_ONLY = 3
CORR_THRESHOLD = 0.95
MEAN_POISON_THRESHOLD = 0.005


class EntityMLP(nn.Module):
    def __init__(self, cat_vocab, cat_dim, n_num, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(v, d) for v, d in zip(cat_vocab, cat_dim)])
        in_dim = n_num + sum(cat_dim)
        layers = []
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        embs = []
        for i, e in enumerate(self.embeds):
            x = x_cat[:, i]
            if i < 2:  # pitcher/batter UNK dropout
                x = x.clone()
                x[torch.rand_like(x, dtype=torch.float) < UNK_P] = 0
            embs.append(e(x))
        z = torch.cat([x_num] + embs, dim=1)
        return self.net(z).squeeze(-1)


def build_fold(train, feats, tr_m, va_m, cats):
    nums = [c for c in feats if c not in cats]
    tr, va = train[tr_m], train[va_m]
    ytr = tr[common.TARGET].values.astype(np.float32)
    yv = va[common.TARGET].values.astype(np.float32)

    cat_vocab = []
    codes = {}
    for c in cats:
        vals = sorted(tr[c].astype(str).unique())
        codes[c] = {v: i + 1 for i, v in enumerate(vals)}
        cat_vocab.append(len(vals) + 1)

    nmean = {c: float(tr[c].mean()) for c in nums}
    nstd = {c: (float(tr[c].std()) if tr[c].std() > 0 else 1.0) for c in nums}

    def to_num(df):
        out = np.zeros((len(df), len(nums)), dtype=np.float32)
        for j, c in enumerate(nums):
            col = df[c].fillna(nmean[c]).values.astype(np.float32)
            out[:, j] = (col - nmean[c]) / nstd[c]
        return out

    def to_cat(df):
        out = np.zeros((len(df), len(cats)), dtype=np.int64)
        for j, c in enumerate(cats):
            out[:, j] = df[c].astype(str).map(lambda v: codes[c].get(v, 0)).values
        return out

    Xn_tr, Xc_tr = torch.tensor(to_num(tr)), torch.tensor(to_cat(tr))
    Xn_va, Xc_va = torch.tensor(to_num(va)), torch.tensor(to_cat(va))
    return Xn_tr, Xc_tr, Xn_va, Xc_va, ytr, yv, cat_vocab, len(nums)


def train_seed(Xn_tr, Xc_tr, Xn_va, Xc_va, ytr, yv, cat_vocab, n_num, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = EntityMLP(cat_vocab, EMB_DIM, n_num).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lossf = nn.BCEWithLogitsLoss()
    yt = torch.tensor(ytr).to(DEVICE)
    n = len(Xn_tr)
    best_bss, best_z, patience = -1e9, None, 0
    for ep in range(MAX_EPOCHS):
        model.train()
        idx = torch.randperm(n)
        for s in range(0, n, BATCH):
            bi = idx[s:s + BATCH]
            xb, cb, yb = Xn_tr[bi].to(DEVICE), Xc_tr[bi].to(DEVICE), yt[bi]
            opt.zero_grad()
            z = model(xb, cb)
            loss = lossf(z, yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            zv = model(Xn_va.to(DEVICE), Xc_va.to(DEVICE)).cpu().numpy()
            bss = common.score(common.sigmoid(zv), yv)
        if bss > best_bss:
            best_bss, best_z, patience = bss, zv.copy(), 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break
    return best_bss, best_z


def run_lgb_fold(train, feats, tr_m, va_m, seeds):
    """한 구성·한 폴드 seeds 시드 로짓 평균."""
    X_tr, y_tr = train.loc[tr_m, feats], train.loc[tr_m, common.TARGET]
    X_va, y_va = train.loc[va_m, feats], train.loc[va_m, common.TARGET]
    zs = []
    for seed in seeds:
        params = dict(common.PARAMS)
        params["seed"] = seed
        dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=LGB_CATS)
        dva = lgb.Dataset(X_va, y_va, categorical_feature=LGB_CATS, reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        zs.append(common.logit(m.predict(X_va, num_iteration=m.best_iteration)))
    return np.mean(zs, axis=0)


def main():
    ap = argparse.ArgumentParser(description="count_platoon_3b2_same LGB10시드+MLP5시드 블렌딩 검증")
    ap.add_argument("--smoke", action="store_true",
                    help="스모크: LGB 1시드(42)×primary 1폴드 + MLP 1시드만 실행")
    args = ap.parse_args()

    t0 = time.time()
    lgb_seeds = list(LGB_SEEDS)
    mlp_seeds = list(MLP_SEEDS)
    folds_run = list(FOLDS)
    smoke = args.smoke
    if smoke:
        lgb_seeds = [42]
        mlp_seeds = [42]
        folds_run = ["primary"]
        print("[--smoke] 축소 실행: LGB 1시드 × primary 1폴드 + MLP 1시드", flush=True)

    print(f"[e6c_blend_platoon3b2] {FEATURE} LGB{len(lgb_seeds)}시드 + "
          f"MLP{len(mlp_seeds)}시드 블렌딩 4폴드 검증 | device: {DEVICE}", flush=True)
    train, _ = common.load_train()
    common.preprocess_for_submission(train)
    test_cols = pd.read_csv("open/data/test.csv", encoding="utf-8-sig", nrows=0).columns
    base_feats = common.get_feature_cols(test_cols)
    cand_feats = list(base_feats) + [FEATURE]
    cats = [c for c in CATS if c in cand_feats]
    print(f"train: {train.shape} | base_feats: {len(base_feats)} | "
          f"cand_feats: {len(cand_feats)} | cats: {len(cats)} | lgb_cats: {LGB_CATS}",
          flush=True)

    # ── 피처 추가 검증 (smoke 필수 확인 사항) ──
    assert FEATURE in train.columns, f"{FEATURE} 미생성"
    fcol = train[FEATURE]
    assert fcol.notna().all(), f"{FEATURE} NaN 존재"
    assert fcol.dtype in (np.int8, np.int64, np.int32), f"{FEATURE} dtype={fcol.dtype}"
    assert len(cand_feats) == len(base_feats) + 1, "cand_feats 크기 불일치"
    assert len(cand_feats) == len(set(cand_feats)), "cand_feats 중복"
    assert all(c in cand_feats for c in cats), "MLP cats 중 cand에 없는 컬럼"
    print(f"  feature OK: {FEATURE} dtype={fcol.dtype} NaN=0 "
          f"shape={train.shape[0]} (cand={len(cand_feats)}=base {len(base_feats)}+1)",
          flush=True)

    isR = train["game_type"] == "R"
    folds = {
        "primary": (train["season"] <= 2023, train["season"] == 2024),
        "r2022": ((train["season"] <= 2021) & isR, (train["season"] == 2022) & isR),
        "r2023": ((train["season"] <= 2022) & isR, (train["season"] == 2023) & isR),
        "r2024": ((train["season"] <= 2023) & isR, (train["season"] == 2024) & isR),
    }

    # ── 1. LGB: base & cand 10시드 캐시 생성 ──
    lgb_z = {"base": {}, "cand": {}}
    for cfg in ("base", "cand"):
        feats = base_feats if cfg == "base" else cand_feats
        for fn in folds_run:
            tr_m, va_m = folds[fn]
            t1 = time.time()
            z = run_lgb_fold(train, feats, tr_m, va_m, lgb_seeds)
            out = f"cache/plato3b2_{cfg}_{fn}_lgb.npy"
            np.save(out, z)
            lgb_z[cfg][fn] = z
            p = common.sigmoid(z)
            yv = train.loc[va_m, common.TARGET].values
            print(f"  [lgb/{cfg}/{fn}] BSS={common.score(p, yv):.1f} "
                  f"pred_mean={p.mean():.4f} 저장 {out} ({time.time()-t1:.0f}s)", flush=True)

    # ── 2. MLP: cand 구성 5시드 ──
    z_mlp_all = {}
    for fn in folds_run:
        tr_m, va_m = folds[fn]
        print(f"\n=== [mlp/{fn}] ===", flush=True)
        (Xn_tr, Xc_tr, Xn_va, Xc_va, ytr, yv, cat_vocab, n_num) = build_fold(
            train, cand_feats, tr_m, va_m, cats)
        zs, bss = [], []
        for seed in mlp_seeds:
            b, z = train_seed(Xn_tr, Xc_tr, Xn_va, Xc_va, ytr, yv, cat_vocab, n_num, seed)
            zs.append(z)
            bss.append(b)
            print(f"  seed={seed} bss={b:.1f}", flush=True)
        z_mlp = np.mean(zs, axis=0)
        np.save(f"cache/plato3b2_cand_{fn}_mlp.npy", z_mlp)
        z_mlp_all[fn] = z_mlp

    # ── per-fold 스코어 ──
    scores = {}
    for fn in folds_run:
        tr_m, va_m = folds[fn]
        yv = train.loc[va_m, common.TARGET].values
        z_base, z_cand, z_mlp = lgb_z["base"][fn], lgb_z["cand"][fn], z_mlp_all[fn]
        s_base = common.score(common.sigmoid(z_base), yv)
        s_cand = common.score(common.sigmoid(z_cand), yv)
        s_mlp = common.score(common.sigmoid(z_mlp), yv)
        corr = float(np.corrcoef(z_cand, z_mlp)[0, 1])
        scores[fn] = dict(
            lgb_base=float(s_base), lgb_cand=float(s_cand), mlp=float(s_mlp),
            corr=corr, gain_adopted=float(s_cand - s_base),
            pred_mean_base=float(common.sigmoid(z_base).mean()),
            pred_mean_cand=float(common.sigmoid(z_cand).mean()),
        )
        print(f"  [{fn}] lgb_base={s_base:.1f} lgb_cand={s_cand:.1f} "
              f"Δ={s_cand-s_base:+.1f} mlp={s_mlp:.1f} corr={corr:.3f}", flush=True)

    if smoke:
        z0 = lgb_z["cand"]["primary"]
        assert np.isfinite(z0).all(), "lgb cand 로짓에 NaN/Inf"
        assert np.isfinite(z_mlp_all["primary"]).all(), "mlp 로짓에 NaN/Inf"
        assert all(np.isfinite(lgb_z["base"][fn]).all() for fn in folds_run), "lgb base 로짓 NaN"
        ok = (len(cand_feats) == len(base_feats) + 1
              and fcol.notna().all()
              and "primary" in scores)
        print(f"\n[--smoke] {'PASS' if ok else 'FAIL'} — LGB base/cand + MLP "
              f"코드 경로·피처 추가·캐시 저장 검증 완료", flush=True)
        return 0 if ok else 1

    # ── 3. primary 최적 w 탐색 (로짓 공간, LGB-cand 단독 대비 증분 측정) ──
    yv_p = train.loc[folds["primary"][1], common.TARGET].values
    z_lc = lgb_z["cand"]["primary"]
    z_lp = z_mlp_all["primary"]

    def sc(w):
        return common.score(common.sigmoid(w * z_lc + (1 - w) * z_lp), yv_p)

    ws = np.linspace(0.5, 1.0, 51)
    best_w = ws[np.argmax([sc(w) for w in ws])]
    s_pri = sc(best_w)
    print(f"\nprimary 최적 w(lgb)={best_w:.3f} → blend={s_pri:.1f} "
          f"(lgb_cand 단독 {scores['primary']['lgb_cand']:.1f}, 증분 "
          f"{s_pri - scores['primary']['lgb_cand']:+.1f})", flush=True)

    # ── 4. R-only 고정 w + per-fold 최적 ──
    rows = {}
    for fn in R_FOLDS:
        yv = train.loc[folds[fn][1], common.TARGET].values
        z_lgb = lgb_z["cand"][fn]
        z_mlp = z_mlp_all[fn]
        s_blend_fix = common.score(common.sigmoid(best_w * z_lgb + (1 - best_w) * z_mlp), yv)
        wl = np.linspace(0.5, 1.0, 51)
        w_opt = wl[np.argmax([common.score(common.sigmoid(w * z_lgb + (1 - w) * z_mlp), yv)
                              for w in wl])]
        s_blend_opt = common.score(common.sigmoid(w_opt * z_lgb + (1 - w_opt) * z_mlp), yv)
        rows[fn] = dict(lgb=scores[fn]["lgb_cand"], mlp=scores[fn]["mlp"],
                        blend_fixed=float(s_blend_fix),
                        gain_fixed=float(s_blend_fix - scores[fn]["lgb_cand"]),
                        w_opt=float(w_opt), blend_opt=float(s_blend_opt),
                        gain_opt=float(s_blend_opt - scores[fn]["lgb_cand"]))
        print(f"  {fn}: lgb={rows[fn]['lgb']:.1f} blend_fixed(w={best_w:.2f})="
              f"{s_blend_fix:.1f} gain={rows[fn]['gain_fixed']:+.1f} | "
              f"opt(w={w_opt:.2f})={s_blend_opt:.1f} gain={rows[fn]['gain_opt']:+.1f}",
              flush=True)

    # ── 5. 강화 게이트 ──
    g_pri = scores["primary"]["gain_adopted"]
    r_imp = sum(1 for fn in R_FOLDS if scores[fn]["gain_adopted"] > 1e-9)
    corr_ok = all(scores[fn]["corr"] < CORR_THRESHOLD for fn in FOLDS)
    dmeans = {fn: scores[fn]["pred_mean_cand"] - scores[fn]["pred_mean_base"]
              for fn in FOLDS}
    max_dm = max(abs(v) for v in dmeans.values())
    mean_ok = max_dm <= MEAN_POISON_THRESHOLD

    lgb_gate_ok = (g_pri >= GATE_PRIMARY) and (r_imp >= GATE_R_ONLY)
    lgb_fail = []
    if g_pri < GATE_PRIMARY:
        lgb_fail.append(f"primary {g_pri:+.1f}<+{GATE_PRIMARY:.0f}")
    if r_imp < GATE_R_ONLY:
        lgb_fail.append(f"R-only {r_imp}/{GATE_R_ONLY}")
    print(f"\nLGB gain gate: primary={g_pri:+.1f} | R-only {r_imp}/3 | "
          f"corr<{CORR_THRESHOLD} {corr_ok} | maxΔmean={max_dm:.5f} "
          f"({'OK' if lgb_gate_ok else '실패: ' + ' '.join(lgb_fail)})", flush=True)

    g_blend_pri = s_pri - scores["primary"]["lgb_cand"]
    blend_r_imp = sum(1 for fn in R_FOLDS if rows[fn]["gain_fixed"] > 0)
    blend_ok = (g_blend_pri > 0) and (blend_r_imp >= GATE_R_ONLY)
    print(f"Blend gate: primary 증분={g_blend_pri:+.1f} | R-only {blend_r_imp}/3 "
          f"({'OK' if blend_ok else '실패'})", flush=True)

    failures = []
    if not lgb_gate_ok:
        failures.append("LGB gain gate: " + " & ".join(lgb_fail))
    if not corr_ok:
        failures.append(f"corr<{CORR_THRESHOLD} 미충족")
    if not mean_ok:
        failures.append(f"maxΔmean {max_dm:.5f}>{MEAN_POISON_THRESHOLD:.3f}")
    if not blend_ok:
        failures.append(f"blend 증분 {g_blend_pri:+.1f}≤0 또는 R-only {blend_r_imp}/3")
    gate_pass = not failures
    verdict = "채택" if gate_pass else "기각"
    verdict_reason = (
        f"10시드 LGB gain(primary {g_pri:+.1f}, R-only {r_imp}/3) & corr<{CORR_THRESHOLD} "
        f"{corr_ok} & maxΔmean {max_dm:.5f}≤{MEAN_POISON_THRESHOLD} & blend(primary 증분 "
        f"{g_blend_pri:+.1f}, R-only {blend_r_imp}/3) 전부 충족"
        if gate_pass else
        f"게이트 미달: {' | '.join(failures)}")
    print(f"\n판정: {verdict} — {verdict_reason}", flush=True)

    # ── 요약 표 ──
    print("\n" + "=" * 110, flush=True)
    print(f"  {'fold':<9s}{'lgb_base':>10s}{'lgb_cand':>10s}{'Δ':>8s}"
          f"{'mlp':>10s}{'corr':>8s}{'blend_fixed':>12s}{'gain':>8s}", flush=True)
    print("-" * 110, flush=True)
    for fn in FOLDS:
        s = scores[fn]
        if fn == "primary":
            bf = s_pri
            gf = g_blend_pri
        else:
            bf = rows[fn]["blend_fixed"]
            gf = rows[fn]["gain_fixed"]
        print(f"  {fn:<9s}{s['lgb_base']:>10.1f}{s['lgb_cand']:>10.1f}"
              f"{s['gain_adopted']:>+8.1f}{s['mlp']:>10.1f}{s['corr']:>8.3f}"
              f"{bf:>12.1f}{gf:>+8.1f}", flush=True)
    print("=" * 110, flush=True)

    # ── JSON 저장 ──
    out = dict(
        adopted_features=[FEATURE],
        blend_formula="z = w*z_lgb_cand + (1-w)*z_mlp (로짓 공간)",
        c_logit=C_LOGIT,
        c_logit_note="제출 정렬용 상수 — 검증 BSS는 적용 전(raw sigmoid) 측정",
        seeds=dict(lgb=lgb_seeds, mlp=mlp_seeds),
        folds=FOLDS,
        scores={fn: dict(lgb_base=s["lgb_base"], lgb_cand=s["lgb_cand"],
                         mlp=s["mlp"], corr=s["corr"], gain_adopted=s["gain_adopted"],
                         pred_mean_base=s["pred_mean_base"],
                         pred_mean_cand=s["pred_mean_cand"])
                for fn, s in scores.items()},
        dmean_per_fold=dmeans, max_abs_dmean=float(max_dm),
        primary_w=float(best_w), primary_blend=float(s_pri),
        primary_gain_blend=float(g_blend_pri),
        r_only={fn: rows[fn] for fn in rows},
        r_only_blend_improved=blend_r_imp,
        gate=dict(
            thresholds=dict(primary=GATE_PRIMARY, r_only=f"{GATE_R_ONLY}/3",
                            corr=CORR_THRESHOLD, max_abs_dmean=MEAN_POISON_THRESHOLD),
            gain_primary=float(g_pri), r_only_improved=r_imp,
            corr_ok=bool(corr_ok), mean_ok=bool(mean_ok),
            lgb_gate_ok=bool(lgb_gate_ok), lgb_fail=lgb_fail,
            blend_gain_primary=float(g_blend_pri),
            blend_r_only_improved=blend_r_imp, blend_ok=bool(blend_ok),
            pass_=bool(gate_pass)),
        verdict=verdict, verdict_reason=verdict_reason,
        total_time=time.time() - t0,
    )
    with open("experiments/e6c_blend_platoon3b2.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n저장: experiments/e6c_blend_platoon3b2.json (총 {time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
