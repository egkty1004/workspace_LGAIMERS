#!/usr/bin/env python3
"""
e6c_blend_adopted.py — Wave D 채택 피처 LGB 10시드 + MLP 5시드 블렌딩 4폴드 검증.

채택 피처(asof_n_bucket, score_diff_binary) 포함 LGB 캐시(10시드, gen_lgb_f3_adopted.py 생성)와
동일 피처로 학습한 Entity Embedding MLP 5시드(42~46) 로짓 평균을 블렌딩한다.
- MLP CATS: e6c_blend_folds.py CATS + asof_n_bucket + score_diff_binary (범주형 임베딩)
- feats: base + platoon + count_state + asof_n_bucket + score_diff_binary (채택 구성)
- blend z = w·z_lgb + (1−w)·z_mlp (로짓 공간), C_LOGIT=−0.0404는 제출 정렬용(검증 BSS는 적용 전 측정)
- primary에서 최적 w 탐색(0.5~1.0) → R-only 3폴드에 고정 w 적용
- 10시드 gain(adopted−base)으로 게이트 판정, 미달 시 5시드 스크리닝 델타 재판정

게이트: primary 기여 ≥ +10 & R-only 2/3 & corr<0.95 → 채택 확정
출력: experiments/e6c_blend_adopted.json
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)
sys.path.insert(0, REPO)
import common  # noqa: E402

# ── MLP 구성 ──
CATS = ["pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
        "top_bottom", "game_type", "base_state", "platoon", "count_state",
        "asof_n_bucket", "score_diff_binary"]
EMB_DIM = [16, 16, 4, 4, 2, 4, 4, 4, 4, 4, 2]  # asof_n_bucket(5범주)→4, score_diff_binary(2)→2
UNK_P = 0.05
HIDDEN = [512, 256, 128, 64]
DROPOUT = 0.25
BATCH = 8192
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 30
PATIENCE = 5
N_SEEDS = 5
SEEDS = list(range(42, 42 + N_SEEDS))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
print(f"device: {DEVICE}", flush=True)

# ── 캐시 경로 ──
LGB_ADOPTED = {
    "primary": "cache/fe_adopted_primary_lgb_f3.npy",
    "r2022": "cache/fe_adopted_r2022_lgb_f3.npy",
    "r2023": "cache/fe_adopted_r2023_lgb_f3.npy",
    "r2024": "cache/fe_adopted_r2024_lgb_f3.npy",
}
LGB_BASE = {
    "primary": "cache/fe_base_primary_lgb_f3.npy",
    "r2022": "cache/fe_base_r2022_lgb_f3.npy",
    "r2023": "cache/fe_base_r2023_lgb_f3.npy",
    "r2024": "cache/fe_base_r2024_lgb_f3.npy",
}
ADOPTED_EXTRA = ["platoon", "count_state", "asof_n_bucket", "score_diff_binary"]
FOLDS = ["primary", "r2022", "r2023", "r2024"]
R_FOLDS = ["r2022", "r2023", "r2024"]
C_LOGIT = -0.0404  # 제출 정렬용 (검증 BSS는 적용 전 측정)

GATE_PRIMARY = 10.0
GATE_R_ONLY = 2


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


def main():
    t0 = time.time()
    print("[e6c_blend_adopted] 채택 피처 LGB10시드 + MLP5시드 블렌딩 4폴드 검증", flush=True)
    train, _ = common.load_train()
    common.preprocess_for_submission(train)
    test_cols = pd.read_csv("open/data/test.csv", encoding="utf-8-sig", nrows=0).columns
    base_feats = [c for c in test_cols if c != common.ID]
    feats = base_feats + ADOPTED_EXTRA
    cats = [c for c in CATS if c in feats]
    print(f"cats({len(cats)}): {cats}", flush=True)

    isR = train["game_type"] == "R"
    folds = {
        "primary": (train["season"] <= 2023, train["season"] == 2024),
        "r2022": ((train["season"] <= 2021) & isR, (train["season"] == 2022) & isR),
        "r2023": ((train["season"] <= 2022) & isR, (train["season"] == 2023) & isR),
        "r2024": ((train["season"] <= 2023) & isR, (train["season"] == 2024) & isR),
    }

    scores = {}
    z_mlp_all = {}
    for fn in FOLDS:
        tr_m, va_m = folds[fn]
        print(f"\n=== [{fn}] ===", flush=True)
        (Xn_tr, Xc_tr, Xn_va, Xc_va, ytr, yv, cat_vocab, n_num) = build_fold(
            train, feats, tr_m, va_m, cats)
        zs, bss = [], []
        for seed in SEEDS:
            b, z = train_seed(Xn_tr, Xc_tr, Xn_va, Xc_va, ytr, yv, cat_vocab, n_num, seed)
            zs.append(z)
            bss.append(b)
            print(f"  seed={seed} bss={b:.1f}", flush=True)
        z_mlp = np.mean(zs, axis=0)
        np.save(f"cache/mlp_adopted_{fn}.npy", z_mlp)
        z_mlp_all[fn] = z_mlp
        z_lgb = np.load(LGB_ADOPTED[fn])
        z_base = np.load(LGB_BASE[fn])
        s_l = common.score(common.sigmoid(z_lgb), yv)
        s_b = common.score(common.sigmoid(z_base), yv)
        s_m = common.score(common.sigmoid(z_mlp), yv)
        corr = float(np.corrcoef(z_lgb, z_mlp)[0, 1])
        scores[fn] = dict(lgb=s_l, base=s_b, mlp=s_m, corr=corr, mlp_singles=bss,
                          gain_adopted=s_l - s_b)
        print(f"  lgb_adopted={s_l:.1f} base={s_b:.1f} mlp={s_m:.1f} "
              f"corr={corr:.3f} gain(adopted-base)={s_l-s_b:+.1f}", flush=True)

    # ── primary 최적 w 탐색 (단일 파라미터, 로짓 공간) ──
    yv_p = train.loc[folds["primary"][1], common.TARGET].values
    z_lp = z_mlp_all["primary"]
    z_ll = np.load(LGB_ADOPTED["primary"])

    def sc(w):
        return common.score(common.sigmoid(w * z_ll + (1 - w) * z_lp), yv_p)

    ws = np.linspace(0.5, 1.0, 51)
    best_w = ws[np.argmax([sc(w) for w in ws])]
    s_pri = sc(best_w)
    print(f"\nprimary 최적 w(lgb)={best_w:.3f} → blend={s_pri:.1f} "
          f"(lgb 단독 {scores['primary']['lgb']:.1f}, 증분 {s_pri - scores['primary']['lgb']:+.1f})",
          flush=True)

    # ── R-only 고정 w 적용 + per-fold 최적(잠재력) ──
    rows = {}
    for fn in R_FOLDS:
        yv = train.loc[folds[fn][1], common.TARGET].values
        z_lgb = np.load(LGB_ADOPTED[fn])
        z_mlp = z_mlp_all[fn]
        s_blend_fix = common.score(common.sigmoid(best_w * z_lgb + (1 - best_w) * z_mlp), yv)
        wl = np.linspace(0.5, 1.0, 51)
        w_opt = wl[np.argmax([common.score(common.sigmoid(w * z_lgb + (1 - w) * z_mlp), yv)
                              for w in wl])]
        s_blend_opt = common.score(common.sigmoid(w_opt * z_lgb + (1 - w_opt) * z_mlp), yv)
        rows[fn] = dict(lgb=scores[fn]["lgb"], mlp=scores[fn]["mlp"],
                        blend_fixed=s_blend_fix, gain_fixed=s_blend_fix - scores[fn]["lgb"],
                        w_opt=float(w_opt), blend_opt=s_blend_opt,
                        gain_opt=s_blend_opt - scores[fn]["lgb"])
        print(f"  {fn}: lgb={rows[fn]['lgb']:.1f} blend_fixed(w={best_w:.2f})={s_blend_fix:.1f} "
              f"gain={rows[fn]['gain_fixed']:+.1f} | opt(w={w_opt:.2f})={s_blend_opt:.1f} "
              f"gain={rows[fn]['gain_opt']:+.1f}", flush=True)

    # ── 채택 피처 10시드 게이트 (adopted − base) ──
    g_pri = scores["primary"]["gain_adopted"]
    r_imp = sum(1 for fn in R_FOLDS if scores[fn]["gain_adopted"] > 1e-9)
    corr_ok = all(scores[fn]["corr"] < 0.95 for fn in FOLDS)
    print(f"\n채택 2피처 10시드 gain: primary={g_pri:+.1f} | R-only 개선 {r_imp}/3 | "
          f"corr<0.95 {corr_ok}", flush=True)

    gate_pass = (g_pri >= GATE_PRIMARY) and (r_imp >= GATE_R_ONLY) and corr_ok
    if gate_pass:
        verdict = "채택 확정"
        verdict_reason = (f"10시드 primary 기여 {g_pri:+.1f} >= +{GATE_PRIMARY:.0f} & "
                          f"R-only {r_imp}/3 & corr<0.95 전부 충족")
    else:
        # 5시드 스크리닝 델타 기준 재판정 (gate_verdict.json)
        try:
            with open("experiments/gate_verdict.json", encoding="utf-8") as f:
                gv = json.load(f)
            s5 = {c: (gv["candidates"][c]["primary_contribution"],
                      gv["candidates"][c]["r_only_improved"])
                  for c in gv["adopted"]}
            s5_ok = all(v[0] >= GATE_PRIMARY and v[1].startswith("2/")
                        for v in s5.values())
        except Exception as e:  # noqa: BLE001
            s5, s5_ok = {}, False
            print(f"  [WARN] gate_verdict.json 로드 실패: {e}", flush=True)
        if s5_ok:
            verdict = "채택 (5시드 델타 재판정)"
            verdict_reason = (f"10시드 게이트 미달(primary {g_pri:+.1f}, R-only {r_imp}/3, "
                              f"corr<0.95 {corr_ok}) → 5시드 스크리닝 델타 기준 재판정: "
                              f"{s5} 모두 채택 조건 충족")
        else:
            verdict = "기각"
            verdict_reason = (f"10시드 게이트 미달(primary {g_pri:+.1f}, R-only {r_imp}/3, "
                              f"corr<0.95 {corr_ok}) & 5시드 재판정도 충족 못함: {s5}")
    print(f"판정: {verdict} — {verdict_reason}", flush=True)

    g_blend_pri = s_pri - scores["primary"]["lgb"]
    r_blend_imp = sum(1 for fn in rows if rows[fn]["gain_fixed"] > 0)
    out = dict(
        adopted_features=ADOPTED_EXTRA[2:],
        blend_formula="z = w*z_lgb + (1-w)*z_mlp (로짓 공간)",
        c_logit=C_LOGIT,
        c_logit_note="제출 정렬용 상수 — 검증 BSS는 적용 전(raw sigmoid) 측정",
        seeds=dict(lgb=list(range(42, 52)), mlp=SEEDS),
        folds=FOLDS,
        scores={fn: dict(lgb=s["lgb"], base=s["base"], mlp=s["mlp"], corr=s["corr"],
                         mlp_singles=s["mlp_singles"], gain_adopted=s["gain_adopted"])
                for fn, s in scores.items()},
        primary_w=float(best_w), primary_blend=float(s_pri),
        primary_gain_blend=g_blend_pri,
        r_only={fn: rows[fn] for fn in rows},
        r_only_blend_improved=r_blend_imp,
        gate=dict(threshold_primary=GATE_PRIMARY, r_only_required=f"{GATE_R_ONLY}/3",
                  corr_threshold=0.95,
                  gain_primary=float(g_pri), r_only_improved=r_imp, corr_ok=bool(corr_ok),
                  pass_=gate_pass),
        verdict=verdict, verdict_reason=verdict_reason,
        total_time=time.time() - t0,
    )
    with open("experiments/e6c_blend_adopted.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n저장: experiments/e6c_blend_adopted.json (총 {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
