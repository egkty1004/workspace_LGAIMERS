#!/usr/bin/env python3
"""
[Exp 6-c] MLP 블렌딩 4폴드 검증 (2026-08-08)

각 폴드에서 Entity Embedding MLP 3시드 로짓 평균 → LGB F3 로짓과 블렌딩.
가중치는 primary에서 정규화(단일 파라미터 w)로 최적화 → R-only 3폴드에
고정 가중치로 적용 (일반화 검증). per-fold 최적 가중치도 잠재력 참고로 보고.

폴드: primary(≤2023→2024) / r2022 / r2023 / r2024
결과: experiments/e6c_blend_results.json
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

CATS = ["pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
        "top_bottom", "game_type", "base_state", "platoon", "count_state"]
EMB_DIM = [16, 16, 4, 4, 2, 4, 4, 4, 4]
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

LGB_Z = {
    "primary": "cache/preds_primary_lgb_f3.npy",
    "r2022": "cache/h2b_r2022_lgb_f3.npy",
    "r2023": "cache/h2b_r2023_lgb_f3.npy",
    "r2024": "cache/h2b_r2024_lgb_f3.npy",
}


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
    print("[Exp 6-c] MLP 블렌딩 4폴드 검증", flush=True)
    train, _ = common.load_train()
    common.preprocess_for_submission(train)
    feats = common.get_feature_cols(
        pd.read_csv("open/data/test.csv", encoding="utf-8-sig", nrows=0).columns)
    cats = [c for c in CATS if c in feats]
    isR = train["game_type"] == "R"
    folds = {
        "primary": (train["season"] <= 2023, train["season"] == 2024),
        "r2022": ((train["season"] <= 2021) & isR, (train["season"] == 2022) & isR),
        "r2023": ((train["season"] <= 2022) & isR, (train["season"] == 2023) & isR),
        "r2024": ((train["season"] <= 2023) & isR, (train["season"] == 2024) & isR),
    }

    scores = {}
    z_mlp_all = {}
    for fn, (tr_m, va_m) in folds.items():
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
        np.save(f"cache/mlp_{fn}.npy", z_mlp)
        z_mlp_all[fn] = z_mlp
        z_lgb = np.load(LGB_Z[fn])
        s_l = common.score(common.sigmoid(z_lgb), yv)
        s_m = common.score(common.sigmoid(z_mlp), yv)
        corr = float(np.corrcoef(z_lgb, z_mlp)[0, 1])
        scores[fn] = dict(lgb=s_l, mlp=s_m, corr=corr, mlp_singles=bss)
        print(f"  lgb={s_l:.1f} mlp={s_m:.1f} corr={corr:.3f}", flush=True)

    # primary 가중치 (정규화, 단일 파라미터 w)
    yv_p = train.loc[folds["primary"][1], common.TARGET].values
    z_lp = z_mlp_all["primary"]
    z_ll = np.load(LGB_Z["primary"])

    def sc(w):
        return common.score(common.sigmoid(w * z_ll + (1 - w) * z_lp), yv_p)

    ws = np.linspace(0.5, 1.0, 51)
    best_w = ws[np.argmax([sc(w) for w in ws])]
    s_pri = sc(best_w)
    print(f"\nprimary 최적 w(lgb)={best_w:.3f} → blend={s_pri:.1f} (lgb 단독 {scores['primary']['lgb']:.1f}, "
          f"증분 {s_pri - scores['primary']['lgb']:+.1f})", flush=True)

    rows = {}
    for fn in ["r2022", "r2023", "r2024"]:
        yv = train.loc[folds[fn][1], common.TARGET].values
        z_lgb = np.load(LGB_Z[fn])
        z_mlp = z_mlp_all[fn]
        s_blend_fix = common.score(common.sigmoid(best_w * z_lgb + (1 - best_w) * z_mlp), yv)
        # per-fold 최적 (잠재력)
        wl = np.linspace(0.5, 1.0, 51)
        w_opt = wl[np.argmax([common.score(common.sigmoid(w * z_lgb + (1 - w) * z_mlp), yv) for w in wl])]
        s_blend_opt = common.score(common.sigmoid(w_opt * z_lgb + (1 - w_opt) * z_mlp), yv)
        rows[fn] = dict(lgb=scores[fn]["lgb"], mlp=scores[fn]["mlp"],
                        blend_fixed=s_blend_fix, gain_fixed=s_blend_fix - scores[fn]["lgb"],
                        w_opt=float(w_opt), blend_opt=s_blend_opt,
                        gain_opt=s_blend_opt - scores[fn]["lgb"])
        print(f"  {fn}: lgb={rows[fn]['lgb']:.1f} blend_fixed(w={best_w:.2f})={s_blend_fix:.1f} "
              f"gain={rows[fn]['gain_fixed']:+.1f} | opt(w={w_opt:.2f})={s_blend_opt:.1f} "
              f"gain={rows[fn]['gain_opt']:+.1f}", flush=True)

    g_pri = s_pri - scores["primary"]["lgb"]
    r_imp = sum(1 for fn in rows if rows[fn]["gain_fixed"] > 0)
    print(f"\nprimary gain={g_pri:+.1f} | R-only 개선 {r_imp}/3 (고정 가중치)", flush=True)
    adopt = g_pri >= 20 and r_imp >= 2
    print(f"판정: {'✅ MLP 블렌딩 채택 후보' if adopt else '❌ 채택 조건 미달'}", flush=True)

    out = dict(scores={fn: dict(lgb=s["lgb"], mlp=s["mlp"], corr=s["corr"],
                                mlp_singles=s["mlp_singles"]) for fn, s in scores.items()},
               primary_w=float(best_w), primary_blend=float(s_pri), primary_gain=g_pri,
               r_only={fn: rows[fn] for fn in rows}, r_only_improved=r_imp,
               adopt=adopt, total_time=time.time() - t0)
    with open("experiments/e6c_blend_results.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n저장: experiments/e6c_blend_results.json (총 {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
