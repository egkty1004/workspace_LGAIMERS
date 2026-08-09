#!/usr/bin/env python3
"""
[Exp 6-c] Entity Embedding MLP — 상관 게이트용 (2026-08-08)

투수/타자/팀 ID + 범주형을 임베딩으로, 수치형은 표준화해 MLP [256,128,64].
primary(≤2023→2024) 폴드 학습 → 로짓 예측 저장 → LGB F3(10시드)와 로짓 공간
피어슨 상관 측정 (블렌딩 게이트: corr<0.95 유망 / >=0.98 기각).

unseen pitcher/batter 대비 학습 중 5% UNK 치환. NaN은 train 평균 채움.
BCE loss, AdamW, early stopping(val BSS, patience 5).

결과: experiments/e6c_results.json, cache/preds_primary_mlp.npy
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
import common  # noqa: E402

CATS = ["pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
        "top_bottom", "game_type", "base_state", "platoon", "count_state"]
EMB_DIM = [16, 16, 4, 4, 2, 4, 4, 4, 4]
UNK_P = 0.05
HIDDEN = [256, 128, 64]
DROPOUT = 0.25
BATCH = 8192
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 30
PATIENCE = 5
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
SEEDS = list(range(42, 42 + N_SEEDS))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {DEVICE}", flush=True)


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

    def forward(self, x_num, x_cat, unk_mask=None):
        embs = []
        for i, e in enumerate(self.embeds):
            x = x_cat[:, i]
            if unk_mask is not None and unk_mask[i]:
                x = x.clone()
                x[torch.rand_like(x, dtype=torch.float) < UNK_P] = 0  # 0 = UNK
            embs.append(e(x))
        z = torch.cat([x_num] + embs, dim=1)
        return self.net(z).squeeze(-1)


def build_primary():
    train, _ = common.load_train()
    common.preprocess_for_submission(train)
    feats = common.get_feature_cols(
        pd.read_csv("data/test.csv", encoding="utf-8-sig", nrows=0).columns)
    cats = [c for c in CATS if c in feats]
    nums = [c for c in feats if c not in cats]
    print(f"cats: {cats}\nnums: {nums} (n={len(nums)})", flush=True)

    tr_m = train["season"] <= 2023
    va_m = train["season"] == 2024
    tr, va = train[tr_m], train[va_m]
    yv = va[common.TARGET].values.astype(np.float32)
    ytr = tr[common.TARGET].values.astype(np.float32)

    # 범주 코드 매핑 (train만 적합, unseen → 0=UNK)
    cat_vocab = []
    codes = {}
    for c in cats:
        vals = sorted(tr[c].astype(str).unique())
        m = {v: i + 1 for i, v in enumerate(vals)}  # 0 = UNK
        cat_vocab.append(len(vals) + 1)
        codes[c] = m

    def to_cat(df):
        out = np.zeros((len(df), len(cats)), dtype=np.int64)
        for j, c in enumerate(cats):
            out[:, j] = df[c].astype(str).map(lambda v: codes[c].get(v, 0)).values
        return out

    # 수치 표준화 (train 적합, NaN = train 평균)
    nmean = {c: float(tr[c].mean()) for c in nums}
    nstd = {c: float(tr[c].std()) if tr[c].std() > 0 else 1.0 for c in nums}

    def to_num(df):
        out = np.zeros((len(df), len(nums)), dtype=np.float32)
        for j, c in enumerate(nums):
            col = df[c].fillna(nmean[c]).values.astype(np.float32)
            out[:, j] = (col - nmean[c]) / nstd[c]
        return out

    Xn_tr, Xc_tr = to_num(tr), to_cat(tr)
    Xn_va, Xc_va = to_num(va), to_cat(va)
    return (torch.tensor(Xn_tr), torch.tensor(Xc_tr),
            torch.tensor(Xn_va), torch.tensor(Xc_va), ytr, yv,
            cat_vocab, len(nums))


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
        tot = 0.0
        for s in range(0, n, BATCH):
            bi = idx[s:s + BATCH]
            xb, cb, yb = Xn_tr[bi].to(DEVICE), Xc_tr[bi].to(DEVICE), yt[bi]
            opt.zero_grad()
            z = model(xb, cb, unk_mask=[True, True] + [False] * (len(CATS) - 2))
            loss = lossf(z, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(bi)
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
    print(f"  seed={seed} best_epoch={ep + 1 - patience} best_bss={best_bss:.2f}", flush=True)
    return best_bss, best_z


def main():
    t0 = time.time()
    print("[Exp 6-c] Entity Embedding MLP", flush=True)
    (Xn_tr, Xc_tr, Xn_va, Xc_va, ytr, yv, cat_vocab, n_num) = build_primary()
    print(f"train: {len(Xn_tr)} | val: {len(Xn_va)} | cats: {len(cat_vocab)} "
          f"vocab={cat_vocab} | nums: {n_num}", flush=True)

    zs, bss_list = [], []
    for seed in SEEDS:
        bss, z = train_seed(Xn_tr, Xc_tr, Xn_va, Xc_va, ytr, yv, cat_vocab, n_num, seed)
        zs.append(z)
        bss_list.append(bss)
    z_ens = np.mean(zs, axis=0)
    p_ens = common.sigmoid(z_ens)
    bss_ens = common.score(p_ens, yv)
    print(f"\nMLP single BSS: {[round(b, 1) for b in bss_list]} | ensemble: {bss_ens:.1f}",
          flush=True)
    np.save("cache/preds_primary_mlp.npy", z_ens)

    z_lgb = np.load("cache/preds_primary_lgb_f3.npy")
    corr = float(np.corrcoef(z_lgb, z_ens)[0, 1])
    print(f"MLP-LGB 로짓 상관 = {corr:.4f}", flush=True)
    if corr < 0.95:
        verdict = "블렌딩 유망 (상관 < 0.95)"
    elif corr >= 0.98:
        verdict = "블렌딩 이득 없음 (>= 0.98)"
    else:
        verdict = "경계 (0.95~0.98) — 블렌딩 시도 후 판단"
    print(f"판정: {verdict}", flush=True)

    out = dict(seeds=SEEDS, bss_single=bss_list, bss_ensemble=float(bss_ens),
               corr_mlp_lgb=corr, verdict=verdict,
               cats=cats if 'cats' in dir() else None,
               total_time=time.time() - t0)
    with open("experiments/e6c_results.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n저장: experiments/e6c_results.json (총 {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
