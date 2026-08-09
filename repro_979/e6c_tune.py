#!/usr/bin/env python3
"""
[Exp 6-c-2] MLP 구성 튜닝 (primary, 1시드, 결정성) (2026-08-08)

후보 구성별 primary BSS 측정 후 최적 구성 선택.
결정성: cudnn.deterministic + torch.use_deterministic_algorithms.
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
UNK_P = 0.05
BATCH = 8192
WEIGHT_DECAY = 1e-4
PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {DEVICE}", flush=True)

CONFIGS = {
    "base":        dict(emb=[16, 16, 4, 4, 2, 4, 4, 4, 4], hidden=[256, 128, 64], lr=1e-3, ep=30, dp=0.25),
    "bigger":      dict(emb=[16, 16, 4, 4, 2, 4, 4, 4, 4], hidden=[512, 256, 128], lr=1e-3, ep=30, dp=0.25),
    "slower":      dict(emb=[16, 16, 4, 4, 2, 4, 4, 4, 4], hidden=[256, 128, 64], lr=5e-4, ep=45, dp=0.25),
    "emb32":       dict(emb=[32, 32, 4, 4, 2, 4, 4, 4, 4], hidden=[256, 128, 64], lr=1e-3, ep=30, dp=0.25),
    "deeper":      dict(emb=[16, 16, 4, 4, 2, 4, 4, 4, 4], hidden=[512, 256, 128, 64], lr=1e-3, ep=30, dp=0.25),
    "highdrop":    dict(emb=[16, 16, 4, 4, 2, 4, 4, 4, 4], hidden=[512, 256, 128], lr=1e-3, ep=30, dp=0.4),
}


class EntityMLP(nn.Module):
    def __init__(self, cat_vocab, cat_dim, n_num, hidden, dropout):
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
            if i < 2:
                x = x.clone()
                x[torch.rand_like(x, dtype=torch.float) < UNK_P] = 0
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
    tr, va = train[train["season"] <= 2023], train[train["season"] == 2024]
    ytr = tr[common.TARGET].values.astype(np.float32)
    yv = va[common.TARGET].values.astype(np.float32)
    cat_vocab, codes = [], {}
    for c in cats:
        vals = sorted(tr[c].astype(str).unique())
        codes[c] = {v: i + 1 for i, v in enumerate(vals)}
        cat_vocab.append(len(vals) + 1)
    nmean = {c: float(tr[c].mean()) for c in nums}
    nstd = {c: (float(tr[c].std()) if tr[c].std() > 0 else 1.0) for c in nums}

    def to_num(df):
        return np.stack([(df[c].fillna(nmean[c]).values.astype(np.float32) - nmean[c]) / nstd[c]
                         for c in nums], axis=1)

    def to_cat(df):
        return np.stack([df[c].astype(str).map(lambda v: codes[c].get(v, 0)).values
                         for c in cats], axis=1)

    return (torch.tensor(to_num(tr)), torch.tensor(to_cat(tr)),
            torch.tensor(to_num(va)), torch.tensor(to_cat(va)), ytr, yv,
            cat_vocab, len(nums))


def train_cfg(Xn_tr, Xc_tr, Xn_va, Xc_va, ytr, yv, cat_vocab, n_num, cfg, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = EntityMLP(cat_vocab, cfg["emb"], n_num, cfg["hidden"], cfg["dp"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=WEIGHT_DECAY)
    lossf = nn.BCEWithLogitsLoss()
    yt = torch.tensor(ytr).to(DEVICE)
    n = len(Xn_tr)
    best_bss, best_z, patience = -1e9, None, 0
    for ep in range(cfg["ep"]):
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
    return best_bss


def main():
    t0 = time.time()
    print("[Exp 6-c-2] MLP 구성 튜닝", flush=True)
    (Xn_tr, Xc_tr, Xn_va, Xc_va, ytr, yv, cat_vocab, n_num) = build_primary()
    print(f"train: {len(Xn_tr)} | val: {len(Xn_va)} | vocab: {cat_vocab} | nums: {n_num}",
          flush=True)
    res = {}
    for name, cfg in CONFIGS.items():
        bss = train_cfg(Xn_tr, Xc_tr, Xn_va, Xc_va, ytr, yv, cat_vocab, n_num, cfg)
        res[name] = dict(bss=float(bss), cfg=cfg)
        print(f"  {name:10s}: primary BSS = {bss:.1f}", flush=True)
    best = max(res, key=lambda k: res[k]["bss"])
    print(f"\n최적 구성: {best} ({res[best]['bss']:.1f})", flush=True)
    with open("experiments/e6c_tune_results.json", "w") as f:
        json.dump(dict(res=res, best=best, total_time=time.time() - t0), f, indent=2, default=float)
    print(f"저장: experiments/e6c_tune_results.json (총 {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
