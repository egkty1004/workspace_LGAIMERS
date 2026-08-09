#!/usr/bin/env python3
"""
[Exp 6-c 배포] 전체 데이터 MLP 학습 + 전처리 저장 (2026-08-08)

전체 데이터(2019~2024)로 deeper 구성 Entity Embedding MLP 10시드 학습.
전처리(cat 코드 매핑, 수치 표준화)와 모델을 submit/model/ 에 저장.

저장:
  submit/model/mlp_prep.pkl  전처리 파라미터 (cats/cat_map/nums/nmean/nstd/vocab)
  submit/model/mlp_s{seed}.pt  torch state_dict (seed 42~51)
  submit/model/mlp_meta.json   구성/시드 정보
"""
import json
import os
import pickle
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
HIDDEN = [512, 256, 128, 64]
DROPOUT = 0.25
BATCH = 8192
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 30
PATIENCE = 5
SEEDS = list(range(42, 52))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
print(f"device: {DEVICE}", flush=True)
MODEL_DIR = "submit/model"


class EntityMLP(nn.Module):
    def __init__(self, cat_vocab, n_num):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(v, d) for v, d in zip(cat_vocab, EMB_DIM)])
        in_dim = n_num + sum(EMB_DIM)
        layers = []
        for h in HIDDEN:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(DROPOUT)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        embs = [e(x_cat[:, i]) for i, e in enumerate(self.embeds)]
        z = torch.cat([x_num] + embs, dim=1)
        return self.net(z).squeeze(-1)


def build_full():
    train, _ = common.load_train()
    common.preprocess_for_submission(train)
    feats = common.get_feature_cols(
        pd.read_csv("data/test.csv", encoding="utf-8-sig", nrows=0).columns)
    cats = [c for c in CATS if c in feats]
    nums = [c for c in feats if c not in cats]

    cat_vocab, cat_map = [], {}
    for c in cats:
        vals = sorted(train[c].astype(str).unique())
        cat_map[c] = {v: i + 1 for i, v in enumerate(vals)}
        cat_vocab.append(len(vals) + 1)
    nmean = {c: float(train[c].mean()) for c in nums}
    nstd = {c: (float(train[c].std()) if train[c].std() > 0 else 1.0) for c in nums}

    prep = dict(cats=cats, cat_map=cat_map, nums=nums, nmean=nmean, nstd=nstd,
                cat_vocab=cat_vocab)
    with open(os.path.join(MODEL_DIR, "mlp_prep.pkl"), "wb") as f:
        pickle.dump(prep, f)

    def to_num(df):
        return np.stack([(df[c].fillna(nmean[c]).values.astype(np.float32) - nmean[c]) / nstd[c]
                         for c in nums], axis=1)

    def to_cat(df):
        return np.stack([df[c].astype(str).map(lambda v: cat_map[c].get(v, 0)).values
                         for c in cats], axis=1)

    ytr = train[common.TARGET].values.astype(np.float32)
    return (torch.tensor(to_num(train)), torch.tensor(to_cat(train)), ytr,
            cat_vocab, len(nums))


def main():
    t0 = time.time()
    print("[Exp 6-c 배포] 전체 데이터 MLP 학습", flush=True)
    Xn, Xc, ytr, cat_vocab, n_num = build_full()
    print(f"full: {len(Xn)} | vocab: {cat_vocab} | nums: {n_num}", flush=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    yt = torch.tensor(ytr).to(DEVICE)
    n = len(Xn)
    meta = dict(seeds=SEEDS, emb_dim=EMB_DIM, hidden=HIDDEN, dropout=DROPOUT,
                lr=LR, max_epochs=MAX_EPOCHS, patience=PATIENCE, unk_p=UNK_P,
                cats=CATS, batch=BATCH, weight_decay=WEIGHT_DECAY)
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = EntityMLP(cat_vocab, n_num).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        lossf = nn.BCEWithLogitsLoss()
        best_loss, best_state, patience = 1e9, None, 0
        for ep in range(MAX_EPOCHS):
            model.train()
            idx = torch.randperm(n)
            tot = 0.0
            for s in range(0, n, BATCH):
                bi = idx[s:s + BATCH]
                xb, cb, yb = Xn[bi].to(DEVICE), Xc[bi].to(DEVICE), yt[bi]
                # UNK dropout for pitcher/batter
                for col in (0, 1):
                    cc = cb[:, col].clone()
                    cc[torch.rand_like(cc, dtype=torch.float) < UNK_P] = 0
                    cb[:, col] = cc
                opt.zero_grad()
                z = model(xb, cb)
                loss = lossf(z, yb)
                loss.backward()
                opt.step()
                tot += loss.item() * len(bi)
            if tot / n < best_loss:
                best_loss = tot / n
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= PATIENCE:
                    break
        torch.save(best_state, os.path.join(MODEL_DIR, f"mlp_s{seed}.pt"))
        print(f"  seed={seed} epochs={ep + 1} best_train_loss={best_loss:.4f}", flush=True)

    with open(os.path.join(MODEL_DIR, "mlp_meta.json"), "w") as f:
        json.dump(dict(meta, prep_vocab=cat_vocab, total_time=time.time() - t0), f, indent=2)
    print(f"\n저장 완료: submit/model/mlp_s*.pt (10개) + mlp_prep.pkl + mlp_meta.json "
          f"(총 {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
