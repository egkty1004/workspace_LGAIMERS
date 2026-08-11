#!/usr/bin/env python3
"""deploy_train_plato3b2.py — count_platoon_3b2_same 포함 F3 전체 데이터 10시드 학습 (LGB + MLP).

submit-to-test 실험 (2-stage gate: 절대 +15 bar 실패 → 2025 리더보드 1-submission live test).
count_platoon_3b2_same(3-2 & 동손, int8 numeric)을 F3 전체 데이터(2019~2024, 1,475,092행)에
추가해 LGB 10시드(42~51) + 챔피언 아키텍처 EntityMLP 10시드(42~51)를 학습한다.

- LGB: feats = base(common.get_feature_cols, F3_EXTRA 10 포함 57) + count_platoon_3b2_same = 58.
        cats = CAT_COLS + ["platoon", "count_state"] (신규 int8은 numeric).
        num_boost_round=109 고정 (챔피언 deploy와 동일 방식, full data는 early_stopping 없음).
- MLP: 챔피언 9범주(EMB_DIM=[16,16,4,4,2,4,4,4,4], HIDDEN=[512,256,128,64], DROPOUT=0.25,
        BATCH=8192, LR=1e-3, WD=1e-4, MAX_EPOCHS=30, PATIENCE=5, unk_p=0.05) + count_platoon_3b2_same
        을 추가 numeric 피처로 사용 (cats에 추가하지 않음 — numeric 유지).

저장:
  submit_plato3b2/model/f3_s{seed}.txt     lgb.Booster.save_model (10개)
  submit_plato3b2/model/mlp_s{seed}.pt     torch state_dict (10개)
  submit_plato3b2/model/mlp_prep.pkl       MLP 전처리 파라미터
  submit_plato3b2/model/mlp_meta.json      MLP 구성/시드 정보
  submit_plato3b2/model/train_meta.json    LGB 구성/피처/시드/라운드 정보
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
import common  # noqa: E402

FEATURE = "count_platoon_3b2_same"

# ── LGB ──
LGB_CATS = common.CAT_COLS + ["platoon", "count_state"]  # 신규 int8은 numeric
LGB_NUM_BOOST_ROUND = 109  # full data 고정 라운드 (GIHO/채택 deploy와 동일 방식)
SEEDS = list(range(42, 52))

# ── MLP (챔피언 아키텍처) ──
MLP_CATS = ["pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

MODEL_DIR = os.path.join(REPO, "submit_plato3b2", "model")


class EntityMLP(nn.Module):
    """챔피언 deploy(e6c_deploy_train.py)와 동일. UNK dropout은 학습 루프에서 적용."""

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


def build_feats(test_cols):
    """base 57(F3_EXTRA 10 포함) + count_platoon_3b2_same = 58."""
    feats = common.get_feature_cols(test_cols) + [FEATURE]
    assert len(feats) == len(set(feats)), "feats 중복"
    return feats


def train_lgb(train, feats):
    """전체 데이터 LGB 10시드 → f3_s{seed}.txt."""
    X = train[feats]
    y = train[common.TARGET]
    os.makedirs(MODEL_DIR, exist_ok=True)
    for seed in SEEDS:
        params = dict(common.PARAMS)
        params["seed"] = seed
        dtr = lgb.Dataset(X, y, categorical_feature=LGB_CATS)
        m = lgb.train(params, dtr, num_boost_round=LGB_NUM_BOOST_ROUND)
        path = os.path.join(MODEL_DIR, f"f3_s{seed}.txt")
        m.save_model(path)
        print(f"  [lgb] seed={seed} trees={m.num_trees()} 저장 {path}", flush=True)
    return X.shape[1]


def build_mlp_full(train, feats):
    """MLP 전처리(prep) + 전체 데이터 텐서. count_platoon_3b2_same은 numeric."""
    cats = [c for c in MLP_CATS if c in feats]
    nums = [c for c in feats if c not in cats]
    assert len(cats) == len(MLP_CATS), f"MLP cats 일부 누락: {set(MLP_CATS) - set(feats)}"
    assert FEATURE in nums, "count_platoon_3b2_same이 numeric에 없음"

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
            cat_vocab, len(nums), cats, nums)


def train_mlp(Xn, Xc, ytr, cat_vocab, n_num):
    """전체 데이터 MLP 10시드 → mlp_s{seed}.pt."""
    yt = torch.tensor(ytr).to(DEVICE)
    n = len(Xn)
    meta = dict(seeds=SEEDS, emb_dim=EMB_DIM, hidden=HIDDEN, dropout=DROPOUT,
                lr=LR, max_epochs=MAX_EPOCHS, patience=PATIENCE, unk_p=UNK_P,
                cats=MLP_CATS, batch=BATCH, weight_decay=WEIGHT_DECAY,
                extra_numeric_feature=FEATURE)
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
                # UNK dropout for pitcher/batter (champion과 동일)
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
        print(f"  [mlp] seed={seed} epochs={ep + 1} best_train_loss={best_loss:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    with open(os.path.join(MODEL_DIR, "mlp_meta.json"), "w") as f:
        json.dump(dict(meta, prep_vocab=cat_vocab, n_num=n_num, total_time=time.time() - t0),
                  f, indent=2)


def main():
    global t0
    t0 = time.time()
    print(f"[deploy_plato3b2] device: {DEVICE} | {FEATURE} 포함 전체 데이터 10시드 학습",
          flush=True)
    train, _ = common.load_train()
    common.preprocess_for_submission(train)
    test_cols = pd.read_csv(os.path.join(REPO, "open", "data", "test.csv"),
                            encoding="utf-8-sig", nrows=0).columns
    feats = build_feats(test_cols)
    base_n = len(common.get_feature_cols(test_cols))
    print(f"train: {train.shape} | feats: {len(feats)} (base {base_n} + {FEATURE}) | "
          f"lgb_cats: {LGB_CATS} | mlp_cats: {len(MLP_CATS)}", flush=True)

    # ── 1. LGB 10시드 ──
    n_feats = train_lgb(train, feats)

    # ── 2. MLP 10시드 ──
    Xn, Xc, ytr, cat_vocab, n_num, cats, nums = build_mlp_full(train, feats)
    print(f"mlp: vocab={cat_vocab} | n_num={n_num} | cats={len(cats)} | nums={len(nums)}",
          flush=True)
    train_mlp(Xn, Xc, ytr, cat_vocab, n_num)

    # ── 3. train_meta.json (LGB 정보 중심) ──
    meta = dict(
        feature=FEATURE,
        seeds=SEEDS,
        num_boost_round=LGB_NUM_BOOST_ROUND,
        best_iter_primary=None,  # full data는 early_stopping 없음 (고정 라운드)
        features=feats,
        base_n=base_n,
        cats=LGB_CATS,
        mlp=dict(cats=MLP_CATS, emb_dim=EMB_DIM, n_num=n_num, prep_vocab=cat_vocab),
        params={k: common.PARAMS[k] for k in [
            "objective", "metric", "learning_rate", "num_leaves",
            "min_data_in_leaf", "feature_fraction", "bagging_fraction",
            "bagging_freq", "num_threads", "verbosity", "deterministic"]},
        total_time=time.time() - t0,
    )
    with open(os.path.join(MODEL_DIR, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n완료: submit_plato3b2/model/f3_s*.txt(10) + mlp_s*.pt(10) + mlp_prep.pkl + "
          f"mlp_meta.json + train_meta.json (총 {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
