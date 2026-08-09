#!/usr/bin/env python3
"""
[Exp 6-c 배포] 제출용 MLP 추론 모듈 (2026-08-08, 채택 피처 확장판)

전처리(mlp_prep.pkl)와 10개 MLP state_dict를 로드해 로짓 예측 제공.
학습(deploy_train_mlp.py)과 동일 아키텍처/전처리 — 재현 일치.

⚠️ 채택 피처(asof_n_bucket, score_diff_binary)가 범주형 임베딩에 추가되어
EMB_DIM이 11개로 확장됨: [16,16,4,4,2,4,4,4,4,4,2].
"""
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

EMB_DIM = [16, 16, 4, 4, 2, 4, 4, 4, 4, 4, 2]
HIDDEN = [512, 256, 128, 64]
DROPOUT = 0.25


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


def load(model_dir, seeds):
    """전처리 + 모델 로드. model_dir에는 mlp_prep.pkl / mlp_s{seed}.pt 존재."""
    with open(os.path.join(model_dir, "mlp_prep.pkl"), "rb") as f:
        prep = pickle.load(f)
    models = []
    for s in seeds:
        m = EntityMLP(prep["cat_vocab"], len(prep["nums"]))
        m.load_state_dict(torch.load(os.path.join(model_dir, f"mlp_s{s}.pt"),
                                     map_location="cpu"))
        m.eval()
        models.append(m)
    return prep, models


def build_input(df, prep):
    """df는 platoon/count_state 포함 전처리 완료 상태여야 함. (x_num, x_cat) 반환."""
    nums, cats = prep["nums"], prep["cats"]
    nmean, nstd = prep["nmean"], prep["nstd"]
    cat_map = prep["cat_map"]
    Xn = np.stack([(df[c].fillna(nmean[c]).astype(np.float32) - nmean[c]) / nstd[c]
                   for c in nums], axis=1)
    Xc = np.stack([df[c].astype(str).map(lambda v: cat_map[c].get(v, 0)).values
                   for c in cats], axis=1)
    return torch.tensor(Xn), torch.tensor(Xc)


def predict_z(df, prep, models, batch=65536):
    """로짓 시드 평균 반환 (확률이 아닌 로짓 — script.py에서 블렌딩/오프셋 적용)."""
    Xn, Xc = build_input(df, prep)
    zs = []
    with torch.no_grad():
        for m in models:
            outs = []
            for i in range(0, len(Xn), batch):
                o = m(Xn[i:i + batch], Xc[i:i + batch])
                outs.append(o.numpy())
            zs.append(np.concatenate(outs))
    return np.mean(zs, axis=0)
