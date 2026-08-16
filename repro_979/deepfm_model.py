#!/usr/bin/env python3
"""deepfm_model.py — frozen pure-PyTorch DeepFM + DCNv2 member model (Task 4).

Aimers9-score-improvement-next-round, Todo 4: canonical DeepFM/DCNv2 implementation.
This module is the SINGLE SOURCE OF TRUTH for every frozen hyperparameter/schema; the
runner (deepfm_runner.py) builds its config digest from the constants below and makes
zero architecture decisions of its own.

FROZEN SPEC (plan §48(5) + Todo 4; the plan INTENTIONALLY supersedes the cited
ideation-ultrabrain.md lines 39-56 defaults where they differ):

  Features      exactly 49 = the 47 official non-ID inputs (after excluding `row_id`)
                + `platoon` + `count_state`, mirroring the existing MLP contract
                column-for-column (qualification_runner.CHAMPION_FEATURES order).
  Categories    nine categorical fields (qualification_runner.MLP_CATS order);
                vocabularies built ONLY from outer-train rows, ID 0 = UNK,
                string keys (e6c_blend_folds.build_fold convention — byte-frozen MLP
                contract mapping: sorted(outer_train[c].astype(str).unique())).
  Numerics      standardization + mean imputation fitted ONLY on outer-train rows;
                zero-variance numerics map to zero; never fit on validation/test rows.
  Embeddings    EVERY categorical field projects to a common 24-d FM vector
                (supersedes ideation's pitcher/batter 24 + team 8 + context 4-8).
  UNK dropout   10% pitcher/batter UNK dropout BEFORE embedding lookup, training-only
                (supersedes ideation's 5% player-ID dropout; keeps 2025 identity
                dependence low).
  Model         linear logit term (numeric Linear + per-category bias embeddings)
                + FM pairwise-dot sum over the 9 embedding vectors
                + three rank-32 DCNv2 low-rank cross layers
                + parallel SiLU/LayerNorm [256,128] tower with ZERO tower dropout
                (supersedes ideation dropout 0.10), all concatenated into one scalar
                BCE head (BCEWithLogitsLoss, mean reduction).
  Optimization  AdamW lr=3e-4, weight_decay=1e-5, batch 4096, max 20 epochs,
                inner-temporal patience 3 (early stopping on inner Brier,
                earliest-epoch tie-break), then REFIT the fixed epoch count on the
                full outer-train window before outer scoring.
  Seeds         screen [52,53] / promotion [52,53,54,55,56]; deterministic seed
                order; ensembles average LOGITS, never probabilities.
  Scoring       deployed formula common.score(clip(sigmoid(z + C_LOGIT), 0.30, 0.70), y),
                C_LOGIT = -0.0404 frozen.

Deployment note (ideation line 54): uses only plain torch ops supported by local
Torch 2.1 and evaluation Torch 2.7 — no extra pip dependencies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ════════════════════════════════════════════════════════════════════
# Frozen spec constants — single source of truth (runner digest covers all)
# ════════════════════════════════════════════════════════════════════

# 49 features — same frozen order as qualification_runner.CHAMPION_FEATURES.
FEATURES: tuple[str, ...] = (
    "season", "game_month", "game_dayofweek", "inning", "top_bottom", "game_type",
    "balls_before", "strikes_before", "outs_before", "run_top_before", "run_bot_before",
    "run_total_before", "score_diff_home", "score_diff_pitcher_team", "runner_on_1b",
    "runner_on_2b", "runner_on_3b", "num_runners_on", "base_state", "home_win_expectancy",
    "away_win_expectancy", "li", "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id", "asof_pitcher_n", "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate", "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate", "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate", "asof_batter_n", "asof_batter_success_rate",
    "asof_batter_middle_rate", "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate", "platoon", "count_state",
)

# Nine categorical fields — same frozen order as qualification_runner.MLP_CATS.
# The first two (pitcher_id, batter_id) are the UNK-dropout fields.
CATS: tuple[str, ...] = (
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id", "top_bottom",
    "game_type", "base_state", "platoon", "count_state",
)

NUMERICS: tuple[str, ...] = tuple(f for f in FEATURES if f not in CATS)

assert len(FEATURES) == 49, f"frozen 49-feature contract violated: {len(FEATURES)}"
assert len(CATS) == 9, f"frozen nine-category contract violated: {len(CATS)}"
assert len(NUMERICS) == 40, f"frozen 40-numeric contract violated: {len(NUMERICS)}"

EMB_DIM = 24                 # common 24-d FM vector for EVERY categorical field
UNK_P = 0.10                 # 10% pitcher/batter UNK dropout before embedding lookup
UNK_DROP_CATS: tuple[str, ...] = ("pitcher_id", "batter_id")
DCN_DEPTH = 3                # three DCNv2 cross layers
DCN_RANK = 32                # low rank 32 (supersedes ideation rank 32|64)
TOWER_HIDDEN: tuple[int, ...] = (256, 128)
TOWER_DROPOUT = 0.0          # ZERO tower dropout (supersedes ideation dropout 0.10)
LR = 0.0003                  # AdamW lr
WEIGHT_DECAY = 0.00001       # AdamW weight decay
BATCH_SIZE = 4096
MAX_EPOCHS = 20
PATIENCE = 3                 # inner-temporal patience on inner Brier, earliest-epoch tie-break
SCREEN_SEEDS: tuple[int, ...] = (52, 53)
PROMOTE_SEEDS: tuple[int, ...] = (52, 53, 54, 55, 56)
C_LOGIT = -0.0404            # frozen global calibration constant (submission alignment)
CLIP_LO, CLIP_HI = 0.30, 0.70
MISSINGNESS_FLAGS: tuple[str, ...] = ()   # frozen: no extra flags inside the 49 features
LOSS = "BCEWithLogitsLoss(reduction='mean')"
SCORE_FORMULA = "common.score(clip(sigmoid(z + C_LOGIT), 0.30, 0.70), y)"
ENSEMBLE_RULE = "average of per-seed LOGITS (never probabilities)"


class CrossLayer(nn.Module):
    """DCNv2 low-rank cross layer: x_{l+1} = x0 * (V (U^T x_l) + b) + x_l, W = U V^T.

    Standard DCN-v2 low-rank parameterization (rank 32, frozen)."""
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.u = nn.Parameter(torch.empty(dim, rank))
        self.v = nn.Parameter(torch.empty(dim, rank))
        self.b = nn.Parameter(torch.zeros(dim))
        nn.init.xavier_uniform_(self.u)
        nn.init.xavier_uniform_(self.v)

    def forward(self, x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        # (B, dim): V (U^T x) + b, then elementwise multiply by x0, residual + x
        proj = (self.v @ (self.u.t() @ x.t())).t()
        return x0 * (proj + self.b) + x


class DeepFMDCNv2(nn.Module):
    """Frozen DeepFM + DCNv2 hybrid member (spec above; zero architecture freedom).

    forward(x_num, x_cat) -> scalar logits of shape (B,):
      linear logit term (numeric Linear + per-category bias embeddings)
      + FM pairwise-dot sum of the nine 24-d embedding vectors
      + three rank-32 DCNv2 cross layers
      + parallel SiLU/LayerNorm [256,128] tower with zero dropout
      all concatenated into one scalar BCE head.
    """
    def __init__(self, cat_vocab, n_num, cat_names=CATS, emb_dim=EMB_DIM, unk_p=UNK_P,
                 unk_drop_cats=UNK_DROP_CATS, dcn_depth=DCN_DEPTH, dcn_rank=DCN_RANK,
                 tower_hidden=TOWER_HIDDEN, tower_dropout=TOWER_DROPOUT):
        super().__init__()
        assert len(cat_vocab) == len(cat_names) == len(CATS) == 9, (
            "frozen nine-category contract: cat_vocab/cat_names must have exactly 9 entries")
        assert tower_dropout == 0.0, "tower dropout is FROZEN to zero — do not raise it"
        self.cat_vocab = tuple(int(v) for v in cat_vocab)
        self.n_num = int(n_num)
        self.emb_dim = int(emb_dim)
        self.unk_p = float(unk_p)
        self.tower_dropout = float(tower_dropout)
        self.unk_drop_idx: tuple[int, ...] = tuple(
            i for i, name in enumerate(cat_names) if name in unk_drop_cats)
        self.embeds = nn.ModuleList([nn.Embedding(v, emb_dim) for v in self.cat_vocab])
        self.cat_bias = nn.ModuleList([nn.Embedding(v, 1) for v in self.cat_vocab])
        self.num_linear = nn.Linear(n_num, 1)
        cross_dim = n_num + len(self.cat_vocab) * emb_dim
        self.cross_layers = nn.ModuleList(
            [CrossLayer(cross_dim, dcn_rank) for _ in range(dcn_depth)])
        tower: list[nn.Module] = []
        in_d = cross_dim
        for h in tower_hidden:
            tower += [nn.Linear(in_d, h), nn.LayerNorm(h), nn.SiLU()]
            in_d = h
        tower.append(nn.Linear(in_d, 1))
        self.tower = nn.Sequential(*tower)
        # scalar BCE head: concat [linear, FM, cross, tower] -> Linear -> logit
        self.head = nn.Linear(1 + 1 + cross_dim + 1, 1)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        batch = x_num.shape[0]
        x = x_cat
        # 10% pitcher/batter UNK dropout BEFORE embedding lookup — training-only.
        if self.training and self.unk_p > 0.0 and self.unk_drop_idx:
            idx = list(self.unk_drop_idx)
            ids = x[:, idx].clone()
            ids[torch.rand_like(ids, dtype=torch.float) < self.unk_p] = 0  # ID 0 = UNK
            x = x.clone()
            x[:, idx] = ids
        embs = torch.stack([e(x[:, i]) for i, e in enumerate(self.embeds)], dim=1)  # (B, 9, 24)
        # linear logit term: numeric linear + per-category bias embeddings
        lin = self.num_linear(x_num)
        for i, b in enumerate(self.cat_bias):
            lin = lin + b(x[:, i])
        # FM pairwise-dot sum: 0.5 * (||sum e_i||^2 - sum_i ||e_i||^2)
        sum_e = embs.sum(dim=1)
        sum_sq = (embs ** 2).sum(dim=(1, 2)).unsqueeze(1)
        fm = 0.5 * ((sum_e ** 2).sum(dim=1, keepdim=True) - sum_sq)
        # shared input for the cross net and the parallel tower
        x0 = torch.cat([x_num, embs.reshape(batch, -1)], dim=1)
        xc = x0
        for layer in self.cross_layers:
            xc = layer(xc, x0)
        t = self.tower(x0)
        z = self.head(torch.cat([lin, fm, xc, t], dim=1))
        return z.squeeze(-1)


class DeepFMPrep:
    """Frozen preprocessing: vocabularies + numeric stats fitted ONLY on outer-train rows.

    - categorical vocabularies: sorted string keys of outer-train rows; ID 0 = UNK;
      unseen/NaN keys map to 0 (mirror of the frozen MLP contract).
    - numerics: mean imputation + z-score standardization from outer-train rows;
      zero-variance numerics map to zero (std<=0 -> divisor 1, mean subtraction zeroes it).
    - transform() never fits anything (validation/test rows are pure transforms).
    """
    def __init__(self, nums, cats, nmean, nstd, cat_maps, cat_vocab):
        self.nums = tuple(nums)
        self.cats = tuple(cats)
        self.nmean = dict(nmean)
        self.nstd = dict(nstd)
        self.cat_maps = {c: dict(m) for c, m in cat_maps.items()}
        self.cat_vocab = tuple(int(v) for v in cat_vocab)
        self.n_num = len(self.nums)

    @classmethod
    def fit(cls, df: pd.DataFrame, nums=NUMERICS, cats=CATS) -> "DeepFMPrep":
        assert len(cats) == len(CATS) == 9, (
            "frozen nine-category contract violated — categorical input must be exactly "
            "the nine frozen fields")
        assert tuple(cats) == CATS, (
            "frozen nine-category ORDER contract violated — categorical fields must be "
            "in the exact frozen order (pitcher_id, batter_id, ...)")
        nmean: dict[str, float] = {}
        nstd: dict[str, float] = {}
        for c in nums:
            std = float(df[c].std())
            nmean[c] = float(df[c].mean())
            nstd[c] = std if std > 0.0 else 1.0  # zero-variance -> divisor 1 (maps to zero)
        cat_maps: dict[str, dict[str, int]] = {}
        cat_vocab: list[int] = []
        for c in cats:
            vals = sorted(df[c].astype(str).unique())  # MLP contract: sorted string keys
            cat_maps[c] = {v: i + 1 for i, v in enumerate(vals)}
            cat_vocab.append(len(vals) + 1)
        return cls(nums, cats, nmean, nstd, cat_maps, cat_vocab)

    def transform_np(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(Xn float32 (n,40), Xc int64 (n,9)) — pure transform, nothing fitted here."""
        n = len(df)
        Xn = np.zeros((n, len(self.nums)), dtype=np.float32)
        for j, c in enumerate(self.nums):
            col = df[c].fillna(self.nmean[c]).values.astype(np.float32)
            Xn[:, j] = (col - self.nmean[c]) / self.nstd[c]
        Xc = np.zeros((n, len(self.cats)), dtype=np.int64)
        for j, c in enumerate(self.cats):
            Xc[:, j] = df[c].astype(str).map(lambda v: self.cat_maps[c].get(v, 0)).values
        return Xn, Xc
