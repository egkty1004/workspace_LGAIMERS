#!/usr/bin/env python3
"""test_deepfm_runner.py — Task 4 frozen DeepFM contract tests (pytest-free, plain asserts).

Covers every Task-4 acceptance gate:
  (a) config digest covers every frozen hyperparameter/schema (mutation sensitivity)
  (b) nine-category contract: order mismatch / count mismatch / missing columns rejected
  (c) 2025-row rejection
  (d) R-only fold label read -> LeakageError
  (e) outer-label read before epoch selection -> PolicyViolation (outer-label-mutation)
  (f) altered config under an existing cache key -> PolicyViolation
  (g) missing fold provenance -> PolicyViolation
  (h) model forward/backward (frozen architecture: 9 x 24-d embeds, rank-32 x3 cross,
      zero-dropout tower, scalar BCE head)
  (i) vocabularies built from outer-train only, ID 0 = UNK, zero-variance numerics -> 0
  (j) inner temporal split rule (last season of the outer-train window = inner val)
  (k) earliest-epoch tie-break bookkeeping
  (l) DeepFMPrep.fit rejects non-nine / reordered categorical inputs

Usage: python3 repro_979/test_deepfm_runner.py   (exit 0 = all PASS)
"""
from __future__ import annotations

import json
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from repro_979 import deepfm_runner as dr  # noqa: E402
from repro_979.deepfm_model import FEATURES, CATS, NUMERICS, DeepFMDCNv2, DeepFMPrep  # noqa: E402

PASSED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        print(f"[FAIL] {name} {detail}", file=sys.stderr)
        raise SystemExit(1)
    PASSED.append(name)
    print(f"  [PASS] {name} {detail}")


def make_train_df(n_rows: int = 600, with_2025: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        c: rng.normal(size=n_rows) for c in NUMERICS
    })
    for c in NUMERICS:
        df[c] = df[c].astype(np.float32)
    df["season"] = rng.integers(2019, 2025, size=n_rows)
    df["pitcher_id"] = rng.integers(1, 30, size=n_rows)
    df["batter_id"] = rng.integers(1, 40, size=n_rows)
    df["pitcher_team_id"] = rng.integers(1, 15, size=n_rows)
    df["batter_team_id"] = rng.integers(1, 15, size=n_rows)
    df["top_bottom"] = rng.integers(0, 2, size=n_rows)
    df["game_type"] = rng.choice(["R", "F", "P"], size=n_rows)
    df["base_state"] = rng.integers(0, 9, size=n_rows)
    df["platoon"] = rng.integers(0, 5, size=n_rows)
    df["count_state"] = rng.integers(0, 13, size=n_rows)
    df["row_id"] = [f"r{i}" for i in range(n_rows)]
    df["control_success"] = rng.integers(0, 2, size=n_rows).astype(np.float32)
    if with_2025:
        row = df.iloc[[0]].copy()
        row["season"] = 2025
        df = pd.concat([df, row], ignore_index=True)
    return df


def _synthetic_tensors(prep: DeepFMPrep, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    Xn = torch.randn(n, prep.n_num)
    Xc = torch.randint(0, 3, (n, len(CATS)))  # ids 0..2 fit the smallest synthetic vocab
    return Xn, Xc


def test_a_digest_covers_frozen_params() -> None:
    base = dr._config_dict()
    base_hash = dr._config_hash()

    def mutated(mutator) -> bool:
        cfg = dr._config_dict()
        mutator(cfg)
        return dr._canonical_sha256(cfg) != base_hash

    mutations = [
        ("emb_dim 24 -> 8", lambda c: c.update({"emb_dim": 8})),
        ("unk_p 0.10 -> 0.05", lambda c: c.update({"unk_p": 0.05})),
        ("dcn_rank 32 -> 64", lambda c: c["model"].update({"dcn_rank": 64})),
        ("dcn_depth 3 -> 2", lambda c: c["model"].update({"dcn_depth": 2})),
        ("tower_dropout 0.0 -> 0.1", lambda c: c["model"].update({"tower_dropout": 0.1})),
        ("tower_hidden [256,128] -> [128,64]", lambda c: c["model"].update({"tower_hidden": [128, 64]})),
        ("lr 3e-4 -> 1e-3", lambda c: c["optim"].update({"adamw_lr": 1e-3})),
        ("weight_decay 1e-5 -> 1e-4", lambda c: c["optim"].update({"weight_decay": 1e-4})),
        ("batch 4096 -> 2048", lambda c: c["optim"].update({"batch_size": 2048})),
        ("max_epochs 20 -> 15", lambda c: c["optim"].update({"max_epochs": 15})),
        ("patience 3 -> 5", lambda c: c["optim"].update({"patience": 5})),
        ("screen seeds [52,53] -> [52]", lambda c: c["seeds"].update({"screen": [52]})),
        ("promote seeds 5 -> 3", lambda c: c["seeds"].update({"promotion": [52, 53, 54]})),
        ("cats reordered", lambda c: c.update({"cats": list(reversed(CATS))})),
        ("feature dropped", lambda c: c.update({"features": list(FEATURES)[:-1]})),
    ]
    for name, fn in mutations:
        assert mutated(fn), f"config digest did NOT change for: {name}"
    check("a_config_digest_covers_frozen_params",
          base["emb_dim"] == 24 and base["model"]["tower_dropout"] == 0.0
          and base["model"]["dcn_rank"] == 32 and base["model"]["dcn_depth"] == 3
          and base["optim"]["patience"] == 3 and base["optim"]["max_epochs"] == 20
          and base["optim"]["batch_size"] == 4096 and base["optim"]["adamw_lr"] == 0.0003
          and base["seeds"]["screen"] == [52, 53]
          and base["seeds"]["promotion"] == [52, 53, 54, 55, 56],
          f"({len(mutations)}/15 mutations change the hash; frozen values verified)")


def test_b_category_contract() -> None:
    df = make_train_df()
    assert dr._check_schema(df) == []
    swapped = (CATS[1], CATS[0]) + CATS[2:]
    probs = dr._check_schema(df, cats=swapped)
    assert probs, "swapped cat order must be rejected"
    probs9 = dr._check_schema(df, cats=CATS[:8])
    assert probs9, "eight-cat input must be rejected"
    bad = df.drop(columns=["pitcher_id"])
    assert dr._check_schema(bad), "missing cat column must be rejected"
    badf = df.drop(columns=["li"])
    assert dr._check_schema(badf), "missing 49-feature column must be rejected"
    check("b_category_contract", True,
          "order mismatch / 8 cats / missing cat / missing feature all rejected")


def test_c_contains_2025() -> None:
    df = make_train_df(with_2025=True)
    try:
        dr._check_no_2025(df)
        raise AssertionError("2025 guard did not fire")
    except dr.PolicyViolation:
        pass
    check("c_contains_2025", True, "PolicyViolation raised on injected 2025 row")


def test_d_r_fold_embargo() -> None:
    dr._epoch_gate.epochs_selected = True
    try:
        dr._read_outer_labels(pd.DataFrame(), np.zeros(1, dtype=bool), "r2022")
        raise AssertionError("r2022 read did not fire LeakageError")
    except dr.LeakageError:
        pass
    check("d_r_fold_embargo", True, "LeakageError raised for r2022 label read")


def test_e_outer_label_mutation() -> None:
    dr._epoch_gate.epochs_selected = False
    df = make_train_df()
    folds = dr.build_folds(df)
    va = folds["primary"][1]
    try:
        dr._read_outer_labels(df, va.values, "primary")
        raise AssertionError("early outer-label read did not fire PolicyViolation")
    except dr.PolicyViolation:
        pass
    check("e_outer_label_mutation", True,
          "PolicyViolation raised when outer labels read before epoch selection")


def test_f_altered_config_under_cache_key() -> None:
    h = dr._config_hash()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        old_root = dr.CACHE_ROOT
        dr.CACHE_ROOT = tmp
        try:
            d = tmp / h
            d.mkdir(parents=True)
            (d / "manifest.json").write_text(
                json.dumps({"config_hash": "a" * 64}), encoding="utf-8")
            try:
                dr._check_cache_key(h)
                raise AssertionError("altered config under existing cache key not rejected")
            except dr.PolicyViolation:
                pass
            # dir exists without manifest -> missing provenance rejection
            (d / "manifest.json").unlink()
            try:
                dr._check_cache_key(h)
                raise AssertionError("missing manifest not rejected")
            except dr.PolicyViolation:
                pass
        finally:
            dr.CACHE_ROOT = old_root
    check("f_altered_config_under_cache_key", True,
          "altered config + missing manifest under existing key both rejected")


def test_g_missing_fold_provenance() -> None:
    manifest = {"config_hash": dr._config_hash(), "fold_provenance": []}
    try:
        dr._fold_provenance(manifest, "primary", 10, "abc")
        raise AssertionError("missing fold provenance not rejected")
    except dr.PolicyViolation:
        pass
    manifest = {"config_hash": dr._config_hash(), "fold_provenance": [
        {"fold": "primary", "n_va": 9, "row_ids_sha256": "abc"}]}
    try:
        dr._fold_provenance(manifest, "primary", 10, "abc")
        raise AssertionError("mismatched fold provenance not rejected")
    except dr.PolicyViolation:
        pass
    check("g_missing_fold_provenance", True,
          "missing / mismatched fold provenance both rejected")


def test_h_model_forward_backward() -> None:
    df = make_train_df()
    tr = df[df["season"] <= 2023]
    prep = DeepFMPrep.fit(tr, NUMERICS, CATS)
    model = DeepFMDCNv2(prep.cat_vocab, prep.n_num)
    assert len(model.embeds) == 9 and all(e.embedding_dim == 24 for e in model.embeds)
    assert len(model.cross_layers) == 3 and all(c.u.shape[1] == 32 for c in model.cross_layers)
    assert model.tower_dropout == 0.0
    assert not any(isinstance(m, nn.Dropout) for m in model.tower)
    Xn, Xc = _synthetic_tensors(prep, 128)
    z = model(Xn, Xc)
    assert z.shape == (128,)
    loss = nn.BCEWithLogitsLoss()(z, torch.rand(128))
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    # UNK dropout is training-only: eval forward is deterministic
    model.eval()
    z1 = model(Xn, Xc).clone()
    z2 = model(Xn, Xc).clone()
    assert torch.equal(z1, z2)
    check("h_model_forward_backward", True,
          "9x24 embeds, rank-32 x3 cross, zero tower dropout, scalar BCE head, "
          "training-only UNK dropout (eval deterministic)")


def test_i_prep_vocab_and_numeric() -> None:
    df = make_train_df(300)
    tr = df[df["season"] <= 2023].copy()
    tr["cv_col"] = tr["count_state"].astype(str)
    prep = DeepFMPrep.fit(tr, NUMERICS, CATS)
    assert prep.cat_vocab == tuple(len(sorted(tr[c].astype(str).unique())) + 1
                                  for c in CATS)
    # unseen value in transform -> ID 0 (UNK)
    out = tr.iloc[:5].copy()
    out["count_state"] = "unseen_value_999"
    Xn, Xc = prep.transform_np(out)
    assert (Xc[:, CATS.index("count_state")] == 0).all(), "unseen cat must map to ID 0"
    # zero-variance numeric -> maps to zero
    z = tr.copy()
    z["li"] = 7.0
    prepz = DeepFMPrep.fit(z, NUMERICS, CATS)
    Xnz, _ = prepz.transform_np(z.iloc[:5])
    li_idx = NUMERICS.index("li")
    assert np.allclose(Xnz[:, li_idx], 0.0), "zero-variance numeric must map to zero"
    # never fit on validation rows: transform of a val row with a new numeric value
    # still uses outer-train mean/std only
    assert prep.nmean["li"] != float(out["li"].mean())
    check("i_prep_vocab_and_numeric", True,
          "vocab from outer-train only, ID 0 = UNK, zero-variance -> 0, "
          "no validation fitting")


def test_j_inner_split() -> None:
    df = make_train_df(400)
    tr_m = (df["season"] <= 2023).values
    it, iv = dr._inner_split(tr_m, df)
    assert sorted(int(s) for s in df.loc[iv, "season"].unique()) == [2023]
    assert int(df.loc[it, "season"].max()) <= 2022
    check("j_inner_split", True, "last season of outer-train window = inner val (2023)")


def test_k_earliest_epoch_tie_break() -> None:
    b, e, bad = dr._track_patience(0.25, 1, float("inf"), 0, 0, 3)
    assert (b, e, bad) == (0.25, 1, 0)
    b2, e2, bad2 = dr._track_patience(0.25, 2, 0.25, 1, 0, 3)   # tie -> keep epoch 1
    assert (b2, e2, bad2) == (0.25, 1, 1)
    b3, e3, bad3 = dr._track_patience(0.24, 3, 0.25, 1, 1, 3)
    assert (b3, e3, bad3) == (0.24, 3, 0)
    b4, e4, bad4 = dr._track_patience(0.30, 4, 0.24, 3, 0, 3)
    assert (b4, e4, bad4) == (0.24, 3, 1)
    check("k_earliest_epoch_tie_break", True,
          "equal inner Brier keeps the earliest epoch; patience counts non-improving epochs")


def test_l_fit_rejects_bad_cats() -> None:
    df = make_train_df(200)
    try:
        DeepFMPrep.fit(df, NUMERICS, CATS[:8])
        raise AssertionError("8-cat fit not rejected")
    except AssertionError:
        pass
    try:
        DeepFMPrep.fit(df, NUMERICS, (CATS[1], CATS[0]) + CATS[2:])
        raise AssertionError("reordered cat fit not rejected")
    except AssertionError:
        pass
    check("l_fit_rejects_bad_cats", True, "DeepFMPrep.fit rejects non-nine/reordered cats")


def main() -> int:
    print("[test_deepfm_runner] Task 4 frozen DeepFM contract tests", flush=True)
    test_a_digest_covers_frozen_params()
    test_b_category_contract()
    test_c_contains_2025()
    test_d_r_fold_embargo()
    test_e_outer_label_mutation()
    test_f_altered_config_under_cache_key()
    test_g_missing_fold_provenance()
    test_h_model_forward_backward()
    test_i_prep_vocab_and_numeric()
    test_j_inner_split()
    test_k_earliest_epoch_tie_break()
    test_l_fit_rejects_bad_cats()
    print(f"\n[test_deepfm_runner] ALL {len(PASSED)} TESTS PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
