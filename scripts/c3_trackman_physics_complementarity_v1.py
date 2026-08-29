#!/usr/bin/env python3
"""Exact four-fit C3 versus C3+physics17 complementarity screen."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "c3_trackman_physics_complementarity_v1.json"
ROW_ID, TARGET = "row_id", "control_success"
DERIVED = frozenset(("platoon", "count_state", "base_out_state_24"))
ORIGINS = ("r2022", "r2023")


class ContractError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                     allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def array_hash(values: Sequence[Any]) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def load_config() -> dict[str, Any]:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    base, physics = cfg["base_c3_features"], cfg["physics_features"]
    cats = ["top_bottom", "game_type", "base_state", "platoon", "count_state", "base_out_state_24"]
    if len(base) != 50 or len(set(base)) != 50 or len(physics) != 17 or len(set(physics)) != 17:
        raise ContractError("50+17 feature contract drift")
    if cfg["categorical_features"] != cats or len(base + physics) != 67:
        raise ContractError("feature/categorical contract drift")
    if list(cfg["origins"]) != list(ORIGINS):
        raise ContractError("origin contract drift")
    if cfg["split"] != {"random_state": 12345, "es_fraction": 0.05, "post_es_refit": False}:
        raise ContractError("split contract drift")
    expected_cat = {"version": "1.2.10", "loss_function": "Logloss", "learning_rate": 0.05,
                    "depth": 6, "l2_leaf_reg": 5, "iterations": 2000,
                    "early_stopping_rounds": 50, "use_best_model": True,
                    "random_seed": 42, "thread_count": 16, "verbose": 0,
                    "allow_writing_files": False}
    if cfg["catboost"] != expected_cat:
        raise ContractError("CatBoost contract drift")
    return cfg


def _outside_repo(path: Path, repo_root: Path) -> None:
    try:
        path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return
    raise ContractError("output directory must be outside repository")


def import_physics_runner(path: Path, expected_sha: str) -> Any:
    if not path.is_file() or path.is_symlink() or sha256_file(path) != expected_sha:
        raise ContractError("physics runner authority drift")
    spec = importlib.util.spec_from_file_location("pinned_physics_runner", path)
    if spec is None or spec.loader is None:
        raise ContractError("physics runner import failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_authorities(c3_contract: Path, physics_runner: Path,
                       physics_config: Path, lookup_json: Path) -> dict[str, Any]:
    cfg = load_config()
    root = c3_contract.resolve().parents[2]
    paths = {
        "c3_report": root / "feature_lab_smoke_report.json",
        "c3_model": c3_contract.resolve().parent / "catboost.cbm",
        "c3_contract": c3_contract.resolve(),
        "c3_parent_zip": root / "final-packages" / "catboost-feature-lab-b1-C3-base-out-submit.zip",
        "physics_runner": physics_runner.resolve(), "physics_config": physics_config.resolve(),
        "physics_lookup": lookup_json.resolve(),
    }
    for name, path in paths.items():
        if not path.is_file() or path.is_symlink():
            raise ContractError(f"authority missing: {name}")
        if sha256_file(path) != cfg["authorities"][f"{name}_sha256"]:
            raise ContractError(f"authority hash drift: {name}")
    c3 = json.loads(c3_contract.read_text(encoding="utf-8"))
    phys = json.loads(physics_config.read_text(encoding="utf-8"))
    lookup = json.loads(lookup_json.read_text(encoding="utf-8"))
    if c3.get("features") != cfg["base_c3_features"] or c3.get("categorical_features") != cfg["categorical_features"]:
        raise ContractError("C3 authority contract mismatch")
    if phys.get("base_c3_features") != cfg["base_c3_features"] or phys.get("physics_features") != cfg["physics_features"]:
        raise ContractError("physics authority contract mismatch")
    if lookup.get("features") != cfg["physics_features"]:
        raise ContractError("lookup feature order mismatch")
    if sorted(lookup.get("states", {})) != [str(year) for year in range(2019, 2026)]:
        raise ContractError("lookup season scope mismatch")
    return {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()}


def raw_feature_columns(cfg: Mapping[str, Any]) -> list[str]:
    return [name for name in cfg["base_c3_features"] if name not in DERIVED]


def read_feature_projection(train_csv: Path, cfg: Mapping[str, Any]) -> pd.DataFrame:
    columns = [ROW_ID, *raw_feature_columns(cfg)]
    if TARGET in columns:
        raise ContractError("target entered feature projection")
    header = pd.read_csv(train_csv, nrows=0, encoding="utf-8-sig")
    if missing := sorted(set(columns) - set(header.columns)):
        raise ContractError(f"missing feature columns: {missing}")
    frame = pd.read_csv(train_csv, usecols=columns, encoding="utf-8-sig")
    if len(frame) != cfg["official_train_rows"]:
        raise ContractError("official train row count drift")
    ids = frame[ROW_ID].astype(str)
    if frame[ROW_ID].isna().any() or ids.duplicated().any():
        raise ContractError("row_id missing/duplicate")
    return frame


def origin_positions(frame: pd.DataFrame, origin: str,
                     cfg: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if origin not in ORIGINS:
        raise ContractError("only r2022/r2023 allowed")
    year = cfg["origins"][origin]["year"]
    season = pd.to_numeric(frame["season"], errors="coerce").to_numpy()
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    train = np.flatnonzero(regular & np.isfinite(season) & (season < year)).astype(np.int64)
    outer = np.flatnonzero(regular & np.isfinite(season) & (season == year)).astype(np.int64)
    expected = cfg["origins"][origin]
    if len(train) != expected["train_rows"] or len(outer) != expected["validation_rows"]:
        raise ContractError(f"{origin} geometry drift")
    return train, outer


def split_positions(source: Sequence[int], cfg: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(source, dtype=np.int64)
    if len(np.unique(positions)) != len(positions) or len(positions) < 2:
        raise ContractError("invalid training positions")
    n_es = int(len(positions) * cfg["split"]["es_fraction"])
    rng = np.random.RandomState(cfg["split"]["random_state"])
    relative_es = np.sort(rng.choice(len(positions), size=n_es, replace=False))
    mask = np.ones(len(positions), dtype=bool)
    mask[relative_es] = False
    fit, es = positions[mask], positions[relative_es]
    if np.intersect1d(fit, es).size or len(fit) + len(es) != len(positions):
        raise ContractError("fit/ES split failure")
    return fit, es


def _scoped_rows(path: Path, positions: Sequence[int], usecols: Sequence[str]) -> pd.DataFrame:
    wanted = frozenset(int(value) for value in positions)
    if not wanted:
        raise ContractError("empty scoped read")
    return pd.read_csv(path, usecols=list(usecols), encoding="utf-8-sig",
                       skiprows=lambda line: line > 0 and (line - 1) not in wanted)


def read_training_labels(train_csv: Path, frame: pd.DataFrame, positions: Sequence[int],
                         origin: str, cfg: Mapping[str, Any]) -> np.ndarray:
    expected, _ = origin_positions(frame, origin, cfg)
    positions = np.asarray(positions, dtype=np.int64)
    if not np.array_equal(positions, expected):
        raise ContractError("training label scope drift")
    scoped = _scoped_rows(train_csv, positions, (ROW_ID, "season", "game_type", TARGET))
    ids = frame.iloc[positions][ROW_ID].astype(str).tolist()
    if scoped[ROW_ID].astype(str).tolist() != ids:
        raise ContractError("training label row mismatch")
    year = cfg["origins"][origin]["year"]
    seasons = pd.to_numeric(scoped["season"], errors="coerce")
    if not seasons.lt(year).all() or seasons.ge(2024).any() or not scoped["game_type"].astype(str).eq("R").all():
        raise ContractError("training label temporal violation")
    target = pd.to_numeric(scoped[TARGET], errors="coerce")
    if target.isna().any() or not target.isin((0, 1)).all():
        raise ContractError("invalid training target")
    return target.to_numpy(np.float64)


class SealedPredictions(NamedTuple):
    origin: str
    positions: tuple[int, ...]
    position_hash: str
    row_id_hash: str
    c3: tuple[float, ...]
    physics17: tuple[float, ...]


def seal_predictions(origin: str, frame: pd.DataFrame, positions: Sequence[int],
                     p0: Sequence[float], p1: Sequence[float], cfg: Mapping[str, Any]) -> SealedPredictions:
    _, expected = origin_positions(frame, origin, cfg)
    positions = np.asarray(positions, dtype=np.int64)
    p0, p1 = np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64)
    if not np.array_equal(positions, expected) or len(p0) != len(positions) or len(p1) != len(positions):
        raise ContractError("outer row parity failure")
    if not np.isfinite(p0).all() or not np.isfinite(p1).all():
        raise ContractError("nonfinite predictions")
    if not ((p0 >= 0).all() and (p0 <= 1).all() and (p1 >= 0).all() and (p1 <= 1).all()):
        raise ContractError("probability range failure")
    ids = frame.iloc[positions][ROW_ID].astype(str).tolist()
    return SealedPredictions(origin, tuple(int(value) for value in positions), array_hash(positions),
                             canonical_hash(ids), tuple(float(v) for v in p0), tuple(float(v) for v in p1))


def read_outer_labels(train_csv: Path, frame: pd.DataFrame, sealed: SealedPredictions,
                      cfg: Mapping[str, Any]) -> np.ndarray:
    if not isinstance(sealed, SealedPredictions):
        raise ContractError("outer labels require matched sealed predictions")
    _, expected = origin_positions(frame, sealed.origin, cfg)
    positions = np.asarray(sealed.positions, dtype=np.int64)
    ids = frame.iloc[positions][ROW_ID].astype(str).tolist()
    if not np.array_equal(positions, expected) or sealed.position_hash != array_hash(positions):
        raise ContractError("sealed position drift")
    if sealed.row_id_hash != canonical_hash(ids):
        raise ContractError("sealed row_id drift")
    scoped = _scoped_rows(train_csv, positions, (ROW_ID, "season", "game_type", TARGET))
    year = cfg["origins"][sealed.origin]["year"]
    if year >= 2024 or scoped[ROW_ID].astype(str).tolist() != ids:
        raise ContractError("outer target identity/scope failure")
    if not pd.to_numeric(scoped["season"], errors="coerce").eq(year).all():
        raise ContractError("outer target season failure")
    target = pd.to_numeric(scoped[TARGET], errors="coerce")
    if target.isna().any() or not target.isin((0, 1)).all():
        raise ContractError("invalid outer target")
    return target.to_numpy(np.float64)


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or float(np.std(left)) == 0 or float(np.std(right)) == 0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else None


def analytic_origin(p0: Sequence[float], p1: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    p0, p1, y = (np.asarray(value, dtype=np.float64) for value in (p0, p1, y))
    if p0.ndim != 1 or len(p0) != len(p1) or len(p0) != len(y) or not len(y):
        raise ContractError("analytic shape mismatch")
    if not np.isfinite(p0).all() or not np.isfinite(p1).all() or not np.isfinite(y).all():
        raise ContractError("analytic nonfinite values")
    d, r = p1 - p0, y - p0
    a, denominator = float(np.mean(r * d)), float(np.mean(d * d))
    raw_w = None if denominator <= 0 else float(a / denominator)
    root = None if denominator <= 0 else float(2 * a / denominator)
    upper = None if a <= 0 or denominator <= 0 else float(min(1, root))
    return {
        "rows": len(y), "c3_brier": float(np.mean((p0-y)**2)),
        "physics17_brier": float(np.mean((p1-y)**2)),
        "c3_mean": float(np.mean(p0)), "c3_std": float(np.std(p0)),
        "physics17_mean": float(np.mean(p1)), "physics17_std": float(np.std(p1)),
        "prediction_correlation": _pearson(p0, p1),
        "delta_mean": float(np.mean(d)), "delta_std": float(np.std(d)),
        "A": a, "D": denominator, "corr_delta_residual": _pearson(d, r),
        "w_star_unclipped": raw_w,
        "w_star": None if raw_w is None else float(np.clip(raw_w, 0, 1)),
        "strict_improvement_root": root, "U": upper,
        "strict_improvement_interval": None if upper is None or upper <= 0 else
            {"lower": 0.0, "upper": upper, "lower_open": True, "upper_open": True},
    }


def finalize(origins: Mapping[str, Mapping[str, Any]],
             vectors: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> dict[str, Any]:
    if any(origins[name]["A"] <= 0 or origins[name]["D"] <= 0 or not origins[name]["U"] for name in ORIGINS):
        return {"result": "COMPLEMENTARITY_STOP", "shared_interval": None,
                "w_shared": None, "direct": {}}
    upper = float(min(origins[name]["U"] for name in ORIGINS))
    weight = upper / 2
    direct, passed = {}, True
    for name in ORIGINS:
        p0, p1, y = vectors[name]
        blend = p0 + weight * (p1-p0)
        base, mixed = float(np.mean((p0-y)**2)), float(np.mean((blend-y)**2))
        ok = bool(np.isfinite(mixed) and mixed < base)
        passed &= ok
        direct[name] = {"c3_brier": base, "blend_brier": mixed,
                        "absolute_brier_improvement": base-mixed, "passed": ok,
                        "blend_probability_hash": array_hash(blend)}
    return {"result": "COMPLEMENTARITY_GO" if passed else "COMPLEMENTARITY_STOP",
            "shared_interval": {"lower": 0.0, "upper": upper, "lower_open": True, "upper_open": True},
            "w_shared": weight, "direct": direct}


def fit_arm(prepared: pd.DataFrame, features: list[str], categories: list[str],
            train_positions: np.ndarray, target: np.ndarray, cfg: Mapping[str, Any],
            model_path: Path) -> tuple[Any, dict[str, Any]]:
    from catboost import CatBoostClassifier, Pool, __version__ as version
    if version != cfg["catboost"]["version"]:
        raise ContractError(f"CatBoost version mismatch: {version}")
    fit, es = split_positions(train_positions, cfg)
    offsets = {int(position): index for index, position in enumerate(train_positions)}
    fit_y = target[[offsets[int(position)] for position in fit]]
    es_y = target[[offsets[int(position)] for position in es]]
    params = {key: value for key, value in cfg["catboost"].items()
              if key not in ("version", "early_stopping_rounds", "use_best_model")}
    model = CatBoostClassifier(**params)
    started = time.time()
    model.fit(Pool(prepared.iloc[fit][features], fit_y,
                   cat_features=[features.index(name) for name in categories]),
              eval_set=Pool(prepared.iloc[es][features], es_y,
                            cat_features=[features.index(name) for name in categories]),
              use_best_model=True, early_stopping_rounds=cfg["catboost"]["early_stopping_rounds"],
              verbose=False)
    if model_path.exists():
        raise ContractError("model path exists")
    model.save_model(str(model_path))
    return model, {"feature_count": len(features), "feature_hash": canonical_hash(features),
                   "fit_rows": len(fit), "es_rows": len(es),
                   "fit_position_hash": array_hash(fit), "es_position_hash": array_hash(es),
                   "tree_count": int(model.tree_count_), "best_iteration": int(model.get_best_iteration()),
                   "model_path": str(model_path), "model_sha256": sha256_file(model_path),
                   "training_seconds": time.time()-started}


def run(train_csv: Path, lookup_json: Path, physics_runner: Path, physics_config: Path,
        c3_contract: Path, output_dir: Path, repo_root: Path) -> Path:
    cfg = load_config()
    _outside_repo(output_dir, repo_root)
    if output_dir.exists():
        raise ContractError("output exists; refusing overwrite")
    authorities = verify_authorities(c3_contract, physics_runner, physics_config, lookup_json)
    module = import_physics_runner(physics_runner, cfg["authorities"]["physics_runner_sha256"])
    physics_cfg = json.loads(physics_config.read_text(encoding="utf-8"))
    lookup = json.loads(lookup_json.read_text(encoding="utf-8"))
    started = time.time()
    frame = read_feature_projection(train_csv, cfg)
    prepared = module.apply_lookup(module.preprocess_c3(frame.drop(columns=[ROW_ID])), lookup, physics_cfg)
    base, physics = list(cfg["base_c3_features"]), list(cfg["physics_features"])
    if list(prepared.loc[:, base+physics].columns) != base+physics:
        raise ContractError("exact 50+17 order failure")
    output_dir.mkdir(parents=True)
    (output_dir / "models").mkdir()
    (output_dir / "predictions").mkdir()
    reports, vectors, fit_count = {}, {}, 0
    for origin in ORIGINS:
        train_pos, outer_pos = origin_positions(frame, origin, cfg)
        train_y = read_training_labels(train_csv, frame, train_pos, origin, cfg)
        evidence, models = {}, {}
        for arm, features in (("C3", base), ("physics17", base+physics)):
            model, item = fit_arm(prepared, features, list(cfg["categorical_features"]), train_pos,
                                  train_y, cfg, output_dir / "models" / f"{origin}_{arm}.cbm")
            models[arm], evidence[arm] = model, item
            fit_count += 1
        for key in ("fit_position_hash", "es_position_hash"):
            if evidence["C3"][key] != evidence["physics17"][key]:
                raise ContractError(f"matched {key} drift")
        p0 = np.asarray(models["C3"].predict_proba(prepared.iloc[outer_pos][base], thread_count=6))[:, 1]
        p1 = np.asarray(models["physics17"].predict_proba(prepared.iloc[outer_pos][base+physics], thread_count=6))[:, 1]
        sealed = seal_predictions(origin, frame, outer_pos, p0, p1, cfg)
        y = read_outer_labels(train_csv, frame, sealed, cfg)
        p0, p1 = np.asarray(sealed.c3), np.asarray(sealed.physics17)
        pred_path = output_dir / "predictions" / f"{origin}_matched_predictions.npz"
        np.savez_compressed(pred_path, c3=p0, physics17=p1)
        analytic = analytic_origin(p0, p1, y)
        evidence["C3"].update({"probability_hash": array_hash(p0), "brier": analytic["c3_brier"],
                               "mean": analytic["c3_mean"], "std": analytic["c3_std"]})
        evidence["physics17"].update({"probability_hash": array_hash(p1),
                                      "brier": analytic["physics17_brier"],
                                      "mean": analytic["physics17_mean"], "std": analytic["physics17_std"]})
        reports[origin] = {"train_rows": len(train_pos), "validation_rows": len(outer_pos),
                           "train_position_hash": array_hash(train_pos),
                           "validation_position_hash": sealed.position_hash,
                           "validation_row_id_hash": sealed.row_id_hash,
                           "training_target_hash": array_hash(train_y), "outer_target_hash": array_hash(y),
                           "arms": evidence, "analytic": analytic, "predictions_path": str(pred_path),
                           "predictions_literal_sha256": sha256_file(pred_path), "row_parity": True,
                           "outer_labels_read_after_both_predictions_sealed": True}
        vectors[origin] = (p0, p1, y)
        del models
    if fit_count != 4:
        raise ContractError("exactly four fits required")
    final = finalize({name: reports[name]["analytic"] for name in ORIGINS}, vectors)
    report = {"contract_version": cfg["contract_version"], "experiment_id": cfg["experiment_id"],
              "protocol_role": cfg["protocol_role"], "recovery_promotion": False,
              "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root,
                                                  text=True).strip(),
              "runner_sha256": sha256_file(Path(__file__).resolve()), "authorities": authorities,
              "feature_contract": {"c3_count": 50, "physics_count": 17, "final_count": 67,
                                   "c3_features": base, "physics_features": physics,
                                   "categorical_features": cfg["categorical_features"]},
              "catboost": cfg["catboost"], "split": cfg["split"], "origins": reports,
              "final": final, "execution": {"fit_count": fit_count,
                                              "elapsed_seconds": time.time()-started},
              "scope": {"primary_target_access": False, "r2024_target_access": False,
                        "2024_target_access": False, "test_access": False,
                        "test_distribution_access": False, "public_access": False,
                        "gpu_used": False, "v93_used": False, "package_built": False,
                        "raw_trackman_scanned": False, "lookup_only": True}}
    report["canonical_report_sha256"] = canonical_hash(report)
    report_path = output_dir / "c3_trackman_physics_complementarity_v1_report.json"
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report_path


def static_contract() -> dict[str, Any]:
    cfg = load_config()
    return {"static_pass": True, "origins": list(ORIGINS), "fit_count": 4,
            "feature_count": len(cfg["base_c3_features"]+cfg["physics_features"])}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    execute = sub.add_parser("run")
    for name in ("train-csv", "lookup-json", "physics-runner", "physics-config",
                 "c3-contract", "output-dir", "repo-root"):
        execute.add_argument(f"--{name}", type=Path, required=name != "repo-root",
                             default=ROOT if name == "repo-root" else None)
    sub.add_parser("static")
    args = parser.parse_args(argv)
    try:
        result: Any = static_contract() if args.command == "static" else {
            "report_path": str(run(args.train_csv.resolve(), args.lookup_json.resolve(),
                                   args.physics_runner.resolve(), args.physics_config.resolve(),
                                   args.c3_contract.resolve(), args.output_dir.resolve(),
                                   args.repo_root.resolve()))}
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except ContractError as exc:
        print(f"CONTRACT_ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
