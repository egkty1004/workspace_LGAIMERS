#!/usr/bin/env python3
"""Bounded, branch-inert diagnostic for an exact v93 TrackMan residual.

The runner reuses two preserved 30,000-row v93 composite-logit caches and the
already reviewed full-origin C3/Physics17 prediction arrays.  It never fits a
model.  Historical labels are read only after the two prediction sources have
been sealed for the exact bounded scope.

This is deliberately not a full-origin v93 experiment.  A positive result is
only ``BOUNDED_SIGNAL_PRESENT`` and cannot authorize promotion, packaging, or
reconstruction of the missing full-origin v93 authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "exact_v93_bounded_trackman_residual_diagnostic_v0.json"
ROW_ID = "row_id"
TARGET = "control_success"
IDENTITY_COLUMNS = (ROW_ID, "season", "game_type")
ORIGINS = ("r2022", "r2023")


class ContractError(RuntimeError):
    """Raised whenever a frozen authority or diagnostic contract drifts."""


JSON = dict[str, Any]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def array_hash(values: Sequence[Any] | np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _finite_float(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _required_file(path: Path, name: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ContractError(f"authority missing or symlinked: {name}")


def load_config(path: Path = CONFIG_PATH) -> JSON:
    """Load and validate the execution-authoritative, frozen JSON contract."""
    _required_file(path, "config")
    try:
        cfg: JSON = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ContractError(f"config unreadable: {exc}") from exc

    if cfg.get("experiment_id") != "exact-v93-bounded-trackman-residual-diagnostic-v0":
        raise ContractError("experiment identity drift")
    if cfg.get("contract_version") != cfg["experiment_id"]:
        raise ContractError("contract version drift")
    if cfg.get("origins") != list(ORIGINS):
        raise ContractError("origin contract drift")
    v93 = cfg.get("v93", {})
    if v93.get("bounded_rows") != 30000 or v93.get("clip") != [0.3, 0.7]:
        raise ContractError("bounded v93 contract drift")
    if v93.get("c_logit") != -0.0461645795229729:
        raise ContractError("v93 C_LOGIT drift")
    if v93.get("blend") != {
        "z_base": "0.65*z_lgb + 0.35*z_mlp",
        "z_blend": "z_base + 0.17991944576662527*(z_ftt - z_base) + 0.43481381354434545*(z_armb - z_base) + 0.0701066994221915*(z_cat - z_base)",
    }:
        raise ContractError("v93 blend formula drift")
    matched = cfg.get("matched_predictions", {})
    if matched.get("keys") != ["c3", "physics17"]:
        raise ContractError("matched prediction key contract drift")
    if cfg.get("source", {}).get("identity_columns") != list(IDENTITY_COLUMNS):
        raise ContractError("identity projection drift")
    complementarity = cfg.get("complementarity_report", {})
    if complementarity.get("canonical_report_sha256") != "364595a50ce2f340702426af84b186b917ad6f2ed6d899c1cea497e09b52a06b":
        raise ContractError("complementarity canonical authority drift")
    if complementarity.get("runner_sha256") != "2fd8570e4b94644f3ddc4369af5d61b57c7e5a85c5a31decd26d7b0570f8bb2e":
        raise ContractError("complementarity runner authority drift")

    alpha = cfg.get("alpha_analysis", {})
    exact_alpha = {
        "domain": [0.0, 1.0],
        "grid_points": 1001,
        "right_derivative_step": 1.0e-6,
        "derivative_tolerance": 1.0e-12,
        "sign_tolerance": 1.0e-12,
        "boundary_abs_tolerance": 1.0e-12,
        "boundary_max_iterations": 80,
        "boundary_method": "bisection_on_first_positive_to_nonpositive_bracket",
        "component_rule": "only first positive connected component adjacent to zero",
    }
    if alpha != exact_alpha:
        raise ContractError("alpha-analysis contract/config divergence")
    for origin in ORIGINS:
        if origin not in v93.get("logit_files", {}) or origin not in matched:
            raise ContractError(f"missing authority entry: {origin}")
        if matched[origin].get("full_origin_rows", 0) <= 0:
            raise ContractError(f"invalid matched row count: {origin}")
    return cfg


def _configured_path(value: str | Path | None, configured: str, base: Path = ROOT) -> Path:
    path = Path(value) if value is not None else Path(configured)
    return path if path.is_absolute() else (base / path)


def verify_v93_meta(meta_path: Path, cfg: Mapping[str, Any]) -> JSON:
    """Verify that the bounded cache metadata pins the two exact v93 arrays."""
    _required_file(meta_path, "v93 metadata")
    expected = cfg["v93"]["meta_sha256"]
    observed = sha256_file(meta_path)
    if observed != expected:
        raise ContractError(f"v93 metadata hash drift: {observed}")
    try:
        meta: JSON = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ContractError(f"v93 metadata unreadable: {exc}") from exc
    if meta.get("bounded") != cfg["v93"]["bounded_rows"]:
        raise ContractError("v93 bounded scope metadata drift")
    if meta.get("c_logit") != cfg["v93"]["c_logit"] or meta.get("clip") != cfg["v93"]["clip"]:
        raise ContractError("v93 deployed metadata drift")
    if meta.get("blend") != cfg["v93"]["blend"]:
        raise ContractError("v93 blend metadata drift")
    for origin in ORIGINS:
        item = meta.get("origins", {}).get(origin, {})
        configured = cfg["v93"]["logit_files"][origin]
        if item.get("logits_digest") != configured["sha256"]:
            raise ContractError(f"v93 metadata logits digest drift: {origin}")
        if item.get("n_rows") != cfg["v93"]["bounded_rows"]:
            raise ContractError(f"v93 metadata row count drift: {origin}")
    return {"path": str(meta_path), "sha256": observed}


def load_v93_logits(origin: str, path: Path, cfg: Mapping[str, Any]) -> np.ndarray:
    """Load one pinned bounded raw composite-logit array."""
    if origin not in ORIGINS:
        raise ContractError("only r2022/r2023 v93 logits are permitted")
    _required_file(path, f"v93 {origin} logits")
    expected = cfg["v93"]["logit_files"][origin]["sha256"]
    observed = sha256_file(path)
    if observed != expected:
        raise ContractError(f"v93 logits hash drift: {origin}")
    try:
        values = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ContractError(f"v93 logits unreadable: {origin}: {exc}") from exc
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (cfg["v93"]["bounded_rows"],) or not np.isfinite(values).all():
        raise ContractError(f"v93 logits shape/finite drift: {origin}")
    return values


def load_matched_predictions(origin: str, path: Path, cfg: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Load the reviewed full-origin C3/Physics17 arrays without row-level output."""
    if origin not in ORIGINS:
        raise ContractError("only r2022/r2023 matched predictions are permitted")
    _required_file(path, f"matched predictions {origin}")
    expected = cfg["matched_predictions"][origin]
    if sha256_file(path) != expected["sha256"]:
        raise ContractError(f"matched prediction archive hash drift: {origin}")
    try:
        loaded = np.load(path, allow_pickle=False)
        keys = list(loaded.files)
        if keys != cfg["matched_predictions"]["keys"]:
            raise ContractError(f"matched prediction keys drift: {origin}")
        p_c3 = np.asarray(loaded["c3"], dtype=np.float64)
        p_phys = np.asarray(loaded["physics17"], dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise ContractError(f"matched predictions unreadable: {origin}: {exc}") from exc
    rows = expected["full_origin_rows"]
    if p_c3.shape != (rows,) or p_phys.shape != (rows,):
        raise ContractError(f"matched prediction row count drift: {origin}")
    if not np.isfinite(p_c3).all() or not np.isfinite(p_phys).all():
        raise ContractError(f"matched prediction nonfinite values: {origin}")
    if not ((p_c3 > 0.0).all() and (p_c3 < 1.0).all() and
            (p_phys > 0.0).all() and (p_phys < 1.0).all()):
        raise ContractError(f"matched prediction logit domain failure: {origin}")
    return p_c3, p_phys


def verify_complementarity_report(report_path: Path, cfg: Mapping[str, Any]) -> JSON:
    """Verify the report that binds each NPZ archive to its reviewed row scope."""
    _required_file(report_path, "complementarity report")
    authority = cfg["complementarity_report"]
    observed = sha256_file(report_path)
    if observed != authority["sha256"]:
        raise ContractError("complementarity report hash drift")
    try:
        report: JSON = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ContractError(f"complementarity report unreadable: {exc}") from exc
    if report.get("canonical_report_sha256") != authority["canonical_report_sha256"]:
        raise ContractError("complementarity canonical report drift")
    if report.get("runner_sha256") != authority["runner_sha256"]:
        raise ContractError("complementarity runner hash drift")
    for origin in ORIGINS:
        item = report.get("origins", {}).get(origin, {})
        expected = cfg["matched_predictions"][origin]
        if item.get("predictions_literal_sha256") != expected["sha256"]:
            raise ContractError(f"complementarity prediction authority drift: {origin}")
        if item.get("validation_rows") != expected["full_origin_rows"]:
            raise ContractError(f"complementarity row count authority drift: {origin}")
        if item.get("validation_position_hash") != expected["full_position_hash"]:
            raise ContractError(f"complementarity position authority drift: {origin}")
        if item.get("validation_row_id_hash") != expected["full_row_id_hash"]:
            raise ContractError(f"complementarity row identity authority drift: {origin}")
        if item.get("row_parity") is not True or item.get("outer_labels_read_after_both_predictions_sealed") is not True:
            raise ContractError(f"complementarity seal authority drift: {origin}")
    return {"path": str(report_path), "sha256": observed,
            "canonical_report_sha256": report.get("canonical_report_sha256"),
            "runner_sha256": report.get("runner_sha256")}


def read_identity_projection(train_csv: Path) -> pd.DataFrame:
    """Read only row identity and origin columns; never reads the target."""
    _required_file(train_csv, "identity source")
    try:
        header = pd.read_csv(train_csv, nrows=0, encoding="utf-8-sig")
        missing = sorted(set(IDENTITY_COLUMNS) - set(header.columns))
        if missing:
            raise ContractError(f"identity projection missing columns: {missing}")
        frame = pd.read_csv(train_csv, usecols=list(IDENTITY_COLUMNS), encoding="utf-8-sig")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ContractError(f"identity projection unreadable: {exc}") from exc
    if frame[ROW_ID].isna().any():
        raise ContractError("row_id missing")
    ids = frame[ROW_ID].astype(str)
    if ids.duplicated().any():
        raise ContractError("row_id duplicate")
    return frame


def origin_positions(frame: pd.DataFrame, origin: str) -> np.ndarray:
    """Return the full regular-season validation positions for one origin."""
    if origin not in ORIGINS:
        raise ContractError("only r2022/r2023 origins are allowed")
    year = 2022 if origin == "r2022" else 2023
    season = pd.to_numeric(frame["season"], errors="coerce").to_numpy(np.float64)
    regular = frame["game_type"].astype(str).eq("R").to_numpy(bool)
    positions = np.flatnonzero(regular & np.isfinite(season) & (season == year)).astype(np.int64)
    if not len(positions):
        raise ContractError(f"empty origin validation scope: {origin}")
    return positions


def validate_origin_scope(frame: pd.DataFrame, origin: str, positions: np.ndarray,
                          cfg: Mapping[str, Any]) -> JSON:
    expected = cfg["matched_predictions"][origin]
    positions = np.asarray(positions, dtype=np.int64)
    if len(positions) != expected["full_origin_rows"]:
        raise ContractError(f"full origin row count drift: {origin}")
    if array_hash(positions) != expected["full_position_hash"]:
        raise ContractError(f"full origin position hash drift: {origin}")
    ids = frame.iloc[positions][ROW_ID].astype(str).tolist()
    if canonical_hash(ids) != expected["full_row_id_hash"]:
        raise ContractError(f"full origin row_id hash drift: {origin}")
    return {"full_rows": len(positions), "full_position_hash": array_hash(positions),
            "full_row_id_hash": canonical_hash(ids)}


def bounded_positions(full_positions: Sequence[int], cfg: Mapping[str, Any]) -> np.ndarray:
    positions = np.asarray(full_positions, dtype=np.int64)
    n = int(cfg["v93"]["bounded_rows"])
    if len(positions) < n:
        raise ContractError("full origin is shorter than bounded authority")
    bounded = positions[:n].copy()
    if len(np.unique(bounded)) != n:
        raise ContractError("bounded positions are not unique")
    return bounded


def logit_probability(probability: Sequence[float]) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or not ((values > 0) & (values < 1)).all():
        raise ContractError("probability cannot be converted to finite logit")
    return np.log(values) - np.log1p(-values)


def stable_sigmoid(logits: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    output = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    negative_exp = np.exp(values[~positive])
    output[~positive] = negative_exp / (1.0 + negative_exp)
    return output


def deployed_probs(z_v93: Sequence[float], residual: Sequence[float], alpha: float,
                   cfg: Mapping[str, Any]) -> np.ndarray:
    """Apply the frozen v93 offset and clipping to an additive logit residual."""
    if not _finite_float(alpha) or not 0.0 <= float(alpha) <= 1.0:
        raise ContractError("alpha outside frozen [0,1] domain")
    z = np.asarray(z_v93, dtype=np.float64)
    d = np.asarray(residual, dtype=np.float64)
    if z.ndim != 1 or d.shape != z.shape or not np.isfinite(z).all() or not np.isfinite(d).all():
        raise ContractError("residual/deployed logit shape or finite failure")
    v93 = cfg["v93"]
    values = stable_sigmoid(z + float(alpha) * d + float(v93["c_logit"]))
    clipped = np.clip(values, float(v93["clip"][0]), float(v93["clip"][1]))
    if not np.isfinite(clipped).all():
        raise ContractError("deployed probabilities are nonfinite")
    return clipped


@dataclass(frozen=True)
class SealedResidualInputs:
    origin: str
    positions: tuple[int, ...]
    position_hash: str
    row_id_hash: str
    z_v93: tuple[float, ...]
    p_c3: tuple[float, ...]
    p_physics17: tuple[float, ...]
    residual: tuple[float, ...]


def seal_residual_inputs(origin: str, frame: pd.DataFrame, positions: Sequence[int],
                         z_v93: Sequence[float], p_c3: Sequence[float],
                         p_physics17: Sequence[float]) -> SealedResidualInputs:
    """Seal exact row scope and both prediction sources before label access."""
    if origin not in ORIGINS:
        raise ContractError("cannot seal a non-selection origin")
    positions = np.asarray(positions, dtype=np.int64)
    z_v93 = np.asarray(z_v93, dtype=np.float64)
    p_c3 = np.asarray(p_c3, dtype=np.float64)
    p_physics17 = np.asarray(p_physics17, dtype=np.float64)
    n = len(positions)
    if n == 0 or any(values.shape != (n,) for values in (z_v93, p_c3, p_physics17)):
        raise ContractError("sealed prediction shape failure")
    if not all(np.isfinite(values).all() for values in (z_v93, p_c3, p_physics17)):
        raise ContractError("sealed prediction nonfinite failure")
    if not ((p_c3 > 0).all() and (p_c3 < 1).all() and
            (p_physics17 > 0).all() and (p_physics17 < 1).all()):
        raise ContractError("sealed source probabilities outside strict logit domain")
    if np.any(positions < 0) or np.any(positions >= len(frame)):
        raise ContractError("sealed positions outside source frame")
    ids = frame.iloc[positions][ROW_ID].astype(str).tolist()
    c3_logit = logit_probability(p_c3)
    physics_logit = logit_probability(p_physics17)
    residual = physics_logit - c3_logit
    if not np.isfinite(residual).all():
        raise ContractError("nonfinite TrackMan residual")
    return SealedResidualInputs(
        origin=origin,
        positions=tuple(int(value) for value in positions),
        position_hash=array_hash(positions),
        row_id_hash=canonical_hash(ids),
        z_v93=tuple(float(value) for value in z_v93),
        p_c3=tuple(float(value) for value in p_c3),
        p_physics17=tuple(float(value) for value in p_physics17),
        residual=tuple(float(value) for value in residual),
    )


def _read_scoped_rows(train_csv: Path, positions: Sequence[int], columns: Sequence[str]) -> pd.DataFrame:
    wanted = frozenset(int(value) for value in positions)
    if not wanted:
        raise ContractError("empty scoped label request")
    try:
        scoped = pd.read_csv(
            train_csv,
            usecols=list(columns),
            encoding="utf-8-sig",
            skiprows=lambda line: line > 0 and (line - 1) not in wanted,
        )
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ContractError(f"scoped label read failed: {exc}") from exc
    if len(scoped) != len(wanted):
        raise ContractError("scoped label row count drift")
    return scoped


def read_outer_labels(train_csv: Path, frame: pd.DataFrame,
                      sealed: SealedResidualInputs) -> np.ndarray:
    """Read only the sealed r2022/r2023 target scope after predictions are sealed."""
    if not isinstance(sealed, SealedResidualInputs) or sealed.origin not in ORIGINS:
        raise ContractError("outer labels require a valid sealed prediction object")
    positions = np.asarray(sealed.positions, dtype=np.int64)
    ids = frame.iloc[positions][ROW_ID].astype(str).tolist()
    if array_hash(positions) != sealed.position_hash or canonical_hash(ids) != sealed.row_id_hash:
        raise ContractError("sealed row identity drift before label access")
    expected_year = 2022 if sealed.origin == "r2022" else 2023
    scoped = _read_scoped_rows(train_csv, positions, (ROW_ID, "season", "game_type", TARGET))
    if scoped[ROW_ID].astype(str).tolist() != ids:
        raise ContractError("outer label row identity mismatch")
    seasons = pd.to_numeric(scoped["season"], errors="coerce")
    if not seasons.eq(expected_year).all() or not scoped["game_type"].astype(str).eq("R").all():
        raise ContractError("outer label temporal scope failure")
    target = pd.to_numeric(scoped[TARGET], errors="coerce")
    if target.isna().any() or not target.isin((0, 1)).all():
        raise ContractError("outer labels invalid")
    return target.to_numpy(dtype=np.float64)


def brier(probability: Sequence[float], target: Sequence[float]) -> float:
    p = np.asarray(probability, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if p.ndim != 1 or p.shape != y.shape or not len(y):
        raise ContractError("Brier shape failure")
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ContractError("Brier nonfinite input")
    return float(np.mean((p - y) ** 2))


def _g_function(z_v93: np.ndarray, residual: np.ndarray, target: np.ndarray,
                cfg: Mapping[str, Any]) -> Callable[[float], float]:
    baseline = brier(deployed_probs(z_v93, residual, 0.0, cfg), target)

    def evaluate(alpha: float) -> float:
        return baseline - brier(deployed_probs(z_v93, residual, alpha, cfg), target)

    return evaluate


def characterize_positive_component(g: Callable[[float], float],
                                    cfg: Mapping[str, Any]) -> JSON:
    """Find only the first positive component adjacent to zero.

    The fixed grid is a sign/bracketing instrument, not a candidate-weight
    sweep.  Once the first non-positive grid point is found, later grid points
    are deliberately ignored.  Bisection refines that first boundary.
    """
    spec = cfg["alpha_analysis"]
    lower, upper = (float(value) for value in spec["domain"])
    grid = np.linspace(lower, upper, int(spec["grid_points"]), dtype=np.float64)
    try:
        values = np.asarray([float(g(float(alpha))) for alpha in grid], dtype=np.float64)
    except (ContractError, ValueError, TypeError, OverflowError) as exc:
        return {"status": "FAIL", "reason": f"grid_evaluation_failed:{exc}"}
    if values.shape != grid.shape or not np.isfinite(values).all():
        return {"status": "FAIL", "reason": "grid_nonfinite"}
    sign_tol = float(spec["sign_tolerance"])
    derivative_step = float(spec["right_derivative_step"])
    try:
        derivative = (float(g(derivative_step)) - float(values[0])) / derivative_step
    except (ContractError, ValueError, TypeError, OverflowError) as exc:
        return {"status": "FAIL", "reason": f"right_derivative_failed:{exc}"}
    if not math.isfinite(derivative) or derivative <= float(spec["derivative_tolerance"]):
        return {"status": "FAIL", "reason": "right_derivative_not_improvement_directed",
                "right_derivative": derivative}
    if abs(float(values[0])) > sign_tol:
        return {"status": "FAIL", "reason": "g_zero_not_zero", "right_derivative": derivative}
    try:
        anchor_value = float(g(derivative_step))
    except (ContractError, ValueError, TypeError, OverflowError) as exc:
        return {"status": "FAIL", "reason": f"positive_anchor_failed:{exc}",
                "right_derivative": derivative}
    if not math.isfinite(anchor_value) or anchor_value <= sign_tol:
        return {"status": "FAIL", "reason": "positive_anchor_not_established",
                "right_derivative": derivative, "positive_anchor_value": anchor_value}

    boundary_index: int | None = None
    for index in range(1, len(values)):
        if values[index] <= sign_tol:
            boundary_index = index
            break

    if boundary_index is None:
        return {
            "status": "PASS",
            "reason": None,
            "right_derivative": derivative,
            "grid_points": len(grid),
            "positive_grid_prefix_points": len(grid) - 1,
            "positive_anchor_alpha": derivative_step,
            "positive_anchor_value": anchor_value,
            "first_nonpositive_grid_index": None,
            "later_positive_components_ignored": False,
            "boundary_established": True,
            "boundary_kind": "domain_upper_endpoint",
            "U": upper,
            "interval": {"lower": lower, "upper": upper, "lower_open": True, "upper_open": True},
        }

    left = derivative_step if boundary_index == 1 else float(grid[boundary_index - 1])
    right = float(grid[boundary_index])
    left_value = anchor_value if boundary_index == 1 else values[boundary_index - 1]
    if left_value <= sign_tol or values[boundary_index] > sign_tol:
        return {"status": "FAIL", "reason": "invalid_first_boundary_bracket",
                "right_derivative": derivative}
    iterations = int(spec["boundary_max_iterations"])
    abs_tol = float(spec["boundary_abs_tolerance"])
    for _ in range(iterations):
        if right - left <= abs_tol:
            break
        middle = (left + right) / 2.0
        try:
            middle_value = float(g(middle))
        except (ContractError, ValueError, TypeError, OverflowError) as exc:
            return {"status": "FAIL", "reason": f"boundary_evaluation_failed:{exc}",
                    "right_derivative": derivative}
        if not math.isfinite(middle_value):
            return {"status": "FAIL", "reason": "boundary_nonfinite", "right_derivative": derivative}
        if middle_value > sign_tol:
            left = middle
        else:
            right = middle
    if right - left > abs_tol:
        return {"status": "FAIL", "reason": "boundary_not_refined",
                "right_derivative": derivative}
    return {
        "status": "PASS",
        "reason": None,
        "right_derivative": derivative,
        "grid_points": len(grid),
        "positive_grid_prefix_points": boundary_index,
        "positive_anchor_alpha": derivative_step,
        "positive_anchor_value": anchor_value,
        "first_nonpositive_grid_index": boundary_index,
        "later_positive_components_ignored": True,
        "boundary_established": True,
        "boundary_bracket": {"lower": float(left),
                              "upper": float(grid[boundary_index])},
        "boundary_width": right - left,
        "U": right,
        "interval": {"lower": lower, "upper": right, "lower_open": True, "upper_open": True},
    }


def analyze_origin(origin: str, z_v93: Sequence[float], p_c3: Sequence[float],
                   p_physics17: Sequence[float], target: Sequence[float],
                   cfg: Mapping[str, Any]) -> JSON:
    z = np.asarray(z_v93, dtype=np.float64)
    c3 = np.asarray(p_c3, dtype=np.float64)
    physics = np.asarray(p_physics17, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if z.ndim != 1 or c3.shape != z.shape or physics.shape != z.shape or y.shape != z.shape:
        raise ContractError(f"origin analysis shape failure: {origin}")
    residual = logit_probability(physics) - logit_probability(c3)
    baseline = deployed_probs(z, residual, 0.0, cfg)
    g = _g_function(z, residual, y, cfg)
    component = characterize_positive_component(g, cfg)
    result: JSON = {
        "rows": len(y),
        "v93_raw_logit_hash": array_hash(z),
        "c3_probability_hash": array_hash(c3),
        "physics17_probability_hash": array_hash(physics),
        "residual_logit_hash": array_hash(residual),
        "deployed_v93_probability_hash": array_hash(baseline),
        "v93_deployed_brier": brier(baseline, y),
        "v93_deployed_mean": float(np.mean(baseline)),
        "analysis": component,
    }
    return result


def finalize_origins(origin_results: Mapping[str, JSON], vectors: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
                     cfg: Mapping[str, Any]) -> JSON:
    """Apply the two-origin diagnostic-only conjunction at U/2."""
    failures = {origin: item["analysis"].get("reason")
                for origin, item in origin_results.items()
                if item["analysis"].get("status") != "PASS"}
    if failures:
        return {"result": "BOUNDED_SIGNAL_ABSENT", "shared_interval": None,
                "alpha_diagnostic": None, "origin_analysis_failures": failures,
                "direct": {}}
    upper = min(float(origin_results[origin]["analysis"]["U"]) for origin in ORIGINS)
    if not math.isfinite(upper) or upper <= 0.0:
        return {"result": "BOUNDED_SIGNAL_ABSENT", "shared_interval": None,
                "alpha_diagnostic": None, "origin_analysis_failures": {}, "direct": {}}
    alpha = upper / 2.0
    direct: JSON = {}
    all_pass = True
    for origin in ORIGINS:
        z, residual, target = vectors[origin]
        baseline = deployed_probs(z, residual, 0.0, cfg)
        blended = deployed_probs(z, residual, alpha, cfg)
        baseline_brier = brier(baseline, target)
        blended_brier = brier(blended, target)
        passed = bool(np.isfinite(blended).all() and blended_brier < baseline_brier)
        all_pass = all_pass and passed
        direct[origin] = {
            "alpha": alpha,
            "baseline_brier": baseline_brier,
            "diagnostic_brier": blended_brier,
            "absolute_brier_improvement": baseline_brier - blended_brier,
            "diagnostic_probability_hash": array_hash(blended),
            "passed": passed,
        }
    return {
        "result": "BOUNDED_SIGNAL_PRESENT" if all_pass else "BOUNDED_SIGNAL_ABSENT",
        "shared_interval": {"lower": 0.0, "upper": upper,
                            "lower_open": True, "upper_open": True},
        "alpha_diagnostic": alpha,
        "origin_analysis_failures": {},
        "direct": direct,
    }


def verify_authorities(cfg: Mapping[str, Any], paths: Mapping[str, Path]) -> JSON:
    meta = verify_v93_meta(paths["v93_meta"], cfg)
    report: JSON = {"v93_meta": meta,
                    "complementarity_report": verify_complementarity_report(
                        paths["complementarity_report"], cfg)}
    for origin in ORIGINS:
        v93_path = paths[f"v93_{origin}"]
        matched_path = paths[f"matched_{origin}"]
        _required_file(v93_path, f"v93 {origin} logits")
        _required_file(matched_path, f"matched {origin} predictions")
        report[f"v93_{origin}"] = {"path": str(v93_path), "sha256": sha256_file(v93_path),
                                    "rows": cfg["v93"]["bounded_rows"]}
        report[f"matched_{origin}"] = {"path": str(matched_path), "sha256": sha256_file(matched_path),
                                        "rows": cfg["matched_predictions"][origin]["full_origin_rows"]}
        if report[f"v93_{origin}"]["sha256"] != cfg["v93"]["logit_files"][origin]["sha256"]:
            raise ContractError(f"v93 authority drift: {origin}")
        if report[f"matched_{origin}"]["sha256"] != cfg["matched_predictions"][origin]["sha256"]:
            raise ContractError(f"matched authority drift: {origin}")
    return report


def build_report(cfg: Mapping[str, Any], git_sha: str, runner_sha: str,
                 config_sha: str, source_projection: JSON, authorities: JSON,
                 origins: JSON, final: JSON) -> JSON:
    report: JSON = {
        "contract_version": cfg["contract_version"],
        "experiment_id": cfg["experiment_id"],
        "protocol_role": cfg["protocol_role"],
        "diagnostic_only": True,
        "recovery_promotion": False,
        "git_sha": git_sha,
        "runner_sha256": runner_sha,
        "config_sha256": config_sha,
        "authorities": authorities,
        "source_projection": source_projection,
        "alpha_analysis_contract": cfg["alpha_analysis"],
        "v93_contract": {
            "bounded_rows": cfg["v93"]["bounded_rows"],
            "c_logit": cfg["v93"]["c_logit"],
            "clip": cfg["v93"]["clip"],
            "blend_formula": cfg["v93"]["blend"],
            "residual_formula": "logit(p_Physics17) - logit(p_C3)",
            "candidate_formula": "z_v93 + alpha * residual",
        },
        "origins": origins,
        "final": final,
        "execution": {"model_fits": 0, "official_data_model_run": False,
                       "prediction_reuse": True, "labels_read_after_seal": True},
        "scope": {
            "primary_target_access": False,
            "r2024_target_access": False,
            "test_access": False,
            "test_distribution_access": False,
            "public_access": False,
            "external_information_access": False,
            "row_level_output": False,
            "models_fit": 0,
            "gpu_used": False,
            "package_built": False,
            "branch_inert": True,
        },
    }
    report["canonical_report_sha256"] = canonical_hash(report)
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    final = report["final"]
    lines = [
        "# Exact v93 Bounded TrackMan Residual Diagnostic v0",
        "",
        "This is bounded first-30k diagnostic evidence only; it is not full-origin v93 evidence.",
        "",
        f"- Result: `{final['result']}`",
        f"- Canonical report SHA-256: `{report['canonical_report_sha256']}`",
        f"- Alpha-analysis grid: `{report['alpha_analysis_contract']['grid_points']}` fixed points on `[0,1]`",
        "- Model fits: `0`",
        "- Recovery promotion: `false`",
        "",
        "## Origin diagnostics",
        "",
    ]
    for origin in ORIGINS:
        item = report["origins"].get(origin, {})
        analysis = item.get("analysis", {})
        lines.extend([
            f"### {origin}",
            f"- Rows: `{item.get('rows')}`",
            f"- Deployed v93 Brier: `{item.get('v93_deployed_brier')}`",
            f"- Right derivative: `{analysis.get('right_derivative')}`",
            f"- First-component status: `{analysis.get('status')}`",
            f"- U: `{analysis.get('U')}`",
        ])
        direct = final.get("direct", {}).get(origin)
        if direct:
            lines.extend([
                f"- Diagnostic alpha: `{direct['alpha']}`",
                f"- Diagnostic Brier: `{direct['diagnostic_brier']}`",
                f"- Absolute improvement: `{direct['absolute_brier_improvement']}`",
            ])
        lines.append("")
    lines.extend([
        "## Scope",
        "",
        "No primary/r2024/test/Public data, model fitting, package construction, or row-level output was used.",
        "",
    ])
    return "\n".join(lines)


def write_report(output_dir: Path, report: JSON) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "exact_v93_bounded_trackman_residual_diagnostic_v0_report.json"
    md_path = output_dir / "exact_v93_bounded_trackman_residual_diagnostic_v0_report.md"
    json_path.write_text(json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def _outside_repo(path: Path, repo_root: Path) -> None:
    try:
        path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return
    raise ContractError("output directory must be outside repository")


def run(train_csv: Path, output_dir: Path, repo_root: Path = ROOT,
        config_path: Path = CONFIG_PATH, v93_meta: Path | None = None,
        v93_r2022: Path | None = None, v93_r2023: Path | None = None,
        matched_r2022: Path | None = None, matched_r2023: Path | None = None) -> Path:
    """Execute the bounded NumPy diagnostic once; no model fitting occurs."""
    cfg = load_config(config_path)
    _outside_repo(output_dir, repo_root)
    if output_dir.exists():
        raise ContractError("output directory exists; refusing overwrite")
    paths = {
        "v93_meta": _configured_path(v93_meta, cfg["v93"]["meta_path"]),
        "v93_r2022": _configured_path(v93_r2022, cfg["v93"]["logit_files"]["r2022"]["path"]),
        "v93_r2023": _configured_path(v93_r2023, cfg["v93"]["logit_files"]["r2023"]["path"]),
        "matched_r2022": _configured_path(matched_r2022, cfg["matched_predictions"]["r2022"]["path"]),
        "matched_r2023": _configured_path(matched_r2023, cfg["matched_predictions"]["r2023"]["path"]),
        "complementarity_report": _configured_path(
            None, cfg["complementarity_report"]["path"]),
    }
    authorities = verify_authorities(cfg, paths)
    meta = verify_v93_meta(paths["v93_meta"], cfg)
    frame = read_identity_projection(train_csv)
    source_projection: JSON = {
        "columns": list(IDENTITY_COLUMNS),
        "rows": len(frame),
        "identity_projection_hash": canonical_hash({
            "row_id_hash": canonical_hash(frame[ROW_ID].astype(str).tolist()),
            "season_hash": array_hash(pd.to_numeric(frame["season"], errors="coerce").to_numpy(np.float64)),
            "game_type_hash": canonical_hash(frame["game_type"].astype(str).tolist()),
        }),
        "target_materialized_before_seal": False,
    }
    origin_reports: JSON = {}
    vectors: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for origin in ORIGINS:
        full = origin_positions(frame, origin)
        full_scope = validate_origin_scope(frame, origin, full, cfg)
        bounded = bounded_positions(full, cfg)
        z = load_v93_logits(origin, paths[f"v93_{origin}"], cfg)
        p_c3_full, p_phys_full = load_matched_predictions(origin, paths[f"matched_{origin}"], cfg)
        p_c3, p_phys = p_c3_full[:len(bounded)], p_phys_full[:len(bounded)]
        sealed = seal_residual_inputs(origin, frame, bounded, z, p_c3, p_phys)
        # The only target reader is deliberately after the seal construction.
        target = read_outer_labels(train_csv, frame, sealed)
        z_bounded = np.asarray(sealed.z_v93, dtype=np.float64)
        c3_bounded = np.asarray(sealed.p_c3, dtype=np.float64)
        phys_bounded = np.asarray(sealed.p_physics17, dtype=np.float64)
        residual = np.asarray(sealed.residual, dtype=np.float64)
        item = analyze_origin(origin, z_bounded, c3_bounded, phys_bounded, target, cfg)
        item.update({"scope": full_scope, "bounded_rows": len(bounded),
                     "bounded_position_hash": sealed.position_hash,
                     "bounded_row_id_hash": sealed.row_id_hash,
                     "bounded_v93_logit_hash": array_hash(z_bounded),
                     "outer_labels_read_after_prediction_seal": True,
                     "target_hash": array_hash(target),
                     "residual_logit_hash": array_hash(residual)})
        origin_reports[origin] = item
        vectors[origin] = (z_bounded, residual, target)
    final = finalize_origins(origin_reports, vectors, cfg)
    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root,
                             capture_output=True, text=True, check=False).stdout.strip() or "unknown"
    report = build_report(cfg, git_sha, sha256_file(Path(__file__).resolve()),
                          sha256_file(config_path), source_projection, authorities,
                          origin_reports, final)
    json_path, _ = write_report(output_dir, report)
    return json_path


def static_contract(config_path: Path = CONFIG_PATH) -> JSON:
    cfg = load_config(config_path)
    source = Path(__file__).read_text(encoding="utf-8").lower()
    forbidden_runtime_references = (
        "cat" + "boost", "recovery_" + "evaluator",
        "trackman_history" + "." + "csv", "test" + "." + "csv",
    )
    present = [token for token in forbidden_runtime_references if token in source]
    if present:
        raise ContractError(f"forbidden runtime dependency token(s): {present}")
    return {"static_pass": True, "origins": list(ORIGINS), "model_fits": 0,
            "bounded_rows": cfg["v93"]["bounded_rows"],
            "grid_points": cfg["alpha_analysis"]["grid_points"],
            "diagnostic_only": True}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    static = sub.add_parser("static")
    static.add_argument("--config", type=Path, default=CONFIG_PATH)
    execute = sub.add_parser("run")
    execute.add_argument("--train-csv", type=Path, required=True)
    execute.add_argument("--output-dir", type=Path, required=True)
    execute.add_argument("--repo-root", type=Path, default=ROOT)
    execute.add_argument("--config", type=Path, default=CONFIG_PATH)
    execute.add_argument("--v93-meta", type=Path)
    execute.add_argument("--v93-r2022", type=Path)
    execute.add_argument("--v93-r2023", type=Path)
    execute.add_argument("--matched-r2022", type=Path)
    execute.add_argument("--matched-r2023", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "static":
            print(json.dumps(static_contract(args.config), sort_keys=True, allow_nan=False))
        else:
            report_path = run(args.train_csv.resolve(), args.output_dir.resolve(), args.repo_root.resolve(),
                              args.config.resolve(), args.v93_meta.resolve() if args.v93_meta else None,
                              args.v93_r2022.resolve() if args.v93_r2022 else None,
                              args.v93_r2023.resolve() if args.v93_r2023 else None,
                              args.matched_r2022.resolve() if args.matched_r2022 else None,
                              args.matched_r2023.resolve() if args.matched_r2023 else None)
            print(json.dumps({"report_path": str(report_path)}, sort_keys=True, allow_nan=False))
        return 0
    except ContractError as exc:
        print(f"CONTRACT_ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
