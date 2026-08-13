#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import tokenize
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

GPU_PATH: Final = "/home/" + "gpu_01/"
EVAL_VERSIONS: Final = {
    "pandas": "2.0.3",
    "numpy": "1.26.4",
    "lightgbm": "4.7.0",
    "torch": "2.1.0",
    "xgboost": "3.2.0",
}
HISTORICAL_REFS: Final = {
    ("repro_979/boundary_check.py", 21): "module docstring preserving the original invocation",
    ("repro_979/gate_verdict.py", 188): "ledger payload preserving the command that produced historical results",
}
FEATURE_COLUMNS: Final = (
    "row_id", "season", "game_month", "game_dayofweek", "inning", "top_bottom",
    "game_type", "balls_before", "strikes_before", "outs_before", "run_top_before",
    "run_bot_before", "run_total_before", "score_diff_home", "score_diff_pitcher_team",
    "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on", "base_state",
    "home_win_expectancy", "away_win_expectancy", "li", "pitcher_id", "batter_id",
    "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id", "asof_pitcher_n",
    "asof_pitcher_success_rate", "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
)


@dataclass(frozen=True, slots=True)
class PathReference:
    path: str
    line: int
    reason: str


def _docstring_lines(source: str) -> set[int]:
    tree = ast.parse(source)
    lines: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not body or not isinstance(body, list):
            continue
        first = body[0]
        if not isinstance(first, ast.Expr) or not isinstance(first.value, ast.Constant):
            continue
        if not isinstance(first.value.value, str):
            continue
        lines.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    return lines


def _comment_lines(source: str) -> set[int]:
    return {
        token.start[0]
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.COMMENT and GPU_PATH in token.string
    }


def scan_paths(project_root: Path, scan_roots: tuple[Path, ...]) -> tuple[list[PathReference], list[PathReference], int]:
    violations: list[PathReference] = []
    historical: list[PathReference] = []
    files = sorted({path for root in scan_roots for suffix in ("*.py", "*.sh") for path in root.rglob(suffix)})
    for path in files:
        source = path.read_text(encoding="utf-8")
        if GPU_PATH not in source:
            continue
        relative = path.relative_to(project_root).as_posix() if path.is_relative_to(project_root) else path.name
        docstrings: set[int] = set()
        comments: set[int] = set()
        if path.suffix == ".py":
            docstrings = _docstring_lines(source)
            comments = _comment_lines(source)
        for line_number, line in enumerate(source.splitlines(), start=1):
            if GPU_PATH not in line:
                continue
            whitelist_reason = HISTORICAL_REFS.get((relative, line_number))
            if whitelist_reason is not None:
                historical.append(PathReference(relative, line_number, whitelist_reason))
            elif line_number in docstrings or line_number in comments or (path.suffix == ".sh" and line.lstrip().startswith("#")):
                historical.append(PathReference(relative, line_number, "documentation or comment"))
            else:
                violations.append(PathReference(relative, line_number, "executable hardcoded GPU path"))
    return violations, historical, len(files)


def check_datasets(project_root: Path) -> tuple[dict[str, bool], list[str]]:
    data_dir = project_root / "repro_979" / "open" / "data"
    expected = {
        "train.csv": [*FEATURE_COLUMNS, "control_success"],
        "test.csv": list(FEATURE_COLUMNS),
        "sample_submission.csv": ["row_id", "control_success"],
    }
    results: dict[str, bool] = {}
    failures: list[str] = []
    for name, expected_header in expected.items():
        path = data_dir / name
        if not path.is_file():
            results[name] = False
            failures.append(f"missing dataset: {path}")
            continue
        with path.open(encoding="utf-8-sig", newline="") as handle:
            actual_header = next(csv.reader(handle), [])
        results[name] = actual_header == expected_header
        if not results[name]:
            failures.append(f"header mismatch: {path}")
    return results, failures


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_champion(project_root: Path) -> tuple[dict[str, str], list[str]]:
    champion = project_root / "team_member_materials" / "GIHO" / "submit979_extract"
    required = [Path("script.py"), Path("model/mlp_prep.pkl")]
    required.extend(Path(f"model/f3_s{seed}.txt") for seed in range(42, 52))
    required.extend(Path(f"model/mlp_s{seed}.pt") for seed in range(42, 52))
    failures = [f"missing champion artifact: {champion / relative}" for relative in required if not (champion / relative).is_file()]
    control_files = [champion / "script.py"]
    model_dir = champion / "model"
    if model_dir.is_dir():
        control_files.extend(sorted(path for path in model_dir.iterdir() if path.is_file()))
    manifest = {
        path.relative_to(champion).as_posix(): sha256(path)
        for path in control_files
        if path.is_file()
    }
    return manifest, failures


def audit_environment() -> tuple[dict[str, str | bool], list[str]]:
    versions: dict[str, str | bool] = {"python": sys.version.split()[0]}
    warnings: list[str] = []
    code = (
        "import json, lightgbm, numpy, pandas, torch, xgboost; "
        "print(json.dumps({'lightgbm': lightgbm.__version__, 'numpy': numpy.__version__, "
        "'pandas': pandas.__version__, 'torch': torch.__version__, 'xgboost': xgboost.__version__, "
        "'cuda_available': torch.cuda.is_available()}))"
    )
    process = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False)
    if process.returncode == 0:
        runtime = json.loads(process.stdout)
        for package, expected in EVAL_VERSIONS.items():
            actual = str(runtime[package])
            versions[package] = actual
            if actual.split("+", maxsplit=1)[0] != expected:
                warnings.append(f"{package} {actual} != evaluation expectation {expected}")
        versions["cuda_available"] = bool(runtime["cuda_available"])
    else:
        for package in EVAL_VERSIONS:
            versions[package] = "import failed"
        warnings.append(f"dependency import audit failed: {process.stderr.strip() or process.stdout.strip()}")
        versions["cuda_available"] = False
    if not versions["cuda_available"]:
        warnings.append("torch CUDA unavailable; evaluation environment provides an L4 GPU with CUDA 12.8")
    return versions, warnings


def run_self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="aimers9-migration-audit-") as directory:
        root = Path(directory)
        injected = root / "injected.py"
        injected.write_text("MODEL = '" + GPU_PATH + "workspace/model.pkl'\n", encoding="utf-8")
        violations, _, scanned = scan_paths(root, (root,))
        injected_exit = 1 if violations else 0
    if injected_exit != 1 or scanned != 1:
        print("[FAIL] self-test: executable GPU path did not produce exit 1")
        return 1
    print("[PASS] self-test: injected executable GPU path produced strict exit 1; temporary file cleaned")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Audit migrated Aimers9 paths, inputs, and champion integrity")
    parser.add_argument("--strict", action="store_true", help="fail on executable hardcoded GPU paths")
    parser.add_argument("--self-test", action="store_true", help="prove strict path detection with a temporary fixture")
    parser.add_argument("--project-root", default=os.environ.get("LGAIMERS_ROOT", Path(__file__).resolve().parent.parent))
    parser.add_argument("--evidence", default=None)
    args = parser.parse_args(argv)
    if args.self_test:
        return run_self_test()

    project_root = Path(args.project_root).expanduser().resolve()
    evidence_path = Path(args.evidence).expanduser().resolve() if args.evidence else project_root / ".omo" / "evidence" / "aimers9-top100" / "task-1-portability.json"
    violations, historical, scanned = scan_paths(project_root, (project_root / "repro_979", project_root / "experiments"))
    datasets, dataset_failures = check_datasets(project_root)
    champion_hashes, champion_failures = check_champion(project_root)
    environment, environment_warnings = audit_environment()
    failures = [*dataset_failures, *champion_failures]
    if args.strict:
        failures.extend(f"{ref.path}:{ref.line}: {ref.reason}" for ref in violations)

    evidence = {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "strict": args.strict,
        "summary": {"scanned_files": scanned, "executable_gpu_paths": len(violations), "historical_gpu_paths": len(historical), "failures": len(failures)},
        "historical_references": [asdict(reference) for reference in historical],
        "dataset_headers": datasets,
        "champion_sha256": champion_hashes,
        "environment": environment,
        "environment_warnings": environment_warnings,
        "failures": failures,
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[{'PASS' if not violations else 'FAIL'}] executable hardcoded GPU paths: {len(violations)} / scanned files: {scanned}")
    for reference in historical:
        print(f"[WARNING] historical reference preserved: {reference.path}:{reference.line} — {reference.reason}")
    print(f"[{'PASS' if all(datasets.values()) else 'FAIL'}] dataset headers: {datasets}")
    print(f"[{'PASS' if not champion_failures else 'FAIL'}] champion control files: {len(champion_hashes)} hashed")
    for warning in environment_warnings:
        print(f"[WARNING] environment: {warning}")
    print(f"[INFO] evidence: {evidence_path}")
    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        print("RESULT: FAIL")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
