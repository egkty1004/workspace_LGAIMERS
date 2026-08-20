#!/usr/bin/env python3
"""Create or audit the Aimers 9 school-GPU development workspace.

This tool never copies or extracts competition assets. It validates external
sources, creates only the ignored symlink bridges required by the current
recovery code, and creates the development Conda environment only in explicit
apply mode. Check mode is read-only.

Passing this development check is not official submission readiness.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


DEFAULT_ENV_NAME = "aimers9-dev"
PYTHON_VERSION = "3.11.15"
TORCH_VERSION = "2.7.1+cu128"
TORCH_CUDA_VERSION = "12.8"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
PYPI_INDEX_URL = "https://pypi.org/simple"
CONDA_SOLVER = "libmamba"

DATA_FILES = ("train.csv", "test.csv", "sample_submission.csv")
PACKAGE_SOURCE_FILES = ("script.py", "common.py", "mlp_model.py", "requirements.txt")
PREP_FILES = (
    "mlp_prep.pkl",
    "ftt_prep.pkl",
    "armb_prep.pkl",
    "catboost_prep.pkl",
)
METADATA_FILES = (
    "mlp_meta.json",
    "ftt_meta.json",
    "armb_meta.json",
    "train_meta.json",
)
SEEDS = tuple(range(42, 52))
FTT_SEEDS = (42, 43, 44)
REQUIRED_MODEL_FILES = (
    *(f"f3_s{seed}.txt" for seed in SEEDS),
    *(f"mlp_s{seed}.pt" for seed in SEEDS),
    *(f"catboost_s{seed}.cbm" for seed in SEEDS),
    *(f"armb_s{seed}.pt" for seed in SEEDS),
    *(f"ftt_s{seed}.pt" for seed in FTT_SEEDS),
)

EXPECTED_PACKAGES = {
    "numpy": "1.26.4",
    "pandas": "2.0.3",
    "scipy": "1.15.3",
    "scikit-learn": "1.8.0",
    "joblib": "1.5.3",
    "threadpoolctl": "3.6.0",
    "lightgbm": "4.7.0",
    "catboost": "1.2.10",
    "xgboost": "3.2.0",
    "torch": TORCH_VERSION,
}

ENV_AUDIT_CODE = """
import importlib.metadata
import json
import platform
import torch

names = %s
print(json.dumps({
    "python": platform.python_version(),
    "packages": {name: importlib.metadata.version(name) for name in names},
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "torch_cuda_available": torch.cuda.is_available(),
}, sort_keys=True))
""" % repr(tuple(EXPECTED_PACKAGES))


class BootstrapError(RuntimeError):
    """Unsafe, incomplete, or conflicting bootstrap state."""


@dataclass(frozen=True)
class V93Contract:
    archive_sha256: str
    registry_model_files: int
    model_artifact_identities: frozenset[str]


@dataclass(frozen=True)
class LinkSpec:
    source: Path
    destination: Path
    label: str


@dataclass(frozen=True)
class GpuInspection:
    ready: bool
    devices: tuple[str, ...]
    processes: tuple[str, ...]
    problems: tuple[str, ...]


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def canonical(path: Path) -> Path:
    return path.expanduser().resolve()


def read_csv_header(path: Path) -> list[str]:
    """Read exactly the CSV header; never load data rows."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle), [])
    except (OSError, UnicodeError, csv.Error) as exc:
        raise BootstrapError(f"cannot read CSV header {path}: {exc}") from exc
    if not header:
        raise BootstrapError(f"empty CSV header: {path}")
    if len(header) != len(set(header)):
        raise BootstrapError(f"duplicate CSV columns: {path}")
    return header


def validate_data_dir(data_dir: Path) -> dict[str, list[str]]:
    data_dir = canonical(data_dir)
    if not data_dir.is_dir():
        raise BootstrapError(f"official data directory is missing: {data_dir}")
    paths = {name: data_dir / name for name in DATA_FILES}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise BootstrapError("missing official data files: " + ", ".join(missing))

    headers = {name: read_csv_header(path) for name, path in paths.items()}
    train = headers["train.csv"]
    test = headers["test.csv"]
    sample = headers["sample_submission.csv"]
    if "row_id" not in train or "control_success" not in train:
        raise BootstrapError("train.csv header must contain row_id and control_success")
    if "row_id" not in test or "control_success" in test:
        raise BootstrapError("test.csv header must contain row_id and exclude control_success")
    if [column for column in train if column != "control_success"] != test:
        raise BootstrapError("train/test feature headers differ")
    if sample != ["row_id", "control_success"]:
        raise BootstrapError(
            "sample_submission.csv header must be exactly row_id,control_success"
        )
    return headers


def load_v93_contract(repo_root: Path) -> V93Contract:
    registry = repo_root / "repro_979" / "recovery_candidate_registry.json"
    evidence = (
        repo_root
        / ".omo"
        / "evidence"
        / "aimers9-top100-recovery"
        / "task-3-baseline-registry.json"
    )
    try:
        payload = json.loads(registry.read_text(encoding="utf-8"))
        baseline = payload["baseline"]
        archive_sha256 = str(baseline["sha256"])
        model_files = int(baseline["model_files"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise BootstrapError(f"cannot read v93 registry contract {registry}: {exc}") from exc
    if not re.fullmatch(r"[0-9a-f]{64}", archive_sha256):
        raise BootstrapError("v93 registry archive SHA-256 is invalid")
    if model_files <= 0:
        raise BootstrapError("v93 registry model_files must be positive")
    try:
        evidence_payload = json.loads(evidence.read_text(encoding="utf-8"))
        evidence_baseline = evidence_payload["baseline"]
        evidence_hashes = evidence_baseline["package_hashes"]["model_files"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise BootstrapError(
            f"cannot read authoritative v93 model identities {evidence}: {exc}"
        ) from exc
    if not isinstance(evidence_hashes, dict):
        raise BootstrapError("authoritative v93 model identity map is not a dictionary")
    identities = frozenset(str(name) for name in evidence_hashes)
    if len(identities) != model_files:
        raise BootstrapError(
            "v93 registry/evidence model-file-count conflict: "
            f"registry={model_files}, evidence={len(identities)}"
        )
    invalid_hashes = [
        name
        for name, digest in evidence_hashes.items()
        if not isinstance(digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
    ]
    if invalid_hashes:
        raise BootstrapError(
            "invalid hashes in authoritative v93 identity evidence: "
            + ", ".join(sorted(invalid_hashes))
        )
    expected_identities = frozenset(
        (*REQUIRED_MODEL_FILES, *PREP_FILES, *METADATA_FILES)
    )
    if identities != expected_identities:
        missing = sorted(expected_identities - identities)
        unexpected = sorted(identities - expected_identities)
        raise BootstrapError(
            f"authoritative v93 identity categories conflict: missing={missing}, "
            f"unexpected={unexpected}"
        )
    if evidence_baseline.get("sha256") != archive_sha256:
        raise BootstrapError("v93 registry and identity evidence archive SHA-256 differ")
    return V93Contract(
        archive_sha256=archive_sha256,
        registry_model_files=model_files,
        model_artifact_identities=identities,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_v93_dir(
    v93_dir: Path,
    contract: V93Contract,
    archive: Path | None = None,
) -> bool:
    """Validate extracted development assets; return archive-provenance status."""
    v93_dir = canonical(v93_dir)
    if not v93_dir.is_dir():
        raise BootstrapError(f"v93 extracted directory is missing: {v93_dir}")

    missing_sources = [
        name for name in PACKAGE_SOURCE_FILES if not (v93_dir / name).is_file()
    ]
    if missing_sources:
        raise BootstrapError(
            "missing v93 package source files: " + ", ".join(missing_sources)
        )

    model_dir = v93_dir / "model"
    if not model_dir.is_dir():
        raise BootstrapError(f"v93 model directory is missing: {model_dir}")
    missing_prep = [
        name for name in PREP_FILES if not (model_dir / name).is_file()
    ]
    if missing_prep:
        raise BootstrapError(
            "missing required v93 prep files: " + ", ".join(missing_prep)
        )
    missing_required = [
        name for name in REQUIRED_MODEL_FILES if not (model_dir / name).is_file()
    ]
    if missing_required:
        raise BootstrapError(
            "missing required v93 model files: " + ", ".join(missing_required)
        )
    missing_metadata = [
        name for name in METADATA_FILES if not (model_dir / name).is_file()
    ]
    if missing_metadata:
        raise BootstrapError(
            "missing required v93 metadata files: " + ", ".join(missing_metadata)
        )

    actual_identities = frozenset(
        path.relative_to(model_dir).as_posix()
        for path in model_dir.rglob("*")
        if path.is_file()
    )
    if actual_identities != contract.model_artifact_identities:
        missing = sorted(contract.model_artifact_identities - actual_identities)
        unexpected = sorted(actual_identities - contract.model_artifact_identities)
        raise BootstrapError(
            f"v93 model/ identity mismatch: missing={missing}, unexpected={unexpected}"
        )

    if archive is None:
        return False
    archive = canonical(archive)
    if not archive.is_file():
        raise BootstrapError(f"v93 archive is missing: {archive}")
    try:
        actual = sha256_file(archive)
    except OSError as exc:
        raise BootstrapError(f"cannot hash v93 archive {archive}: {exc}") from exc
    if actual != contract.archive_sha256:
        raise BootstrapError(
            f"v93 archive SHA-256 mismatch: {actual} != {contract.archive_sha256}"
        )
    return True


def build_link_specs(
    repo_root: Path, data_dir: Path, v93_dir: Path
) -> list[LinkSpec]:
    repo_root = canonical(repo_root)
    data_dir = canonical(data_dir)
    v93_dir = canonical(v93_dir)
    data_destination = repo_root / "repro_979" / "open" / "data"
    v93_destination = repo_root / "repro_979" / "cache" / "v93_extract_verify"
    if data_dir == data_destination:
        raise BootstrapError("official data source cannot be the recovery bridge directory")
    if v93_dir == v93_destination:
        raise BootstrapError("v93 source cannot be the recovery bridge destination")
    specs = [
        LinkSpec(data_dir / name, data_destination / name, f"official data: {name}")
        for name in DATA_FILES
    ]
    specs.append(LinkSpec(v93_dir, v93_destination, "v93 extracted assets"))
    return specs


def _resolved_link_target(path: Path) -> Path:
    raw_target = Path(os.readlink(path))
    target = raw_target if raw_target.is_absolute() else path.parent / raw_target
    return target.resolve(strict=False)


def link_state(spec: LinkSpec) -> str:
    expected = spec.source.resolve(strict=True)
    destination = spec.destination
    if destination.is_symlink():
        actual = _resolved_link_target(destination)
        if actual != expected:
            raise BootstrapError(
                f"wrong or broken symlink for {spec.label}: {destination} -> {actual}; "
                f"expected {expected}"
            )
        return "ready"
    if destination.exists():
        raise BootstrapError(
            f"refusing to overwrite non-symlink destination for {spec.label}: "
            f"{destination}"
        )
    return "missing"


def validate_parent_paths(repo_root: Path, specs: Sequence[LinkSpec]) -> None:
    repo_root = canonical(repo_root)
    for spec in specs:
        current = spec.destination.parent
        while current != repo_root:
            if current.is_symlink():
                raise BootstrapError(f"refusing symlinked destination parent: {current}")
            if current.exists() and not current.is_dir():
                raise BootstrapError(f"destination parent is not a directory: {current}")
            if repo_root not in current.parents:
                raise BootstrapError(f"destination escapes repository: {spec.destination}")
            current = current.parent


def inspect_links(repo_root: Path, specs: Sequence[LinkSpec]) -> dict[str, str]:
    validate_parent_paths(repo_root, specs)
    return {spec.label: link_state(spec) for spec in specs}


def apply_links(repo_root: Path, specs: Sequence[LinkSpec]) -> dict[str, str]:
    """Create only absent links after the entire link plan has passed inspection."""
    states = inspect_links(repo_root, specs)
    for spec in specs:
        if states[spec.label] == "ready":
            continue
        spec.destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            spec.destination.symlink_to(
                spec.source, target_is_directory=spec.source.is_dir()
            )
        except FileExistsError as exc:
            raise BootstrapError(
                f"destination appeared during apply: {spec.destination}"
            ) from exc
    return inspect_links(repo_root, specs)


def _run(
    command: Sequence[str], *, check: bool = True, stage: str = "command"
) -> subprocess.CompletedProcess[str]:
    rendered_command = " ".join(command)
    try:
        return subprocess.run(
            list(command), check=check, capture_output=True, text=True, timeout=600
        )
    except subprocess.TimeoutExpired as exc:
        raise BootstrapError(
            f"{stage} timed out after {exc.timeout} seconds: {rendered_command}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        detail_suffix = f"; output: {detail[-2000:]}" if detail else ""
        raise BootstrapError(
            f"{stage} failed with exit code {exc.returncode}: "
            f"{rendered_command}{detail_suffix}"
        ) from exc
    except OSError as exc:
        raise BootstrapError(
            f"{stage} could not start: {rendered_command}: {exc}"
        ) from exc


def conda_executable() -> str:
    executable = shutil.which("conda")
    if executable is None:
        raise BootstrapError("conda is not available on PATH")
    return executable


def conda_env_exists(conda: str, env_name: str) -> bool:
    result = _run((conda, "env", "list", "--json"))
    try:
        environments = json.loads(result.stdout)["envs"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise BootstrapError("cannot parse 'conda env list --json'") from exc
    return any(Path(path).name == env_name for path in environments)


def parse_audit_output(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "packages" in value:
            return value
    raise BootstrapError("environment audit did not emit JSON")


def validate_environment_report(report: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    if report.get("python") != PYTHON_VERSION:
        problems.append(f"python {report.get('python')} != {PYTHON_VERSION}")
    packages = report.get("packages")
    if not isinstance(packages, dict):
        problems.append("package version map missing")
    else:
        for name, expected in EXPECTED_PACKAGES.items():
            if packages.get(name) != expected:
                problems.append(f"{name} {packages.get(name)} != {expected}")
    if report.get("torch_version") != TORCH_VERSION:
        problems.append(
            f"torch.__version__ {report.get('torch_version')} != {TORCH_VERSION}"
        )
    if report.get("torch_cuda_version") != TORCH_CUDA_VERSION:
        problems.append(
            f"torch.version.cuda {report.get('torch_cuda_version')} != "
            f"{TORCH_CUDA_VERSION}"
        )
    if report.get("torch_cuda_available") is not True:
        problems.append(
            f"torch.cuda.is_available() {report.get('torch_cuda_available')} != True"
        )
    return problems


def audit_environment(conda: str, env_name: str) -> dict[str, Any]:
    _run(
        (
            conda,
            "run",
            "-n",
            env_name,
            "python",
            "-m",
            "pip",
            "check",
        )
    )
    result = _run(
        (
            conda,
            "run",
            "--no-capture-output",
            "-n",
            env_name,
            "python",
            "-c",
            ENV_AUDIT_CODE,
        )
    )
    report = parse_audit_output(result.stdout)
    problems = validate_environment_report(report)
    if problems:
        raise BootstrapError("environment audit failed: " + "; ".join(problems))
    return report


def ensure_environment(repo_root: Path, env_name: str) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", env_name):
        raise BootstrapError(f"unsafe Conda environment name: {env_name!r}")
    conda = conda_executable()
    if conda_env_exists(conda, env_name):
        return audit_environment(conda, env_name)

    manifest = repo_root / "environment" / "aimers9-dev.yml"
    requirements = repo_root / "environment" / "aimers9-dev-requirements.txt"
    _run(
        (
            conda,
            "env",
            "create",
            "--solver",
            CONDA_SOLVER,
            "--name",
            env_name,
            "--file",
            str(manifest),
        ),
        stage=f"fresh Conda environment creation (solver={CONDA_SOLVER})",
    )
    python = (conda, "run", "-n", env_name, "python", "-m", "pip", "install")
    _run(
        (*python, "-r", str(requirements)),
        stage="pinned development dependency installation",
    )
    _run(
        (
            *python,
            "--index-url",
            TORCH_INDEX_URL,
            "--extra-index-url",
            PYPI_INDEX_URL,
            f"torch=={TORCH_VERSION}",
        ),
        stage="CUDA 12.8 Torch installation",
    )
    return audit_environment(conda, env_name)


def check_environment(env_name: str) -> dict[str, Any]:
    conda = conda_executable()
    if not conda_env_exists(conda, env_name):
        raise BootstrapError(f"Conda environment does not exist: {env_name}")
    return audit_environment(conda, env_name)


def inspect_gpu() -> GpuInspection:
    """Report GPU state without selecting a device or allocating CUDA tensors."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        print("[GPU] nvidia-smi unavailable; GPU readiness not verified")
        return GpuInspection(
            ready=False,
            devices=(),
            processes=(),
            problems=("nvidia-smi is unavailable",),
        )

    device_result = _run(
        (
            nvidia_smi,
            "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,"
            "utilization.gpu",
            "--format=csv,noheader,nounits",
        ),
        check=False,
    )
    process_result = _run(
        (
            nvidia_smi,
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ),
        check=False,
    )
    device_lines = tuple(
        line.strip() for line in device_result.stdout.splitlines() if line.strip()
    )
    process_lines = tuple(
        line.strip() for line in process_result.stdout.splitlines() if line.strip()
    )
    problems: list[str] = []
    usable_devices: list[str] = []
    if device_result.returncode != 0:
        detail = device_result.stderr.strip() or f"exit {device_result.returncode}"
        problems.append(f"nvidia-smi device query failed: {detail}")
    else:
        for line in device_lines:
            fields = [field.strip() for field in next(csv.reader([line]))]
            if len(fields) != 7 or not all(fields[index] for index in (0, 1, 2)):
                continue
            try:
                total_memory = float(fields[3])
                free_memory = float(fields[5])
            except ValueError:
                continue
            if total_memory > 0 and free_memory >= 0:
                usable_devices.append(line)
        if not usable_devices:
            problems.append("nvidia-smi reported no usable GPU device")
    if process_result.returncode != 0:
        detail = process_result.stderr.strip() or f"exit {process_result.returncode}"
        problems.append(f"nvidia-smi process query failed: {detail}")

    print(
        "[GPU] devices:\n"
        + ("\n".join(device_lines) if device_lines else "none reported")
    )
    print(
        "[GPU] compute processes:\n"
        + ("\n".join(process_lines) if process_lines else "none reported")
    )
    if problems:
        print("[GPU] NOT READY: " + "; ".join(problems))
    else:
        print(
            "[GPU] inspection passed; no device selected. Inspect occupancy "
            "before later GPU work"
        )
    return GpuInspection(
        ready=not problems,
        devices=tuple(usable_devices),
        processes=process_lines,
        problems=tuple(problems),
    )


def print_link_states(states: dict[str, str]) -> None:
    for label, state in states.items():
        print(f"[LINK] {state.upper():7s} {label}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "apply"))
    parser.add_argument("--repo-root", type=Path, default=default_repo_root())
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--v93-dir", type=Path, required=True)
    parser.add_argument("--v93-archive", type=Path, default=None)
    parser.add_argument("--env-name", default=DEFAULT_ENV_NAME)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        repo_root = canonical(args.repo_root)
        data_dir = canonical(args.data_dir or repo_root / "데이터" / "open")
        v93_dir = canonical(args.v93_dir)
        contract = load_v93_contract(repo_root)

        headers = validate_data_dir(data_dir)
        archive_verified = validate_v93_dir(v93_dir, contract, args.v93_archive)
        specs = build_link_specs(repo_root, data_dir, v93_dir)
        states = inspect_links(repo_root, specs)

        print(f"[DATA] validated headers only: {', '.join(headers)}")
        print(
            "[V93] authoritative structural identities valid: "
            f"weights={len(REQUIRED_MODEL_FILES)}, prep={len(PREP_FILES)}, "
            f"metadata={len(METADATA_FILES)}, "
            f"registry_model_files={contract.registry_model_files}"
        )
        if archive_verified:
            print("[V93] archive SHA-256 verified; archive provenance check passed")
        else:
            print(
                "[V93] archive not supplied; extracted structure is "
                "development-ready, archive provenance is NOT proven"
            )
        print_link_states(states)
        gpu = inspect_gpu()
        if not gpu.ready:
            raise BootstrapError("GPU inspection failed: " + "; ".join(gpu.problems))

        if args.mode == "check":
            environment = check_environment(args.env_name)
            missing = [label for label, state in states.items() if state != "ready"]
            if missing:
                raise BootstrapError(
                    "recovery bridges are not ready: " + ", ".join(missing)
                )
        else:
            environment = ensure_environment(repo_root, args.env_name)
            states = apply_links(repo_root, specs)
            print_link_states(states)

        print(
            f"[ENV] {args.env_name}: Python {environment['python']}, "
            f"Torch {environment['torch_version']}, "
            f"CUDA build {environment['torch_cuda_version']}"
        )
        print("[READY] school-GPU development bootstrap passed")
        print("[NOTICE] this does NOT establish official submission readiness")
        return 0
    except BootstrapError as exc:
        print(f"[NOT READY] {exc}", file=sys.stderr)
        print("[NOTICE] no claim of official submission readiness", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
