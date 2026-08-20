from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bootstrap_gpu.py"
SPEC = importlib.util.spec_from_file_location("bootstrap_gpu", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
bootstrap_gpu = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bootstrap_gpu
SPEC.loader.exec_module(bootstrap_gpu)


class BootstrapFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="aimers9-bootstrap-test-"
        )
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.data = self.root / "official" / "open"
        self.v93 = self.root / "external" / "v93"
        self.repo.mkdir(parents=True)
        self.data.mkdir(parents=True)
        self.v93.mkdir(parents=True)
        registry = self.repo / "repro_979" / "recovery_candidate_registry.json"
        registry.parent.mkdir(parents=True)
        registry.write_text(
            json.dumps(
                {
                    "baseline": {
                        "sha256": "a" * 64,
                        "model_files": 51,
                    }
                }
            ),
            encoding="utf-8",
        )
        identities = (
            *bootstrap_gpu.REQUIRED_MODEL_FILES,
            *bootstrap_gpu.PREP_FILES,
            *bootstrap_gpu.METADATA_FILES,
        )
        evidence = (
            self.repo
            / ".omo"
            / "evidence"
            / "aimers9-top100-recovery"
            / "task-3-baseline-registry.json"
        )
        evidence.parent.mkdir(parents=True)
        evidence.write_text(
            json.dumps(
                {
                    "baseline": {
                        "sha256": "a" * 64,
                        "package_hashes": {
                            "model_files": {
                                name: "b" * 64 for name in identities
                            }
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
        self._write_valid_data()
        self._write_valid_v93()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_valid_data(self) -> None:
        (self.data / "train.csv").write_text(
            "row_id,feature,control_success\n", encoding="utf-8"
        )
        (self.data / "test.csv").write_text(
            "row_id,feature\n", encoding="utf-8"
        )
        (self.data / "sample_submission.csv").write_text(
            "row_id,control_success\n", encoding="utf-8"
        )

    def _write_valid_v93(self) -> None:
        for name in bootstrap_gpu.PACKAGE_SOURCE_FILES:
            (self.v93 / name).touch()
        model = self.v93 / "model"
        model.mkdir()
        for name in bootstrap_gpu.REQUIRED_MODEL_FILES:
            (model / name).touch()
        for name in bootstrap_gpu.PREP_FILES:
            (model / name).touch()
        for name in bootstrap_gpu.METADATA_FILES:
            (model / name).touch()


class DataValidationTests(BootstrapFixture):
    def test_header_only_contract_passes(self) -> None:
        headers = bootstrap_gpu.validate_data_dir(self.data)
        self.assertEqual(headers["test.csv"], ["row_id", "feature"])

    def test_train_test_header_mismatch_fails(self) -> None:
        (self.data / "test.csv").write_text(
            "row_id,different\n1,secret\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(
            bootstrap_gpu.BootstrapError, "feature headers differ"
        ):
            bootstrap_gpu.validate_data_dir(self.data)

    def test_sample_header_must_be_exact(self) -> None:
        (self.data / "sample_submission.csv").write_text(
            "control_success,row_id\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(
            bootstrap_gpu.BootstrapError, "must be exactly"
        ):
            bootstrap_gpu.validate_data_dir(self.data)


class V93ValidationTests(BootstrapFixture):
    def test_authoritative_evidence_defines_all_identities(self) -> None:
        contract = bootstrap_gpu.load_v93_contract(self.repo)
        self.assertEqual(contract.registry_model_files, 51)
        self.assertEqual(len(contract.model_artifact_identities), 51)

    def test_registry_and_evidence_identity_counts_must_match(self) -> None:
        registry = self.repo / "repro_979" / "recovery_candidate_registry.json"
        payload = json.loads(registry.read_text(encoding="utf-8"))
        payload["baseline"]["model_files"] = 50
        registry.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(
            bootstrap_gpu.BootstrapError,
            "registry/evidence model-file-count conflict",
        ):
            bootstrap_gpu.load_v93_contract(self.repo)

    def test_extracted_contract_does_not_claim_archive_provenance(self) -> None:
        contract = bootstrap_gpu.load_v93_contract(self.repo)
        self.assertFalse(
            bootstrap_gpu.validate_v93_dir(self.v93, contract)
        )

    def test_unrelated_file_outside_model_directory_is_ignored(self) -> None:
        (self.v93 / "unrelated_extracted_tree_file.txt").touch()
        contract = bootstrap_gpu.load_v93_contract(self.repo)
        self.assertFalse(
            bootstrap_gpu.validate_v93_dir(self.v93, contract)
        )

    def test_missing_required_model_fails(self) -> None:
        (self.v93 / "model" / "armb_s42.pt").unlink()
        contract = bootstrap_gpu.load_v93_contract(self.repo)
        with self.assertRaisesRegex(
            bootstrap_gpu.BootstrapError, "missing required"
        ):
            bootstrap_gpu.validate_v93_dir(self.v93, contract)

    def test_missing_required_prep_fails_separately(self) -> None:
        (self.v93 / "model" / "catboost_prep.pkl").unlink()
        contract = bootstrap_gpu.load_v93_contract(self.repo)
        with self.assertRaisesRegex(
            bootstrap_gpu.BootstrapError, "missing required v93 prep files"
        ):
            bootstrap_gpu.validate_v93_dir(self.v93, contract)

    def test_missing_required_metadata_fails_separately(self) -> None:
        (self.v93 / "model" / "train_meta.json").unlink()
        contract = bootstrap_gpu.load_v93_contract(self.repo)
        with self.assertRaisesRegex(
            bootstrap_gpu.BootstrapError, "missing required v93 metadata files"
        ):
            bootstrap_gpu.validate_v93_dir(self.v93, contract)

    def test_arbitrary_filler_identity_is_rejected(self) -> None:
        (self.v93 / "model" / "additional_contract_file_0.bin").touch()
        contract = bootstrap_gpu.load_v93_contract(self.repo)
        with self.assertRaisesRegex(
            bootstrap_gpu.BootstrapError, "model/ identity mismatch"
        ):
            bootstrap_gpu.validate_v93_dir(self.v93, contract)

    def test_optional_archive_hash_is_checked(self) -> None:
        archive = self.root / "v93.zip"
        archive.touch()
        contract = bootstrap_gpu.load_v93_contract(self.repo)
        with mock.patch.object(
            bootstrap_gpu, "sha256_file", return_value="a" * 64
        ):
            self.assertTrue(
                bootstrap_gpu.validate_v93_dir(
                    self.v93, contract, archive
                )
            )
        with mock.patch.object(
            bootstrap_gpu, "sha256_file", return_value="b" * 64
        ):
            with self.assertRaisesRegex(
                bootstrap_gpu.BootstrapError, "SHA-256 mismatch"
            ):
                bootstrap_gpu.validate_v93_dir(
                    self.v93, contract, archive
                )


class LinkSafetyTests(BootstrapFixture):
    def test_apply_is_idempotent_and_creates_only_approved_bridges(
        self,
    ) -> None:
        specs = bootstrap_gpu.build_link_specs(
            self.repo, self.data, self.v93
        )
        first = bootstrap_gpu.apply_links(self.repo, specs)
        second = bootstrap_gpu.apply_links(self.repo, specs)
        self.assertTrue(
            all(state == "ready" for state in first.values())
        )
        self.assertEqual(first, second)
        bridge = self.repo / "repro_979" / "open" / "data"
        self.assertEqual(
            sorted(path.name for path in bridge.iterdir()),
            ["sample_submission.csv", "test.csv", "train.csv"],
        )
        self.assertFalse((bridge / "trackman_history.csv").exists())

    def test_conflicting_file_is_not_overwritten(self) -> None:
        specs = bootstrap_gpu.build_link_specs(
            self.repo, self.data, self.v93
        )
        destination = specs[0].destination
        destination.parent.mkdir(parents=True)
        destination.write_text("user file", encoding="utf-8")
        with self.assertRaisesRegex(
            bootstrap_gpu.BootstrapError, "refusing to overwrite"
        ):
            bootstrap_gpu.apply_links(self.repo, specs)
        self.assertEqual(
            destination.read_text(encoding="utf-8"), "user file"
        )

    def test_wrong_and_broken_symlinks_fail(self) -> None:
        specs = bootstrap_gpu.build_link_specs(
            self.repo, self.data, self.v93
        )
        destination = specs[0].destination
        destination.parent.mkdir(parents=True)
        destination.symlink_to(self.root / "wrong.csv")
        with self.assertRaisesRegex(
            bootstrap_gpu.BootstrapError, "wrong or broken"
        ):
            bootstrap_gpu.inspect_links(self.repo, specs)

    def test_symlinked_destination_parent_fails(self) -> None:
        external_parent = self.root / "unexpected-parent"
        external_parent.mkdir()
        open_dir = self.repo / "repro_979" / "open"
        open_dir.parent.mkdir(parents=True, exist_ok=True)
        open_dir.symlink_to(external_parent, target_is_directory=True)
        specs = bootstrap_gpu.build_link_specs(
            self.repo, self.data, self.v93
        )
        with self.assertRaisesRegex(
            bootstrap_gpu.BootstrapError,
            "symlinked destination parent",
        ):
            bootstrap_gpu.inspect_links(self.repo, specs)


class CliModeTests(BootstrapFixture):
    def _environment_report(self) -> dict[str, object]:
        return {
            "python": bootstrap_gpu.PYTHON_VERSION,
            "packages": dict(bootstrap_gpu.EXPECTED_PACKAGES),
            "torch_version": bootstrap_gpu.TORCH_VERSION,
            "torch_cuda_version": bootstrap_gpu.TORCH_CUDA_VERSION,
            "torch_cuda_available": True,
        }

    def _gpu_ready(self) -> bootstrap_gpu.GpuInspection:
        return bootstrap_gpu.GpuInspection(
            ready=True,
            devices=("0, NVIDIA L4, GPU-test, 23034, 0, 23034, 0",),
            processes=(),
            problems=(),
        )

    def _arguments(self, mode: str) -> list[str]:
        return [
            mode,
            "--repo-root",
            str(self.repo),
            "--data-dir",
            str(self.data),
            "--v93-dir",
            str(self.v93),
        ]

    def test_check_mode_is_read_only(self) -> None:
        with (
            mock.patch.object(
                bootstrap_gpu, "inspect_gpu", return_value=self._gpu_ready()
            ),
            mock.patch.object(
                bootstrap_gpu,
                "check_environment",
                return_value=self._environment_report(),
            ),
        ):
            self.assertEqual(bootstrap_gpu.main(self._arguments("check")), 2)
        self.assertFalse(
            (self.repo / "repro_979" / "open" / "data").exists()
        )
        self.assertFalse(
            (
                self.repo
                / "repro_979"
                / "cache"
                / "v93_extract_verify"
            ).exists()
        )

    def test_apply_mode_uses_safe_link_apply_and_is_idempotent(self) -> None:
        with (
            mock.patch.object(
                bootstrap_gpu, "inspect_gpu", return_value=self._gpu_ready()
            ),
            mock.patch.object(
                bootstrap_gpu,
                "ensure_environment",
                return_value=self._environment_report(),
            ),
        ):
            self.assertEqual(bootstrap_gpu.main(self._arguments("apply")), 0)
            self.assertEqual(bootstrap_gpu.main(self._arguments("apply")), 0)

    def test_failed_gpu_inspection_blocks_apply_before_environment(self) -> None:
        failure = bootstrap_gpu.GpuInspection(
            ready=False,
            devices=(),
            processes=(),
            problems=("nvidia-smi failed",),
        )
        with (
            mock.patch.object(
                bootstrap_gpu, "inspect_gpu", return_value=failure
            ),
            mock.patch.object(
                bootstrap_gpu, "ensure_environment"
            ) as ensure_environment,
        ):
            self.assertEqual(bootstrap_gpu.main(self._arguments("apply")), 2)
            ensure_environment.assert_not_called()


class EnvironmentAuditTests(unittest.TestCase):
    def test_exact_torch_and_cuda_build_are_required(self) -> None:
        report = {
            "python": bootstrap_gpu.PYTHON_VERSION,
            "packages": dict(bootstrap_gpu.EXPECTED_PACKAGES),
            "torch_version": bootstrap_gpu.TORCH_VERSION,
            "torch_cuda_version": bootstrap_gpu.TORCH_CUDA_VERSION,
            "torch_cuda_available": True,
        }
        self.assertEqual(
            bootstrap_gpu.validate_environment_report(report), []
        )
        report["torch_cuda_version"] = "12.6"
        self.assertIn(
            "torch.version.cuda 12.6 != 12.8",
            bootstrap_gpu.validate_environment_report(report),
        )

    def test_cuda_unavailable_is_not_ready(self) -> None:
        report = {
            "python": bootstrap_gpu.PYTHON_VERSION,
            "packages": dict(bootstrap_gpu.EXPECTED_PACKAGES),
            "torch_version": bootstrap_gpu.TORCH_VERSION,
            "torch_cuda_version": bootstrap_gpu.TORCH_CUDA_VERSION,
            "torch_cuda_available": False,
        }
        self.assertIn(
            "torch.cuda.is_available() False != True",
            bootstrap_gpu.validate_environment_report(report),
        )


class GpuInspectionTests(unittest.TestCase):
    def test_missing_nvidia_smi_is_not_ready(self) -> None:
        with mock.patch.object(
            bootstrap_gpu.shutil, "which", return_value=None
        ):
            result = bootstrap_gpu.inspect_gpu()
        self.assertFalse(result.ready)
        self.assertIn("nvidia-smi is unavailable", result.problems)

    def test_failed_nvidia_smi_is_not_ready(self) -> None:
        failed = subprocess.CompletedProcess(
            args=["nvidia-smi"], returncode=1, stdout="", stderr="driver error"
        )
        with (
            mock.patch.object(
                bootstrap_gpu.shutil,
                "which",
                return_value="/usr/bin/nvidia-smi",
            ),
            mock.patch.object(bootstrap_gpu, "_run", return_value=failed),
        ):
            result = bootstrap_gpu.inspect_gpu()
        self.assertFalse(result.ready)
        self.assertTrue(
            any("query failed" in problem for problem in result.problems)
        )

    def test_no_reported_gpu_device_is_not_ready(self) -> None:
        no_devices = subprocess.CompletedProcess(
            args=["nvidia-smi"], returncode=0, stdout="", stderr=""
        )
        with (
            mock.patch.object(
                bootstrap_gpu.shutil,
                "which",
                return_value="/usr/bin/nvidia-smi",
            ),
            mock.patch.object(
                bootstrap_gpu, "_run", return_value=no_devices
            ),
        ):
            result = bootstrap_gpu.inspect_gpu()
        self.assertFalse(result.ready)
        self.assertIn(
            "nvidia-smi reported no usable GPU device", result.problems
        )

    def test_structured_gpu_success_does_not_select_a_device(self) -> None:
        devices = subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout="0, NVIDIA L4, GPU-test, 23034, 10, 23024, 0\n",
            stderr="",
        )
        processes = subprocess.CompletedProcess(
            args=["nvidia-smi"], returncode=0, stdout="", stderr=""
        )
        with (
            mock.patch.object(
                bootstrap_gpu.shutil,
                "which",
                return_value="/usr/bin/nvidia-smi",
            ),
            mock.patch.object(
                bootstrap_gpu, "_run", side_effect=(devices, processes)
            ),
        ):
            result = bootstrap_gpu.inspect_gpu()
        self.assertTrue(result.ready)
        self.assertEqual(len(result.devices), 1)


if __name__ == "__main__":
    unittest.main()
