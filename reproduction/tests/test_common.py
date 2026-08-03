from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from reproduction.common import (
    Job,
    _is_julia_command,
    _sanitize_julia_environment,
    arithmetic_mean,
    collect_logs,
    materialize_archive_section,
    prepare_mode_output,
    run_jobs,
    sample_std,
)


class StatisticsTests(unittest.TestCase):
    def test_arithmetic_mean_and_sample_std(self) -> None:
        self.assertEqual(arithmetic_mean([1.0, 2.0, 3.0]), 2.0)
        self.assertAlmostEqual(sample_std([1.0, 2.0, 3.0]), 1.0)
        self.assertEqual(sample_std([4.0]), 0.0)


class ArchiveTests(unittest.TestCase):
    def test_materializes_only_requested_section(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "logs.zip"
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr(
                    "bundle/experiments_logs/3-2-networkflow/job/stdout.log",
                    "network\n",
                )
                zipped.writestr(
                    "bundle/experiments_logs/3-3-opf_distrun/job/stdout.log",
                    "opf\n",
                )
            with materialize_archive_section(archive, "3-2-networkflow") as section:
                logs = collect_logs(section)
                self.assertEqual(len(logs), 1)
                self.assertEqual(logs[0].read_text(encoding="utf-8"), "network\n")


class ModeOutputTests(unittest.TestCase):
    def test_modes_are_isolated_and_smoke_is_marked_validation_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / "section"
            smoke = prepare_mode_output(base, "smoke")
            full = prepare_mode_output(base, "full")

            self.assertEqual(smoke, base / "smoke")
            self.assertEqual(full, base / "full")
            self.assertNotEqual(smoke, full)
            smoke_profile = json.loads(
                (smoke / "artifact_profile.json").read_text(encoding="utf-8")
            )
            full_profile = json.loads(
                (full / "artifact_profile.json").read_text(encoding="utf-8")
            )
            self.assertEqual(smoke_profile["artifact_scope"], "validation_only_subset")
            self.assertIn("not the complete paper", smoke_profile["publication_warning"])
            self.assertEqual(full_profile["artifact_scope"], "fresh_paper_grid")
            self.assertIsNone(full_profile["publication_warning"])
    def test_appendix_modes_have_explicit_artifact_scopes(self) -> None:
        expected = {
            "table": "static_configuration_export",
            "parse": "caller_supplied_parse_input",
            "archived-source": "reported_source_preservation",
            "full": "fresh_paper_grid",
        }
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / "appendix"
            for mode, scope in expected.items():
                with self.subTest(mode=mode):
                    output = prepare_mode_output(base, mode)
                    profile = json.loads(
                        (output / "artifact_profile.json").read_text(encoding="utf-8")
                    )
                    self.assertEqual(profile["artifact_scope"], scope)

    def test_explicit_mode_directory_is_not_duplicated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            smoke = Path(temporary) / "smoke"
            self.assertEqual(prepare_mode_output(smoke, "smoke"), smoke.resolve())

    def test_unknown_mode_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "Unsupported output mode"):
                prepare_mode_output(Path(temporary), "preview")


class JobTests(unittest.TestCase):
    def test_julia_environment_removes_only_active_conda_libraries(self) -> None:
        conda_prefix = "/opt/reviewer-conda"
        source = {
            "CONDA_PREFIX": conda_prefix,
            "LD_LIBRARY_PATH": os.pathsep.join(
                (
                    f"{conda_prefix}/lib",
                    "/opt/cuda/lib64",
                    f"{conda_prefix}/lib64",
                    "/opt/gurobi/lib",
                )
            ),
        }

        sanitized, removed = _sanitize_julia_environment(source)

        self.assertEqual(
            removed,
            [f"{conda_prefix}/lib", f"{conda_prefix}/lib64"],
        )
        self.assertEqual(
            sanitized["LD_LIBRARY_PATH"],
            os.pathsep.join(("/opt/cuda/lib64", "/opt/gurobi/lib")),
        )
        self.assertIn(f"{conda_prefix}/lib", source["LD_LIBRARY_PATH"])
        self.assertTrue(_is_julia_command(("/path/to/julia", "--version")))
        self.assertTrue(_is_julia_command(("/path/to/julia-lts", "--version")))
        self.assertFalse(_is_julia_command((sys.executable, "--version")))

    def test_julia_adjustment_is_applied_and_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_julia = root / "julia"
            fake_julia.write_text(
                f"#!{sys.executable}\n"
                "import os\n"
                "print(os.environ.get('LD_LIBRARY_PATH', '<unset>'))\n",
                encoding="utf-8",
            )
            fake_julia.chmod(0o755)
            conda_prefix = root / "conda"
            conda_lib = str(conda_prefix / "lib")
            kept_lib = str(root / "kept")

            result = run_jobs(
                (Job("julia_env", (str(fake_julia),)),),
                log_root=root / "logs",
                cwd=root,
                environment={
                    "CONDA_PREFIX": str(conda_prefix),
                    "LD_LIBRARY_PATH": os.pathsep.join((conda_lib, kept_lib)),
                },
            )

            self.assertEqual(result[0].returncode, 0)
            self.assertEqual(
                Path(result[0].log_path).read_text(encoding="utf-8").strip(),
                kept_lib,
            )
            command = json.loads(
                (root / "logs" / "julia_env.command.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                command["environment_adjustments"]["removed_ld_library_path_entries"],
                [conda_lib],
            )
            result_marker = root / "logs" / "julia_env.result.json"
            legacy_result = json.loads(result_marker.read_text(encoding="utf-8"))
            legacy_result.pop("environment_adjustments")
            result_marker.write_text(
                json.dumps(legacy_result),
                encoding="utf-8",
            )
            resumed = run_jobs(
                (Job("julia_env", (str(fake_julia),)),),
                log_root=root / "logs",
                cwd=root,
                environment={
                    "CONDA_PREFIX": str(conda_prefix),
                    "LD_LIBRARY_PATH": os.pathsep.join((conda_lib, kept_lib)),
                },
            )
            self.assertTrue(resumed[0].skipped)

    def test_merged_log_and_resume_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = Job(
                "nested/example",
                (
                    sys.executable,
                    "-c",
                    "import sys; print('stdout'); print('stderr', file=sys.stderr)",
                ),
            )
            first = run_jobs((job,), log_root=root, cwd=root)
            self.assertEqual(first[0].returncode, 0)
            self.assertFalse(first[0].skipped)
            log = Path(first[0].log_path).read_text(encoding="utf-8")
            self.assertIn("stdout", log)
            self.assertIn("stderr", log)
            second = run_jobs((job,), log_root=root, cwd=root)
            self.assertTrue(second[0].skipped)
            marker = root / "nested" / "example.result.json"
            self.assertEqual(json.loads(marker.read_text(encoding="utf-8"))["returncode"], 0)

    def test_rejects_parent_traversal_job_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ValueError):
                run_jobs(
                    (Job("../escape", (sys.executable, "-c", "pass")),),
                    log_root=root,
                    cwd=root,
                )


if __name__ == "__main__":
    unittest.main()
