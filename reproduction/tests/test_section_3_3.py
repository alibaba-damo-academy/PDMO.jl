from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from reproduction import section_3_3


def _log_text(
    *,
    case: str = "case30",
    partition_number: int = 3,
    objective: float = section_3_3.OBJECTIVE_FINGERPRINTS["case30"],
    partition_times: tuple[float, float, float] = (1.0, 2.0, 3.0),
    admm_times: tuple[float, float, float] = (4.0, 5.0, 6.0),
    iterations: tuple[int, int, int] = (10, 20, 40),
) -> str:
    lines = [
        "Running Distributed DCOPF",
        f" case file = /matpower/{case}.m",
        f" num. Partitions = {partition_number}",
        " admmSolver = original",
        " initialRho = 100.0",
        " tol = 1e-4",
        " maxIter = 1000000",
        " timeLimit = 7200.0",
        " logInterval = 100",
        " r_value = 10000.0",
        " seed = 126",
        "Run Bipartite ADMM with threads = 16",
        f"True DC objective: {objective}",
    ]
    for index, method in enumerate(section_3_3.METHODS):
        lines.extend(
            (
                " Solver Status = ADMM_TERMINATION_OPTIMAL",
                f" Objective = {100.0 + index}",
                f" Pres (L2) = {1.0e-5 + index * 1.0e-6}",
                " Pres (LInf) = 1e-6",
                " Dres (L2) = 2e-5",
                " Dres (LInf) = 2e-6",
                f" Stop. Iter = {iterations[index]}",
                f" Total Time = {admm_times[index]}",
                " True Obj. Diff = 1e-8",
            )
        )
    lines.append(f"SUMMARY OF RESULTS for partitions: {partition_number}")
    for index, method in enumerate(section_3_3.METHODS):
        lines.append(
            f" {method} | {partition_times[index]} | {iterations[index]} | "
            f"{admm_times[index]} | {100.0 + index} |"
        )
    return "\n".join(lines) + "\n"


def _raw_comparison_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    fresh_rows: list[dict[str, object]] = []
    archive_rows: list[dict[str, object]] = []
    counter = 100
    for spec in section_3_3.FRESH_FULL_SPECS:
        for method in section_3_3.METHODS:
            key = (*spec.key, method)
            archive_censored = key in section_3_3.ARCHIVE_TIME_CENSORED_KEYS
            archived = {
                "figure": spec.figure,
                "case": spec.case,
                "partition_number": spec.partition_number,
                "method": method,
                "admm_solver": spec.solver,
                "initial_rho": spec.rho,
                "tolerance": section_3_3.TOLERANCE,
                "max_iterations": section_3_3.MAX_ITERATIONS,
                "time_limit_seconds": section_3_3.TIME_LIMIT_SECONDS,
                "log_interval": section_3_3.LOG_INTERVAL,
                "r_value": section_3_3.R_VALUE,
                "seed": section_3_3.SEED,
                "threads": 16,
                "iterations": counter,
                "termination_status": (
                    "ADMM_TERMINATION_TIME_LIMIT"
                    if archive_censored
                    else "ADMM_TERMINATION_OPTIMAL"
                ),
                "partition_time_seconds": float(counter) / 10.0,
                "admm_time_seconds": float(counter),
            }
            fresh = dict(archived)
            fresh["partition_time_seconds"] = float(archived["partition_time_seconds"]) + 7.0
            fresh["admm_time_seconds"] = float(archived["admm_time_seconds"]) + 11.0
            if archive_censored:
                # These differences must be reported but not gated.
                fresh["iterations"] = counter + 999
                fresh["termination_status"] = "ADMM_TERMINATION_OPTIMAL"
            archive_rows.append(archived)
            fresh_rows.append(fresh)
            counter += 1
    return fresh_rows, archive_rows


def _semantic_archive_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    counter = 1_000
    for spec in section_3_3.ARCHIVED_SPECS:
        assert spec.archive_job_id is not None
        for method in section_3_3.METHODS:
            time_censored = (
                spec.figure,
                spec.case,
                spec.partition_number,
                method,
            ) in section_3_3.ARCHIVE_TIME_CENSORED_KEYS
            rows.append(
                {
                    "archive_folder": spec.archive_folder,
                    "archive_job_id": spec.archive_job_id,
                    "figure": spec.figure,
                    "case": spec.case,
                    "partition_number": spec.partition_number,
                    "method": method,
                    "admm_solver": spec.solver,
                    "initial_rho": spec.rho,
                    "tolerance": section_3_3.TOLERANCE,
                    "max_iterations": section_3_3.MAX_ITERATIONS,
                    "time_limit_seconds": section_3_3.TIME_LIMIT_SECONDS,
                    "log_interval": section_3_3.LOG_INTERVAL,
                    "r_value": section_3_3.R_VALUE,
                    "seed": section_3_3.SEED,
                    "threads": 16,
                    "true_dc_objective": section_3_3.OBJECTIVE_FINGERPRINTS[spec.case],
                    "iterations": counter,
                    "termination_status": (
                        "ADMM_TERMINATION_TIME_LIMIT"
                        if time_censored
                        else "ADMM_TERMINATION_OPTIMAL"
                    ),
                    "partition_time_seconds": counter / 100.0,
                    "admm_time_seconds": counter / 10.0,
                }
            )
            counter += 1
    return rows


class Section33ParserTests(unittest.TestCase):
    def _parse(self, text: str):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stdout.log"
            path.write_text(text, encoding="utf-8")
            return section_3_3._parse_log(path)

    def test_last_complete_invocation_parses_three_long_rows(self) -> None:
        earlier = _log_text(
            case="case57",
            objective=section_3_3.OBJECTIVE_FINGERPRINTS["case57"],
        )
        spec, rows = self._parse(earlier + _log_text())

        self.assertEqual(spec.key, (13, "case30", 3))
        self.assertEqual([row["method"] for row in rows], ["BFS", "MILP", "GNN"])
        self.assertEqual([row["iterations"] for row in rows], [10, 20, 40])
        self.assertEqual([row["termination_status"] for row in rows], [
            "ADMM_TERMINATION_OPTIMAL",
            "ADMM_TERMINATION_OPTIMAL",
            "ADMM_TERMINATION_OPTIMAL",
        ])
        self.assertAlmostEqual(float(rows[0]["total_time_seconds"]), 5.0)
        self.assertAlmostEqual(float(rows[2]["primal_residual_l2"]), 1.2e-5)

    def test_header_and_summary_partition_mismatch_is_rejected(self) -> None:
        text = _log_text().replace(
            "SUMMARY OF RESULTS for partitions: 3",
            "SUMMARY OF RESULTS for partitions: 4",
        )

        with self.assertRaisesRegex(ValueError, "disagrees with summary"):
            self._parse(text)


class Section33AggregationTests(unittest.TestCase):
    def test_means_and_within_group_normalization(self) -> None:
        _, first = self._parse(_log_text())
        _, second = self._parse(
            _log_text(
                partition_number=4,
                partition_times=(3.0, 4.0, 5.0),
                admm_times=(6.0, 7.0, 8.0),
                iterations=(30, 40, 60),
            )
        )

        aggregates = section_3_3._aggregate(first + second)
        by_method = {row["method"]: row for row in aggregates}

        self.assertEqual(len(aggregates), 3)
        self.assertEqual(by_method["BFS"]["sample_count"], 2)
        self.assertAlmostEqual(float(by_method["BFS"]["partition_time_seconds"]), 2.0)
        self.assertAlmostEqual(float(by_method["BFS"]["admm_time_seconds"]), 5.0)
        self.assertAlmostEqual(float(by_method["BFS"]["iterations"]), 20.0)
        self.assertAlmostEqual(float(by_method["BFS"]["normalized_total_time"]), 7.0 / 11.0)
        self.assertAlmostEqual(float(by_method["BFS"]["normalized_iterations"]), 20.0 / 50.0)
        self.assertEqual(by_method["GNN"]["normalized_total_time"], 1.0)
        self.assertEqual(by_method["GNN"]["normalized_iterations"], 1.0)

    def _parse(self, text: str):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stdout.log"
            path.write_text(text, encoding="utf-8")
            return section_3_3._parse_log(path)

    def test_archived_identifier_names_zip_member_not_temporary_path(self) -> None:
        identifier = section_3_3._archive_log_identifier(
            Path("/review/experiments_logs.zip"),
            {"figure": 14, "case": "case57", "partition_number": 13},
        )

        self.assertEqual(
            identifier,
            "/review/experiments_logs.zip::experiments_logs/3-3-opf_distrun/"
            "case57_flip_2000/51523865/stdout.log",
        )

    def test_smoke_validation_and_exact_raw_reference(self) -> None:
        _, rows = self._parse(
            _log_text(
                partition_times=(0.0, 5.587, 4.841),
                admm_times=(2.71, 1.03, 1.02),
                iterations=(136, 136, 136),
            )
        )

        validation = section_3_3._validate_rows(
            rows,
            section_3_3.SMOKE_SPECS,
            require_archived_statuses=False,
            check_objectives=True,
        )
        comparison = section_3_3._reference_comparison(
            section_3_3._aggregate(rows),
            rows,
            smoke_profile=True,
        )

        self.assertEqual(validation["status"], "passed")
        self.assertEqual(validation["row_count"], 3)
        self.assertTrue(comparison["summary"]["all_iteration_values_match"])
        self.assertEqual(comparison["summary"]["compared_entries"], 3)
        self.assertTrue(all(entry["partition_number"] == 3 for entry in comparison["entries"]))

    def test_validation_report_retains_errors(self) -> None:
        _, rows = self._parse(_log_text())
        rows[0]["termination_status"] = ""

        validation = section_3_3._validate_rows(
            rows,
            section_3_3.SMOKE_SPECS,
            require_archived_statuses=False,
            check_objectives=True,
        )

        self.assertEqual(validation["status"], "failed")
        self.assertTrue(any("unexpected status" in error for error in validation["errors"]))

    def test_wrong_seed_is_rejected(self) -> None:
        _, rows = self._parse(_log_text())
        rows[0]["seed"] = 125
        validation = section_3_3._validate_rows(
            rows,
            section_3_3.SMOKE_SPECS,
            require_archived_statuses=False,
            check_objectives=True,
        )
        self.assertEqual(validation["status"], "failed")
        self.assertTrue(any("invalid seed" in error for error in validation["errors"]))

    def test_wrong_thread_count_is_rejected(self) -> None:
        _, rows = self._parse(_log_text())
        rows[0]["threads"] = 8

        validation = section_3_3._validate_rows(
            rows,
            section_3_3.SMOKE_SPECS,
            require_archived_statuses=False,
            check_objectives=True,
        )

        self.assertEqual(validation["status"], "failed")
        self.assertTrue(any("invalid threads" in error for error in validation["errors"]))

    def test_job_command_passes_seed_126_explicitly(self) -> None:
        args = argparse.Namespace(julia="julia", threads=16)
        jobs = section_3_3._jobs_for_specs(
            args,
            section_3_3.SMOKE_SPECS,
            {"case30": Path("/tmp/case30.m")},
        )
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].command[-1], "126")
    def test_matpower_input_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "case30.m").write_text("function mpc = case30\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "pinned 8.1"):
                section_3_3._matpower_inputs(root, section_3_3.SMOKE_SPECS)

    def test_smoke_iteration_mismatch_is_enforced(self) -> None:
        comparison = {
            "summary": {
                "all_iteration_values_match": False,
            }
        }

        with self.assertRaisesRegex(SystemExit, "smoke point"):
            section_3_3._enforce_iteration_reference(
                comparison,
                mode="smoke",
                smoke_profile=True,
            )
        section_3_3._enforce_iteration_reference(
            comparison,
            mode="parse",
            smoke_profile=False,
        )

    def test_reference_mismatch_is_in_final_validation_report(self) -> None:
        validation = {"status": "passed", "errors": []}
        comparison = {
            "summary": {
                "all_iteration_values_match": False,
            }
        }

        merged = section_3_3._merge_reference_validation(
            validation,
            comparison,
            mode="smoke",
            smoke_profile=True,
        )

        self.assertEqual(merged["status"], "failed")
        self.assertTrue(any("smoke point" in error for error in merged["errors"]))
        self.assertEqual(merged["reference_summary"], comparison["summary"])



class Section33Figure14ProfileTests(unittest.TestCase):
    def test_full_uses_published_profile_but_remains_fresh(self) -> None:
        archived = [spec for spec in section_3_3.ARCHIVED_SPECS if spec.figure == 14]
        fresh = [spec for spec in section_3_3.FRESH_FULL_SPECS if spec.figure == 14]

        self.assertEqual([spec.partition_number for spec in archived], list(range(13, 19)))
        self.assertTrue(all(spec.rho == 2000.0 for spec in archived))
        self.assertTrue(all(spec.archive_folder == "case57_flip_2000" for spec in archived))
        self.assertEqual(
            [spec.archive_job_id for spec in archived],
            [str(value) for value in range(51523865, 51523871)],
        )
        self.assertTrue(all(spec.rho == 2000.0 for spec in fresh))
        self.assertTrue(all(spec.archive_job_id is None for spec in fresh))
        self.assertTrue(
            all(spec.archive_job_id is None for spec in section_3_3.FRESH_FULL_SPECS)
        )
        self.assertEqual(
            section_3_3._experiment_profile(section_3_3.ARCHIVED_SPECS)["classification"],
            "historical_reported_archive",
        )
        self.assertEqual(
            section_3_3._experiment_profile(section_3_3.FRESH_FULL_SPECS)["classification"],
            "fresh_published_profile",
        )
        self.assertTrue(
            all(
                spec.rho == 1000.0
                for spec in section_3_3.CAPTION_TYPO_SPECS
                if spec.figure == 14
            )
        )
        self.assertEqual(
            section_3_3._experiment_profile(section_3_3.CAPTION_TYPO_SPECS)[
                "classification"
            ],
            "caption_typo_parse_only",
        )

    def test_parse_profile_detection_does_not_conflate_rhos(self) -> None:
        self.assertIs(
            section_3_3._parse_mode_expected_specs(section_3_3.ARCHIVED_SPECS),
            section_3_3.FRESH_FULL_SPECS,
        )
        self.assertIs(
            section_3_3._parse_mode_expected_specs(section_3_3.FRESH_FULL_SPECS),
            section_3_3.FRESH_FULL_SPECS,
        )
        self.assertIs(
            section_3_3._parse_mode_expected_specs(section_3_3.CAPTION_TYPO_SPECS),
            section_3_3.CAPTION_TYPO_SPECS,
        )

    def test_fresh_full_command_uses_published_rho_2000(self) -> None:
        spec = next(spec for spec in section_3_3.FRESH_FULL_SPECS if spec.figure == 14)
        args = argparse.Namespace(julia="julia", threads=16)
        job = section_3_3._jobs_for_specs(
            args, (spec,), {"case57": Path("/tmp/case57.m")}
        )[0]

        self.assertIn("2000.0", job.command)
        self.assertNotIn("1000.0", job.command)

    def test_full_requires_archive_for_raw_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = argparse.Namespace(
                mode="full", archive=Path(temporary) / "missing.zip"
            )
            with self.assertRaisesRegex(SystemExit, "requires experiments_logs.zip"):
                section_3_3._require_full_archive(args)

    def test_full_rejects_non_paper_threads_before_output_or_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "experiments_logs.zip"
            archive.write_bytes(b"placeholder")
            output = root / "output"

            with mock.patch.object(section_3_3, "run_jobs") as run_jobs:
                with self.assertRaisesRegex(SystemExit, "requires --threads 16"):
                    section_3_3.main(
                        (
                            "--mode", "full",
                            "--archive", str(archive),
                            "--output", str(output),
                            "--threads", "8",
                        )
                    )

                run_jobs.assert_not_called()
            self.assertFalse(output.exists())

    def test_full_archive_preflight_enforces_strict_reference_before_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "experiments_logs.zip"
            archive.write_bytes(b"placeholder")
            output = root / "full"
            output.mkdir()
            args = argparse.Namespace(
                archive=archive,
                julia="julia",
                mode="full",
            )
            row_validation = {
                "status": "passed",
                "errors": [],
                "warnings": [],
            }
            reference_comparison = {
                "summary": {
                    "all_iteration_values_match": True,
                    "figure_14_reference_applicable": True,
                    "all_figure_14_reference_values_match": False,
                }
            }

            with (
                mock.patch.object(
                    section_3_3,
                    "materialize_archive_section",
                    return_value=nullcontext(root / "selected"),
                ),
                mock.patch.object(
                    section_3_3,
                    "_parse_archived_logs",
                    return_value=(list(section_3_3.ARCHIVED_SPECS), []),
                ),
                mock.patch.object(
                    section_3_3,
                    "_archive_semantic_manifest",
                    return_value=(
                        {
                            "schema": section_3_3.ARCHIVE_SEMANTIC_SCHEMA,
                            "records": [],
                        },
                        {
                            "status": "passed",
                            "errors": [],
                            "semantic_sha256_matches": True,
                            "source_bindings_match": True,
                        },
                    ),
                ) as semantic_manifest,
                mock.patch.object(
                    section_3_3,
                    "_validate_rows",
                    return_value=row_validation,
                ) as validate_rows,
                mock.patch.object(section_3_3, "_aggregate", return_value=[]),
                mock.patch.object(
                    section_3_3,
                    "_reference_comparison",
                    return_value=reference_comparison,
                ) as compare_reference,
                mock.patch.object(
                    section_3_3, "write_provenance"
                ) as write_failure_provenance,
            ):
                with self.assertRaisesRegex(SystemExit, "component times"):
                    section_3_3._full_archive_preflight(args, output)

            semantic_manifest.assert_called_once()
            validate_rows.assert_called_once()
            compare_reference.assert_called_once()
            write_failure_provenance.assert_called_once()
            self.assertEqual(
                write_failure_provenance.call_args.kwargs["inputs"],
                (archive,),
            )

            comparison_payload = json.loads(
                (output / "archive_reference_comparison.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(
                comparison_payload["summary"]["all_figure_14_reference_values_match"]
            )
            validation_payload = json.loads(
                (output / "archive_reference_validation.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(validation_payload["status"], "failed")
            self.assertTrue(
                any(
                    "component times" in error
                    for error in validation_payload["errors"]
                )
            )

    def test_full_preflight_stops_at_semantic_digest_before_row_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "experiments_logs.zip"
            archive.write_bytes(b"placeholder")
            output = root / "full"
            output.mkdir()
            args = argparse.Namespace(archive=archive, julia="julia", mode="full")
            semantic_validation = {
                "status": "failed",
                "errors": ["Archive semantic SHA256 mismatch"],
                "semantic_sha256_matches": False,
                "source_bindings_match": True,
            }

            with (
                mock.patch.object(
                    section_3_3,
                    "materialize_archive_section",
                    return_value=nullcontext(root / "selected"),
                ),
                mock.patch.object(
                    section_3_3,
                    "_parse_archived_logs",
                    return_value=(list(section_3_3.ARCHIVED_SPECS), []),
                ),
                mock.patch.object(
                    section_3_3,
                    "_archive_semantic_manifest",
                    return_value=(
                        {
                            "schema": section_3_3.ARCHIVE_SEMANTIC_SCHEMA,
                            "records": [],
                        },
                        semantic_validation,
                    ),
                ),
                mock.patch.object(section_3_3, "_validate_rows") as validate_rows,
                mock.patch.object(
                    section_3_3, "_reference_comparison"
                ) as compare_reference,
                mock.patch.object(section_3_3, "write_provenance") as provenance,
            ):
                with self.assertRaisesRegex(SystemExit, "semantic validation failed"):
                    section_3_3._full_archive_preflight(args, output)

            validate_rows.assert_not_called()
            compare_reference.assert_not_called()
            provenance.assert_called_once()
            validation_payload = json.loads(
                (output / "archive_reference_validation.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(validation_payload["status"], "failed")
            self.assertTrue(
                any("SHA256 mismatch" in error for error in validation_payload["errors"])
            )

    def test_published_reference_values_are_rho_2000_values(self) -> None:
        self.assertEqual(
            section_3_3.REFERENCE_FIGURE_14[13]["BFS"],
            (10120.0, 0.0, 165.04),
        )
        self.assertEqual(
            section_3_3.REFERENCE_FIGURE_14[18]["GNN"],
            (34046.0, 4.854, 703.55),
        )

    def test_reported_source_panels_are_preserved_byte_exactly(self) -> None:
        payloads = {
            "opf_flip_iterations.png": b"published iterations panel",
            "opf_flip_stacked_time.png": b"published stacked-time panel",
        }
        expected = {
            name: (
                f"copy-{index}.png",
                hashlib.sha256(payload).hexdigest(),
            )
            for index, (name, payload) in enumerate(payloads.items())
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "archive" / "case57_flip_2000"
            source.mkdir(parents=True)
            for name, payload in payloads.items():
                (source / name).write_bytes(payload)
            with mock.patch.object(
                section_3_3, "PUBLISHED_FIGURE_14_SOURCE_PANELS", expected
            ):
                validation = section_3_3._preserve_reported_figure_14_panels(
                    root / "archive", root / "output"
                )

            self.assertEqual(validation["status"], "passed")
            for name, payload in payloads.items():
                target_name = expected[name][0]
                self.assertEqual(
                    (root / "output" / "reported_source" / target_name).read_bytes(),
                    payload,
                )


class Section33ArchiveSemanticManifestTests(unittest.TestCase):
    def test_real_archive_matches_independent_semantic_digest(self) -> None:
        archive = section_3_3.REPO_ROOT / "experiments_logs.zip"
        if not archive.is_file():
            self.skipTest("experiments_logs.zip is not present in this checkout")

        with section_3_3.materialize_archive_section(
            archive, section_3_3.ARCHIVE_SECTION
        ) as section_root:
            specs, rows = section_3_3._parse_archived_logs(section_root)
        manifest, validation = section_3_3._archive_semantic_manifest(
            rows, archive=archive
        )

        self.assertEqual(len(specs), 54)
        self.assertEqual(len(manifest["records"]), 162)
        self.assertEqual(validation["status"], "passed")
        self.assertTrue(validation["source_bindings_match"])
        self.assertEqual(validation["actual_canonical_json_bytes"], 75_278)
        self.assertEqual(
            validation["observed_semantic_sha256"],
            "c2a9cd539527ca3ebcf6873573159e4de5b10b425c8e2a1d23ff288bd95b6d35",
        )
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "archive_semantic_manifest.json"
            section_3_3._write_archive_semantic_manifest(manifest_path, manifest)
            manifest_bytes = manifest_path.read_bytes()
        self.assertEqual(len(manifest_bytes), 75_278)
        self.assertEqual(
            hashlib.sha256(manifest_bytes).hexdigest(),
            "c2a9cd539527ca3ebcf6873573159e4de5b10b425c8e2a1d23ff288bd95b6d35",
        )

    def test_offsetting_raw_iteration_mutations_fail_digest(self) -> None:
        rows = _semantic_archive_rows()
        _, initial = section_3_3._archive_semantic_manifest(rows)
        expected_digest = initial["observed_semantic_sha256"]
        expected_bytes = initial["actual_canonical_json_bytes"]
        constant_patches = (
            mock.patch.object(
                section_3_3, "ARCHIVE_SEMANTIC_EXPECTED_SHA256", expected_digest
            ),
            mock.patch.object(
                section_3_3,
                "ARCHIVE_SEMANTIC_EXPECTED_CANONICAL_BYTES",
                expected_bytes,
            ),
        )
        with constant_patches[0], constant_patches[1]:
            _, baseline = section_3_3._archive_semantic_manifest(rows)
        self.assertEqual(baseline["status"], "passed")

        mutated = [dict(row) for row in rows]
        first = next(
            row
            for row in mutated
            if (row["case"], row["partition_number"], row["method"])
            == ("case30", 3, "BFS")
        )
        second = next(
            row
            for row in mutated
            if (row["case"], row["partition_number"], row["method"])
            == ("case30", 4, "BFS")
        )
        original_sum = int(first["iterations"]) + int(second["iterations"])
        first["iterations"] = int(first["iterations"]) + 1
        second["iterations"] = int(second["iterations"]) - 1
        self.assertEqual(
            int(first["iterations"]) + int(second["iterations"]), original_sum
        )

        with constant_patches[0], constant_patches[1]:
            _, validation = section_3_3._archive_semantic_manifest(mutated)
        self.assertEqual(validation["status"], "failed")
        self.assertTrue(validation["source_bindings_match"])
        self.assertFalse(validation["semantic_sha256_matches"])

    def test_swapped_job_sources_fail_binding_even_if_digest_is_rebased(self) -> None:
        rows = [dict(row) for row in _semantic_archive_rows()]
        p3_id = section_3_3.ARCHIVED_JOB_IDS["case30"][0]
        p4_id = section_3_3.ARCHIVED_JOB_IDS["case30"][1]
        for row in rows:
            if row["archive_folder"] != "case30":
                continue
            if row["archive_job_id"] == p3_id:
                row["archive_job_id"] = p4_id
            elif row["archive_job_id"] == p4_id:
                row["archive_job_id"] = p3_id

        _, observed = section_3_3._archive_semantic_manifest(rows)
        with (
            mock.patch.object(
                section_3_3,
                "ARCHIVE_SEMANTIC_EXPECTED_SHA256",
                observed["observed_semantic_sha256"],
            ),
            mock.patch.object(
                section_3_3,
                "ARCHIVE_SEMANTIC_EXPECTED_CANONICAL_BYTES",
                observed["actual_canonical_json_bytes"],
            ),
        ):
            _, validation = section_3_3._archive_semantic_manifest(rows)

        self.assertTrue(validation["semantic_sha256_matches"])
        self.assertFalse(validation["source_bindings_match"])
        self.assertEqual(validation["status"], "failed")
        self.assertTrue(
            any("partition_number" in error for error in validation["errors"])
        )

    def test_actual_log_path_is_bound_before_content_classification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            section_root = Path(temporary)
            path = section_root / "case30" / "51455118" / "stdout.log"
            path.parent.mkdir(parents=True)
            path.write_text(_log_text(partition_number=4), encoding="utf-8")

            spec, rows = section_3_3._parse_archived_log(section_root, path)

        self.assertEqual(spec.partition_number, 4)
        self.assertTrue(all(row["archive_folder"] == "case30" for row in rows))
        self.assertTrue(all(row["archive_job_id"] == "51455118" for row in rows))
        errors = section_3_3._archive_source_binding_errors(rows)
        self.assertTrue(
            any(
                "case30/51455118" in error and "partition_number=4" in error
                for error in errors
            )
        )


class Section33RawArchiveComparisonTests(unittest.TestCase):
    def test_exact_159_rows_are_gated_and_three_censored_rows_are_reported(self) -> None:
        fresh, archived = _raw_comparison_rows()

        comparison = section_3_3._raw_archive_comparison(fresh, archived)

        self.assertEqual(comparison["status"], "passed")
        summary = comparison["summary"]
        self.assertEqual(summary["expected_raw_row_count"], 162)
        self.assertTrue(summary["exact_key_identity"])
        self.assertTrue(summary["all_configurations_match"])
        self.assertEqual(summary["non_censored_row_count"], 159)
        self.assertEqual(summary["non_censored_iteration_match_count"], 159)
        self.assertEqual(summary["non_censored_status_match_count"], 159)
        self.assertEqual(summary["archive_time_censored_row_count"], 3)
        self.assertTrue(summary["timings_are_informational_only"])
        censored = [
            entry
            for entry in comparison["entries"]
            if entry["classification"] == "archive_time_censored"
        ]
        self.assertEqual(len(censored), 3)
        self.assertTrue(all(not entry["iterations_equality_enforced"] for entry in censored))
        self.assertTrue(all(not entry["status_equality_enforced"] for entry in censored))
        self.assertTrue(all(not entry["timing_equality_enforced"] for entry in censored))

    def test_non_censored_iteration_and_status_mismatch_fail(self) -> None:
        fresh, archived = _raw_comparison_rows()
        fresh[0]["iterations"] = int(fresh[0]["iterations"]) + 1
        fresh[0]["termination_status"] = "ADMM_TERMINATION_TIME_LIMIT"

        comparison = section_3_3._raw_archive_comparison(fresh, archived)

        self.assertEqual(comparison["status"], "failed")
        self.assertFalse(comparison["summary"]["all_non_censored_iterations_match"])
        self.assertFalse(comparison["summary"]["all_non_censored_statuses_match"])
        self.assertTrue(any("iterations" in error for error in comparison["errors"]))
        self.assertTrue(any("status" in error for error in comparison["errors"]))

    def test_configuration_mismatch_fails_even_on_censored_row(self) -> None:
        fresh, archived = _raw_comparison_rows()
        censored_key = next(iter(section_3_3.ARCHIVE_TIME_CENSORED_KEYS))
        row = next(item for item in fresh if section_3_3._raw_row_key(item) == censored_key)
        row["seed"] = 125

        comparison = section_3_3._raw_archive_comparison(fresh, archived)

        self.assertEqual(comparison["status"], "failed")
        self.assertFalse(comparison["summary"]["all_configurations_match"])
        self.assertTrue(any("configuration mismatch" in error for error in comparison["errors"]))

    def test_archived_reconstruction_still_requires_exact_retained_times(self) -> None:
        comparison = {
            "summary": {
                "all_iteration_values_match": True,
                "figure_14_reference_applicable": True,
                "all_figure_14_iteration_values_match": True,
                "all_figure_14_reference_values_match": False,
            }
        }

        self.assertTrue(
            any(
                "component times" in error
                for error in section_3_3._reference_validation_errors(
                    comparison, mode="archived", smoke_profile=False
                )
            )
        )

if __name__ == "__main__":
    unittest.main()
