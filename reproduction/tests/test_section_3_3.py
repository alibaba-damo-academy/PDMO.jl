from __future__ import annotations

import argparse
import hashlib
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
    def test_archived_and_fresh_specs_are_explicitly_distinct(self) -> None:
        archived = [spec for spec in section_3_3.ARCHIVED_SPECS if spec.figure == 14]
        fresh = [spec for spec in section_3_3.FRESH_FULL_SPECS if spec.figure == 14]

        self.assertEqual([spec.partition_number for spec in archived], list(range(13, 19)))
        self.assertTrue(all(spec.rho == 2000.0 for spec in archived))
        self.assertTrue(all(spec.archive_folder == "case57_flip_2000" for spec in archived))
        self.assertEqual(
            [spec.archive_job_id for spec in archived],
            [str(value) for value in range(51523865, 51523871)],
        )
        self.assertTrue(all(spec.rho == 1000.0 for spec in fresh))
        self.assertTrue(all(spec.archive_job_id is None for spec in fresh))
        self.assertEqual(
            section_3_3._experiment_profile(section_3_3.ARCHIVED_SPECS)["classification"],
            "historical_reported_archive",
        )
        self.assertEqual(
            section_3_3._experiment_profile(section_3_3.FRESH_FULL_SPECS)["classification"],
            "manuscript_literal_fresh",
        )

    def test_parse_profile_detection_does_not_conflate_rhos(self) -> None:
        self.assertIs(
            section_3_3._parse_mode_expected_specs(section_3_3.ARCHIVED_SPECS),
            section_3_3.ARCHIVED_SPECS,
        )
        self.assertIs(
            section_3_3._parse_mode_expected_specs(section_3_3.FRESH_FULL_SPECS),
            section_3_3.FRESH_FULL_SPECS,
        )

    def test_fresh_full_command_keeps_manuscript_rho_1000(self) -> None:
        spec = next(spec for spec in section_3_3.FRESH_FULL_SPECS if spec.figure == 14)
        args = argparse.Namespace(julia="julia", threads=16)
        job = section_3_3._jobs_for_specs(
            args, (spec,), {"case57": Path("/tmp/case57.m")}
        )[0]

        self.assertIn("1000.0", job.command)
        self.assertNotIn("2000.0", job.command)

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

if __name__ == "__main__":
    unittest.main()
