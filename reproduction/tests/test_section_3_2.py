from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

from reproduction import section_3_2
from reproduction.common import JobResult


def _method_block(
    method: str,
    *,
    partition_time: float,
    initialization_time: float,
    admm_time: float,
    iterations: tuple[int, ...],
    stop_iter: int | None,
    status: str = "ADMM_TERMINATION_OPTIMAL",
    milp_solve_status: str | None = None,
    milp_solution_status: str = "feasible",
    milp_fingerprint: tuple[int, int, int, int] | None = None,
) -> str:
    titles = {
        "basic": "LP formulation",
        "bfs": "BFS bipartization",
        "milp": "MILP bipartization",
    }
    lines = [f"Running Bipartite ADMM with {titles[method]}..."]
    if method == "milp" and milp_solve_status is not None:
        lines.extend(
            (
                "Running HiGHS 1.11.0",
                "Solving report",
                f"  Status            {milp_solve_status}",
                "  Primal bound      10",
                "  Dual bound        9",
                "  Gap               1% (tolerance: 1%)",
                f"  Solution status   {milp_solution_status}",
            )
        )
    if method == "basic":
        lines.append(
            "ADMMBipartiteGraph: The graph is already bipartite; "
            "skip bipartization algorithm."
        )
    else:
        kind = f"{method.upper()}_BIPARTIZATION"
        lines.append(f"ADMMBipartiteGraph: {kind} took {partition_time} seconds")
    if method == "milp" and milp_fingerprint is not None:
        nodes, left, right, edges = milp_fingerprint
        lines.extend(
            (
                "Summary of ADMM Bipartitie Graph:",
                f"    Number of nodes             = {nodes}",
                f"    Parition size (left, right) = ({left}, {right})",
                f"    Number of edges             = {edges}",
            )
        )
    lines.append(f"ADMM: initialization took {initialization_time} seconds")
    for iteration in iterations:
        lines.append(f" {iteration} 1 2 3 4 5 6 7")
    lines.append(f" Solver Status = {status}")
    if stop_iter is not None:
        lines.append(f" Stop. Iter = {stop_iter}")
    lines.append(f" Total Time = {admm_time}")
    return "\n".join(lines) + "\n"


def _log_text(
    seed: int,
    *,
    include_methods: tuple[str, ...] = section_3_2.METHODS,
    use_stop_iter: bool = True,
    scale: float = 1.0,
) -> str:
    text = [
        "Generating random instance: nodes=200 arcs=2000",
        f" seed = {seed}",
        " solver     = original",
        " maxIter    = 100000",
        " initialRho = 1.0",
        " timeLimit  = 3600.0",
        " logInterval= 1000",
    ]
    for index, method in enumerate(include_methods, 1):
        last_iteration = seed * 10 + index
        text.append(
            _method_block(
                method,
                partition_time=scale * index,
                initialization_time=scale * (index + 0.5),
                admm_time=scale * (index + 2.0),
                iterations=(last_iteration - 1, last_iteration),
                stop_iter=last_iteration if use_stop_iter else None,
            ).rstrip("\n")
        )
    return "\n".join(text) + "\n"


class Section32ParserTests(unittest.TestCase):
    def _parse(self, text: str):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stdout.log"
            path.write_text(text, encoding="utf-8")
            return section_3_2.parse_log(path, "synthetic/stdout.log")

    def test_complete_log_parses_three_long_rows(self) -> None:
        rows, metadata, errors = self._parse(_log_text(1))

        self.assertEqual(errors, [])
        self.assertEqual(metadata["nodes"], 200)
        self.assertEqual(metadata["arcs"], 2000)
        self.assertEqual(metadata["seed"], 1)
        self.assertEqual(
            {setting: metadata[setting] for setting in section_3_2.PAPER_SETTINGS},
            section_3_2.PAPER_SETTINGS,
        )
        self.assertEqual([row["method"] for row in rows], ["basic", "bfs", "milp"])
        self.assertEqual(rows[0]["partition_time_sec"], 0.0)
        self.assertEqual(rows[1]["partition_kind"], "BFS_BIPARTIZATION")
        self.assertEqual(rows[2]["partition_kind"], "MILP_BIPARTIZATION")
        self.assertEqual([row["iterations"] for row in rows], [11, 12, 13])
        self.assertTrue(all(row["iteration_source"] == "stop_iter" for row in rows))
        self.assertAlmostEqual(float(rows[1]["total_time_sec"]), 6.0)

    def test_stderr_timings_before_stdout_method_banners_are_reassociated(self) -> None:
        timing_lines = [
            "ADMMBipartiteGraph: The graph is already bipartite; skip bipartization algorithm.",
            "ADMM: initialization took 1.5 seconds",
            "ADMMBipartiteGraph: BFS_BIPARTIZATION took 2.0 seconds",
            "ADMM: initialization took 2.5 seconds",
            "ADMMBipartiteGraph: MILP_BIPARTIZATION took 3.0 seconds",
            "ADMM: initialization took 3.5 seconds",
        ]
        stdout_lines = [
            line
            for line in _log_text(1).splitlines()
            if "ADMMBipartiteGraph:" not in line
            and "ADMM: initialization took" not in line
        ]
        rows, _, errors = self._parse("\n".join(timing_lines + stdout_lines) + "\n")

        self.assertEqual(errors, [])
        self.assertEqual(
            [row["initialization_time_sec"] for row in rows],
            [1.5, 2.5, 3.5],
        )
        self.assertEqual([row["partition_time_sec"] for row in rows], [0.0, 2.0, 3.0])
        self.assertEqual(rows[1]["partition_kind"], "BFS_BIPARTIZATION")
        self.assertEqual(rows[2]["partition_kind"], "MILP_BIPARTIZATION")

    def test_stderr_first_already_bipartite_timings_follow_method_order(self) -> None:
        timing_lines: list[str] = []
        for initialization in (1.5, 2.5, 3.5):
            timing_lines.extend(
                (
                    "ADMMBipartiteGraph: The graph is already bipartite; "
                    "skip bipartization algorithm.",
                    f"ADMM: initialization took {initialization} seconds",
                )
            )
        stdout_lines = [
            line
            for line in _log_text(1).splitlines()
            if "ADMMBipartiteGraph:" not in line
            and "ADMM: initialization took" not in line
        ]

        rows, _, errors = self._parse("\n".join(timing_lines + stdout_lines) + "\n")

        self.assertEqual(errors, [])
        self.assertEqual([row["partition_time_sec"] for row in rows], [0.0, 0.0, 0.0])
        self.assertEqual(
            [row["initialization_time_sec"] for row in rows],
            [1.5, 2.5, 3.5],
        )

    def test_skip_timing_before_its_banner_does_not_attach_to_prior_method(self) -> None:
        blocks = {
            method: _method_block(
                method,
                partition_time=float(index),
                initialization_time=index + 0.5,
                admm_time=index + 2.0,
                iterations=(10 + index,),
                stop_iter=10 + index,
            ).splitlines()
            for index, method in enumerate(section_3_2.METHODS, 1)
        }
        skip_line = (
            "ADMMBipartiteGraph: The graph is already bipartite; "
            "skip bipartization algorithm."
        )
        for method in ("bfs", "milp"):
            blocks[method] = [
                skip_line if "ADMMBipartiteGraph:" in line else line
                for line in blocks[method]
            ]

        def without_timing(method: str) -> list[str]:
            return [
                line
                for line in blocks[method]
                if "ADMMBipartiteGraph:" not in line
                and "ADMM: initialization took" not in line
            ]

        def timing(method: str) -> list[str]:
            return [
                line
                for line in blocks[method]
                if "ADMMBipartiteGraph:" in line
                or "ADMM: initialization took" in line
            ]

        lines = [
            "Generating random instance: nodes=200 arcs=2000",
            " seed = 1",
            *blocks["basic"],
            *timing("bfs"),
            *without_timing("bfs"),
            *timing("milp"),
            *without_timing("milp"),
        ]
        rows, _, errors = self._parse("\n".join(lines) + "\n")

        self.assertEqual(errors, [])
        self.assertEqual(
            [row["initialization_time_sec"] for row in rows],
            [1.5, 2.5, 3.5],
        )
        self.assertEqual([row["partition_time_sec"] for row in rows], [0.0, 0.0, 0.0])

    def test_latest_progress_iteration_is_used_when_stop_line_is_missing(self) -> None:
        rows, _, errors = self._parse(_log_text(2, use_stop_iter=False))

        self.assertEqual(errors, [])
        self.assertEqual([row["iterations"] for row in rows], [21, 22, 23])
        self.assertEqual([row["latest_progress_iter"] for row in rows], [21, 22, 23])
        self.assertTrue(
            all(row["iteration_source"] == "latest_progress" for row in rows)
        )
        self.assertTrue(
            all(
                "used latest progress-table" in row["parse_warnings"]
                for row in rows
            )
        )

    def test_milp_reports_parse_optimal_and_time_limit_feasible(self) -> None:
        for nodes, seed, expected_status, expected_time_limit in (
            (200, 1, "Optimal", False),
            (400, 1, "Time limit reached", True),
        ):
            with self.subTest(nodes=nodes, seed=seed):
                rows, _, errors = self._parse(_paper_log_text(nodes, seed))
                milp = next(row for row in rows if row["method"] == "milp")
                self.assertEqual(errors, [])
                self.assertTrue(milp["milp_metadata_available"])
                self.assertEqual(milp["milp_solve_status"], expected_status)
                self.assertIs(milp["milp_time_limit"], expected_time_limit)
                self.assertTrue(milp["milp_solution_feasible"])
                self.assertTrue(milp["milp_partition_valid"])

    def test_milp_report_rejects_missing_duplicate_unknown_or_infeasible(
        self,
    ) -> None:
        canonical = _paper_log_text(200, 1)
        mutations = {
            "missing": canonical.replace("Solving report\n", "", 1),
            "duplicate": canonical.replace(
                "Solving report\n",
                "Solving report\nSolving report\n",
                1,
            ),
            "unknown": canonical.replace(
                "Status            Optimal",
                "Status            Interrupted",
                1,
            ),
            "infeasible": canonical.replace(
                "Solution status   feasible",
                "Solution status   infeasible",
                1,
            ),
        }
        for mutation, text in mutations.items():
            with self.subTest(mutation=mutation):
                _, _, errors = self._parse(text)
                self.assertTrue(errors)
                self.assertTrue(
                    any(
                        "HiGHS" in error
                        or "Solving report" in error
                        for error in errors
                    )
                )

    def test_missing_method_section_is_rejected(self) -> None:
        rows, metadata, parse_errors = self._parse(
            _log_text(1, include_methods=("basic", "bfs"))
        )
        _, grid_errors = section_3_2.validate_grid(rows, [metadata], "parse")

        self.assertTrue(any("expected method order" in error for error in parse_errors))
        self.assertTrue(any("expected methods" in error for error in grid_errors))


class Section32AggregationTests(unittest.TestCase):
    def test_arithmetic_means_sample_std_and_status_counts(self) -> None:
        parsed_rows: list[dict[str, object]] = []
        for seed, scale in ((1, 1.0), (2, 2.0)):
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / f"seed_{seed}.log"
                path.write_text(_log_text(seed, scale=scale), encoding="utf-8")
                rows, _, errors = section_3_2.parse_log(path, path.name)
            self.assertEqual(errors, [])
            parsed_rows.extend(rows)

        summaries = section_3_2.summarize(parsed_rows)
        by_method = {row["method"]: row for row in summaries}

        self.assertEqual(len(summaries), 3)
        basic = by_method["basic"]
        self.assertEqual(basic["n_runs"], 2)
        self.assertEqual(basic["partition_time_mean"], 0.0)
        self.assertAlmostEqual(float(basic["initialization_time_mean"]), 2.25)
        self.assertAlmostEqual(
            float(basic["initialization_time_std"]), math.sqrt(1.125)
        )
        self.assertAlmostEqual(float(basic["admm_time_mean"]), 4.5)
        self.assertAlmostEqual(float(basic["admm_time_std"]), math.sqrt(4.5))
        self.assertAlmostEqual(float(basic["iterations_mean"]), 16.0)
        self.assertAlmostEqual(float(basic["iterations_std"]), math.sqrt(50.0))
        self.assertEqual(basic["status_counts"], "ADMM_TERMINATION_OPTIMAL:2")

        bfs = by_method["bfs"]
        self.assertAlmostEqual(float(bfs["partition_time_mean"]), 3.0)
        self.assertAlmostEqual(float(bfs["admm_time_mean"]), 6.0)
        self.assertAlmostEqual(float(bfs["total_time_mean"]), 9.0)

    @staticmethod
    def _conclusion_summaries() -> list[dict[str, object]]:
        summaries: list[dict[str, object]] = []
        for nodes in section_3_2.PAPER_NODES:
            total_times = (
                {"basic": 100.0, "bfs": 120.0, "milp": 130.0}
                if nodes == 200
                else {"basic": 300.0, "bfs": 100.0, "milp": 110.0}
            )
            iterations = {"basic": 100.0, "bfs": 200.0, "milp": 150.0}
            for method in section_3_2.METHODS:
                summaries.append(
                    {
                        "nodes": nodes,
                        "method": method,
                        "n_runs": 10,
                        "iterations_mean": iterations[method],
                        "total_time_mean": total_times[method],
                        "partition_time_mean": 0.0,
                        "initialization_time_mean": 0.0,
                        "admm_time_mean": total_times[method],
                        "status_counts": "ADMM_TERMINATION_OPTIMAL:10",
                    }
                )
        return summaries

    def test_complete_fresh_grid_enforces_robust_conclusions_only(self) -> None:
        for mode in ("full", "parse"):
            with self.subTest(mode=mode):
                report, errors = section_3_2.evaluate_conclusion_consistency(
                    self._conclusion_summaries(), mode
                )

                self.assertEqual(errors, [])
                self.assertTrue(report["complete_paper_grid"])
                self.assertTrue(report["enforced_for_this_mode"])
                self.assertEqual(report["summary"]["enforced_count"], 19)
                self.assertEqual(report["summary"]["enforced_failed"], 0)
                self.assertEqual(report["summary"]["required_pattern_count"], 19)
                self.assertEqual(report["summary"]["required_pattern_failed"], 0)
                self.assertTrue(
                    report["summary"]["all_enforced_checks_passed"]
                )
                self.assertEqual(report["summary"]["informational_count"], 6)
                self.assertEqual(report["summary"]["informational_passed"], 4)
                crossover = [
                    check
                    for check in report["checks"]
                    if check["category"] == "timing_crossover"
                ]
                self.assertEqual([check["nodes"] for check in crossover], [500, 600])
                self.assertTrue(all(not check["enforced"] for check in crossover))
                self.assertTrue(all(not check["passed"] for check in crossover))

    def test_each_robust_conclusion_failure_is_enforced(self) -> None:
        mutations = (
            ((200, "basic"), "iterations_mean", 250.0, "basic_iterations_less_than_bfs"),
            ((200, "milp"), "iterations_mean", 250.0, "milp_iterations_less_than_bfs"),
            ((500, "bfs"), "total_time_mean", 350.0, "bfs_total_time_less_than_basic"),
            ((500, "milp"), "total_time_mean", 350.0, "milp_total_time_less_than_basic"),
        )
        for key, field, value, check_id in mutations:
            with self.subTest(check_id=check_id):
                summaries = self._conclusion_summaries()
                row = next(
                    row
                    for row in summaries
                    if (row["nodes"], row["method"]) == key
                )
                row[field] = value

                report, errors = section_3_2.evaluate_conclusion_consistency(
                    summaries, "full"
                )

                check = next(
                    check
                    for check in report["checks"]
                    if check["id"] == check_id and check["nodes"] == key[0]
                )
                self.assertTrue(check["enforced"])
                self.assertFalse(check["passed"])
                self.assertTrue(errors)
                self.assertFalse(
                    report["summary"]["all_enforced_checks_passed"]
                )

    def test_n300_n400_runtime_relations_are_informational(self) -> None:
        summaries = self._conclusion_summaries()
        for row in summaries:
            if row["nodes"] in (300, 400) and row["method"] in ("bfs", "milp"):
                row["total_time_mean"] = 350.0

        report, errors = section_3_2.evaluate_conclusion_consistency(
            summaries, "full"
        )

        self.assertEqual(errors, [])
        early_runtime = [
            check
            for check in report["checks"]
            if check["nodes"] in (300, 400)
            and check["category"] == "scalability_pattern"
        ]
        self.assertEqual(len(early_runtime), 4)
        self.assertTrue(all(check["informational"] for check in early_runtime))
        self.assertTrue(all(not check["enforced"] for check in early_runtime))
        self.assertTrue(all(not check["passed"] for check in early_runtime))
        self.assertTrue(report["summary"]["all_enforced_checks_passed"])

    def test_archived_grid_records_n400_exception_without_failing(self) -> None:
        summaries = [
            {
                "nodes": nodes,
                "method": method,
                "n_runs": 10,
                **reference,
            }
            for (nodes, method), reference in section_3_2.REFERENCE_MEANS.items()
        ]

        report, errors = section_3_2.evaluate_conclusion_consistency(
            summaries, "archived"
        )

        self.assertEqual(errors, [])
        self.assertTrue(report["complete_paper_grid"])
        self.assertFalse(report["enforced_for_this_mode"])
        self.assertEqual(report["summary"]["required_pattern_failed"], 0)
        n400_runtime = [
            check
            for check in report["checks"]
            if check["nodes"] == 400
            and check["category"] == "scalability_pattern"
        ]
        self.assertEqual(len(n400_runtime), 2)
        self.assertTrue(all(not check["passed"] for check in n400_runtime))
        self.assertTrue(all(not check["enforced"] for check in n400_runtime))
        self.assertTrue(all(check["informational"] for check in n400_runtime))

    def test_reference_report_records_conclusion_checks(self) -> None:
        comparison, errors = section_3_2.compare_reference(
            self._conclusion_summaries(), "full"
        )

        self.assertEqual(errors, [])
        self.assertTrue(
            comparison["conclusion_consistency"]["summary"]
            ["all_enforced_checks_passed"]
        )


class Section32SeedContractTests(unittest.TestCase):
    def test_smoke_and_full_commands_use_exact_seed_grids(self) -> None:
        parser = section_3_2.build_parser()
        smoke = section_3_2.build_jobs(parser.parse_args(("--mode", "smoke")))
        full = section_3_2.build_jobs(parser.parse_args(("--mode", "full")))

        self.assertEqual(len(smoke), 1)
        self.assertEqual(len(full), 50)
        self.assertEqual(smoke[0].name, "nodes_0200/seed_01")
        self.assertEqual(smoke[0].command.count("--timeLimit"), 1)
        self.assertEqual(smoke[0].command[smoke[0].command.index("--seed") + 1], "1")
        observed = {
            (
                int(job.command[job.command.index("--random") + 1]),
                int(job.command[job.command.index("--seed") + 1]),
            )
            for job in full
        }
        expected = {
            (nodes, seed)
            for nodes in section_3_2.PAPER_NODES
            for seed in section_3_2.PAPER_SEEDS
        }
        self.assertEqual(observed, expected)
        self.assertTrue(all(job.command.count("--timeLimit") == 1 for job in full))

    def test_parse_rejects_nonpaper_seed(self) -> None:
        rows, metadata, parse_errors = Section32ParserTests()._parse(_log_text(999))
        self.assertEqual(parse_errors, [])
        _, grid_errors = section_3_2.validate_grid(rows, [metadata], "parse")
        self.assertTrue(any("non-paper jobs" in error for error in grid_errors))

    def test_full_missing_archive_reaches_job_layer_and_records_skip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            missing_archive = root / "missing-experiments-logs.zip"

            def fake_run_jobs(jobs, *, log_root, **_):
                _write_full_grid_logs(log_root)
                return [
                    JobResult(
                        name=job.name,
                        command=job.command,
                        log_path=str(log_root / f"{job.name}.log"),
                        returncode=0,
                        elapsed_seconds=0.0,
                        skipped=False,
                    )
                    for job in jobs
                ]

            with mock.patch.object(
                section_3_2, "run_jobs", side_effect=fake_run_jobs
            ) as run_jobs:
                returncode = section_3_2.main(
                    (
                        "--mode",
                        "full",
                        "--archive",
                        str(missing_archive),
                        "--output",
                        str(output),
                        "--no-plots",
                    )
                )

            self.assertEqual(returncode, 0)
            run_jobs.assert_called_once()
            jobs = run_jobs.call_args.args[0]
            self.assertEqual(len(jobs), 50)
            observed = {
                (
                    int(job.command[job.command.index("--random") + 1]),
                    int(job.command[job.command.index("--seed") + 1]),
                )
                for job in jobs
            }
            self.assertEqual(
                observed,
                {
                    (nodes, seed)
                    for nodes in section_3_2.PAPER_NODES
                    for seed in section_3_2.PAPER_SEEDS
                },
            )

            mode_output = output / "full"
            raw = json.loads(
                (mode_output / "raw_archive_comparison.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(raw["status"], "skipped")
            self.assertFalse(raw["required"])
            self.assertFalse(raw["applied"])
            self.assertEqual(
                raw["reason_code"],
                section_3_2.ARCHIVE_OPTIONAL_SKIP_REASON,
            )
            self.assertEqual(raw["validation_errors"], [])

            validation = json.loads(
                (mode_output / "validation.json").read_text(encoding="utf-8")
            )
            self.assertTrue(validation["valid"])
            self.assertEqual(validation["errors"], [])
            self.assertTrue(validation["paper_settings"]["all_match"])
            raw_reference = validation["raw_archive_reference"]
            self.assertTrue(raw_reference["applicable"])
            self.assertFalse(raw_reference["required"])
            self.assertFalse(raw_reference["applied"])
            self.assertEqual(raw_reference["status"], "skipped")
            self.assertEqual(raw_reference["error_count"], 0)
            self.assertFalse(
                any("archive" in error.lower() for error in validation["errors"])
            )

            provenance = json.loads(
                (mode_output / "provenance.json").read_text(encoding="utf-8")
            )
            self.assertFalse(
                any(
                    Path(record["path"]) == missing_archive.resolve()
                    for record in provenance["inputs"]
                )
            )
            self.assertTrue(
                any(
                    note.startswith(section_3_2.ARCHIVE_OPTIONAL_SKIP_REASON)
                    for note in provenance["notes"]
                )
            )

    def test_full_rejects_invalid_archive_before_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            archive = root / "experiments_logs.zip"
            archive.write_bytes(b"not a zip file")

            with self.assertRaisesRegex(SystemExit, "valid experiments_logs.zip"):
                section_3_2.main(
                    (
                        "--mode",
                        "full",
                        "--archive",
                        str(archive),
                        "--output",
                        str(output),
                        "--no-plots",
                    )
                )

            self.assertFalse(output.exists())

    def test_full_rejects_nonpaper_threads_before_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"

            with self.assertRaisesRegex(SystemExit, "--threads 16"):
                section_3_2.main(
                    (
                        "--mode",
                        "full",
                        "--threads",
                        "8",
                        "--archive",
                        str(root / "missing.zip"),
                        "--output",
                        str(output),
                        "--no-plots",
                    )
                )

            self.assertFalse(output.exists())

    def test_smoke_reference_is_exact_and_mutation_is_rejected(self) -> None:
        rows, _, parse_errors = Section32ParserTests()._parse(_log_text(1))
        self.assertEqual(parse_errors, [])
        for row in rows:
            reference = section_3_2.REFERENCE_SMOKE_N200_SEED1[str(row["method"])]
            row["iterations"] = reference["iterations"]
            row["termination_status"] = reference["termination_status"]
        comparison, errors = section_3_2.compare_reference([], "smoke", rows)
        self.assertEqual(errors, [])
        self.assertTrue(
            comparison["smoke_reference"]["all_iteration_and_status_values_match"]
        )

        rows[0]["iterations"] = 999
        _, errors = section_3_2.compare_reference([], "smoke", rows)
        self.assertTrue(any("smoke iteration mismatch" in error for error in errors))


def _raw_archive_grid() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    ordinal = 0
    for nodes in section_3_2.PAPER_NODES:
        for seed in section_3_2.PAPER_SEEDS:
            scale_folder, run_id = section_3_2.ARCHIVE_PAPER_JOBS[(nodes, seed)]
            for method in section_3_2.METHODS:
                ordinal += 1
                key = (nodes, seed, method)
                partition = float(ordinal % 7)
                admm_time = float(ordinal * 2)
                iterations, status = section_3_2.ARCHIVE_EXPECTED_OUTCOMES[key]
                milp_metadata: dict[str, object | None] = {
                    field: None for field in section_3_2.MILP_METADATA_FIELDS
                }
                if method == "milp":
                    fingerprint = (
                        section_3_2.ARCHIVE_MILP_PARTITION_FINGERPRINTS[
                            (nodes, seed)
                        ]
                    )
                    time_limit = (
                        nodes,
                        seed,
                    ) in section_3_2.ARCHIVE_MILP_TIME_LIMIT_PAIRS
                    milp_metadata.update(
                        {
                            "milp_metadata_available": True,
                            "milp_source_nodes": nodes,
                            "milp_source_arcs": section_3_2.PAPER_ARCS,
                            "milp_source_seed": seed,
                            "milp_solve_status": (
                                "Time limit reached"
                                if time_limit
                                else "Optimal"
                            ),
                            "milp_time_limit": time_limit,
                            "milp_primal_bound": 10.0,
                            "milp_dual_bound": 9.0,
                            "milp_gap_percent": 1.0,
                            "milp_solution_status": "feasible",
                            "milp_solution_feasible": True,
                            "milp_partition_nodes": fingerprint[0],
                            "milp_partition_left": fingerprint[1],
                            "milp_partition_right": fingerprint[2],
                            "milp_partition_edges": fingerprint[3],
                            "milp_partition_fingerprint": ":".join(
                                str(value) for value in fingerprint
                            ),
                            "milp_partition_valid": True,
                        }
                    )
                rows.append(
                    {
                        "scale_folder": scale_folder,
                        "run_id": run_id,
                        "nodes": nodes,
                        "arcs": section_3_2.PAPER_ARCS,
                        "seed": seed,
                        "method": method,
                        "partition_time_sec": partition,
                        "initialization_time_sec": float(ordinal % 5),
                        "admm_time_sec": admm_time,
                        "total_time_sec": partition + admm_time,
                        "termination_status": status,
                        "iterations": iterations,
                        **milp_metadata,
                        "stdout_log": (
                            f"/retained/3-2-networkflow/{scale_folder}/"
                            f"{run_id}/stdout.log"
                        ),
                    }
                )
    return rows


def _raw_row(
    rows: list[dict[str, object]], key: tuple[int, int, str]
) -> dict[str, object]:
    return next(
        row
        for row in rows
        if (row["nodes"], row["seed"], row["method"]) == key
    )


def _write_archive_zip(
    path: Path,
    rows: list[dict[str, object]],
    *,
    command_overrides: dict[tuple[int, int], str] | None = None,
    stdout_overrides: dict[tuple[int, int], str] | None = None,
) -> None:
    stream = io.StringIO()
    fields = (
        "scale_folder",
        "run_id",
        "nodes",
        "arcs",
        "seed",
        "method",
        "partition_time_sec",
        "init_time_sec",
        "algorithm_time_sec",
        "termination_status",
        "iterations",
        "stdout_log",
    )
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "scale_folder": row["scale_folder"],
                "run_id": row["run_id"],
                "nodes": row["nodes"],
                "arcs": row["arcs"],
                "seed": row["seed"],
                "method": row["method"],
                "partition_time_sec": row["partition_time_sec"],
                "init_time_sec": row["initialization_time_sec"],
                "algorithm_time_sec": row["admm_time_sec"],
                "termination_status": row["termination_status"],
                "iterations": row["iterations"],
                "stdout_log": row["stdout_log"],
            }
        )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(section_3_2.ARCHIVE_RUNS_MEMBER, stream.getvalue())
        jobs = {
            (
                int(row["nodes"]),
                int(row["seed"]),
                str(row["scale_folder"]),
                int(row["run_id"]),
            )
            for row in rows
        }
        for nodes, seed, scale_folder, run_id in sorted(jobs):
            command = " ".join(
                section_3_2._expected_archive_command_tokens(nodes, seed)
            )
            if command_overrides and (nodes, seed) in command_overrides:
                command = command_overrides[(nodes, seed)]
            prefix = (
                f"experiments_logs/3-2-networkflow/{scale_folder}/{run_id}"
            )
            archive.writestr(f"{prefix}/cmd", command)
            stdout = _archive_milp_stdout(nodes, seed)
            if stdout_overrides and (nodes, seed) in stdout_overrides:
                stdout = stdout_overrides[(nodes, seed)]
            archive.writestr(f"{prefix}/stdout.log", stdout)


def _archive_milp_stdout(
    nodes: int,
    seed: int,
    *,
    solve_status: str | None = None,
    solution_status: str = "feasible",
    fingerprint: tuple[int, int, int, int] | None = None,
) -> str:
    if solve_status is None:
        solve_status = (
            "Time limit reached"
            if (nodes, seed) in section_3_2.ARCHIVE_MILP_TIME_LIMIT_PAIRS
            else "Optimal"
        )
    if fingerprint is None:
        fingerprint = section_3_2.ARCHIVE_MILP_PARTITION_FINGERPRINTS[
            (nodes, seed)
        ]
    iterations, status = section_3_2.ARCHIVE_EXPECTED_OUTCOMES[
        (nodes, seed, "milp")
    ]
    return "\n".join(
        (
            f"Generating random instance: nodes={nodes} arcs=2000",
            f" seed = {seed}",
            _method_block(
                "milp",
                partition_time=60.0,
                initialization_time=0.1,
                admm_time=1.0,
                iterations=(iterations,),
                stop_iter=iterations,
                status=status,
                milp_solve_status=solve_status,
                milp_solution_status=solution_status,
                milp_fingerprint=fingerprint,
            ),
        )
    )


def _paper_log_text(nodes: int, seed: int) -> str:
    text = [
        f"Generating random instance: nodes={nodes} arcs=2000",
        f" seed = {seed}",
        " solver     = original",
        " maxIter    = 100000",
        " initialRho = 1.0",
        " timeLimit  = 3600.0",
        " logInterval= 1000",
    ]
    for index, method in enumerate(section_3_2.METHODS, 1):
        iterations, status = section_3_2.ARCHIVE_EXPECTED_OUTCOMES[
            (nodes, seed, method)
        ]
        text.append(
            _method_block(
                method,
                partition_time=float(index),
                initialization_time=index + 0.5,
                admm_time=(
                    10.0
                    if nodes >= 300 and method == "basic"
                    else index + 2.0
                ),
                iterations=(iterations,),
                stop_iter=iterations,
                status=status,
                milp_solve_status=(
                    (
                        "Time limit reached"
                        if (nodes, seed)
                        in section_3_2.ARCHIVE_MILP_TIME_LIMIT_PAIRS
                        else "Optimal"
                    )
                    if method == "milp"
                    else None
                ),
                milp_fingerprint=(
                    section_3_2.ARCHIVE_MILP_PARTITION_FINGERPRINTS[
                        (nodes, seed)
                    ]
                    if method == "milp"
                    else None
                ),
            ).rstrip("\n")
        )
    return "\n".join(text) + "\n"


def _write_full_grid_logs(root: Path) -> list[dict[str, object]]:
    for nodes in section_3_2.PAPER_NODES:
        for seed in section_3_2.PAPER_SEEDS:
            text = _paper_log_text(nodes, seed)
            path = root / f"nodes_{nodes:04d}" / f"seed_{seed:02d}.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    rows, _, errors = section_3_2.parse_logs(sorted(root.rglob("*.log")), root)
    if errors:
        raise AssertionError(errors)
    for row in rows:
        nodes = int(row["nodes"])
        seed = int(row["seed"])
        scale_folder, run_id = section_3_2.ARCHIVE_PAPER_JOBS[(nodes, seed)]
        row["scale_folder"] = scale_folder
        row["run_id"] = run_id
        row["stdout_log"] = (
            f"/retained/3-2-networkflow/{scale_folder}/{run_id}/stdout.log"
        )
    return rows


class Section32RawArchiveValidationTests(unittest.TestCase):
    def _assert_full_preflight_rejects(
        self,
        rows: list[dict[str, object]],
        pattern: str,
        *,
        command_overrides: dict[tuple[int, int], str] | None = None,
        stdout_overrides: dict[tuple[int, int], str] | None = None,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "experiments_logs.zip"
            output = root / "output"
            _write_archive_zip(
                archive,
                rows,
                command_overrides=command_overrides,
                stdout_overrides=stdout_overrides,
            )
            with mock.patch.object(
                section_3_2,
                "run_jobs",
                side_effect=AssertionError("jobs started before archive preflight"),
            ):
                with self.assertRaisesRegex(SystemExit, pattern):
                    section_3_2.main(
                        (
                            "--mode",
                            "full",
                            "--archive",
                            str(archive),
                            "--output",
                            str(output),
                            "--no-plots",
                        )
                    )
            self.assertFalse(output.exists())

    def test_stable_iteration_and_status_mismatch_fails(self) -> None:
        archived = _raw_archive_grid()
        fresh = [dict(row) for row in archived]
        row = _raw_row(fresh, (300, 4, "bfs"))
        row["iterations"] = int(row["iterations"]) + 1
        row["termination_status"] = "ADMM_TERMINATION_ITERATION_LIMIT"

        comparison = section_3_2.build_raw_archive_comparison(fresh, archived)
        errors = section_3_2.raw_archive_validation_errors(comparison)

        self.assertTrue(any("iterations" in error for error in errors))
        self.assertTrue(any("termination_status" in error for error in errors))
        self.assertEqual(comparison["summary"]["compared_keys"], 150)
        self.assertEqual(comparison["summary"]["stable_outcome_rows"], 115)
        self.assertEqual(comparison["summary"]["enforced_outcome_checks"], 230)
        self.assertEqual(comparison["summary"]["enforced_identity_checks"], 150)
        self.assertFalse(comparison["summary"]["all_required_checks_passed"])

    def test_archive_arcs_contract_and_identity_are_enforced(self) -> None:
        archived = _raw_archive_grid()
        fresh = [dict(row) for row in archived]
        row = _raw_row(archived, (400, 7, "milp"))
        row["arcs"] = 1999

        comparison = section_3_2.build_raw_archive_comparison(fresh, archived)
        errors = section_3_2.raw_archive_validation_errors(comparison)

        self.assertTrue(any("arcs value is not 2000" in error for error in errors))
        self.assertTrue(any("/arcs" in error for error in errors))
        self.assertFalse(comparison["summary"]["key_integrity_valid"])

    def test_archive_time_limit_row_is_censored_and_does_not_fail(self) -> None:
        archived = _raw_archive_grid()
        fresh = [dict(row) for row in archived]
        row = _raw_row(fresh, section_3_2.PAPER_CENSORED_KEY)
        row["iterations"] = int(row["iterations"]) + 999
        row["termination_status"] = "ADMM_TERMINATION_OPTIMAL"
        row["admm_time_sec"] = float(row["admm_time_sec"]) / 2.0

        comparison = section_3_2.build_raw_archive_comparison(fresh, archived)
        errors = section_3_2.raw_archive_validation_errors(comparison)

        self.assertEqual(errors, [])
        self.assertEqual(comparison["summary"]["censored_row_count"], 35)
        self.assertEqual(comparison["summary"]["admm_censored_row_count"], 1)
        self.assertEqual(
            comparison["summary"]["milp_partition_censored_row_count"], 34
        )
        self.assertEqual(comparison["summary"]["partition_censored_row_count"], 34)
        self.assertEqual(comparison["summary"]["censored_outcome_row_count"], 35)
        censored = next(
            item
            for item in comparison["censored_rows"]
            if (item["nodes"], item["seed"], item["method"])
            == section_3_2.PAPER_CENSORED_KEY
        )
        self.assertEqual(
            (censored["nodes"], censored["seed"], censored["method"]),
            section_3_2.PAPER_CENSORED_KEY,
        )
        outcome_checks = [
            check
            for check in comparison["checks"]
            if check["key"] == "nodes=500/seed=6/basic"
            and check["field"] in {"iterations", "termination_status"}
        ]
        self.assertEqual(len(outcome_checks), 2)
        self.assertTrue(all(not check["enforced"] for check in outcome_checks))
        self.assertTrue(comparison["summary"]["all_required_checks_passed"])

    def test_uncensored_milp_outcome_mismatch_fails(self) -> None:
        archived = _raw_archive_grid()
        fresh = [dict(row) for row in archived]
        row = _raw_row(fresh, (200, 1, "milp"))
        row["iterations"] = int(row["iterations"]) + 1

        comparison = section_3_2.build_raw_archive_comparison(fresh, archived)
        errors = section_3_2.raw_archive_validation_errors(comparison)

        self.assertTrue(any("/iterations" in error for error in errors))
        self.assertFalse(comparison["summary"]["all_required_checks_passed"])

    def test_censored_milp_outcome_mismatch_passes_but_is_reported(self) -> None:
        archived = _raw_archive_grid()
        fresh = [dict(row) for row in archived]
        key = (400, 6, "milp")
        row = _raw_row(fresh, key)
        row["iterations"] = int(row["iterations"]) - 390
        row["termination_status"] = "ADMM_TERMINATION_ITERATION_LIMIT"
        row["milp_primal_bound"] = 9.5

        comparison = section_3_2.build_raw_archive_comparison(fresh, archived)
        errors = section_3_2.raw_archive_validation_errors(comparison)

        self.assertEqual(errors, [])
        censored = next(
            item
            for item in comparison["censored_rows"]
            if (item["nodes"], item["seed"], item["method"]) == key
        )
        self.assertEqual(censored["censor_type"], "milp_partition_time_limit")
        self.assertEqual(censored["archive_iterations"], 13107)
        self.assertEqual(censored["observed_iterations"], 12717)
        outcome_checks = [
            check
            for check in comparison["checks"]
            if check["key"] == "nodes=400/seed=6/milp"
            and check["field"] in {"iterations", "termination_status"}
        ]
        self.assertEqual(len(outcome_checks), 2)
        self.assertTrue(all(not check["enforced"] for check in outcome_checks))
        self.assertTrue(any(not check["passed"] for check in outcome_checks))
        self.assertTrue(comparison["summary"]["all_required_checks_passed"])

    def test_fresh_optimal_replacing_archived_partition_timeout_passes(self) -> None:
        archived = _raw_archive_grid()
        fresh = [dict(row) for row in archived]
        row = _raw_row(fresh, (300, 4, "milp"))
        row["milp_solve_status"] = "Optimal"
        row["milp_time_limit"] = False

        comparison = section_3_2.build_raw_archive_comparison(fresh, archived)
        errors = section_3_2.raw_archive_validation_errors(comparison)

        self.assertEqual(errors, [])
        checks = [
            check
            for check in comparison["checks"]
            if check["key"] == "nodes=300/seed=4/milp"
            and check["field"] in {"milp_solve_status", "milp_time_limit"}
        ]
        self.assertEqual(len(checks), 2)
        self.assertTrue(all(not check["enforced"] for check in checks))
        self.assertTrue(all(not check["passed"] for check in checks))

    def test_fresh_timeout_replacing_archive_optimal_partition_fails(self) -> None:
        archived = _raw_archive_grid()
        fresh = [dict(row) for row in archived]
        row = _raw_row(fresh, (200, 1, "milp"))
        row["milp_solve_status"] = "Time limit reached"
        row["milp_time_limit"] = True

        comparison = section_3_2.build_raw_archive_comparison(fresh, archived)
        errors = section_3_2.raw_archive_validation_errors(comparison)

        self.assertTrue(any("milp_solve_status" in error for error in errors))
        self.assertTrue(any("milp_time_limit" in error for error in errors))
        self.assertFalse(comparison["summary"]["all_required_checks_passed"])

    def test_censored_partition_count_change_is_informational(self) -> None:
        archived = _raw_archive_grid()
        fresh = [dict(row) for row in archived]
        row = _raw_row(fresh, (600, 3, "milp"))
        row["milp_partition_nodes"] = 2514
        row["milp_partition_left"] = 557
        row["milp_partition_right"] = 1957
        row["milp_partition_edges"] = 3914
        row["milp_partition_fingerprint"] = "2514:557:1957:3914"
        row["milp_partition_valid"] = True

        comparison = section_3_2.build_raw_archive_comparison(fresh, archived)
        errors = section_3_2.raw_archive_validation_errors(comparison)

        self.assertEqual(errors, [])
        check = next(
            check
            for check in comparison["checks"]
            if check["key"] == "nodes=600/seed=3/milp"
            and check["field"] == "milp_partition_fingerprint"
        )
        self.assertFalse(check["enforced"])
        self.assertFalse(check["passed"])

    def test_censored_milp_missing_or_infeasible_partition_fails(self) -> None:
        for mutation in ("missing", "infeasible"):
            with self.subTest(mutation=mutation):
                archived = _raw_archive_grid()
                fresh = [dict(row) for row in archived]
                row = _raw_row(fresh, (500, 2, "milp"))
                if mutation == "missing":
                    row["milp_metadata_available"] = False
                    row["milp_partition_fingerprint"] = None
                    row["milp_partition_valid"] = False
                else:
                    row["milp_solution_status"] = "infeasible"
                    row["milp_solution_feasible"] = False

                comparison = section_3_2.build_raw_archive_comparison(
                    fresh, archived
                )
                errors = section_3_2.raw_archive_validation_errors(comparison)

                self.assertTrue(errors)
                self.assertTrue(
                    any(
                        "milp_" in error
                        or "partition" in error
                        for error in errors
                    )
                )
                self.assertFalse(
                    comparison["summary"]["all_required_checks_passed"]
                )

    def test_echoed_paper_setting_mismatch_fails_metadata_validation(self) -> None:
        parser_tests = Section32ParserTests()
        text = _log_text(1).replace(" maxIter    = 100000", " maxIter    = 999")
        rows, metadata, parse_errors = parser_tests._parse(text)
        self.assertEqual(parse_errors, [])

        report, grid_errors = section_3_2.validate_grid(rows, [metadata], "smoke")

        self.assertTrue(any("maxIter=100000" in error for error in grid_errors))
        self.assertEqual(report["paper_settings"]["mismatch_count"], 1)
        self.assertFalse(report["paper_settings"]["all_match"])

    def test_validated_archive_loader_rejects_wrong_censor_contract(self) -> None:
        archived = _raw_archive_grid()
        row = _raw_row(archived, section_3_2.PAPER_CENSORED_KEY)
        row["termination_status"] = "ADMM_TERMINATION_OPTIMAL"
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "experiments_logs.zip"
            _write_archive_zip(archive, archived)

            with self.assertRaisesRegex(ValueError, "archive outcome mismatch"):
                section_3_2.load_validated_archive_runs(archive)

    def test_full_preflight_rejects_mutated_scale_folder_and_run_id(self) -> None:
        for mutation, value in (
            ("scale_folder", "99999999"),
            ("run_id", 99_999_999),
        ):
            with self.subTest(mutation=mutation):
                archived = _raw_archive_grid()
                for row in archived:
                    if int(row["nodes"]) == 300 and int(row["seed"]) == 4:
                        row[mutation] = value
                        row["stdout_log"] = (
                            f"/retained/3-2-networkflow/{row['scale_folder']}/"
                            f"{row['run_id']}/stdout.log"
                        )
                self._assert_full_preflight_rejects(
                    archived, "archive job mapping mismatch"
                )

    def test_full_preflight_rejects_mutated_stable_iteration(self) -> None:
        archived = _raw_archive_grid()
        row = _raw_row(archived, (400, 7, "bfs"))
        row["iterations"] = int(row["iterations"]) + 1

        self._assert_full_preflight_rejects(
            archived, "archive outcome mismatch"
        )

    def test_full_preflight_rejects_mutated_archived_command_config(self) -> None:
        command = " ".join(
            section_3_2._expected_archive_command_tokens(600, 9)
        ).replace("--maxIter 100000", "--maxIter 99999")

        self._assert_full_preflight_rejects(
            _raw_archive_grid(),
            "archived command mismatch",
            command_overrides={(600, 9): command},
        )

    def test_full_preflight_rejects_wrong_archive_milp_censor_manifest(
        self,
    ) -> None:
        pair = (400, 1)
        stdout = _archive_milp_stdout(
            *pair,
            solve_status="Optimal",
        )

        self._assert_full_preflight_rejects(
            _raw_archive_grid(),
            "archive MILP censor manifest mismatch",
            stdout_overrides={pair: stdout},
        )

    def test_full_preflight_rejects_swapped_archive_stdout_identity(self) -> None:
        first = (400, 1)
        second = (400, 7)

        self._assert_full_preflight_rejects(
            _raw_archive_grid(),
            "archive stdout identity mismatch",
            stdout_overrides={
                first: _archive_milp_stdout(*second),
                second: _archive_milp_stdout(*first),
            },
        )

    def test_complete_grid_parse_records_archive_hash_and_raw_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs = root / "logs"
            archived = _write_full_grid_logs(logs)
            archive = root / "experiments_logs.zip"
            _write_archive_zip(archive, archived)
            output = root / "output"

            returncode = section_3_2.main(
                (
                    "--mode",
                    "parse",
                    "--logs",
                    str(logs),
                    "--archive",
                    str(archive),
                    "--output",
                    str(output),
                    "--no-plots",
                )
            )

            mode_output = output / "parse"
            self.assertEqual(returncode, 0)
            raw_path = mode_output / "raw_archive_comparison.json"
            self.assertTrue(raw_path.is_file())
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            self.assertTrue(raw["applied"])
            self.assertTrue(raw["summary"]["all_required_checks_passed"])
            provenance = json.loads(
                (mode_output / "provenance.json").read_text(encoding="utf-8")
            )
            archive_record = next(
                record
                for record in provenance["inputs"]
                if Path(record["path"]) == archive.resolve()
            )
            expected_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
            self.assertEqual(archive_record["sha256"], expected_hash)

    def test_complete_grid_parse_load_failure_is_not_marked_applied(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs = root / "logs"
            _write_full_grid_logs(logs)
            output = root / "output"

            returncode = section_3_2.main(
                (
                    "--mode",
                    "parse",
                    "--logs",
                    str(logs),
                    "--archive",
                    str(root / "missing.zip"),
                    "--output",
                    str(output),
                    "--no-plots",
                )
            )

            mode_output = output / "parse"
            self.assertEqual(returncode, 1)
            raw = json.loads(
                (mode_output / "raw_archive_comparison.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(raw["applied"])
            validation = json.loads(
                (mode_output / "validation.json").read_text(encoding="utf-8")
            )
            self.assertFalse(validation["raw_archive_reference"]["applied"])
            self.assertFalse(validation["valid"])


if __name__ == "__main__":
    unittest.main()
