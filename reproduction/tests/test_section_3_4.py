from __future__ import annotations

from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from reproduction import section_3_4
from reproduction import section_3_4_impl


def _method_marker(method: str) -> str:
    if method == "basic":
        return "Solving classic distributed opt problem..."
    if method == "bfs":
        return "Solving distributed opt problem with BFS bipartization..."
    if method.startswith("milp_"):
        return (
            "Solving distributed opt problem with MILP bipartization "
            f"gap={section_3_4.METHOD_GAP[method]}..."
        )
    return "Solving distributed opt problem with GNN-Pycall bipartization..."


def _summary_label(method: str) -> str:
    if method == "basic":
        return "Classic"
    if method == "bfs":
        return "BFS"
    if method.startswith("milp_"):
        return f"MILP({section_3_4.METHOD_GAP[method]})"
    return "GNN-Pycall"


def _complete_log(*, status: str = "ADMM_TERMINATION_OPTIMAL") -> str:
    lines = [
        "Running Distributed Opt",
        "  numberNodes = 50",
        "  kappa = 0.1",
        "  n = 500",
        "  m = 250",
        "  solver = original",
        "  initialRho = 10.0",
        "  maxIter = 100000",
        "  logInterval = 1000",
        "  seed = 111",
    ]
    summary_rows = []
    for index, method in enumerate(section_3_4.PAPER_METHODS):
        graph_nodes = 100 + index
        graph_edges = 150 + index
        left = 50
        right = graph_nodes - left
        graph_partition = 0.0 if method in {"basic", "bfs"} else index + 0.25
        summary_partition = graph_partition + (0.001 if graph_partition else 0.0)
        iterations = 1000 - 10 * index
        admm_time = 20.0 - index
        lines.append(_method_marker(method))
        lines.append("[ Info: Run Bipartite ADMM with threads = 16.")
        if method == "basic":
            lines.append(
                "[ Info: ADMMBipartiteGraph: The graph is already bipartite; "
                "skip bipartization algorithm."
            )
        else:
            kind = "GNN_BIPARTIZATION" if method == "gnn" else (
                "BFS_BIPARTIZATION" if method == "bfs" else "MILP_BIPARTIZATION"
            )
            lines.append(
                f"[ Info: ADMMBipartiteGraph: {kind} took {graph_partition:.2f} seconds."
            )
        if method.startswith("milp_"):
            gap_percent = 100.0 * section_3_4.METHOD_GAP[method]
            lines.extend(
                (
                    "Solving report",
                    "  Status            Optimal",
                    f"  Gap               0.50% (tolerance: {gap_percent:g}%)",
                    "  Timing            1.25 (total)",
                )
            )
        lines.extend(
            (
                "[ Info: Summary of ADMM Bipartitie Graph:",
                f"    Number of nodes             = {graph_nodes}",
                f"    Parition size (left, right) = ({left}, {right})",
                f"    Number of edges             = {graph_edges}",
                "[ Info: ADMM: initialization took 1.50 seconds",
                "[ Info: ADMM Summary:",
                f"    Solver Status   =   {status}",
                "    Objective       =   1.2345e+02",
                "    Pres (L2)       =   1.0e-05",
                "    Pres (LInf)     =   9.0e-07",
                "    Dres (L2)       =   8.0e-07",
                "    Dres (LInf)     =   7.0e-08",
                f"    Stop. Iter      = {iterations}",
                f"    Total Time      = {admm_time:.2f}",
            )
        )
        summary_rows.append(
            f"{_summary_label(method):>15} | {summary_partition:.3f} | "
            f"{iterations:5d} | {admm_time:9.2f} | 123.4500 |"
        )
    lines.extend(
        (
            "SUMMARY OF RESULTS",
            "Method | BipT | Iters | ADMM Time | ADMM Obj |",
            *summary_rows,
        )
    )
    return "\n".join(lines) + "\n"


def _raw_full_grid() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for solver_index, solver in enumerate(section_3_4.PAPER_SOLVERS):
        for number_nodes in section_3_4.PAPER_NODE_COUNTS:
            for seed in section_3_4_impl.PAPER_SEEDS:
                for method_index, method in enumerate(section_3_4.PAPER_METHODS):
                    right = 100 + method_index + seed // 111
                    key = (solver, number_nodes, seed, method)
                    is_milp = method.startswith("milp_")
                    records.append(
                        {
                            "solver": solver,
                            "number_nodes": number_nodes,
                            "seed": seed,
                            "method": method,
                            "n": 500,
                            "m": 250,
                            "kappa": seed / (number_nodes * 1000.0),
                            "initial_rho": 10.0,
                            "max_iter": 100000,
                            "log_interval": 1000,
                            "threads": 16,
                            "mip_gap": section_3_4.METHOD_GAP.get(method),
                            "mip_status": "Optimal" if is_milp else None,
                            "mip_achieved_gap_percent": 0.5 if is_milp else None,
                            "mip_solver_time_seconds": 1.0 if is_milp else None,
                            "partition_time_seconds": float(method_index),
                            "graph_partition_time_seconds": float(method_index),
                            "graph_nodes": number_nodes + right,
                            "partition_left": number_nodes,
                            "partition_right": right,
                            "graph_edges": 2 * right + method_index,
                            "admm_status": (
                                "ADMM_TERMINATION_TIME_LIMIT"
                                if key == section_3_4_impl.PAPER_ADMM_CENSORED_KEY
                                else "ADMM_TERMINATION_OPTIMAL"
                            ),
                            "iterations": (
                                1000
                                + 100 * solver_index
                                + number_nodes
                                + seed
                                + method_index
                            ),
                            "admm_time_seconds": 10.0 + method_index,
                            "admm_initialization_seconds": 1.0,
                            "objective": 123.45,
                            "primal_residual_l2": 1e-5,
                            "primal_residual_linf": 1e-7,
                            "dual_residual_l2": 1e-6,
                            "dual_residual_linf": 1e-8,
                        }
                    )
    return records


def _raw_row(
    records: list[dict[str, object]],
    key: tuple[str, int, int, str],
) -> dict[str, object]:
    return next(
        row
        for row in records
        if (
            row["solver"],
            row["number_nodes"],
            row["seed"],
            row["method"],
        )
        == key
    )


def _archive_reference_grid() -> list[dict[str, object]]:
    records = _raw_full_grid()
    jobs = {
        (solver, number_nodes, seed): (batch, job_id)
        for batch, job_id, solver, number_nodes, seed in section_3_4_impl.ARCHIVE_PAPER_JOBS
    }
    for record in records:
        key = (
            str(record["solver"]),
            int(record["number_nodes"]),
            int(record["seed"]),
            str(record["method"]),
        )
        batch, job_id = jobs[key[:3]]
        record["archive_batch"] = batch
        record["run_id"] = job_id
        if key[3].startswith("milp_"):
            record["mip_status"] = (
                section_3_4_impl.MIP_TIME_LIMIT_STATUS
                if key in section_3_4_impl.ARCHIVE_MIP_TIME_CENSORED_KEYS
                else "Optimal"
            )
        record["admm_status"] = (
            section_3_4_impl.ADMM_TIME_LIMIT_STATUS
            if key in section_3_4_impl.ARCHIVE_ADMM_TIME_CENSORED_KEYS
            else "ADMM_TERMINATION_OPTIMAL"
        )
    return records


def _write_archive_manifest_files(root: Path, *, skip_index: int | None = None) -> None:
    for index, (batch, job_id, *_metadata) in enumerate(
        section_3_4_impl.ARCHIVE_PAPER_JOBS
    ):
        if index == skip_index:
            continue
        path = root / batch / job_id / "stdout.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")


class Section34ParserTests(unittest.TestCase):
    def test_python_option_is_available_for_gnn_method_sets(self) -> None:
        parser = section_3_4_impl.build_parser()
        args = parser.parse_args(
            (
                "--mode",
                "smoke",
                "--smoke-methods",
                "gnn",
                "--python",
                "/tmp/python3.9",
            )
        )

        self.assertEqual(args.pdmo_python, Path("/tmp/python3.9"))
        self.assertEqual(section_3_4_impl.validate_arguments(args), ("gnn",))

    def test_current_and_archived_labels_canonicalize_to_paper_methods(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stdout.log"
            path.write_text(_complete_log(), encoding="utf-8")
            rows = section_3_4.parse_log(path)

        self.assertEqual([row["method"] for row in rows], list(section_3_4.PAPER_METHODS))
        self.assertEqual(rows[0]["method_display"], "Basic")
        self.assertEqual(rows[-1]["method"], "gnn")
        self.assertEqual(rows[-1]["admm_status"], "ADMM_TERMINATION_OPTIMAL")
        self.assertEqual(rows[2]["mip_status"], "Optimal")
        self.assertEqual(rows[2]["mip_achieved_gap_percent"], 0.5)
        self.assertEqual({row["threads"] for row in rows}, {16})

    def test_thread_echo_is_bound_to_each_method_block(self) -> None:
        bfs_header = (
            _method_marker("bfs")
            + "\n[ Info: Run Bipartite ADMM with threads = 16."
        )
        text = _complete_log().replace(
            bfs_header,
            _method_marker("bfs")
            + "\n[ Info: Run Bipartite ADMM with threads = 8.",
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stdout.log"
            path.write_text(text, encoding="utf-8")
            rows = section_3_4.parse_log(path)

        threads = {str(row["method"]): row["threads"] for row in rows}
        self.assertEqual(threads["basic"], 16)
        self.assertEqual(threads["bfs"], 8)
        self.assertEqual(threads["gnn"], 16)

    def test_summary_precision_is_kept_separate_from_graph_log_time(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stdout.log"
            path.write_text(_complete_log(), encoding="utf-8")
            rows = section_3_4.parse_log(path)

        milp = next(row for row in rows if row["method"] == "milp_0.01")
        self.assertEqual(milp["graph_partition_time_seconds"], 2.25)
        self.assertEqual(milp["partition_time_seconds"], 2.251)
        self.assertAlmostEqual(float(milp["average_degree"]), 304.0 / 102.0)

    def test_buffered_highs_reports_are_keyed_by_tolerance(self) -> None:
        reports: list[str] = []
        report_pattern = re.compile(r"Solving report\n(?:[^\n]*\n){3}")
        body = report_pattern.sub(
            lambda match: reports.append(match.group(0)) or "", _complete_log()
        )
        native_reports = "".join(
            report.replace("0.50%", f"{0.5 + index:.2f}%").replace(
                "1.25 (total)", f"{1.25 + index:.2f}"
            )
            for index, report in reversed(list(enumerate(reports)))
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stdout.log"
            path.write_text(native_reports + body, encoding="utf-8")
            rows = section_3_4.parse_log(path)

        milp_rows = [row for row in rows if str(row["method"]).startswith("milp_")]
        self.assertEqual([row["mip_status"] for row in milp_rows], ["Optimal"] * 4)
        self.assertEqual(
            [row["mip_achieved_gap_percent"] for row in milp_rows], [0.5, 1.5, 2.5, 3.5]
        )
        self.assertEqual(
            [row["mip_solver_time_seconds"] for row in milp_rows], [1.25, 2.25, 3.25, 4.25]
        )
        self.assertTrue(all(row["complete"] for row in milp_rows))

    def test_missing_highs_report_does_not_parse_admm_status(self) -> None:
        parsed = section_3_4.parse_method_block(
            "Solver Status = ADMM_TERMINATION_OPTIMAL", "milp_0.1", 0.1
        )

        self.assertIsNone(parsed["mip_status"])
        self.assertIsNone(parsed["mip_achieved_gap_percent"])
        self.assertIsNone(parsed["mip_solver_time_seconds"])

    def test_graph_values_parse_when_stderr_headers_precede_method_banner(self) -> None:
        block = """
            Number of nodes             = 50
            Number of edges             = 132
            Number of nodes             = 105
            Parition size (left, right) = (60, 45)
            Number of edges             = 187
        """

        parsed = section_3_4.parse_graph_block(block)

        self.assertEqual(parsed["graph_nodes"], 105)
        self.assertEqual(parsed["partition_left"], 60)
        self.assertEqual(parsed["partition_right"], 45)
        self.assertEqual(parsed["graph_edges"], 187)
        self.assertAlmostEqual(float(parsed["average_degree"]), 374.0 / 105.0)

    def test_bipartite_counts_follow_delayed_logger_header(self) -> None:
        block = """
            [ Info: Summary of ADMM Bipartitie Graph:
            [ Info: ADMMBipartiteGraph: BFS_BIPARTIZATION took 0.00 seconds.
            [ Info: ADMM: initialization took 3.52 seconds
            [ Info: Infeasibility: Pres (L2) = 1.4630e-05
            [ Info: ADMMBipartiteGraph: GNN_BIPARTIZATION took 4.51 seconds.
            [ Info: ADMM: initialization took 2.40 seconds
            [ Info: Infeasibility: Pres (L2) = 1.0542e-05
            Number of nodes             = 50
            Number of edges             = 132
            Number of nodes             = 81
            Parition size (left, right) = (36, 45)
            Number of edges             = 163
            Solver Status   = ADMM_TERMINATION_OPTIMAL
            Pres (L2)       = 1.0452e-05
            Pres (LInf)     = 9.9940e-07
            [ Info: Infeasibility: Pres (L2) = 9.9999e-05
        """

        parsed = section_3_4.parse_method_block(block, "gnn", None)

        self.assertEqual(parsed["graph_nodes"], 81)
        self.assertEqual(parsed["partition_left"], 36)
        self.assertEqual(parsed["partition_right"], 45)
        self.assertEqual(parsed["graph_edges"], 163)
        self.assertEqual(parsed["graph_partition_time_seconds"], 4.51)
        self.assertEqual(parsed["admm_initialization_seconds"], 2.40)
        self.assertEqual(parsed["primal_residual_l2"], 1.0452e-05)


class Section34AggregationTests(unittest.TestCase):
    def _records(self) -> list[dict[str, object]]:
        records = []
        for seed, scale in ((111, 1.0), (222, 2.0)):
            records.extend(
                (
                    {
                        "solver": "original",
                        "number_nodes": 50,
                        "seed": seed,
                        "n": 500,
                        "m": 250,
                        "initial_rho": 10.0,
                        "max_iter": 100000,
                        "log_interval": 1000,
                        "method": "basic",
                        "iterations": 100.0 * scale,
                        "partition_time_seconds": 0.0,
                        "graph_partition_time_seconds": 0.0,
                        "admm_time_seconds": 20.0 * scale,
                        "admm_status": (
                            "ADMM_TERMINATION_TIME_LIMIT"
                            if seed == 222
                            else "ADMM_TERMINATION_OPTIMAL"
                        ),
                        "graph_nodes": 150,
                        "partition_left": 50,
                        "partition_right": 100,
                        "graph_edges": 200,
                        "average_degree": 400.0 / 150.0,
                        "balance": 0.5,
                    },
                    {
                        "solver": "original",
                        "number_nodes": 50,
                        "seed": seed,
                        "n": 500,
                        "m": 250,
                        "initial_rho": 10.0,
                        "max_iter": 100000,
                        "log_interval": 1000,
                        "method": "bfs",
                        "iterations": 50.0 * scale,
                        "partition_time_seconds": 1.0 * scale,
                        "graph_partition_time_seconds": 1.0 * scale,
                        "admm_time_seconds": 10.0 * scale,
                        "admm_status": "ADMM_TERMINATION_OPTIMAL",
                        "graph_nodes": 100,
                        "partition_left": 50,
                        "partition_right": 50,
                        "graph_edges": 160,
                        "average_degree": 3.2,
                        "balance": 1.0,
                    },
                )
            )
        return records

    def test_arithmetic_aggregation_normalization_and_time_limit_counts(self) -> None:
        rows = section_3_4_impl.aggregate_runs(self._records())
        by_method = {row["method"]: row for row in rows}
        self.assertEqual(by_method["basic"]["mean_iterations"], 150.0)
        self.assertEqual(by_method["basic"]["normalized_iterations"], 1.0)
        self.assertEqual(by_method["basic"]["n_time_limit"], 1)
        self.assertEqual(by_method["bfs"]["mean_total_time_seconds"], 16.5)
        self.assertAlmostEqual(by_method["bfs"]["normalized_total_time"], 16.5 / 30.0)

    def test_table_features_and_valid_time_limit_policy(self) -> None:
        records = self._records()
        table = section_3_4_impl.build_table_1(records)
        basic = next(row for row in table if row["method"] == "basic")
        bfs = next(row for row in table if row["method"] == "bfs")
        self.assertEqual(basic["average_original_edges"], 100.0)
        self.assertAlmostEqual(float(bfs["mean_average_degree"]), 3.2)
        validation = section_3_4_impl.validate_records(
            records,
            mode="parse",
            selected_methods=("basic", "bfs"),
            failed_job_names=(),
        )
        self.assertTrue(validation["all_time_limit_rows_valid"])
        self.assertEqual(len(validation["time_limit_rows"]), 1)


class Section34SeedContractTests(unittest.TestCase):
    def test_smoke_and_full_commands_use_exact_seed_grids(self) -> None:
        parser = section_3_4_impl.build_parser()
        smoke_args = parser.parse_args(("--mode", "smoke"))
        smoke_methods = section_3_4_impl.validate_arguments(smoke_args)
        smoke = section_3_4_impl.build_jobs(smoke_args, smoke_methods)
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "archive.zip"
            archive.write_bytes(b"")
            full_args = parser.parse_args(
                ("--mode", "full", "--archive", str(archive))
            )
            full_methods = section_3_4_impl.validate_arguments(full_args)
            full = section_3_4_impl.build_jobs(full_args, full_methods)

        self.assertEqual(len(smoke), 1)
        self.assertEqual(smoke[0].name, "original/nodes_50/seed_111")
        self.assertEqual(smoke[0].command[-4], "111")
        self.assertEqual(len(full), 30)
        observed = {
            (
                job.name.split("/")[0],
                int(job.name.split("/")[1].removeprefix("nodes_")),
                int(job.name.split("/")[2].removeprefix("seed_")),
            )
            for job in full
        }
        expected = {
            (solver, nodes, seed)
            for solver in section_3_4.PAPER_SOLVERS
            for nodes in section_3_4.PAPER_NODE_COUNTS
            for seed in section_3_4_impl.PAPER_SEEDS
        }
        self.assertEqual(observed, expected)

    def test_fresh_mode_rejects_nonpaper_configuration(self) -> None:
        parser = section_3_4_impl.build_parser()
        args = parser.parse_args(("--mode", "smoke", "--rho", "99"))
        with self.assertRaisesRegex(SystemExit, "paper configuration"):
            section_3_4_impl.validate_arguments(args)

    def test_full_requires_archive_for_raw_comparison(self) -> None:
        parser = section_3_4_impl.build_parser()
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing.zip"
            args = parser.parse_args(
                ("--mode", "full", "--archive", str(missing))
            )

            with self.assertRaisesRegex(SystemExit, "raw archive consistency check"):
                section_3_4_impl.validate_arguments(args)

    def test_full_rejects_nonpaper_thread_count_before_running(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "archive.zip"
            archive.write_bytes(b"")
            with patch.object(section_3_4_impl, "prepare_mode_output") as prepare:
                with patch.object(section_3_4_impl, "run_jobs") as run_jobs:
                    with self.assertRaisesRegex(SystemExit, "--threads=8"):
                        section_3_4_impl.main(
                            (
                                "--mode",
                                "full",
                                "--archive",
                                str(archive),
                                "--threads",
                                "8",
                            )
                        )

            prepare.assert_not_called()
            run_jobs.assert_not_called()

    def test_parse_rejects_nonpaper_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stdout.log"
            path.write_text(
                _complete_log().replace("  seed = 111", "  seed = 999"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SystemExit, "non-paper"):
                section_3_4_impl.parse_logs((path,), "parse")

    def test_zero_parse_records_are_rejected(self) -> None:
        validation = section_3_4_impl.validate_records(
            (),
            mode="parse",
            selected_methods=("basic",),
            failed_job_names=(),
        )
        self.assertEqual(validation["status"], "failed")
        self.assertTrue(any("No paper" in error for error in validation["errors"]))

    def test_smoke_fingerprints_are_enforced_without_table_means(self) -> None:
        records = [
            record
            for record in Section34AggregationTests()._records()
            if record["seed"] == 111
        ]
        for record in records:
            reference = section_3_4.SMOKE_REFERENCE[str(record["method"])]
            record["iterations"] = reference["iterations"]
            record["graph_nodes"] = reference["graph_nodes"]
            record["graph_edges"] = reference["graph_edges"]
        aggregate = section_3_4_impl.aggregate_runs(records)
        table = section_3_4_impl.build_table_1(records)
        comparison = section_3_4_impl.build_reference_comparison(
            records, aggregate, table, "smoke"
        )
        errors = section_3_4_impl.reference_validation_errors(
            comparison,
            mode="smoke",
            selected_methods=("basic", "bfs"),
        )
        self.assertEqual(errors, [])
        self.assertEqual(comparison["summary"]["compared"], 6)
        self.assertTrue(comparison["summary"]["all_enforced_checks_passed"])

        records[0]["iterations"] = 999
        comparison = section_3_4_impl.build_reference_comparison(
            records,
            section_3_4_impl.aggregate_runs(records),
            section_3_4_impl.build_table_1(records),
            "smoke",
        )
        errors = section_3_4_impl.reference_validation_errors(
            comparison,
            mode="smoke",
            selected_methods=("basic", "bfs"),
        )
        self.assertTrue(any("Reference mismatch" in error for error in errors))

    def test_milp_smoke_drift_is_enforced(self) -> None:
        reference = section_3_4.SMOKE_REFERENCE["milp_0.2"]
        record = {
            "solver": "original",
            "number_nodes": 50,
            "seed": 111,
            "method": "milp_0.2",
            "iterations": reference["iterations"] - 1,
            "graph_nodes": reference["graph_nodes"],
            "graph_edges": reference["graph_edges"],
        }
        comparison = section_3_4_impl.build_reference_comparison(
            (record,), (), (), "smoke"
        )
        errors = section_3_4_impl.reference_validation_errors(
            comparison, mode="smoke", selected_methods=("milp_0.2",)
        )

        self.assertTrue(any("Reference mismatch" in error for error in errors))
        self.assertFalse(comparison["summary"]["all_compared_checks_passed"])
        self.assertFalse(comparison["summary"]["all_enforced_checks_passed"])

    def test_inconsistent_graph_partition_is_rejected(self) -> None:
        record = Section34AggregationTests()._records()[0]
        record["graph_nodes"] = 999

        validation = section_3_4_impl.validate_records(
            (record,),
            mode="parse",
            selected_methods=("basic",),
            failed_job_names=(),
        )

        self.assertTrue(
            any("graph_nodes does not equal" in error for error in validation["errors"])
        )

    def test_blank_mip_status_is_rejected(self) -> None:
        record = Section34AggregationTests()._records()[0]
        record.update(
            {
                "method": "milp_0.01",
                "mip_status": "",
                "mip_achieved_gap_percent": 0.5,
                "mip_solver_time_seconds": 1.25,
            }
        )

        validation = section_3_4_impl.validate_records(
            (record,),
            mode="parse",
            selected_methods=("milp_0.01",),
            failed_job_names=(),
        )

        self.assertTrue(any("missing fields: mip_status" in error for error in validation["errors"]))


class Section34RawArchiveComparisonTests(unittest.TestCase):
    def test_exact_full_grid_passes_while_timing_drift_is_informational(self) -> None:
        archived = _raw_full_grid()
        fresh = [dict(row) for row in archived]
        for row in fresh:
            row["partition_time_seconds"] = float(row["partition_time_seconds"]) + 5.0
            row["admm_time_seconds"] = float(row["admm_time_seconds"]) + 10.0

        comparison = section_3_4_impl.build_raw_archive_comparison(fresh, archived)

        self.assertEqual(section_3_4_impl.raw_archive_validation_errors(comparison), [])
        self.assertTrue(comparison["summary"]["all_enforced_checks_passed"])
        timing_checks = [
            check
            for check in comparison["checks"]
            if check["category"] == "timing_informational"
        ]
        self.assertTrue(timing_checks)
        self.assertTrue(any(not check["passed"] for check in timing_checks))
        self.assertTrue(all(not check["enforced"] for check in timing_checks))

    def test_stable_structure_and_outcome_drift_are_enforced(self) -> None:
        archived = _raw_full_grid()
        fresh = [dict(row) for row in archived]
        bfs = _raw_row(fresh, ("original", 50, 111, "bfs"))
        bfs["graph_edges"] = int(bfs["graph_edges"]) + 1
        bfs["iterations"] = int(bfs["iterations"]) + 1

        comparison = section_3_4_impl.build_raw_archive_comparison(fresh, archived)
        errors = section_3_4_impl.raw_archive_validation_errors(comparison)

        self.assertTrue(any("graph_edges" in error for error in errors))
        self.assertTrue(any("iterations" in error for error in errors))
        self.assertFalse(comparison["summary"]["all_enforced_checks_passed"])

    def test_either_milp_time_limit_censors_partition_and_downstream_outcome(self) -> None:
        archived = _raw_full_grid()
        fresh = [dict(row) for row in archived]
        key = ("original", 200, 111, "milp_0.01")
        archived_row = _raw_row(archived, key)
        fresh_row = _raw_row(fresh, key)
        archived_row["mip_status"] = "Time limit reached"
        fresh_row["graph_nodes"] = int(fresh_row["graph_nodes"]) + 3
        fresh_row["partition_right"] = int(fresh_row["partition_right"]) + 3
        fresh_row["graph_edges"] = int(fresh_row["graph_edges"]) + 5
        fresh_row["iterations"] = int(fresh_row["iterations"]) + 100
        fresh_row["admm_status"] = "ADMM_TERMINATION_TIME_LIMIT"

        comparison = section_3_4_impl.build_raw_archive_comparison(fresh, archived)

        self.assertEqual(section_3_4_impl.raw_archive_validation_errors(comparison), [])
        censored = next(
            row for row in comparison["censored_rows"] if row["key"].endswith("milp_0.01")
        )
        self.assertTrue(censored["mip_censored"])
        relevant = [
            check
            for check in comparison["checks"]
            if check["key"] == "original/N=200/seed=111/milp_0.01"
            and check["field"] in {"graph_nodes", "graph_edges", "iterations"}
        ]
        self.assertTrue(relevant)
        self.assertTrue(all(not check["enforced"] for check in relevant))

    def test_admm_censored_paper_row_still_enforces_structure(self) -> None:
        archived = _raw_full_grid()
        fresh = [dict(row) for row in archived]
        row = _raw_row(fresh, section_3_4_impl.PAPER_ADMM_CENSORED_KEY)
        row["iterations"] = int(row["iterations"]) + 500
        row["admm_status"] = "ADMM_TERMINATION_OPTIMAL"

        comparison = section_3_4_impl.build_raw_archive_comparison(fresh, archived)
        self.assertEqual(section_3_4_impl.raw_archive_validation_errors(comparison), [])

        row["graph_edges"] = int(row["graph_edges"]) + 1
        comparison = section_3_4_impl.build_raw_archive_comparison(fresh, archived)
        errors = section_3_4_impl.raw_archive_validation_errors(comparison)
        self.assertTrue(any("graph_edges" in error for error in errors))

    def test_unexpected_admm_time_limit_outside_paper_censor_is_enforced(self) -> None:
        archived = _raw_full_grid()
        fresh = [dict(row) for row in archived]
        row = _raw_row(fresh, ("original", 50, 111, "bfs"))
        row["admm_status"] = "ADMM_TERMINATION_TIME_LIMIT"

        comparison = section_3_4_impl.build_raw_archive_comparison(fresh, archived)
        errors = section_3_4_impl.raw_archive_validation_errors(comparison)

        self.assertTrue(any("admm_status" in error for error in errors))

    def test_original_and_doubly_must_share_seeded_basic_instance(self) -> None:
        archived = _raw_full_grid()
        fresh = [dict(row) for row in archived]
        key = ("doubly", 100, 222, "basic")
        for records in (archived, fresh):
            row = _raw_row(records, key)
            row["graph_nodes"] = int(row["graph_nodes"]) + 1
            row["partition_right"] = int(row["partition_right"]) + 1

        comparison = section_3_4_impl.build_raw_archive_comparison(fresh, archived)
        errors = section_3_4_impl.raw_archive_validation_errors(comparison)

        self.assertTrue(any("same_instance/N=100/seed=222/basic" in error for error in errors))

    def test_missing_fresh_archive_key_is_rejected(self) -> None:
        archived = _raw_full_grid()
        fresh = [dict(row) for row in archived[:-1]]

        comparison = section_3_4_impl.build_raw_archive_comparison(fresh, archived)
        errors = section_3_4_impl.raw_archive_validation_errors(comparison)

        self.assertTrue(any("missing from fresh full run" in error for error in errors))


class Section34ArchiveManifestTests(unittest.TestCase):
    def test_exact_thirty_paper_logs_are_selected_and_extra_jobs_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_archive_manifest_files(root)
            extra = root / "admm_50_40959569" / "99999999" / "stdout.log"
            extra.parent.mkdir(parents=True)
            extra.write_text("not a paper job", encoding="utf-8")

            paths = section_3_4_impl.archived_paper_log_paths(root)

        expected = [
            root / batch / job_id / "stdout.log"
            for batch, job_id, *_metadata in section_3_4_impl.ARCHIVE_PAPER_JOBS
        ]
        self.assertEqual(paths, expected)
        self.assertEqual(len(paths), 30)
        self.assertNotIn(extra, paths)

    def test_missing_exact_paper_job_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_archive_manifest_files(root, skip_index=0)

            with self.assertRaisesRegex(SystemExit, "51362366/stdout.log"):
                section_3_4_impl.archived_paper_log_paths(root)

    def test_job_manifest_ties_each_seed_to_its_scheduler_job(self) -> None:
        records = _archive_reference_grid()
        report = section_3_4_impl.validate_archive_job_manifest(records)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["observed_job_count"], 30)

        first_job = section_3_4_impl.ARCHIVE_PAPER_JOBS[0][:2]
        for record in records:
            if (record["archive_batch"], record["run_id"]) == first_job:
                record["seed"] = 999
        report = section_3_4_impl.validate_archive_job_manifest(records)
        self.assertEqual(report["status"], "failed")
        self.assertTrue(any("51362366" in error for error in report["errors"]))

    def test_exact_archive_censor_status_manifest_is_enforced(self) -> None:
        records = _archive_reference_grid()
        report = section_3_4_impl.validate_archive_censor_status_manifest(records)

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["observed_mip_time_limit_count"], 48)
        self.assertEqual(report["observed_admm_time_limit_count"], 1)

    def test_missing_and_unexpected_archive_censors_are_rejected(self) -> None:
        records = _archive_reference_grid()
        expected_mip = _raw_row(records, ("original", 100, 111, "milp_0.01"))
        unexpected_mip = _raw_row(records, ("original", 50, 111, "milp_0.2"))
        unexpected_admm = _raw_row(records, ("original", 50, 111, "bfs"))
        expected_mip["mip_status"] = "Optimal"
        unexpected_mip["mip_status"] = section_3_4_impl.MIP_TIME_LIMIT_STATUS
        unexpected_admm["admm_status"] = section_3_4_impl.ADMM_TIME_LIMIT_STATUS

        report = section_3_4_impl.validate_archive_censor_status_manifest(records)

        self.assertEqual(report["status"], "failed")
        self.assertEqual(len(report["missing_mip_time_limit_keys"]), 1)
        self.assertEqual(len(report["unexpected_mip_time_limit_keys"]), 1)
        self.assertEqual(len(report["unexpected_admm_time_limit_keys"]), 1)

    def test_archive_preflight_rejects_nonpaper_thread_count(self) -> None:
        records = _archive_reference_grid()
        first_job = section_3_4_impl.ARCHIVE_PAPER_JOBS[0][:2]
        for record in records:
            if (record["archive_batch"], record["run_id"]) == first_job:
                record["threads"] = 8

        validation = section_3_4_impl.validate_records(
            records,
            mode="archived",
            selected_methods=section_3_4.PAPER_METHODS,
            failed_job_names=(),
        )

        self.assertEqual(validation["status"], "failed")
        self.assertTrue(
            any(
                "original/N=50/seed=111: threads=8, expected 16" in error
                for error in validation["errors"]
            )
        )

    def test_canonical_manifest_rejects_stable_iteration_mutation(self) -> None:
        records = _archive_reference_grid()
        baseline = section_3_4_impl.build_archive_canonical_manifest(records)
        expected_digest = str(baseline["observed_sha256"])
        with patch.object(
            section_3_4_impl,
            "ARCHIVE_PAPER_CANONICAL_SHA256",
            expected_digest,
        ):
            accepted = section_3_4_impl.build_archive_canonical_manifest(records)
            basic = _raw_row(records, ("original", 50, 111, "basic"))
            basic["iterations"] = int(basic["iterations"]) + 1
            mutated = section_3_4_impl.build_archive_canonical_manifest(records)
            _, _, _, preflight = (
                section_3_4_impl.build_archive_reference_validation(records)
            )

        self.assertEqual(accepted["status"], "passed")
        self.assertEqual(mutated["status"], "failed")
        self.assertNotEqual(mutated["observed_sha256"], expected_digest)
        self.assertTrue(
            any("canonical row digest mismatch" in error for error in mutated["errors"])
        )
        self.assertEqual(
            preflight["archive_canonical_manifest"]["status"],
            "failed",
        )
        self.assertTrue(
            any("canonical row digest mismatch" in error for error in preflight["errors"])
        )

    def test_canonical_manifest_binds_rows_to_archive_job_path(self) -> None:
        records = _archive_reference_grid()
        baseline = section_3_4_impl.build_archive_canonical_manifest(records)
        expected_digest = str(baseline["observed_sha256"])
        first_job = section_3_4_impl.ARCHIVE_PAPER_JOBS[0][:2]
        first_record = next(
            record
            for record in baseline["records"]
            if record["archive_batch"] == first_job[0]
            and record["run_id"] == first_job[1]
        )
        self.assertEqual(
            first_record["source_member"],
            "3-4-distributed/admm_50_40959569/51362366/stdout.log",
        )

        second_job = section_3_4_impl.ARCHIVE_PAPER_JOBS[1][:2]
        for record in records:
            if (record["archive_batch"], record["run_id"]) == first_job:
                record["run_id"] = second_job[1]
            elif (record["archive_batch"], record["run_id"]) == second_job:
                record["run_id"] = first_job[1]
        with patch.object(
            section_3_4_impl,
            "ARCHIVE_PAPER_CANONICAL_SHA256",
            expected_digest,
        ):
            mutated = section_3_4_impl.build_archive_canonical_manifest(records)
        job_report = section_3_4_impl.validate_archive_job_manifest(records)

        self.assertEqual(mutated["status"], "failed")
        self.assertEqual(job_report["status"], "failed")
        self.assertEqual(job_report["missing_jobs"], [])
        self.assertEqual(job_report["unexpected_jobs"], [])
        self.assertTrue(any(first_job[1] in error for error in job_report["errors"]))
        self.assertTrue(any(second_job[1] in error for error in job_report["errors"]))

    def test_archive_preflight_failure_happens_before_fresh_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "archive.zip"
            archive.write_bytes(b"")
            output = root / "output"
            full_output = output / "full"
            full_output.mkdir(parents=True)
            sentinel = full_output / "sentinel.txt"
            sentinel.write_text("keep this", encoding="utf-8")
            failed_validation = {"status": "failed", "errors": ["bad archive"]}
            with patch.object(
                section_3_4_impl,
                "load_archive_reference",
                return_value=([], [], [], {}, failed_validation),
            ) as load_archive:
                with patch.object(section_3_4_impl, "run_jobs") as run_jobs:
                    with self.assertRaisesRegex(SystemExit, "archive preflight failed"):
                        section_3_4_impl.main(
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

            load_archive.assert_called_once_with(archive)
            run_jobs.assert_not_called()
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep this")
            self.assertFalse((full_output / "raw").exists())
            self.assertFalse((full_output / "figures").exists())

    def test_archive_reference_loader_persists_preflight_artifacts(self) -> None:
        comparison = {"summary": {"all_enforced_checks_passed": True}}
        validation = {
            "status": "passed",
            "errors": [],
            "reference_summary": comparison["summary"],
            "archive_canonical_manifest": {
                "status": "passed",
                "records": [],
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            with patch.object(
                section_3_4_impl, "materialize_archive_section"
            ) as materialize:
                materialize.return_value.__enter__.return_value = root
                with patch.object(
                    section_3_4_impl, "archived_paper_log_paths", return_value=[]
                ):
                    with patch.object(
                        section_3_4_impl, "parse_logs", return_value=[]
                    ):
                        with patch.object(
                            section_3_4_impl,
                            "build_archive_reference_validation",
                            return_value=([], [], comparison, validation),
                        ):
                            section_3_4_impl.load_archive_reference(
                                root / "archive.zip", output=output
                            )

            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "archive_reference_runs.csv",
                    "archive_reference_aggregate.csv",
                    "archive_reference_table_1.csv",
                    "archive_reference_comparison.json",
                    "archive_reference_manifest.json",
                    "archive_reference_validation.json",
                },
            )



class Section34PublicationTests(unittest.TestCase):
    def test_failed_validation_does_not_publish_paper_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with patch.object(section_3_4_impl, "write_table_1") as write_table:
                with patch.object(section_3_4_impl, "make_figure") as make_figure:
                    table_paths, figure_paths = (
                        section_3_4_impl.publish_validated_artifacts(
                            output,
                            (),
                            (),
                            {"status": "failed", "errors": ["mismatch"]},
                            no_plots=False,
                        )
                    )

            self.assertEqual(table_paths, [])
            self.assertEqual(figure_paths, [])
            write_table.assert_not_called()
            make_figure.assert_not_called()

    def test_passed_validation_publishes_table_and_respects_no_plots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            expected_table = output / "table_1.csv"
            with patch.object(
                section_3_4_impl,
                "write_table_1",
                return_value=[expected_table],
            ) as write_table:
                with patch.object(section_3_4_impl, "make_figure") as make_figure:
                    table_paths, figure_paths = (
                        section_3_4_impl.publish_validated_artifacts(
                            output,
                            (),
                            (),
                            {"status": "passed", "errors": []},
                            no_plots=True,
                        )
                    )

            self.assertEqual(table_paths, [expected_table])
            self.assertEqual(figure_paths, [])
            write_table.assert_called_once()
            make_figure.assert_not_called()


if __name__ == "__main__":
    unittest.main()
