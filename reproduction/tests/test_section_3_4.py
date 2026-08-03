from __future__ import annotations

from pathlib import Path
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
        "[ Info: Run Bipartite ADMM with threads = 16.",
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
            lines.extend(
                (
                    "Solving report",
                    "  Status            Optimal",
                    "  Gap               0.50% (tolerance: 1%)",
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

    def test_summary_precision_is_kept_separate_from_graph_log_time(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stdout.log"
            path.write_text(_complete_log(), encoding="utf-8")
            rows = section_3_4.parse_log(path)

        milp = next(row for row in rows if row["method"] == "milp_0.01")
        self.assertEqual(milp["graph_partition_time_seconds"], 2.25)
        self.assertEqual(milp["partition_time_seconds"], 2.251)
        self.assertAlmostEqual(float(milp["average_degree"]), 304.0 / 102.0)

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
        full_args = parser.parse_args(("--mode", "full"))
        smoke_methods = section_3_4_impl.validate_arguments(smoke_args)
        full_methods = section_3_4_impl.validate_arguments(full_args)
        smoke = section_3_4_impl.build_jobs(smoke_args, smoke_methods)
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
