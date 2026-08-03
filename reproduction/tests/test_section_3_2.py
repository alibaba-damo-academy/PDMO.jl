from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

from reproduction import section_3_2


def _method_block(
    method: str,
    *,
    partition_time: float,
    initialization_time: float,
    admm_time: float,
    iterations: tuple[int, ...],
    stop_iter: int | None,
    status: str = "ADMM_TERMINATION_OPTIMAL",
) -> str:
    titles = {
        "basic": "LP formulation",
        "bfs": "BFS bipartization",
        "milp": "MILP bipartization",
    }
    lines = [f"Running Bipartite ADMM with {titles[method]}..."]
    if method == "basic":
        lines.append(
            "ADMMBipartiteGraph: The graph is already bipartite; "
            "skip bipartization algorithm."
        )
    else:
        kind = f"{method.upper()}_BIPARTIZATION"
        lines.append(f"ADMMBipartiteGraph: {kind} took {partition_time} seconds")
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
        self.assertTrue(all(row["iteration_source"] == "latest_progress" for row in rows))
        self.assertTrue(all("used latest progress-table" in row["parse_warnings"] for row in rows))

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


if __name__ == "__main__":
    unittest.main()
