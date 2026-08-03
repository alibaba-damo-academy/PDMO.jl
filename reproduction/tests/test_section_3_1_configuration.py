from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from reproduction.section_3_1_impl import (
    JULIA_DRIVER,
    PAPER_MAX_ITER,
    build_parser,
    validate_exact_configuration,
)


class ExactSection31ConfigurationTests(unittest.TestCase):
    def _write_fixture(self, root: Path, solver: str = "DOUBLY_LINEARIZED_SOLVER") -> None:
        configuration = {
            "algorithm": {
                "column_initialization": "cyclic_unshuffled",
                "row_initialization": "all_in_cluster_1",
                "passes": 5,
                "tie_break": "smallest_cluster_index",
                "row_grouping": "final_row_cluster",
                "bfs_traversal_order": "lexicographic_node_then_edge",
                "empty_cluster_repair": False,
                "early_stopping": False,
            },
            "k": 4,
            "rho": 1000.0,
            "global_seed": 126,
            "solver": solver,
            "max_iter": 200,
            "log_interval": 1,
            "apply_scaling": False,
            "threads": 16,
            "row_cluster": [1, 2, 3, 4],
            "column_cluster": [1, 2, 3, 4],
            "group_rows": [[1], [2], [3], [4]],
            "column_groups": [[1], [2], [3], [4]],
            "group_blocks": [[1], [2], [3], [4]],
        }
        graphs = {
            "configuration": {
                "k": 4,
                "passes": 5,
                "global_seed": 126,
                "mip_rel_gap": 0.01,
                "mip_time_limit_seconds": 60.0,
                "mip_heuristic_effort": 0.2,
            },
            "original": {"nodes": [1], "edges": [1]},
            "bfs": {"nodes": [1], "edges": [1]},
            "milp": {"nodes": [1], "edges": [1]},
        }
        terminal = {
            method: {
                "status": "ADMM_TERMINATION_ITERATION_LIMIT",
                "stop_iter": 200,
                "exported_iterations": [1, 200],
            }
            for method in ("Basic", "BFS", "MILP")
        }
        (root / "exact_configuration.json").write_text(json.dumps(configuration))
        (root / "graphs.json").write_text(json.dumps(graphs))
        (root / "terminal.json").write_text(json.dumps(terminal))

    def test_literal_manuscript_configuration_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_fixture(root)
            validation = validate_exact_configuration(root)
            self.assertEqual(validation["status"], "passed")

    def test_full_limit_matches_original_experiment_driver(self) -> None:
        self.assertEqual(PAPER_MAX_ITER, 100_000)
        args = build_parser().parse_args(("--mode", "full"))
        self.assertEqual(args.max_iter, 100_000)
        self.assertIn(": 100_000", JULIA_DRIVER.read_text(encoding="utf-8"))

    def test_wrong_solver_identity_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_fixture(root, solver="ORIGINAL_ADMM_SUBPROBLEM_SOLVER")
            validation = validate_exact_configuration(root)
            self.assertEqual(validation["status"], "failed")
            self.assertTrue(any("solver=" in error for error in validation["errors"]))

    def test_wrong_global_seed_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_fixture(root)
            path = root / "exact_configuration.json"
            configuration = json.loads(path.read_text())
            configuration["global_seed"] = 42
            path.write_text(json.dumps(configuration))
            validation = validate_exact_configuration(root)
            self.assertEqual(validation["status"], "failed")
            self.assertTrue(any("global_seed" in error for error in validation["errors"]))


if __name__ == "__main__":
    unittest.main()
