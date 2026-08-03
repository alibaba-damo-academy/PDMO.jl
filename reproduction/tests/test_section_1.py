from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from reproduction.section_1_impl import discover_parse_inputs, validate_rows


def smoke_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for formulation, label in (
        ("12", "breaking 1st constraint"),
        ("23", "breaking 2nd constraint"),
        ("31", "breaking 3rd constraint"),
    ):
        for display_iteration in range(1, 101):
            pres = 1.0 / display_iteration
            dres = float("inf") if display_iteration == 1 else 2.0 / display_iteration
            rows.append(
                {
                    "rho": 100.0,
                    "formulation": formulation,
                    "label": label,
                    "iteration": display_iteration,
                    "actual_iteration": display_iteration - 1,
                    "pres_l2": pres,
                    "dres_l2": dres,
                    "residual_sum": pres + dres,
                    "status": "ADMM_TERMINATION_ITERATION_LIMIT",
                    "stop_iter": 10_000,
                    "solve_max_iter": 10_000,
                    "plot_cutoff": 100,
                    "solver": "OriginalADMMSubproblemSolver",
                    "seed": 126,
                }
            )
    return rows


class Figure2ValidationTests(unittest.TestCase):
    def test_exact_legacy_cutoff_grid_passes(self) -> None:
        validation = validate_rows(smoke_rows(), ((100.0, 100),))
        self.assertEqual(validation["status"], "passed")
        self.assertEqual(validation["row_count"], 300)
        self.assertEqual(validation["trajectory_count"], 3)

    def test_shifted_history_and_short_solve_are_rejected(self) -> None:
        rows = smoke_rows()
        rows[0]["actual_iteration"] = 1
        rows[0]["solve_max_iter"] = 100
        validation = validate_rows(rows, ((100.0, 100),))
        self.assertEqual(validation["status"], "failed")
        joined = "\n".join(validation["errors"])
        self.assertIn("actual iterations", joined)
        self.assertIn("solve_max_iter", joined)

    def test_wrong_seed_is_rejected(self) -> None:
        rows = smoke_rows()
        rows[0]["seed"] = 42
        validation = validate_rows(rows, ((100.0, 100),))
        self.assertEqual(validation["status"], "failed")
        self.assertTrue(any("seed" in error for error in validation["errors"]))

    def test_merged_output_wins_over_nested_raw_csvs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            merged = root / "residuals.csv"
            nested = root / "raw" / "rho_100" / "residuals.csv"
            nested.parent.mkdir(parents=True)
            merged.write_text("merged\n", encoding="utf-8")
            nested.write_text("nested\n", encoding="utf-8")
            self.assertEqual(discover_parse_inputs(root), [merged])


if __name__ == "__main__":
    unittest.main()
