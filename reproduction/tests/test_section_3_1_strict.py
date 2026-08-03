from __future__ import annotations

import unittest

from reproduction.section_3_1_impl import reference_comparison, validate_trajectories


def complete_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method in ("Basic", "BFS", "MILP"):
        for iteration in (1, 2):
            rows.append(
                {
                    "method": method,
                    "iteration": iteration,
                    "pres_l2": 1.0 / iteration,
                    "dres_l2": 2.0 / iteration,
                    "status": "ADMM_TERMINATION_ITERATION_LIMIT",
                    "stop_iter": 2,
                    "admm_time_seconds": 0.25,
                }
            )
    return rows


class StrictSection31ValidationTests(unittest.TestCase):
    def test_completed_iteration_histories_pass(self) -> None:
        validation = validate_trajectories(
            complete_rows(), mode="smoke", max_iter=2
        )
        self.assertEqual(validation["status"], "passed")
        self.assertEqual(validation["row_count"], 6)

    def test_initialization_row_and_solver_fallback_status_are_rejected(self) -> None:
        rows = complete_rows()
        rows.insert(
            0,
            {
                **rows[0],
                "iteration": 0,
                "status": "ADMM_TERMINATION_UNSPECIFIED",
            },
        )
        validation = validate_trajectories(rows, mode="smoke", max_iter=2)
        self.assertEqual(validation["status"], "failed")
        joined = "\n".join(validation["errors"])
        self.assertIn("expected exactly completed iterations", joined)
        self.assertIn("inconsistent statuses", joined)

    def test_fresh_reference_is_explicitly_unavailable(self) -> None:
        validation = validate_trajectories(
            complete_rows(), mode="smoke", max_iter=2
        )
        comparison = reference_comparison(validation, "smoke")
        self.assertFalse(comparison["reference_checks_available"])
        self.assertIsNone(comparison["all_checks_passed"])
        self.assertTrue(comparison["not_applicable_reason"])


if __name__ == "__main__":
    unittest.main()
