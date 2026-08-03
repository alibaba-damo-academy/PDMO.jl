from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from reproduction.section_3_1 import METHODS, parse_archived_logs, validate_rows


def _iteration(iteration: int, primal: float, dual: float) -> str:
    # iter, objective, objective-change, Pres(L2), Pres(LInf), Dres(L2),
    # Dres(LInf), elapsed -- matching the archived progress table.
    return (
        f"{iteration:8d} 1.0 2.0 {primal:.4e} 4.0 "
        f"{dual:.4e} 6.0 7.0\n"
    )


class ArchivedResidualParserTests(unittest.TestCase):
    def test_parses_all_three_methods_and_terminal_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for method in METHODS:
                text = (
                    _iteration(0, 1.0, 2.0)
                    + _iteration(500, 0.5, 0.25)
                    + "    Solver Status = ADMM_TERMINATION_OPTIMAL\n"
                    + "    Stop. Iter = 500\n"
                    + "    Total Time = 3.5\n"
                )
                (root / f"{method.lower()}.txt").write_text(text, encoding="utf-8")
            rows = parse_archived_logs(root)
            validate_rows(rows)
            self.assertEqual(len(rows), 6)
            self.assertEqual({row["method"] for row in rows}, set(METHODS))
            self.assertTrue(all(row["stop_iter"] == 500 for row in rows))
            self.assertTrue(all(row["status"] == "ADMM_TERMINATION_OPTIMAL" for row in rows))


if __name__ == "__main__":
    unittest.main()

