import subprocess
import unittest
from unittest.mock import patch

from reproduction.section_3_3 import _check_python_dependencies


class GNNPythonVersionTests(unittest.TestCase):
    @patch("reproduction.section_3_3.subprocess.run")
    def test_python_38_is_rejected_before_import_probe(self, run_mock):
        run_mock.return_value = subprocess.CompletedProcess(
            args=("python3.8",), returncode=0, stdout="3.8\n"
        )

        with self.assertRaises(SystemExit) as caught:
            _check_python_dependencies("python3.8")

        self.assertIn("Python 3.9-3.11", str(caught.exception))
        run_mock.assert_called_once()

    @patch("reproduction.section_3_3.subprocess.run")
    def test_python_39_with_pinned_imports_is_accepted(self, run_mock):
        run_mock.side_effect = (
            subprocess.CompletedProcess(
                args=("python3.9",), returncode=0, stdout="3.9\n"
            ),
            subprocess.CompletedProcess(
                args=("python3.9",),
                returncode=0,
                stdout="2.8.0 2.6.1 2.0.2\n",
            ),
        )

        _check_python_dependencies("python3.9")

        self.assertEqual(run_mock.call_count, 2)


