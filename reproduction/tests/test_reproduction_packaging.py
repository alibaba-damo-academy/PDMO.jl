import contextlib
import hashlib
import io
import re
import unittest
from pathlib import Path

from reproduction import (
    appendix_a,
    section_1,
    section_3_1,
    section_3_2,
    section_3_3,
    section_3_4,
)
from reproduction.common import DEFAULT_ENLIGHT_HARD_MPS, DEFAULT_MATPOWER_DIR
from reproduction.section_3_1_impl import (
    build_parser as build_section_3_1_parser,
)
from reproduction.section_3_1_reported import validate_official_mps

REPO_ROOT = Path(__file__).resolve().parents[2]
REPRODUCTION_ROOT = REPO_ROOT / "reproduction"


class ReproductionPackagingTests(unittest.TestCase):
    def test_root_readme_hands_off_to_reproduction_guide(self):
        text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("[reproduction guide](reproduction/README.md)", text)

    def test_guide_names_entry_points_and_dependency_families(self):
        text = (REPRODUCTION_ROOT / "README.md").read_text(encoding="utf-8")
        for entry_point in (
            "section_1.py",
            "section_3_1.py",
            "section_3_2.py",
            "section_3_3.py",
            "section_3_4.py",
            "appendix_a.py",
        ):
            self.assertIn(entry_point, text)
        for dependency in (
            "Python 3.10",
            "Julia 1.11.5",
            "torch==2.8.0",
            "torch-geometric==2.6.1",
            "numpy==2.0.2",
            "TeX Live 2025",
            "experiments_logs.zip",
        ):
            self.assertIn(dependency, text)

    def test_guide_makes_archive_optional_for_fresh_full_runs(self):
        text = (REPRODUCTION_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn(
            "Fresh `smoke` and `full` runs do not require `experiments_logs.zip`",
            text,
        )
        self.assertIn("optional row comparison as skipped", text)
        self.assertNotIn(
            "Full mode requires `experiments_logs.zip`",
            text,
        )

    def test_all_public_entry_points_dispatch_to_current_help(self):
        expected_modes = (
            (section_1, ("archived", "parse", "smoke", "full")),
            (
                section_3_1,
                ("archived", "parse", "smoke", "full", "reported"),
            ),
            (section_3_2, ("archived", "parse", "smoke", "full")),
            (section_3_3, ("archived", "parse", "smoke", "full")),
            (section_3_4, ("archived", "parse", "smoke", "full")),
            (appendix_a, ("table", "parse", "archived-source", "full")),
        )
        for module, modes in expected_modes:
            with self.subTest(module=module.__name__):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    with self.assertRaises(SystemExit) as caught:
                        module.main(("--help",))
                self.assertEqual(caught.exception.code, 0)
                help_text = output.getvalue()
                self.assertIn("usage:", help_text)
                for mode in modes:
                    self.assertIn(mode, help_text)

    def test_section_3_4_julia_driver_rejects_loose_arguments(self):
        text = (REPRODUCTION_ROOT / "julia" / "section_3_4.jl").read_text(
            encoding="utf-8"
        )
        self.assertIn("length(args) in 8:11 || usage_error()", text)
        self.assertIn("initialRho must be positive", text)
        self.assertIn("mipHeuristicEffort must lie in [0, 1]", text)

    def test_tested_julia_lock_snapshots_are_portable(self):
        snapshots = {
            "PDMO.Manifest.toml": "9a8efa225a41c900b4c1d592c3e73987d4db1375068c1c5a72cdeb321d324b67",
            "advanced.Manifest.toml": "be70f2e80a271886ff027e6817bba6b92f1491618ac84f0478bc33e3bd09f79d",
        }
        for filename, expected_sha256 in snapshots.items():
            path = REPRODUCTION_ROOT / "julia_manifests" / filename
            payload = path.read_bytes()
            text = payload.decode("utf-8")
            self.assertEqual(hashlib.sha256(payload).hexdigest(), expected_sha256)
            self.assertIn('julia_version = "1.11.5"', text)
            self.assertIsNone(re.search(r'^path = "/', text, flags=re.MULTILINE))
            self.assertRegex(
                text,
                r"(?ms)^\[\[deps\.HiGHS\]\].*?^version = \"1\.18\.0\"$",
            )
            self.assertRegex(
                text,
                (
                    r"(?ms)^\[\[deps\.HiGHS_jll\]\].*?"
                    r"^version = \"1\.11\.0\+1\"$"
                ),
            )

    def test_bundled_problem_inputs_have_pinned_hashes(self):
        expected = {
            "miplib/enlight_hard.mps.gz": "942168c2126a2a91ae3ec1ededea59bc1af0cad55f94223edf4c03d20e831f66",
            "matpower/case30.m": "3d9030311259b553be85d02336b7e1bcb24ec04775bee6671bdb62d18e4e2137",
            "matpower/case57.m": "2218325a6e8fe6c7b8b28202f523670459268075a6fd41b4959d66f17d47d28b",
            "matpower/case89pegase.m": "7eb25c591f04a08dcd99ab433451054eaafd8bb3d7999d9279a8ddb23d8ffe58",
            "matpower/case118.m": "bc2e6f22b4b9e776572885ee4b50e4f4ab2ee0c5577e9126e86d906f14c4b5f7",
            "matpower/case300.m": "69a90280e999ef533d94656e0fbc08311f1347c962dd2753ff2005ff5e3f9ac5",
            "matpower/case1888rte.m": "df675cd826bb300e91596795ee3258a70deb81f0329dbc49ccf24c6048668037",
        }
        for relative, expected_sha256 in expected.items():
            path = REPRODUCTION_ROOT / "instances" / relative
            self.assertTrue(path.is_file(), path)
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), expected_sha256
            )
        for relative in (
            "instances/README.md",
            "instances/SHA256SUMS",
            "instances/matpower/MATPOWER-PACKAGE-LICENSE",
            "instances/matpower/MATPOWER-CITATION",
        ):
            path = REPRODUCTION_ROOT / relative
            self.assertTrue(path.is_file(), path)


    def test_fresh_clis_default_to_bundled_problem_inputs(self):
        self.assertTrue(DEFAULT_ENLIGHT_HARD_MPS.is_absolute())
        self.assertTrue(DEFAULT_MATPOWER_DIR.is_absolute())
        self.assertEqual(
            build_section_3_1_parser().parse_args(("--mode", "reported")).mps,
            DEFAULT_ENLIGHT_HARD_MPS,
        )
        self.assertEqual(
            section_3_3.build_parser().parse_args(("--mode", "smoke")).matpower_dir,
            DEFAULT_MATPOWER_DIR,
        )
        self.assertEqual(
            validate_official_mps(DEFAULT_ENLIGHT_HARD_MPS)["status"], "passed"
        )
        inputs = section_3_3._matpower_inputs(
            DEFAULT_MATPOWER_DIR, section_3_3.FRESH_FULL_SPECS
        )
        self.assertEqual(set(inputs), set(section_3_3.FIGURE_13_CASES))


if __name__ == "__main__":
    unittest.main()
