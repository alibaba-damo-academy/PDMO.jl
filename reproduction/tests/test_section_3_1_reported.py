from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, PngImagePlugin

from reproduction.section_3_1_impl import main
from reproduction.section_3_1_reported import (
    pixel_sha256,
    validate_reported_driver_source,
    validate_reported_panels,
)


class ReportedSection31Tests(unittest.TestCase):
    def test_public_historical_driver_is_pinned_and_configured(self) -> None:
        validation = validate_reported_driver_source()

        self.assertEqual(validation["status"], "passed")
        self.assertFalse(validation["errors"])
        self.assertTrue(
            all(check["matched"] for check in validation["configuration_checks"])
        )
        configuration = validation["expected_configuration"]
        self.assertEqual(configuration["global_seed"], 126)
        self.assertEqual(configuration["column_clusters"], 4)
        self.assertEqual(configuration["alternating_passes"], 10)
        self.assertTrue(configuration["force_split_single_block"])
        self.assertTrue(configuration["promote_pairwise_rows"])

    def test_pixel_hash_ignores_png_container_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.png"
            second = root / "second.png"
            image = Image.new("RGBA", (3, 2), (17, 34, 51, 255))
            image.save(first)
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("reviewer-note", "same decoded pixels")
            image.save(second, pnginfo=metadata, compress_level=9)

            self.assertNotEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(pixel_sha256(first), pixel_sha256(second))

    def test_panel_validator_detects_one_changed_pixel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            panel = root / "matrix_original.png"
            Image.new("RGBA", (2, 2), (1, 2, 3, 255)).save(panel)
            expected, _ = pixel_sha256(panel)

            passed = validate_reported_panels(
                root, {"matrix_original.png": expected}
            )
            self.assertEqual(passed["status"], "passed")

            changed = Image.open(panel).convert("RGBA")
            changed.putpixel((0, 0), (9, 2, 3, 255))
            changed.save(panel)
            failed = validate_reported_panels(
                root, {"matrix_original.png": expected}
            )
            self.assertEqual(failed["status"], "failed")
            self.assertFalse(failed["all_published_pixel_hashes_match"])

    def test_reported_cli_reaches_preflight_without_name_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wrong_mps = root / "wrong.mps"
            wrong_mps.write_text("NAME WRONG\nENDATA\n", encoding="utf-8")
            output_base = root / "results"

            returncode = main(
                (
                    "--mode",
                    "reported",
                    "--mps",
                    str(wrong_mps),
                    "--output",
                    str(output_base),
                    "--julia",
                    sys.executable,
                )
            )

            self.assertEqual(returncode, 1)
            reported = output_base / "reported"
            validation = json.loads(
                (reported / "validation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(validation["status"], "failed")
            self.assertIn("official enlight_hard", "\n".join(validation["errors"]))
            profile = json.loads(
                (reported / "artifact_profile.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                profile["artifact_scope"], "fresh_reported_historical_workflow"
            )


