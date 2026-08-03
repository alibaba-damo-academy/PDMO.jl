from __future__ import annotations

import csv
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from reproduction import appendix_a


class AppendixATableTests(unittest.TestCase):
    def test_canonical_table_matches_paper_configuration(self) -> None:
        by_name = {row.hyperparameter: row.value_setting for row in appendix_a.TABLE_2_ROWS}

        self.assertEqual(len(appendix_a.TABLE_2_ROWS), 17)
        self.assertEqual(by_name["GNN Type"], "GINE")
        self.assertEqual(by_name["Number of GNN Layers"], "40")
        self.assertEqual(by_name["Hidden Dimension"], "64")
        self.assertEqual(by_name["Optimizer"], "AdamW")
        self.assertEqual(by_name["Number of Epochs"], "1000")
        self.assertEqual(by_name["Dataset Size"], "10,000 training, 400 test")

    def test_exports_csv_markdown_and_latex(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            paths = appendix_a.export_table_2(output)

            self.assertTrue(all(path.is_file() for path in paths))
            with paths[0].open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 17)
            self.assertIn("| GNN Type | GINE |", paths[1].read_text(encoding="utf-8"))
            self.assertIn(r"\begin{tabular}", paths[2].read_text(encoding="utf-8"))

    def test_table_mode_writes_profile_validation_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_base = Path(temporary) / "output"
            result = appendix_a.main(
                ("--mode", "table", "--output", str(output_base))
            )
            output = output_base / "table"

            self.assertEqual(result, 0)
            self.assertTrue((output / "table_2.csv").is_file())
            self.assertTrue((output / "validation.json").is_file())
            self.assertTrue((output / "provenance.json").is_file())
            profile = json.loads(
                (output / "artifact_profile.json").read_text(encoding="utf-8")
            )
            self.assertEqual(profile["artifact_scope"], "static_configuration_export")


class AppendixAAccuracyTests(unittest.TestCase):
    def _csv(self, root: Path, text: str) -> Path:
        path = root / "accuracy.csv"
        path.write_text(text, encoding="utf-8")
        return path

    def test_fraction_and_percent_inputs_normalize_identically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fraction = self._csv(root, "epoch,accuracy\n0,0.5\n10,0.75\n1000,0.9\n")
            fraction_rows = appendix_a.read_accuracy_csv(fraction)
            percent = self._csv(root, "epoch,accuracy\n0,50\n10,75\n1000,90\n")
            percent_rows = appendix_a.read_accuracy_csv(
                percent, accuracy_scale="percent"
            )

        self.assertEqual(fraction_rows, percent_rows)
        self.assertEqual(fraction_rows[-1]["accuracy_percent"], 90.0)

    def test_custom_column_names_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._csv(Path(temporary), "step,test_acc\n1,0.4\n2,0.6\n")
            rows = appendix_a.read_accuracy_csv(
                path, epoch_column="step", accuracy_column="test_acc"
            )

        self.assertEqual([row["epoch"] for row in rows], [1, 2])

    def test_rejects_nonincreasing_epochs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._csv(Path(temporary), "epoch,accuracy\n1,0.5\n1,0.6\n")
            with self.assertRaisesRegex(SystemExit, "strictly increasing"):
                appendix_a.read_accuracy_csv(path)

    def test_rejects_epoch_outside_paper_range(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._csv(Path(temporary), "epoch,accuracy\n1,0.5\n1001,0.6\n")
            with self.assertRaisesRegex(SystemExit, "outside"):
                appendix_a.read_accuracy_csv(path)

    def test_rejects_nonfinite_or_out_of_range_accuracy(self) -> None:
        invalid_values = ("nan", "inf", "1.01", "-0.01")
        for value in invalid_values:
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temporary:
                path = self._csv(
                    Path(temporary), f"epoch,accuracy\n1,0.5\n2,{value}\n"
                )
                with self.assertRaises(SystemExit):
                    appendix_a.read_accuracy_csv(path)

    def test_parse_mode_writes_normalized_data_validation_reference_and_figures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._csv(root, "epoch,accuracy\n0,0.5\n500,0.8\n1000,0.9\n")
            output_base = root / "output"

            result = appendix_a.main(
                (
                    "--mode",
                    "parse",
                    "--accuracy-csv",
                    str(source),
                    "--output",
                    str(output_base),
                )
            )
            output = output_base / "parse"

            self.assertEqual(result, 0)
            self.assertTrue((output / "figure_18_accuracy.csv").is_file())
            self.assertTrue((output / "figures" / "figure_18.png").is_file())
            self.assertTrue((output / "figures" / "figure_18.pdf").is_file())
            self.assertTrue((output / "reference.json").is_file())
            self.assertTrue((output / "provenance.json").is_file())
            validation = json.loads((output / "validation.json").read_text(encoding="utf-8"))
            self.assertEqual(validation["status"], "passed")
            self.assertEqual(
                validation["figure_18"]["numeric_reference_comparison"], "unavailable"
            )
            self.assertFalse(validation["figure_18"]["is_retraining_result"])


class AppendixAArchivedSourceTests(unittest.TestCase):
    PNG_BYTES = b"\x89PNG\r\n\x1a\nstrict-source-test"

    def test_directory_source_is_copied_and_marked_source_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            raster = source / "gnn_training_accuracy.png"
            raster.write_bytes(self.PNG_BYTES)
            output_base = root / "output"

            result = appendix_a.main(
                (
                    "--mode",
                    "archived-source",
                    "--arxiv-source",
                    str(source),
                    "--allow-source-hash-mismatch",
                    "--output",
                    str(output_base),
                )
            )
            output = output_base / "archived-source"

            self.assertEqual(result, 0)
            copied = output / "figures" / "figure_18_reported_source.png"
            self.assertEqual(copied.read_bytes(), self.PNG_BYTES)
            self.assertFalse((output / "figures" / "figure_18.png").exists())
            manifest = json.loads(
                (output / "archived_source_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["classification"], "reported-source-only")
            self.assertFalse(manifest["is_retraining_result"])
            self.assertTrue(manifest["copy_hash_matches"])
            self.assertFalse(manifest["source_hash_matches_expected"])
            validation = json.loads(
                (output / "validation.json").read_text(encoding="utf-8")
            )
            self.assertTrue(validation["figure_18"]["source_hash_override_used"])

    def test_tar_source_and_explicit_member_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.tar.gz"
            with tarfile.open(source, mode="w:gz") as archive:
                info = tarfile.TarInfo("paper/plots/final_curve.png")
                info.size = len(self.PNG_BYTES)
                archive.addfile(info, io.BytesIO(self.PNG_BYTES))
            output_base = root / "output"

            appendix_a.main(
                (
                    "--mode",
                    "archived-source",
                    "--arxiv-source",
                    str(source),
                    "--source-member",
                    "paper/plots/final_curve.png",
                    "--allow-source-hash-mismatch",
                    "--output",
                    str(output_base),
                )
            )

            output = output_base / "archived-source"
            copied = output / "figures" / "figure_18_reported_source.png"
            self.assertEqual(copied.read_bytes(), self.PNG_BYTES)

    def test_hash_mismatch_is_refused_without_explicit_override(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / appendix_a.EXPECTED_REPORTED_RASTER_MEMBER).parent.mkdir(parents=True)
            (source / appendix_a.EXPECTED_REPORTED_RASTER_MEMBER).write_bytes(self.PNG_BYTES)
            output_base = root / "output"

            with self.assertRaisesRegex(SystemExit, "does not match verified arXiv v1"):
                appendix_a.main(
                    (
                        "--mode",
                        "archived-source",
                        "--arxiv-source",
                        str(source),
                        "--output",
                        str(output_base),
                    )
                )

            output = output_base / "archived-source"
            validation = json.loads(
                (output / "validation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(validation["status"], "failed_source_hash_mismatch")
            manifest = json.loads(
                (output / "archived_source_manifest.json").read_text(encoding="utf-8")
            )
            self.assertFalse(manifest["source_hash_matches_expected"])

    def test_ambiguous_source_requires_explicit_member(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            (source / "gnn_accuracy_a.png").write_bytes(self.PNG_BYTES)
            (source / "gnn_accuracy_b.png").write_bytes(self.PNG_BYTES)

            with self.assertRaisesRegex(SystemExit, "Multiple equally likely"):
                appendix_a.read_reported_raster(source)


class AppendixAFullModeTests(unittest.TestCase):
    def test_full_mode_fails_actionably_without_fabricating_figure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_base = Path(temporary) / "output"

            with self.assertRaisesRegex(SystemExit, "10,400 generated graphs"):
                appendix_a.main(("--mode", "full", "--output", str(output_base)))

            output = output_base / "full"
            validation = json.loads((output / "validation.json").read_text(encoding="utf-8"))
            self.assertEqual(validation["status"], "failed_missing_training_assets")
            self.assertFalse(validation["figure_18"]["generated"])
            self.assertFalse((output / "figures" / "figure_18.png").exists())
            self.assertTrue((output / "table_2.csv").is_file())
            self.assertTrue((output / "reference.json").is_file())
            self.assertTrue((output / "provenance.json").is_file())


if __name__ == "__main__":
    unittest.main()
