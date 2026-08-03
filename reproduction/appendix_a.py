#!/usr/bin/env python3
"""Strict reproduction support for Appendix A, Table 2, and Figure 18.

The repository contains the trained inference checkpoint, but it does not
contain the final training program, the 10,400 training/test graphs, or the
per-epoch accuracy history used for Figure 18.  This entry point therefore
keeps three materially different operations explicit:

``table``
    Export paper Table 2 from canonical constants.
``parse``
    Validate a user-supplied epoch/accuracy CSV and render Figure 18.
``archived-source``
    Copy and hash the already-rendered Figure 18 raster from an explicitly
    supplied arXiv source directory/tarball.  This is source preservation,
    not experimental reproduction or retraining.
``full``
    Fail with an inventory of the original training assets that are still
    needed.  The command never fabricates an accuracy curve from the final
    checkpoint or digitizes the paper figure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import platform
import subprocess
import sys
import tarfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

try:
    from .common import (
        REPO_ROOT,
        REPRODUCTION_ROOT,
        get_pyplot,
        prepare_mode_output,
        sha256_file,
        write_csv,
        write_json,
    )
except ImportError:
    from common import (  # type: ignore
        REPO_ROOT,
        REPRODUCTION_ROOT,
        get_pyplot,
        prepare_mode_output,
        sha256_file,
        write_csv,
        write_json,
    )


DEFAULT_OUTPUT = REPRODUCTION_ROOT / "output" / "appendix_a"
MAX_EPOCHS = 1000
TRAINING_GRAPHS = 10_000
TEST_GRAPHS = 400
TOTAL_GRAPHS = TRAINING_GRAPHS + TEST_GRAPHS
INFERENCE_CHECKPOINT = (
    REPO_ROOT / "advanced" / "src" / "gnn" / "GNN" / "model_weights_10.pth"
)


@dataclass(frozen=True)
class Table2Row:
    category: str
    hyperparameter: str
    value_setting: str
    description: str


TABLE_2_ROWS: tuple[Table2Row, ...] = (
    Table2Row(
        "Model Architecture",
        "GNN Type",
        "GINE",
        "Graph Isomorphism Network with Edge features",
    ),
    Table2Row(
        "Model Architecture",
        "Number of GNN Layers",
        "40",
        "Depth of the model (message passing steps)",
    ),
    Table2Row(
        "Model Architecture",
        "Hidden Dimension",
        "64",
        "Dimensionality of node embeddings",
    ),
    Table2Row(
        "Model Architecture",
        "Activation Function",
        "ELU",
        "Non-linear activation in MLPs",
    ),
    Table2Row(
        "Model Architecture",
        "Normalization",
        "LayerNorm",
        "Applied before GNN layer and feed-forward network",
    ),
    Table2Row(
        "Model Architecture",
        "Residual Connection",
        "Yes",
        "Skip connection in each GNN block",
    ),
    Table2Row(
        "Model Architecture",
        "Dropout Rate",
        "0.5",
        "Applied after each feed-forward network",
    ),
    Table2Row(
        "Training Configuration",
        "Optimizer",
        "AdamW",
        "Adam with decoupled weight decay",
    ),
    Table2Row(
        "Training Configuration",
        "Learning Rate",
        "1e-5",
        "Initial learning rate",
    ),
    Table2Row(
        "Training Configuration",
        "Weight Decay",
        "1e-5",
        "L2 regularization strength",
    ),
    Table2Row(
        "Training Configuration",
        "Batch Size",
        "64",
        "Graphs per batch",
    ),
    Table2Row(
        "Training Configuration",
        "Number of Epochs",
        str(MAX_EPOCHS),
        "Maximum training epochs",
    ),
    Table2Row(
        "Training Configuration",
        "Learning Rate Schedule",
        "Cosine Annealing",
        "Cosine decay",
    ),
    Table2Row(
        "Data & Features",
        "Node Feature Dimension",
        "20",
        "Initial node feature dimension (all-ones vector)",
    ),
    Table2Row(
        "Data & Features",
        "Edge Feature Dimension",
        "15",
        "Edge feature dimension (statistical, structural, geometric)",
    ),
    Table2Row(
        "Data & Features",
        "Graph Size (Training)",
        "20 nodes per graph",
        "Fixed number of nodes in each generated graph",
    ),
    Table2Row(
        "Data & Features",
        "Dataset Size",
        "10,000 training, 400 test",
        "Total 10,400 generated QP instances",
    ),
)

TABLE_FIELDS = ("category", "hyperparameter", "value_setting", "description")
ACCURACY_FIELDS = ("epoch", "accuracy_fraction", "accuracy_percent")
RASTER_SUFFIXES = frozenset((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"))
MAX_SOURCE_RASTER_BYTES = 100 * 1024 * 1024
EXPECTED_REPORTED_RASTER_MEMBER = "plots/gnn_test_accuracy.png"
EXPECTED_REPORTED_RASTER_SHA256 = (
    "b93e66ce848b2801e567da74e63edc265e448d18217ef2cf63fea546775dc476"
)

MISSING_TRAINING_ASSETS: tuple[str, ...] = (
    "the final synthetic-QP generator, including graph-topology sampling and ground-truth label construction",
    "the exact data-generation and model-training random seeds",
    "the 10,400 generated graphs, or a complete deterministic recipe that recreates them",
    "the exact ordered 10,000/400 train/test split",
    "the final AdamW/cosine-annealing training entry point and a pinned Python/PyTorch environment",
    "the per-epoch test-accuracy history (or deterministic logging code) used to draw Figure 18",
)

NO_NUMERIC_REFERENCE_REASON = (
    "Neither this repository nor the internal experiment archive contains the final "
    "per-epoch test-accuracy history. The committed .pth file is an inference state "
    "dictionary, so it cannot recover earlier epoch accuracies."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("table", "parse", "archived-source", "full"),
        default="table",
        help=(
            "table: export Table 2; parse: validate --accuracy-csv and plot Figure 18; "
            "archived-source: preserve a raster from --arxiv-source; full: fail until "
            "the original training assets are supplied"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Artifact directory (never recursively deleted).",
    )
    parser.add_argument(
        "--accuracy-csv",
        type=Path,
        help="Epoch/test-accuracy CSV required by --mode parse.",
    )
    parser.add_argument(
        "--epoch-column",
        default="epoch",
        help="Name of the epoch column in --accuracy-csv (default: epoch).",
    )
    parser.add_argument(
        "--accuracy-column",
        default="accuracy",
        help="Name of the test-accuracy column in --accuracy-csv (default: accuracy).",
    )
    parser.add_argument(
        "--accuracy-scale",
        choices=("fraction", "percent"),
        default="fraction",
        help="Interpret accuracy as [0,1] fractions or [0,100] percentages.",
    )
    parser.add_argument(
        "--arxiv-source",
        type=Path,
        help="Explicit arXiv source tarball or extracted source directory.",
    )
    parser.add_argument(
        "--source-member",
        help=(
            "Raster path relative to the source root/tarball. Required when automatic "
            "Figure 18 candidate detection is ambiguous."
        ),
    )
    parser.add_argument(
        "--allow-source-hash-mismatch",
        action="store_true",
        help="Explicitly allow a source raster whose SHA256 differs from the verified arXiv v1 asset.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="In parse mode, validate/write normalized CSV without rendering PNG/PDF.",
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if not args.epoch_column.strip():
        raise SystemExit("--epoch-column must not be empty")
    if not args.accuracy_column.strip():
        raise SystemExit("--accuracy-column must not be empty")
    if args.epoch_column == args.accuracy_column:
        raise SystemExit("--epoch-column and --accuracy-column must be different")

    if args.mode != "archived-source" and args.allow_source_hash_mismatch:
        raise SystemExit("--allow-source-hash-mismatch is valid only with --mode archived-source")

    if args.mode == "parse":
        if args.accuracy_csv is None:
            raise SystemExit("--mode parse requires --accuracy-csv /path/to/epoch_accuracy.csv")
        if args.arxiv_source is not None or args.source_member is not None:
            raise SystemExit("--mode parse does not accept --arxiv-source/--source-member")
    elif args.mode == "archived-source":
        if args.arxiv_source is None:
            raise SystemExit(
                "--mode archived-source requires --arxiv-source /path/to/source.tar "
                "or /path/to/extracted/source"
            )
        if args.accuracy_csv is not None:
            raise SystemExit("--mode archived-source does not accept --accuracy-csv")
    else:
        if args.accuracy_csv is not None or args.arxiv_source is not None or args.source_member is not None:
            raise SystemExit(
                f"--mode {args.mode} does not accept --accuracy-csv, --arxiv-source, "
                "or --source-member"
            )


def prepare_appendix_output(path: Path, mode: str) -> Path:
    """Return a mode-isolated output directory, matching the other CLIs."""

    return prepare_mode_output(path, mode)


def _markdown_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def export_table_2(output: Path) -> tuple[Path, Path, Path]:
    records = [asdict(row) for row in TABLE_2_ROWS]
    csv_path = output / "table_2.csv"
    markdown_path = output / "table_2.md"
    latex_path = output / "table_2.tex"
    write_csv(csv_path, TABLE_FIELDS, records)

    markdown_lines = (
        "# Table 2: Model hyperparameters and training configuration",
        "",
        "| Category | Hyperparameter | Value/Setting | Description |",
        "|---|---|---|---|",
        *(
            "| " + " | ".join(
                _markdown_escape(value)
                for value in (
                    row.category,
                    row.hyperparameter,
                    row.value_setting,
                    row.description,
                )
            ) + " |"
            for row in TABLE_2_ROWS
        ),
        "",
    )
    markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")

    latex_lines = [
        r"\begin{tabular}{llll}",
        r"\hline",
        r"Category & Hyperparameter & Value/Setting & Description \\",
        r"\hline",
    ]
    for row in TABLE_2_ROWS:
        latex_lines.append(
            " & ".join(
                _latex_escape(value)
                for value in (
                    row.category,
                    row.hyperparameter,
                    row.value_setting,
                    row.description,
                )
            )
            + r" \\"
        )
    latex_lines.extend((r"\hline", r"\end{tabular}", ""))
    latex_path.write_text("\n".join(latex_lines), encoding="utf-8")
    return csv_path, markdown_path, latex_path


def read_accuracy_csv(
    path: Path,
    *,
    epoch_column: str = "epoch",
    accuracy_column: str = "accuracy",
    accuracy_scale: str = "fraction",
) -> list[dict[str, float | int]]:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"Accuracy CSV not found: {source}")
    if accuracy_scale not in ("fraction", "percent"):
        raise ValueError(f"Unsupported accuracy scale: {accuracy_scale}")

    records: list[dict[str, float | int]] = []
    previous_epoch: int | None = None
    with source.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise SystemExit(f"Accuracy CSV has no header: {source}")
        if len(fieldnames) != len(set(fieldnames)):
            raise SystemExit(f"Accuracy CSV has duplicate column names: {source}")
        missing = [name for name in (epoch_column, accuracy_column) if name not in fieldnames]
        if missing:
            raise SystemExit(
                f"Accuracy CSV is missing column(s) {', '.join(missing)}; "
                f"available columns: {', '.join(fieldnames)}"
            )

        for line_number, raw in enumerate(reader, start=2):
            epoch_text = (raw.get(epoch_column) or "").strip()
            accuracy_text = (raw.get(accuracy_column) or "").strip()
            if not epoch_text or not accuracy_text:
                raise SystemExit(
                    f"Accuracy CSV line {line_number} has an empty epoch or accuracy value"
                )
            try:
                epoch_number = float(epoch_text)
                accuracy_number = float(accuracy_text)
            except ValueError as error:
                raise SystemExit(
                    f"Accuracy CSV line {line_number} has a non-numeric epoch/accuracy"
                ) from error
            if not math.isfinite(epoch_number) or not math.isfinite(accuracy_number):
                raise SystemExit(
                    f"Accuracy CSV line {line_number} contains NaN or an infinite value"
                )
            if not epoch_number.is_integer():
                raise SystemExit(
                    f"Accuracy CSV line {line_number} epoch must be an integer: {epoch_text}"
                )
            epoch = int(epoch_number)
            if not 0 <= epoch <= MAX_EPOCHS:
                raise SystemExit(
                    f"Accuracy CSV line {line_number} epoch {epoch} is outside [0, {MAX_EPOCHS}]"
                )
            if previous_epoch is not None and epoch <= previous_epoch:
                raise SystemExit(
                    f"Accuracy CSV epochs must be strictly increasing; line {line_number} "
                    f"has {epoch} after {previous_epoch}"
                )

            if accuracy_scale == "fraction":
                if not 0.0 <= accuracy_number <= 1.0:
                    raise SystemExit(
                        f"Accuracy CSV line {line_number} accuracy {accuracy_number:g} is "
                        "outside [0, 1] for --accuracy-scale fraction"
                    )
                accuracy_fraction = accuracy_number
            else:
                if not 0.0 <= accuracy_number <= 100.0:
                    raise SystemExit(
                        f"Accuracy CSV line {line_number} accuracy {accuracy_number:g} is "
                        "outside [0, 100] for --accuracy-scale percent"
                    )
                accuracy_fraction = accuracy_number / 100.0

            records.append(
                {
                    "epoch": epoch,
                    "accuracy_fraction": accuracy_fraction,
                    "accuracy_percent": 100.0 * accuracy_fraction,
                }
            )
            previous_epoch = epoch

    if len(records) < 2:
        raise SystemExit("Accuracy CSV must contain at least two data rows")
    return records


def plot_figure_18(rows: Sequence[dict[str, float | int]], output: Path) -> tuple[Path, Path]:
    pyplot = get_pyplot()
    figure, axis = pyplot.subplots(figsize=(6.4, 4.2))
    axis.plot(
        [int(row["epoch"]) for row in rows],
        [float(row["accuracy_percent"]) for row in rows],
        linewidth=2.0,
        color="#1f77b4",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Test accuracy (%)")
    axis.set_xlim(0, MAX_EPOCHS)
    axis.set_ylim(0, 100)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    png_path = output / "figures" / "figure_18.png"
    pdf_path = output / "figures" / "figure_18.pdf"
    figure.savefig(png_path, dpi=200)
    figure.savefig(pdf_path)
    pyplot.close(figure)
    return png_path, pdf_path


def _normalized_member_name(name: str) -> str:
    candidate = PurePosixPath(name.replace("\\", "/"))
    parts = tuple(part for part in candidate.parts if part not in ("", "."))
    if candidate.is_absolute() or not parts or ".." in parts:
        raise SystemExit(f"Unsafe or empty source member path: {name!r}")
    return PurePosixPath(*parts).as_posix()


def _raster_score(name: str) -> int:
    path = PurePosixPath(name)
    if path.suffix.lower() not in RASTER_SUFFIXES:
        return -1
    stem = path.stem.lower().replace("-", "_")
    score = 0
    if stem in ("figure_18", "figure18", "fig_18", "fig18"):
        score += 100
    if "figure_18" in stem or "figure18" in stem or "fig_18" in stem or "fig18" in stem:
        score += 50
    if "training" in stem or "train" in stem:
        score += 12
    if "accuracy" in stem:
        score += 12
    if "gnn" in stem:
        score += 8
    return score if score > 0 else -1


def _select_candidate(names: Sequence[str], requested: str | None) -> str:
    normalized_to_original: dict[str, str] = {}
    for name in names:
        normalized = _normalized_member_name(name)
        if normalized in normalized_to_original:
            raise SystemExit(f"Duplicate normalized source member: {normalized}")
        normalized_to_original[normalized] = name

    if requested is not None:
        normalized = _normalized_member_name(requested)
        original = normalized_to_original.get(normalized)
        if original is None:
            raise SystemExit(f"Figure 18 source member not found: {requested}")
        if PurePosixPath(normalized).suffix.lower() not in RASTER_SUFFIXES:
            raise SystemExit(
                f"Figure 18 source member is not a supported raster ({', '.join(sorted(RASTER_SUFFIXES))}): "
                f"{requested}"
            )
        return original

    verified = normalized_to_original.get(EXPECTED_REPORTED_RASTER_MEMBER)
    if verified is not None:
        return verified

    ranked = sorted(
        ((score, name) for name in names if (score := _raster_score(name)) >= 0),
        key=lambda item: (-item[0], _normalized_member_name(item[1])),
    )
    if not ranked:
        raise SystemExit(
            "No Figure 18 raster candidate was found in the supplied arXiv source. "
            "Pass --source-member relative/path/to/the/reported/raster.png."
        )
    best_score = ranked[0][0]
    best = [name for score, name in ranked if score == best_score]
    if len(best) != 1:
        display = "\n  - ".join(_normalized_member_name(name) for name in best)
        raise SystemExit(
            "Multiple equally likely Figure 18 rasters were found. Select one with "
            f"--source-member:\n  - {display}"
        )
    return best[0]


def read_reported_raster(
    source: Path,
    *,
    source_member: str | None = None,
) -> tuple[str, bytes, str]:
    root = source.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"arXiv source not found: {root}")

    if root.is_dir():
        candidates = [
            path.relative_to(root).as_posix()
            for path in sorted(root.rglob("*"))
            if path.is_file() and not path.is_symlink()
        ]
        selected = _select_candidate(candidates, source_member)
        selected_path = (root / selected).resolve()
        try:
            selected_path.relative_to(root)
        except ValueError as error:
            raise SystemExit(f"Source member escapes source directory: {selected}") from error
        data = selected_path.read_bytes()
        source_kind = "directory"
    elif root.is_file():
        try:
            with tarfile.open(root, mode="r:*") as archive:
                members = [member for member in archive.getmembers() if member.isfile()]
                selected_name = _select_candidate(
                    [member.name for member in members], source_member
                )
                matches = [member for member in members if member.name == selected_name]
                if len(matches) != 1:
                    raise SystemExit(f"Ambiguous tar member: {selected_name}")
                member = matches[0]
                if member.size > MAX_SOURCE_RASTER_BYTES:
                    raise SystemExit(
                        f"Refusing Figure 18 raster larger than {MAX_SOURCE_RASTER_BYTES} bytes: "
                        f"{member.size}"
                    )
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise SystemExit(f"Could not read tar member: {selected_name}")
                data = extracted.read(MAX_SOURCE_RASTER_BYTES + 1)
                selected = _normalized_member_name(selected_name)
        except tarfile.ReadError as error:
            raise SystemExit(
                f"--arxiv-source must be an extracted directory or readable tar archive: {root}"
            ) from error
        source_kind = "tar"
    else:
        raise SystemExit(f"Unsupported arXiv source path: {root}")

    if len(data) > MAX_SOURCE_RASTER_BYTES:
        raise SystemExit(
            f"Refusing Figure 18 raster larger than {MAX_SOURCE_RASTER_BYTES} bytes"
        )
    if not data:
        raise SystemExit(f"Figure 18 source raster is empty: {selected}")
    return selected, data, source_kind


def preserve_reported_raster(
    source: Path,
    output: Path,
    *,
    source_member: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    selected, data, source_kind = read_reported_raster(
        source, source_member=source_member
    )
    digest = hashlib.sha256(data).hexdigest()
    normalized_selected = _normalized_member_name(selected)
    expected_name_matches = normalized_selected == EXPECTED_REPORTED_RASTER_MEMBER
    expected_hash_matches = digest == EXPECTED_REPORTED_RASTER_SHA256
    suffix = PurePosixPath(selected).suffix.lower()
    destination = output / "figures" / f"figure_18_reported_source{suffix}"
    destination.write_bytes(data)
    copied_digest = sha256_file(destination)
    manifest: dict[str, Any] = {
        "classification": "reported-source-only",
        "is_retraining_result": False,
        "is_numerical_reproduction": False,
        "source_root": str(source.expanduser().resolve()),
        "source_kind": source_kind,
        "source_member": selected,
        "source_member_size_bytes": len(data),
        "source_member_sha256": digest,
        "expected_source_member": EXPECTED_REPORTED_RASTER_MEMBER,
        "source_member_matches_expected_name": expected_name_matches,
        "expected_source_member_sha256": EXPECTED_REPORTED_RASTER_SHA256,
        "source_hash_matches_expected": expected_hash_matches,
        "copied_path": str(destination.resolve()),
        "copied_sha256": copied_digest,
        "copy_hash_matches": digest == copied_digest,
        "warning": (
            "This raster is copied from the supplied arXiv source. It is the reported "
            "output only and is not evidence that GNN training was rerun."
        ),
    }
    write_json(output / "archived_source_manifest.json", manifest)
    return destination, manifest


def build_reference() -> dict[str, Any]:
    checkpoint: dict[str, Any] = {
        "path": str(INFERENCE_CHECKPOINT),
        "exists": INFERENCE_CHECKPOINT.is_file(),
        "role": "final inference state dictionary; not an epoch history",
    }
    if INFERENCE_CHECKPOINT.is_file():
        checkpoint.update(
            {
                "size_bytes": INFERENCE_CHECKPOINT.stat().st_size,
                "sha256": sha256_file(INFERENCE_CHECKPOINT),
            }
        )
    return {
        "paper_artifacts": ["Appendix A, Table 2", "Appendix A, Figure 18"],
        "table_2": {
            "canonical_rows": [asdict(row) for row in TABLE_2_ROWS],
            "row_count": len(TABLE_2_ROWS),
        },
        "figure_18": {
            "caption": "Training accuracy of GNN under hyperparameters in Table 2.",
            "x_quantity": "epoch",
            "y_quantity": "test accuracy",
            "allowed_epoch_range": [0, MAX_EPOCHS],
            "reported_source_member": EXPECTED_REPORTED_RASTER_MEMBER,
            "reported_source_sha256": EXPECTED_REPORTED_RASTER_SHA256,
            "reported_source_version": "arXiv v1",
            "numeric_reference_available": False,
            "numeric_reference_unavailable_reason": NO_NUMERIC_REFERENCE_REASON,
        },
        "training_dataset": {
            "training_graphs": TRAINING_GRAPHS,
            "test_graphs": TEST_GRAPHS,
            "total_graphs": TOTAL_GRAPHS,
        },
        "committed_checkpoint": checkpoint,
        "full_retraining_assets_available": False,
        "missing_training_assets": list(MISSING_TRAINING_ASSETS),
    }


def _git_capture(*arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ("git", *arguments),
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_appendix_provenance(
    output: Path,
    *,
    args: argparse.Namespace,
    inputs: Sequence[Path],
    notes: Sequence[str],
) -> Path:
    input_records: list[dict[str, Any]] = []
    for path in inputs:
        resolved = path.expanduser().resolve()
        record: dict[str, Any] = {
            "path": str(resolved),
            "exists": resolved.exists(),
            "kind": "directory" if resolved.is_dir() else "file",
        }
        if resolved.is_file():
            record.update(
                {
                    "size_bytes": resolved.stat().st_size,
                    "sha256": sha256_file(resolved),
                }
            )
        input_records.append(record)

    artifacts: list[dict[str, Any]] = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "provenance.json":
            artifacts.append(
                {
                    "path": str(path.relative_to(output)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    payload = {
        "section": "Appendix A / Table 2 / Figure 18",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "arguments": _json_safe(vars(args)),
        "repository": {
            "path": str(REPO_ROOT),
            "git_commit": _git_capture("rev-parse", "HEAD"),
            "git_status_short": _git_capture("status", "--short"),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "inputs": input_records,
        "artifacts": artifacts,
        "notes": list(notes),
    }
    path = output / "provenance.json"
    write_json(path, payload)
    return path


def _table_validation() -> dict[str, Any]:
    parameters = [row.hyperparameter for row in TABLE_2_ROWS]
    return {
        "row_count": len(TABLE_2_ROWS),
        "expected_row_count": 17,
        "row_count_matches": len(TABLE_2_ROWS) == 17,
        "hyperparameters_are_unique": len(parameters) == len(set(parameters)),
        "all_fields_nonempty": all(
            row.category and row.hyperparameter and row.value_setting and row.description
            for row in TABLE_2_ROWS
        ),
    }


def _full_mode_message(output: Path) -> str:
    inventory = "\n".join(f"  - {item}" for item in MISSING_TRAINING_ASSETS)
    return (
        "Appendix A --mode full cannot retrain Figure 18 because the original "
        "training assets are not present. Required assets:\n"
        f"{inventory}\n"
        "The committed final checkpoint cannot reconstruct test accuracy at earlier epochs.\n"
        "After recovering the original training project, add a deterministic training driver "
        "and its pinned environment; until then use --mode parse with the original epoch/accuracy "
        "CSV, or --mode archived-source with an explicit arXiv source tar/directory. Those modes "
        "do not claim retraining.\n"
        f"A machine-readable failed validation was written below {output}."
    )


def run(args: argparse.Namespace) -> int:
    validate_arguments(args)
    output = prepare_appendix_output(args.output, args.mode)
    export_table_2(output)
    write_json(output / "reference.json", build_reference())
    inputs: list[Path] = []
    notes = [
        "Table 2 is exported verbatim from canonical constants transcribed from the paper.",
        NO_NUMERIC_REFERENCE_REASON,
    ]
    validation: dict[str, Any] = {
        "mode": args.mode,
        "table_2": _table_validation(),
    }

    if args.mode == "table":
        validation.update(
            {
                "status": "passed",
                "figure_18": {
                    "generated": False,
                    "reason": "table mode exports Table 2 only",
                },
            }
        )
    elif args.mode == "parse":
        accuracy_path = args.accuracy_csv.expanduser().resolve()
        rows = read_accuracy_csv(
            accuracy_path,
            epoch_column=args.epoch_column,
            accuracy_column=args.accuracy_column,
            accuracy_scale=args.accuracy_scale,
        )
        inputs.append(accuracy_path)
        write_csv(output / "figure_18_accuracy.csv", ACCURACY_FIELDS, rows)
        figure_paths: list[str] = []
        if not args.no_plots:
            figure_paths = [
                str(path.relative_to(output))
                for path in plot_figure_18(rows, output)
            ]
        validation.update(
            {
                "status": "passed",
                "figure_18": {
                    "generated": not args.no_plots,
                    "artifacts": figure_paths,
                    "source_kind": "user-supplied epoch/accuracy history",
                    "is_retraining_result": False,
                    "row_count": len(rows),
                    "epochs_strictly_increasing": True,
                    "epoch_min": int(rows[0]["epoch"]),
                    "epoch_max": int(rows[-1]["epoch"]),
                    "epochs_within_paper_range": True,
                    "accuracies_finite_and_in_range": True,
                    "accuracy_fraction_min": min(
                        float(row["accuracy_fraction"]) for row in rows
                    ),
                    "accuracy_fraction_max": max(
                        float(row["accuracy_fraction"]) for row in rows
                    ),
                    "numeric_reference_comparison": "unavailable",
                    "numeric_reference_unavailable_reason": NO_NUMERIC_REFERENCE_REASON,
                },
            }
        )
        notes.append(
            "Parse mode validates and plots supplied history; it does not rerun model training."
        )
    elif args.mode == "archived-source":
        source_path = args.arxiv_source.expanduser().resolve()
        inputs.append(source_path)
        destination, manifest = preserve_reported_raster(
            source_path,
            output,
            source_member=args.source_member,
        )
        source_hash_matches = bool(manifest["source_hash_matches_expected"])
        source_hash_accepted = source_hash_matches or args.allow_source_hash_mismatch
        validation.update(
            {
                "status": (
                    "passed_source_only"
                    if source_hash_accepted
                    else "failed_source_hash_mismatch"
                ),
                "figure_18": {
                    "generated": False,
                    "copied_reported_raster": str(destination.relative_to(output)),
                    "classification": "reported-source-only",
                    "is_retraining_result": False,
                    "is_numerical_reproduction": False,
                    "copy_hash_matches": manifest["copy_hash_matches"],
                    "source_member_sha256": manifest["source_member_sha256"],
                    "expected_source_member_sha256": EXPECTED_REPORTED_RASTER_SHA256,
                    "source_hash_matches_expected": source_hash_matches,
                    "source_hash_override_requested": args.allow_source_hash_mismatch,
                    "source_hash_override_used": (
                        args.allow_source_hash_mismatch and not source_hash_matches
                    ),
                },
            }
        )
        notes.append(
            "Archived-source mode copied the reported raster byte-for-byte; it is not retraining."
        )
        if not source_hash_accepted:
            write_json(output / "validation.json", validation)
            notes.append(
                "Archived-source mode refused an unverified raster; no mismatch override was supplied."
            )
            write_appendix_provenance(output, args=args, inputs=inputs, notes=notes)
            raise SystemExit(
                f"Selected Figure 18 raster SHA256 {manifest['source_member_sha256']} does not "
                f"match verified arXiv v1 SHA256 {EXPECTED_REPORTED_RASTER_SHA256}. "
                "Pass --allow-source-hash-mismatch only to audit a deliberately different source."
            )
    else:
        validation.update(
            {
                "status": "failed_missing_training_assets",
                "figure_18": {
                    "generated": False,
                    "is_retraining_result": False,
                    "missing_training_assets": list(MISSING_TRAINING_ASSETS),
                },
            }
        )
        write_json(output / "validation.json", validation)
        notes.append("Full mode stopped before generating Figure 18; no curve was fabricated.")
        write_appendix_provenance(output, args=args, inputs=inputs, notes=notes)
        raise SystemExit(_full_mode_message(output))

    write_json(output / "validation.json", validation)
    write_appendix_provenance(output, args=args, inputs=inputs, notes=notes)
    print(f"Wrote Appendix A artifacts to {output}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
