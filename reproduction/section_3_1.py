#!/usr/bin/env python3
"""Reproduce Section 3.1 / Figures 7--10 for ``enlight_hard``."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys

if __package__:
    from . import common as _common
    sys.modules.setdefault("common", _common)
from pathlib import Path

from common import (
    DEFAULT_ENLIGHT_HARD_MPS,
    REPRODUCTION_ROOT,
    REPO_ROOT,
    Job,
    add_common_arguments,
    failed_jobs,
    get_pyplot,
    julia_command,
    materialize_archive_section,
    prepare_output,
    run_jobs,
    validate_common_arguments,
    write_csv,
    write_json,
    write_provenance,
)


METHODS = ("Basic", "BFS", "MILP")
METHOD_STYLES = {"Basic": "-", "BFS": ":", "MILP": "--"}
METHOD_COLORS = {"Basic": "#2980b9", "BFS": "#f39c12", "MILP": "#27ae60"}
ITERATION_RE = re.compile(
    r"^\s*(\d+)\s+([0-9eE+\-.]+)\s+([0-9eE+\-.]+)\s+"
    r"([0-9eE+\-.]+)\s+([0-9eE+\-.]+)\s+([0-9eE+\-.]+)\s+"
    r"([0-9eE+\-.]+)\s+([0-9eE+\-.]+)"
)
RESIDUAL_FIELDS = (
    "method",
    "iteration",
    "pres_l2",
    "dres_l2",
    "status",
    "stop_iter",
    "admm_time_seconds",
    "source_log",
)


def parse_archived_logs(root: Path) -> list[dict[str, object]]:
    files = {method: root / f"{method.lower()}.txt" for method in METHODS}
    rows: list[dict[str, object]] = []
    for method, path in files.items():
        if not path.is_file():
            raise SystemExit(f"Missing archived Section 3.1 log: {path}")
        status = ""
        stop_iter: int | None = None
        total_time: float | None = None
        parsed: list[dict[str, object]] = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            match = ITERATION_RE.match(line)
            if match:
                parsed.append(
                    {
                        "method": method,
                        "iteration": int(match.group(1)),
                        "pres_l2": float(match.group(4)),
                        "dres_l2": float(match.group(6)),
                        "source_log": str(path),
                    }
                )
            match = re.search(r"Solver Status\s*=\s*(ADMM_TERMINATION_[A-Z_]+)", line)
            if match:
                status = match.group(1)
            match = re.search(r"Stop\.\s*Iter\s*=\s*(\d+)", line)
            if match:
                stop_iter = int(match.group(1))
            match = re.search(r"Total Time\s*=\s*([0-9.eE+\-]+)", line)
            if match:
                total_time = float(match.group(1))
        if not parsed:
            raise SystemExit(f"No iteration rows parsed from {path}")
        for row in parsed:
            row["status"] = status
            row["stop_iter"] = stop_iter if stop_iter is not None else parsed[-1]["iteration"]
            row["admm_time_seconds"] = total_time if total_time is not None else ""
        rows.extend(parsed)
    return rows


def parse_structured(root: Path) -> list[dict[str, object]]:
    candidates = sorted(root.rglob("residuals.csv"))
    if not candidates:
        raise SystemExit(f"No residuals.csv found below {root}")
    rows: list[dict[str, object]] = []
    for path in candidates:
        with path.open(newline="", encoding="utf-8") as stream:
            for raw in csv.DictReader(stream):
                rows.append(
                    {
                        "method": raw["method"],
                        "iteration": int(raw["iteration"]),
                        "pres_l2": float(raw["pres_l2"]),
                        "dres_l2": float(raw["dres_l2"]),
                        "status": raw.get("status", ""),
                        "stop_iter": int(raw["stop_iter"]),
                        "admm_time_seconds": float(raw["admm_time_seconds"]),
                        "source_log": str(path),
                    }
                )
    return rows


def validate_rows(rows: list[dict[str, object]]) -> None:
    present = {str(row["method"]) for row in rows}
    missing = set(METHODS) - present
    if missing:
        raise SystemExit(f"Missing residual trajectories for: {sorted(missing)}")
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        iterations = [int(row["iteration"]) for row in selected]
        if iterations != sorted(set(iterations)):
            raise SystemExit(f"Iteration sequence for {method} is duplicated or unsorted")


def plot_residuals(rows: list[dict[str, object]], output: Path) -> None:
    pyplot = get_pyplot()
    figure, axes = pyplot.subplots(1, 2, figsize=(12.8, 4.5))
    panels = (("pres_l2", "Primal residual", 500), ("dres_l2", "Dual residual", 1))
    for axis, (field, ylabel, stride) in zip(axes, panels):
        for method in METHODS:
            selected = [
                row
                for row in rows
                if row["method"] == method
                and (int(row["iteration"]) == 0 or int(row["iteration"]) % stride == 0)
                and float(row[field]) > 0
                and math.isfinite(float(row[field]))
            ]
            axis.semilogy(
                [int(row["iteration"]) for row in selected],
                [float(row[field]) for row in selected],
                linestyle=METHOD_STYLES[method],
                color=METHOD_COLORS[method],
                linewidth=2.0,
                label=method,
            )
        axis.set_xlabel("Iteration")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", linestyle="--", alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / "figures" / "figure_10.png", dpi=200)
    figure.savefig(output / "figures" / "figure_10.pdf")
    pyplot.close(figure)

    for field, name, stride in (("pres_l2", "primal_residuals", 500), ("dres_l2", "dual_residuals", 1)):
        figure, axis = pyplot.subplots(figsize=(8, 4.5))
        for method in METHODS:
            selected = [
                row
                for row in rows
                if row["method"] == method
                and (int(row["iteration"]) == 0 or int(row["iteration"]) % stride == 0)
                and float(row[field]) > 0
                and math.isfinite(float(row[field]))
            ]
            axis.semilogy(
                [int(row["iteration"]) for row in selected],
                [float(row[field]) for row in selected],
                METHOD_STYLES[method],
                color=METHOD_COLORS[method],
                linewidth=2.0,
                label=method,
            )
        axis.set_xlabel("Iteration")
        axis.set_ylabel("Pres(2)" if field == "pres_l2" else "Dres(2)")
        axis.grid(True, which="both", linestyle="--", alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / "figures" / f"{name}.png", dpi=200)
        pyplot.close(figure)


def _short_label(node_id: str) -> str:
    variable = re.match(r"VariableNode\((.+)\)", node_id)
    if variable:
        return variable.group(1)
    constraint = re.match(r"ConstraintNode\((.+)\)", node_id)
    if constraint:
        return f"C{constraint.group(1)}"
    if node_id.startswith("ADMMNodeConvertedFromEdge"):
        match = re.search(r"(?:TwoBlockEdge|MultiblockEdge)\(([^,)]+)", node_id)
        return f"aux({match.group(1)})" if match else "aux"
    return node_id


def _draw_graph(axis, graph: dict[str, object], title: str) -> None:
    nodes = {item["id"]: item for item in graph["nodes"]}
    left = list(graph.get("left", []))
    right = list(graph.get("right", []))
    positions: dict[str, tuple[float, float]] = {}
    if left or right:
        for x_value, partition in ((0.0, left), (1.0, right)):
            for index, node_id in enumerate(partition):
                y_value = 0.5 if len(partition) == 1 else 1.0 - index / (len(partition) - 1)
                positions[node_id] = (x_value, y_value)
    else:
        node_ids = sorted(nodes)
        for index, node_id in enumerate(node_ids):
            angle = 2.0 * math.pi * index / max(1, len(node_ids))
            positions[node_id] = (math.cos(angle), math.sin(angle))
    for edge in graph["edges"]:
        u, v = edge["u"], edge["v"]
        if u not in positions or v not in positions:
            continue
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        axis.plot((x1, x2), (y1, y2), color="#666666", linewidth=1.1, zorder=1)
    colors = {"variable": "#8ecae6", "constraint": "#90ee90", "auxiliary": "#f4a261"}
    for node_id, item in nodes.items():
        if node_id not in positions:
            continue
        x_value, y_value = positions[node_id]
        axis.scatter(
            (x_value,),
            (y_value,),
            s=1250,
            color=colors.get(item.get("kind", ""), "#d3d3d3"),
            edgecolors="#333333",
            linewidths=0.9,
            zorder=2,
        )
        axis.text(x_value, y_value, _short_label(node_id), ha="center", va="center", fontsize=8, zorder=3)
    axis.set_title(title)
    axis.axis("off")


def plot_structural_figures(raw_dir: Path, output: Path) -> list[str]:
    pyplot = get_pyplot()
    warnings: list[str] = []
    original_matrix = raw_dir / "matrix_original.png"
    permuted_matrix = raw_dir / "matrix_coclustered.png"
    block_matrix = raw_dir / "matrix_coclustered_stacked.png"
    graph_path = raw_dir / "graphs.json"
    required = (original_matrix, permuted_matrix, graph_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        warnings.append("Structural figures unavailable; missing: " + ", ".join(missing))
        return warnings
    graphs = json.loads(graph_path.read_text(encoding="utf-8"))

    figure, axes = pyplot.subplots(1, 2, figsize=(12, 5.5))
    for axis, path, title in zip(axes, (original_matrix, permuted_matrix), ("(a) Original A", "(b) Permuted A")):
        axis.imshow(pyplot.imread(path))
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output / "figures" / "figure_7.png", dpi=200)
    figure.savefig(output / "figures" / "figure_7.pdf")
    pyplot.close(figure)

    figure, axes = pyplot.subplots(1, 2, figsize=(12, 5.2))
    if block_matrix.is_file():
        axes[0].imshow(pyplot.imread(block_matrix))
        axes[0].set_title("(a) Block view of permuted A")
        axes[0].axis("off")
    else:
        axes[0].text(0.5, 0.5, "Block-view image unavailable", ha="center", va="center")
        axes[0].axis("off")
    _draw_graph(axes[1], graphs["original"], "(b) Graph representation")
    figure.tight_layout()
    figure.savefig(output / "figures" / "figure_8.png", dpi=200)
    figure.savefig(output / "figures" / "figure_8.pdf")
    pyplot.close(figure)

    figure, axes = pyplot.subplots(1, 2, figsize=(12, 5.2))
    _draw_graph(axes[0], graphs["milp"], "(a) By MILP")
    _draw_graph(axes[1], graphs["bfs"], "(b) By BFS")
    figure.tight_layout()
    figure.savefig(output / "figures" / "figure_9.png", dpi=200)
    figure.savefig(output / "figures" / "figure_9.pdf")
    pyplot.close(figure)
    return warnings


def _legacy_main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, default_output=REPRODUCTION_ROOT / "output" / "section_3_1")
    parser.add_argument(
        "--mps",
        type=Path,
        default=DEFAULT_ENLIGHT_HARD_MPS,
        help="Path to enlight_hard.mps or enlight_hard.mps.gz; overrides the bundled input.",
    )
    parser.add_argument("--max-iter", type=int, default=1_000_000, help="Fresh full-run iteration limit.")
    args = parser.parse_args()
    validate_common_arguments(args)
    if args.max_iter < 1:
        raise SystemExit("--max-iter must be positive")
    output = prepare_output(args.output)
    jobs: list[Job] = []
    notes: list[str] = []

    if args.mode in ("smoke", "full"):
        if not args.mps.expanduser().is_file():
            raise SystemExit(
                f"MPS input not found: {args.mps}; override the bundled input with --mps PATH."
            )
        mps = args.mps.expanduser().resolve()
        raw_dir = output / "raw" / "enlight_hard"
        driver = REPRODUCTION_ROOT / "julia" / "section_3_1_exact.jl"
        max_iter = min(args.max_iter, 200) if args.mode == "smoke" else args.max_iter
        jobs = [
            Job(
                "enlight_hard/run",
                julia_command(args.julia, args.threads, driver, mps, raw_dir, max_iter),
            )
        ]
        results = run_jobs(
            jobs,
            log_root=output / "raw",
            cwd=REPO_ROOT,
            workers=1,
            resume=not args.no_resume,
        )
        if failed_jobs(results):
            raise SystemExit(f"Section 3.1 Julia driver failed; inspect {output / 'raw'}")
        rows = parse_structured(raw_dir)
        structural_root: Path | None = raw_dir
        inputs = [mps]
    elif args.mode == "archived":
        with materialize_archive_section(args.archive, "3-1-enlight_hard") as archive_root:
            rows = parse_archived_logs(archive_root)
        structural_root = None
        inputs = [args.archive]
        notes.append("The supplied archive contains Figure 10 logs but no MPS file or Figure 7--9 source data.")
    else:
        parse_root = args.logs.expanduser().resolve()
        if list(parse_root.rglob("residuals.csv")):
            rows = parse_structured(parse_root)
            graph_candidates = list(parse_root.rglob("graphs.json"))
            structural_root = graph_candidates[0].parent if graph_candidates else None
        else:
            rows = parse_archived_logs(parse_root)
            structural_root = None
        inputs = []

    validate_rows(rows)
    write_csv(output / "residuals.csv", RESIDUAL_FIELDS, rows)
    if not args.no_plots:
        plot_residuals(rows, output)
        if structural_root is not None:
            notes.extend(plot_structural_figures(structural_root, output))
    manifest = {
        "paper_configuration": {
            "instance": "enlight_hard (continuous LP relaxation)",
            "column_clusters": 4,
            "alternating_passes": 5,
            "solver": "doubly linearized ADMM / FLiP-ADMM",
            "rho": 1000.0,
            "primal_plot_stride": 500,
            "dual_plot_stride": 1,
        },
        "available_figures": sorted(path.name for path in (output / "figures").glob("figure_*.png")),
        "notes": notes,
    }
    write_json(output / "manifest.json", manifest)
    write_provenance(
        output,
        section="Section 3.1 / Figures 7--10",
        args=args,
        jobs=jobs,
        inputs=inputs,
        notes=(
            "The MIPLIB instance is read through GenericLP and reproduced as its continuous LP relaxation.",
            "The reviewer driver uses the paper's k=4 and five alternating passes; it does not use the later ten-pass demo defaults.",
            *notes,
        ),
    )
    print(f"Wrote Section 3.1 artifacts to {output}")
    return 0


def main(argv=None) -> int:
    """Dispatch imports and script execution to the strict reviewer CLI."""

    try:
        from .section_3_1_impl import main as strict_main
    except ImportError:
        from section_3_1_impl import main as strict_main
    return strict_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

