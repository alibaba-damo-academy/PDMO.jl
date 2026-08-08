"""Strict execution and validation layer for Section 3.1."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

try:
    from .common import (
        DEFAULT_ENLIGHT_HARD_MPS,
        REPO_ROOT,
        REPRODUCTION_ROOT,
        Job,
        add_common_arguments,
        failed_jobs,
        job_results_as_dicts,
        julia_command,
        materialize_archive_section,
        prepare_mode_output,
        run_jobs,
        validate_common_arguments,
        write_csv,
        write_json,
        write_provenance,
    )
    from .section_3_1 import (
        METHODS,
        RESIDUAL_FIELDS,
        parse_archived_logs,
        plot_residuals,
        plot_structural_figures,
    )
    from .section_3_1_reported import run_reported_mode, validate_official_mps
except ImportError:
    from common import (  # type: ignore
        DEFAULT_ENLIGHT_HARD_MPS,
        REPO_ROOT,
        REPRODUCTION_ROOT,
        Job,
        add_common_arguments,
        failed_jobs,
        job_results_as_dicts,
        julia_command,
        materialize_archive_section,
        prepare_mode_output,
        run_jobs,
        validate_common_arguments,
        write_csv,
        write_json,
        write_provenance,
    )
    from section_3_1 import (  # type: ignore
        METHODS,
        RESIDUAL_FIELDS,
        parse_archived_logs,
        plot_residuals,
        plot_structural_figures,
    )
    from section_3_1_reported import run_reported_mode, validate_official_mps  # type: ignore


ARCHIVE_SECTION = "3-1-enlight_hard"
JULIA_DRIVER = REPRODUCTION_ROOT / "julia" / "section_3_1_exact.jl"
DEFAULT_OUTPUT = REPRODUCTION_ROOT / "output" / "section_3_1"
PAPER_MAX_ITER = 100_000
SMOKE_MAX_ITER = 200
PAPER_GLOBAL_SEED = 126
PAPER_MIP_REL_GAP = 0.01
PAPER_MIP_TIME_LIMIT = 60.0
PAPER_MIP_HEURISTIC_EFFORT = 0.2
PAPER_TERMINALS = {
    "Basic": ("ADMM_TERMINATION_OPTIMAL", 72_462),
    "BFS": ("ADMM_TERMINATION_OPTIMAL", 53_700),
    "MILP": ("ADMM_TERMINATION_OPTIMAL", 16_049),
}
ALLOWED_STATUSES = {
    "ADMM_TERMINATION_OPTIMAL",
    "ADMM_TERMINATION_ITERATION_LIMIT",
    "ADMM_TERMINATION_TIME_LIMIT",
}
STRUCTURAL_ARTIFACTS = (
    "residuals.csv",
    "terminal.json",
    "exact_configuration.json",
    "graphs.json",
    "matrix_original.png",
    "matrix_coclustered.png",
    "matrix_coclustered_stacked.png",
)
STRUCTURED_FIELDS = {
    "method",
    "iteration",
    "pres_l2",
    "dres_l2",
    "status",
    "stop_iter",
    "admm_time_seconds",
}


def _stable_archive_source(archive: Path, method: str) -> str:
    member = Path("experiments_logs") / ARCHIVE_SECTION / f"{method.lower()}.txt"
    return f"{archive.expanduser().resolve()}::{member.as_posix()}"


def read_structured_file(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = STRUCTURED_FIELDS.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing CSV fields {sorted(missing)}")
        for raw in reader:
            try:
                iteration = int(raw["iteration"])
                stop_iter = int(raw["stop_iter"])
                pres = float(raw["pres_l2"])
                dres = float(raw["dres_l2"])
                elapsed = float(raw["admm_time_seconds"])
            except ValueError as error:
                raise ValueError(f"{path}: invalid numeric trajectory field") from error
            rows.append(
                {
                    "method": raw["method"],
                    "iteration": iteration,
                    "pres_l2": pres,
                    "dres_l2": dres,
                    "status": raw["status"],
                    "stop_iter": stop_iter,
                    "admm_time_seconds": elapsed,
                    "source_log": str(path.resolve()),
                }
            )
    if not rows:
        raise ValueError(f"{path}: no residual trajectories")
    return rows


def discover_structured_input(root: Path) -> Path | None:
    path = root.expanduser().resolve()
    if path.is_file():
        return path if path.name == "residuals.csv" else None
    if not path.is_dir():
        return None
    merged = path / "residuals.csv"
    if merged.is_file():
        return merged
    candidates = sorted(path.rglob("residuals.csv"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise SystemExit(
            f"Multiple residuals.csv inputs found below {path}; pass the desired CSV directly."
        )
    return None


def validate_trajectories(
    rows: Sequence[Mapping[str, object]],
    *,
    mode: str,
    max_iter: int | None,
) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("method") or "")].append(row)

    missing = sorted(set(METHODS).difference(groups))
    extra = sorted(set(groups).difference(METHODS))
    if missing:
        errors.append(f"Missing trajectories: {missing}")
    if extra:
        errors.append(f"Unexpected trajectories: {extra}")

    method_summary: dict[str, dict[str, object]] = {}
    for method in METHODS:
        selected = sorted(groups.get(method, ()), key=lambda row: int(row["iteration"]))
        if not selected:
            continue
        iterations = [int(row["iteration"]) for row in selected]
        stop_values = {int(row["stop_iter"]) for row in selected}
        status_values = {str(row["status"]) for row in selected}
        time_values = {float(row["admm_time_seconds"]) for row in selected}
        if len(stop_values) != 1:
            errors.append(f"{method}: inconsistent stop_iter values {sorted(stop_values)}")
            continue
        stop_iter = next(iter(stop_values))
        if (
            len(iterations) != stop_iter
            or iterations[0] != 1
            or iterations[-1] != stop_iter
            or any(right != left + 1 for left, right in zip(iterations, iterations[1:]))
        ):
            errors.append(
                f"{method}: expected exactly completed iterations 1:{stop_iter}, "
                f"got {len(iterations)} rows spanning {iterations[0]}:{iterations[-1]}"
            )
        if len(status_values) != 1:
            errors.append(f"{method}: inconsistent statuses {sorted(status_values)}")
            status = ""
        else:
            status = next(iter(status_values))
            if status not in ALLOWED_STATUSES:
                errors.append(f"{method}: unsupported terminal status {status!r}")
        if len(time_values) != 1 or any(not math.isfinite(value) or value < 0 for value in time_values):
            errors.append(f"{method}: invalid ADMM elapsed-time metadata")

        for row in selected:
            iteration = int(row["iteration"])
            for field in ("pres_l2", "dres_l2"):
                value = float(row[field])
                if not math.isfinite(value) or value < 0:
                    errors.append(f"{method} iteration {iteration}: invalid {field}={value}")

        if max_iter is not None and stop_iter > max_iter:
            errors.append(f"{method}: stop_iter {stop_iter} exceeds configured max_iter {max_iter}")
        if status == "ADMM_TERMINATION_ITERATION_LIMIT" and max_iter is not None and stop_iter != max_iter:
            errors.append(
                f"{method}: iteration-limit status at {stop_iter}, configured max_iter is {max_iter}"
            )
        if status == "ADMM_TERMINATION_TIME_LIMIT":
            warnings.append(f"{method}: time-limit trajectory is retained in the figure")

        method_summary[method] = {
            "status": status,
            "stop_iter": stop_iter,
            "admm_time_seconds": next(iter(time_values)) if len(time_values) == 1 else None,
            "row_count": len(selected),
        }

    if mode == "archived":
        for method, (expected_status, expected_stop) in PAPER_TERMINALS.items():
            summary = method_summary.get(method)
            if summary is None:
                continue
            if summary["status"] != expected_status or summary["stop_iter"] != expected_stop:
                errors.append(
                    f"{method}: archived terminal is "
                    f"({summary['status']}, {summary['stop_iter']}), expected "
                    f"({expected_status}, {expected_stop})"
                )

    return {
        "status": "passed" if not errors else "failed",
        "mode": mode,
        "row_count": len(rows),
        "method_summary": method_summary,
        "errors": errors,
        "warnings": warnings,
    }


def validate_exact_configuration(raw_dir: Path) -> dict[str, object]:
    errors: list[str] = []
    try:
        configuration = json.loads((raw_dir / "exact_configuration.json").read_text())
        graphs = json.loads((raw_dir / "graphs.json").read_text())
        terminal = json.loads((raw_dir / "terminal.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        return {"status": "failed", "errors": [f"Could not read exact metadata: {error}"]}

    algorithm = configuration.get("algorithm", {})
    expected = {
        "column_initialization": "cyclic_unshuffled",
        "row_initialization": "all_in_cluster_1",
        "passes": 5,
        "tie_break": "smallest_cluster_index",
        "row_grouping": "final_row_cluster",
        "bfs_traversal_order": "lexicographic_node_then_edge",
        "empty_cluster_repair": False,
        "early_stopping": False,
    }
    for field, value in expected.items():
        if algorithm.get(field) != value:
            errors.append(f"algorithm.{field}={algorithm.get(field)!r}, expected {value!r}")
    for field, value in (
        ("k", 4),
        ("rho", 1000.0),
        ("global_seed", PAPER_GLOBAL_SEED),
    ):
        if configuration.get(field) != value:
            errors.append(f"{field}={configuration.get(field)!r}, expected {value!r}")

    for field, value in (
        ("solver", "DOUBLY_LINEARIZED_SOLVER"),
        ("log_interval", 1),
        ("apply_scaling", False),
    ):
        if configuration.get(field) != value:
            errors.append(f"{field}={configuration.get(field)!r}, expected {value!r}")
    if not isinstance(configuration.get("max_iter"), int) or configuration["max_iter"] < 1:
        errors.append("max_iter must be a positive integer")
    if not isinstance(configuration.get("threads"), int) or configuration["threads"] < 1:
        errors.append("threads must be a positive integer")

    row_cluster = configuration.get("row_cluster", [])
    column_cluster = configuration.get("column_cluster", [])
    if set(row_cluster) != {1, 2, 3, 4}:
        errors.append("row_cluster must use exactly labels 1,2,3,4")
    if set(column_cluster) != {1, 2, 3, 4}:
        errors.append("column_cluster must use exactly labels 1,2,3,4")

    row_groups = configuration.get("group_rows", [])
    column_groups = configuration.get("column_groups", [])
    group_blocks = configuration.get("group_blocks", [])
    if len(row_groups) != 4 or any(not group for group in row_groups):
        errors.append("Expected four nonempty final row clusters/block constraints")
    if len(column_groups) != 4 or any(not group for group in column_groups):
        errors.append("Expected four nonempty final column clusters/block variables")
    if len(group_blocks) != 4 or any(not group for group in group_blocks):
        errors.append("Expected four nonempty row-group support sets")
    if len(row_groups) == 4:
        flattened_rows = [index for group in row_groups for index in group]
        if len(flattened_rows) != len(set(flattened_rows)) or set(flattened_rows) != set(range(1, len(row_cluster) + 1)):
            errors.append("Final row groups must be a disjoint, complete row partition")
    if len(column_groups) == 4:
        flattened_columns = [index for group in column_groups for index in group]
        if len(flattened_columns) != len(set(flattened_columns)) or set(flattened_columns) != set(range(1, len(column_cluster) + 1)):
            errors.append("Final column groups must be a disjoint, complete column partition")
    graph_configuration = graphs.get("configuration", {})
    expected_graph_configuration = {
        "k": 4,
        "passes": 5,
        "global_seed": PAPER_GLOBAL_SEED,
        "mip_rel_gap": PAPER_MIP_REL_GAP,
        "mip_time_limit_seconds": PAPER_MIP_TIME_LIMIT,
        "mip_heuristic_effort": PAPER_MIP_HEURISTIC_EFFORT,
    }
    for field, value in expected_graph_configuration.items():
        if graph_configuration.get(field) != value:
            errors.append(
                f"graphs.configuration.{field}={graph_configuration.get(field)!r}, "
                f"expected {value!r}"
            )
    for graph_name in ("original", "bfs", "milp"):
        payload = graphs.get(graph_name, {})
        if not payload.get("nodes") or not payload.get("edges"):
            errors.append(f"graphs.json has an empty {graph_name} graph")
    for method in METHODS:
        details = terminal.get(method)
        if not isinstance(details, dict):
            errors.append(f"terminal.json is missing {method}")
            continue
        status = details.get("status")
        stop_iter = details.get("stop_iter")
        if status not in ALLOWED_STATUSES:
            errors.append(f"terminal.json has unsupported {method} status {status!r}")
        if not isinstance(stop_iter, int) or stop_iter < 1:
            errors.append(f"terminal.json has invalid {method} stop_iter {stop_iter!r}")
        if details.get("exported_iterations") != [1, stop_iter]:
            errors.append(f"terminal.json has inconsistent {method} export range")

    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "configuration": configuration,
        "terminal": terminal,
    }


def reference_comparison(
    trajectory_validation: Mapping[str, object], mode: str
) -> dict[str, object]:
    summaries = trajectory_validation.get("method_summary", {})
    checks = []
    if mode == "archived" and isinstance(summaries, dict):
        for method, (expected_status, expected_stop) in PAPER_TERMINALS.items():
            observed = summaries.get(method, {})
            passed = (
                observed.get("status") == expected_status
                and observed.get("stop_iter") == expected_stop
            )
            checks.append(
                {
                    "method": method,
                    "expected_status": expected_status,
                    "observed_status": observed.get("status"),
                    "expected_stop_iter": expected_stop,
                    "observed_stop_iter": observed.get("stop_iter"),
                    "passed": passed,
                }
            )
    checks_available = mode == "archived"
    return {
        "reference": "Retained later Section 3.1 terminal summaries (not submitted Figure 10)",
        "mode": mode,
        "checks": checks,
        "reference_checks_available": checks_available,
        "all_checks_passed": (
            all(check["passed"] for check in checks) if checks_available else None
        ),
        "not_applicable_reason": (
            None
            if checks_available
            else "Fresh smoke/full executes the literal manuscript 4-by-4 co-clustering, while the retained archive used a distinct 4-variable/5-constraint pipeline; use --mode reported for the submitted Figure 10."
        ),
        "fidelity_note": (
            "The retained archive is a later experiment with four block variables and five "
            "constraints; its terminals and trajectories differ from the submitted Figure 10. "
            "Archived mode reconstructs only that retained comparison, smoke/full executes "
            "the literal five-pass manuscript algorithm, and reported mode reruns the pinned "
            "ten-pass workflow that generated the submitted panels."
        ),
    }


def _fresh_artifacts_valid(raw_dir: Path) -> bool:
    if not all((raw_dir / name).is_file() for name in STRUCTURAL_ARTIFACTS):
        return False
    try:
        rows = read_structured_file(raw_dir / "residuals.csv")
    except (OSError, ValueError):
        return False
    trajectories = validate_trajectories(rows, mode="parse", max_iter=None)
    configuration = validate_exact_configuration(raw_dir)
    return trajectories["status"] == "passed" and configuration["status"] == "passed"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce Section 3.1 / Figures 7--10 for enlight_hard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_arguments(parser, default_output=DEFAULT_OUTPUT, extra_modes=("reported",))
    parser.add_argument(
        "--mps",
        type=Path,
        default=DEFAULT_ENLIGHT_HARD_MPS,
        help="Path to enlight_hard.mps or enlight_hard.mps.gz; overrides the bundled input.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=PAPER_MAX_ITER,
        help=(
            "Paper-pinned ADMM iteration limit for full/reported mode; smoke is "
            "always capped at 200"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_common_arguments(args)
    if args.max_iter < 1:
        raise SystemExit("--max-iter must be positive")
    if args.mode == "full" and args.max_iter != PAPER_MAX_ITER:
        raise SystemExit(
            f"--mode full requires the original experiment max iteration limit {PAPER_MAX_ITER}; "
            "use --mode smoke for a shortened validation run"
        )
    output = prepare_mode_output(args.output, args.mode)
    if args.mode == "reported":
        return run_reported_mode(args, output)

    jobs: list[Job] = []
    job_results = []
    notes: list[str] = []
    raw_dir: Path | None = None
    configuration_validation: dict[str, object] | None = None
    mps_validation: dict[str, object] | None = None

    if args.mode in {"smoke", "full"}:
        if not args.mps.expanduser().is_file():
            raise SystemExit(
                f"MPS input not found: {args.mps}. The default is the bundled "
                f"{DEFAULT_ENLIGHT_HARD_MPS}; override it with --mps PATH."
            )
        mps = args.mps.expanduser().resolve()
        mps_validation = validate_official_mps(mps)
        if mps_validation["status"] != "passed":
            formatted = "\n".join(
                f"  - {error}" for error in mps_validation["errors"]
            )
            raise SystemExit(f"MPS input validation failed:\n{formatted}")

        raw_dir = output / "raw" / "enlight_hard"
        max_iter = min(args.max_iter, SMOKE_MAX_ITER) if args.mode == "smoke" else args.max_iter
        jobs = [
            Job(
                "enlight_hard/run",
                julia_command(args.julia, args.threads, JULIA_DRIVER, mps, raw_dir, max_iter),
            )
        ]
        job_results = run_jobs(
            jobs,
            log_root=output / "raw",
            cwd=REPO_ROOT,
            workers=1,
            resume=not args.no_resume and _fresh_artifacts_valid(raw_dir),
        )
        write_json(output / "job_results.json", job_results_as_dicts(job_results))
        if failed_jobs(job_results):
            raise SystemExit(f"Section 3.1 Julia driver failed; inspect {output / 'raw'}")
        if not _fresh_artifacts_valid(raw_dir):
            raise SystemExit("Section 3.1 Julia driver omitted one or more required artifacts")
        try:
            rows = read_structured_file(raw_dir / "residuals.csv")
        except (OSError, ValueError) as error:
            raise SystemExit(str(error)) from error
        configuration_validation = validate_exact_configuration(raw_dir)
        structural_root = raw_dir
        inputs = [mps, JULIA_DRIVER]
        validation_max_iter = max_iter
    elif args.mode == "archived":
        with materialize_archive_section(args.archive, ARCHIVE_SECTION) as archive_root:
            rows = parse_archived_logs(archive_root)
        for row in rows:
            row["source_log"] = _stable_archive_source(args.archive, str(row["method"]))
        structural_root = None
        inputs = [args.archive]
        validation_max_iter = PAPER_MAX_ITER
        notes.append(
            "The supplied archive contains later Section 3.1 logs, not the submitted Figure 10 source run."
        )
    else:
        parse_root = args.logs.expanduser().resolve()
        structured = discover_structured_input(parse_root)
        if structured is not None:
            try:
                rows = read_structured_file(structured)
            except (OSError, ValueError) as error:
                raise SystemExit(str(error)) from error
            candidate_root = structured.parent
            if (candidate_root / "graphs.json").is_file():
                structural_root = candidate_root
                configuration_validation = validate_exact_configuration(candidate_root)
            else:
                structural_root = None
        else:
            rows = parse_archived_logs(parse_root)
            structural_root = None
        inputs = [parse_root]
        validation_max_iter = None

    trajectory_validation = validate_trajectories(
        rows,
        mode=args.mode,
        max_iter=validation_max_iter,
    )
    combined_errors = list(trajectory_validation["errors"])
    if configuration_validation is not None:
        combined_errors.extend(configuration_validation["errors"])
    validation = {
        **trajectory_validation,
        "configuration_validation": configuration_validation,
        "mps_validation": mps_validation,
        "errors": combined_errors,
        "status": "passed" if not combined_errors else "failed",
    }

    write_csv(output / "residuals.csv", RESIDUAL_FIELDS, rows)
    write_json(output / "validation.json", validation)
    comparison = reference_comparison(trajectory_validation, args.mode)
    write_json(output / "reference_comparison.json", comparison)

    if not args.no_plots and validation["status"] == "passed":
        plot_residuals(list(rows), output)
        if structural_root is not None:
            notes.extend(plot_structural_figures(structural_root, output))
    manifest = {
        "paper_configuration": {
            "instance": "enlight_hard (continuous LP relaxation)",
            "column_clusters": 4,
            "row_clusters": 4,
            "alternating_passes": 5,
            "column_initialization": "cyclic, unshuffled",
            "tie_break": "smallest cluster index",
            "row_grouping": "final row cluster",
            "solver": "doubly linearized ADMM / FLiP-ADMM",
            "rho": 1000.0,
            "primal_plot_stride": 500,
            "dual_plot_stride": 1,
        },
        "available_figures": sorted(path.name for path in (output / "figures").glob("figure_*.png")),
        "fidelity_note": comparison["fidelity_note"],
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
            "The MIPLIB instance is represented as its continuous LP relaxation.",
            "Fresh mode implements the manuscript's literal k=4/five-pass co-clustering locally.",
            "Global RNG seed 126 matches the original enlight_hard experiment and controls the Basic initial point.",
            comparison["fidelity_note"],
            *notes,
        ),
    )

    if validation["errors"]:
        for error in validation["errors"]:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Wrote validated Section 3.1 artifacts to {output}")
    return 0


__all__ = (
    "discover_structured_input",
    "main",
    "read_structured_file",
    "reference_comparison",
    "validate_exact_configuration",
    "validate_trajectories",
)
