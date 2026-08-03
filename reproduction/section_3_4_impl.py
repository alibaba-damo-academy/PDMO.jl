"""Execution and artifact layer for :mod:`reproduction.section_3_4`.

Kept separate only to permit an add-only implementation in environments where
the managed patch helper cannot update a newly created file.  The public
reviewer entry point remains ``reproduction/section_3_4.py``.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

try:
    from .common import (
        PAPER_SEEDS,
        Job,
        add_common_arguments,
        arithmetic_mean,
        collect_logs,
        failed_jobs,
        get_pyplot,
        job_results_as_dicts,
        julia_command,
        legacy_workdir,
        prepare_mode_output,
        run_jobs,
        sample_std,
        source_logs,
        validate_common_arguments,
        write_csv,
        write_json,
        write_provenance,
    )
    from .section_3_4 import (
        DEFAULT_OUTPUT,
        FULL_RUNTIME_NOTE,
        GNN_MODEL,
        JULIA_DRIVER,
        JULIA_METHOD_TOKEN,
        METHOD_COLORS,
        METHOD_DISPLAY,
        METHOD_GAP,
        METHOD_LIGHT_COLORS,
        OBJECTIVE_FIDELITY_WARNING,
        PAPER_METHODS,
        PAPER_NODE_COUNTS,
        PAPER_SOLVERS,
        REFERENCE_FIGURES,
        REFERENCE_TABLE,
        RUN_FIELDS,
        SECTION_ARCHIVE_PATH,
        SMOKE_REFERENCE,
        optional_float,
        optional_int,
        parse_log,
        parse_method_spec,
    )
    from .section_3_3 import (
        _check_pycall_dependencies,
        _check_python_dependencies,
        _resolve_python,
    )
except ImportError:
    from common import (  # type: ignore
        PAPER_SEEDS,
        Job,
        add_common_arguments,
        arithmetic_mean,
        collect_logs,
        failed_jobs,
        get_pyplot,
        job_results_as_dicts,
        julia_command,
        legacy_workdir,
        prepare_mode_output,
        run_jobs,
        sample_std,
        source_logs,
        validate_common_arguments,
        write_csv,
        write_json,
        write_provenance,
    )
    from section_3_4 import (  # type: ignore
        DEFAULT_OUTPUT,
        FULL_RUNTIME_NOTE,
        GNN_MODEL,
        JULIA_DRIVER,
        JULIA_METHOD_TOKEN,
        METHOD_COLORS,
        METHOD_DISPLAY,
        METHOD_GAP,
        METHOD_LIGHT_COLORS,
        OBJECTIVE_FIDELITY_WARNING,
        PAPER_METHODS,
        PAPER_NODE_COUNTS,
        PAPER_SOLVERS,
        REFERENCE_FIGURES,
        REFERENCE_TABLE,
        RUN_FIELDS,
        SECTION_ARCHIVE_PATH,
        SMOKE_REFERENCE,
        optional_float,
        optional_int,
        parse_log,
        parse_method_spec,
    )
    from section_3_3 import (  # type: ignore
        _check_pycall_dependencies,
        _check_python_dependencies,
        _resolve_python,
    )


AGGREGATE_FIELDS = (
    "solver",
    "number_nodes",
    "method",
    "method_display",
    "mip_gap",
    "n_runs",
    "seeds",
    "mean_iterations",
    "std_iterations",
    "normalized_iterations",
    "mean_partition_time_seconds",
    "std_partition_time_seconds",
    "mean_admm_time_seconds",
    "std_admm_time_seconds",
    "mean_total_time_seconds",
    "std_total_time_seconds",
    "normalized_partition_time",
    "normalized_admm_time",
    "normalized_total_time",
    "n_optimal",
    "n_time_limit",
)

PAPER_DECISION_DIMENSION = 500
PAPER_MEASUREMENTS_PER_AGENT = 250
PAPER_RHO = 10.0
PAPER_MAX_ITER = 100_000
PAPER_LOG_INTERVAL = 1_000
PAPER_MIP_HEURISTIC_EFFORT = 0.2
PAPER_MIP_TIME_LIMIT = 60.0

TABLE_FIELDS = (
    "number_nodes",
    "average_original_edges",
    "method",
    "method_display",
    "n_seeds",
    "seeds",
    "mean_average_degree",
    "mean_balance",
    "mean_partition_time_seconds",
)


def paper_run_filter(record: Mapping[str, object], mode: str) -> bool:
    number_nodes = optional_int(record.get("number_nodes"))
    seed = optional_int(record.get("seed"))
    solver = str(record.get("solver") or "").lower()
    method = str(record.get("method") or "")
    if number_nodes not in PAPER_NODE_COUNTS or seed not in PAPER_SEEDS:
        return False
    if solver not in PAPER_SOLVERS or method not in PAPER_METHODS:
        return False
    if mode == "smoke":
        return number_nodes == 50 and seed == 111 and solver == "original"
    return True


def record_sort_key(record: Mapping[str, object]) -> tuple[object, ...]:
    solver = str(record.get("solver") or "")
    solver_index = PAPER_SOLVERS.index(solver) if solver in PAPER_SOLVERS else len(PAPER_SOLVERS)
    number_nodes = optional_int(record.get("number_nodes")) or 0
    seed = optional_int(record.get("seed")) or 0
    method = str(record.get("method") or "")
    method_index = PAPER_METHODS.index(method) if method in PAPER_METHODS else len(PAPER_METHODS)
    return solver_index, number_nodes, seed, method_index


def parse_logs(log_paths: Sequence[Path], mode: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    rejected_paper_methods: list[tuple[object, ...]] = []
    for path in log_paths:
        try:
            parsed = parse_log(path)
        except OSError as error:
            print(f"warning: could not read {path}: {error}", file=sys.stderr)
            continue
        for record in parsed:
            if paper_run_filter(record, mode):
                records.append(record)
                continue
            if mode != "parse" or str(record.get("method") or "") not in PAPER_METHODS:
                continue
            metadata = (
                record.get("source_log"),
                record.get("solver"),
                record.get("number_nodes"),
                record.get("seed"),
                record.get("method"),
            )
            number_nodes = optional_int(record.get("number_nodes"))
            seed = optional_int(record.get("seed"))
            solver = str(record.get("solver") or "").lower()
            if (
                number_nodes not in PAPER_NODE_COUNTS
                or seed not in PAPER_SEEDS
                or solver not in PAPER_SOLVERS
            ):
                rejected_paper_methods.append(metadata)
    if rejected_paper_methods:
        formatted = "; ".join(map(str, rejected_paper_methods[:5]))
        raise SystemExit(
            "Parse input contains paper-method rows with a non-paper solver, node count, "
            f"or seed; first={formatted}"
        )
    return sorted(records, key=record_sort_key)


def aggregate_runs(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int, str], list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        solver = str(record.get("solver") or "")
        number_nodes = optional_int(record.get("number_nodes"))
        method = str(record.get("method") or "")
        if number_nodes is None or not method:
            continue
        values = (
            optional_float(record.get("iterations")),
            optional_float(record.get("partition_time_seconds")),
            optional_float(record.get("admm_time_seconds")),
        )
        if any(value is None for value in values):
            continue
        groups[(solver, number_nodes, method)].append(record)

    rows: list[dict[str, object]] = []
    for solver in PAPER_SOLVERS:
        for number_nodes in PAPER_NODE_COUNTS:
            panel_rows: list[dict[str, object]] = []
            for method in PAPER_METHODS:
                group = groups.get((solver, number_nodes, method), [])
                if not group:
                    continue
                iterations = [float(record["iterations"]) for record in group]
                partition_times = [float(record["partition_time_seconds"]) for record in group]
                admm_times = [float(record["admm_time_seconds"]) for record in group]
                total_times = [part + admm for part, admm in zip(partition_times, admm_times)]
                statuses = [str(record.get("admm_status") or "") for record in group]
                seeds = sorted(
                    seed
                    for seed in (optional_int(record.get("seed")) for record in group)
                    if seed is not None
                )
                panel_rows.append(
                    {
                        "solver": solver,
                        "number_nodes": number_nodes,
                        "method": method,
                        "method_display": METHOD_DISPLAY[method],
                        "mip_gap": METHOD_GAP.get(method),
                        "n_runs": len(group),
                        "seeds": ";".join(str(seed) for seed in seeds),
                        "mean_iterations": arithmetic_mean(iterations),
                        "std_iterations": sample_std(iterations),
                        "mean_partition_time_seconds": arithmetic_mean(partition_times),
                        "std_partition_time_seconds": sample_std(partition_times),
                        "mean_admm_time_seconds": arithmetic_mean(admm_times),
                        "std_admm_time_seconds": sample_std(admm_times),
                        "mean_total_time_seconds": arithmetic_mean(total_times),
                        "std_total_time_seconds": sample_std(total_times),
                        "n_optimal": sum(status == "ADMM_TERMINATION_OPTIMAL" for status in statuses),
                        "n_time_limit": sum(status == "ADMM_TERMINATION_TIME_LIMIT" for status in statuses),
                    }
                )

            if not panel_rows:
                continue
            max_iterations = max(float(row["mean_iterations"]) for row in panel_rows)
            max_total_time = max(float(row["mean_total_time_seconds"]) for row in panel_rows)
            for row in panel_rows:
                row["normalized_iterations"] = float(row["mean_iterations"]) / max_iterations
                row["normalized_partition_time"] = (
                    float(row["mean_partition_time_seconds"]) / max_total_time
                )
                row["normalized_admm_time"] = float(row["mean_admm_time_seconds"]) / max_total_time
                row["normalized_total_time"] = float(row["mean_total_time_seconds"]) / max_total_time
                rows.append(row)
    return rows


def build_table_1(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    original = [record for record in records if record.get("solver") == "original"]
    basic_edges: dict[tuple[int, int], float] = {}
    for record in original:
        if record.get("method") != "basic":
            continue
        number_nodes = optional_int(record.get("number_nodes"))
        seed = optional_int(record.get("seed"))
        left = optional_float(record.get("partition_left"))
        right = optional_float(record.get("partition_right"))
        if number_nodes is not None and seed is not None and left is not None and right is not None:
            basic_edges[(number_nodes, seed)] = max(left, right)

    rows: list[dict[str, object]] = []
    for number_nodes in PAPER_NODE_COUNTS:
        original_edges = [edges for (nodes, _), edges in basic_edges.items() if nodes == number_nodes]
        average_original_edges = arithmetic_mean(original_edges) if original_edges else None
        for method in PAPER_METHODS:
            group = [
                record
                for record in original
                if optional_int(record.get("number_nodes")) == number_nodes
                and record.get("method") == method
                and optional_float(record.get("average_degree")) is not None
                and optional_float(record.get("balance")) is not None
            ]
            if not group:
                continue
            degrees = [float(record["average_degree"]) for record in group]
            balances = [float(record["balance"]) for record in group]
            graph_times = []
            for record in group:
                value = optional_float(record.get("graph_partition_time_seconds"))
                if value is None:
                    value = optional_float(record.get("partition_time_seconds"))
                if value is not None:
                    graph_times.append(value)
            seeds = sorted(
                seed
                for seed in (optional_int(record.get("seed")) for record in group)
                if seed is not None
            )
            rows.append(
                {
                    "number_nodes": number_nodes,
                    "average_original_edges": average_original_edges,
                    "method": method,
                    "method_display": METHOD_DISPLAY[method],
                    "n_seeds": len(group),
                    "seeds": ";".join(str(seed) for seed in seeds),
                    "mean_average_degree": arithmetic_mean(degrees),
                    "mean_balance": arithmetic_mean(balances),
                    "mean_partition_time_seconds": arithmetic_mean(graph_times) if graph_times else None,
                }
            )
    return rows


def format_number(value: object, digits: int = 2) -> str:
    number = optional_float(value)
    return "" if number is None else f"{number:.{digits}f}"


def write_table_1(output: Path, rows: Sequence[Mapping[str, object]]) -> list[Path]:
    csv_path = output / "table_1.csv"
    write_csv(csv_path, TABLE_FIELDS, rows)

    markdown_path = output / "table_1.md"
    lines = [
        "# Table 1: Average features of bipartite graphs",
        "",
        "| N | Avg. edges | Partition | Avg. degree | Balance | Time(P), s | Seeds |",
        "|---:|---:|:---|---:|---:|---:|---:|",
    ]
    for row in rows:
        average_edges = optional_float(row.get("average_original_edges"))
        lines.append(
            "| {nodes} | {edges} | {method} | {degree} | {balance} | {time} | {seeds} |".format(
                nodes=row["number_nodes"],
                edges=f"{average_edges:.0f}" if average_edges is not None else "",
                method=row["method_display"],
                degree=format_number(row.get("mean_average_degree")),
                balance=format_number(row.get("mean_balance")),
                time=format_number(row.get("mean_partition_time_seconds")),
                seeds=row["n_seeds"],
            )
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    latex_path = output / "table_1.tex"
    latex_lines = [
        "\\begin{tabular}{rrlrrrr}",
        "\\hline",
        "$|V|$ & Avg. $|E|$ & Partition & $2|\\widehat E|/|\\widehat V|$ & Balance & Time(P) [s] & Seeds \\\\",
        "\\hline",
    ]
    for row in rows:
        average_edges = optional_float(row.get("average_original_edges"))
        method = str(row["method_display"]).replace("%", "\\%")
        latex_lines.append(
            "{} & {} & {} & {} & {} & {} & {} \\\\".format(
                row["number_nodes"],
                f"{average_edges:.0f}" if average_edges is not None else "",
                method,
                format_number(row.get("mean_average_degree")),
                format_number(row.get("mean_balance")),
                format_number(row.get("mean_partition_time_seconds")),
                row["n_seeds"],
            )
        )
    latex_lines.extend(("\\hline", "\\end{tabular}"))
    latex_path.write_text("\n".join(latex_lines) + "\n", encoding="utf-8")
    return [csv_path, markdown_path, latex_path]


def make_figure(
    aggregate: Sequence[Mapping[str, object]],
    solver: str,
    figure_number: int,
    output: Path,
) -> list[Path]:
    pyplot = get_pyplot()
    from matplotlib.patches import Patch

    index = {
        (str(row["solver"]), int(row["number_nodes"]), str(row["method"])): row
        for row in aggregate
    }
    fig, axes = pyplot.subplots(3, 2, figsize=(14, 12), constrained_layout=False)
    panel_letter = 0
    solver_title = "Original ADMM" if solver == "original" else "FLiP-ADMM"
    for row_index, number_nodes in enumerate(PAPER_NODE_COUNTS):
        available = [
            method for method in PAPER_METHODS if (solver, number_nodes, method) in index
        ]
        for column, metric_kind in enumerate(("iterations", "time")):
            axis = axes[row_index][column]
            letter = chr(ord("a") + panel_letter)
            panel_letter += 1
            if not available:
                axis.text(0.5, 0.5, "Not run in this profile", ha="center", va="center")
                axis.set_axis_off()
                continue
            x_values = list(range(len(available)))
            colors = [METHOD_COLORS[method] for method in available]
            labels = [METHOD_DISPLAY[method] for method in available]
            if metric_kind == "iterations":
                heights = [float(index[(solver, number_nodes, method)]["normalized_iterations"]) for method in available]
                axis.bar(x_values, heights, color=colors, edgecolor="black", linewidth=0.35)
                axis.set_ylabel("Normalized average iterations")
                noun = "iteration"
            else:
                partition = [
                    float(index[(solver, number_nodes, method)]["normalized_partition_time"])
                    for method in available
                ]
                admm = [
                    float(index[(solver, number_nodes, method)]["normalized_admm_time"])
                    for method in available
                ]
                light = [METHOD_LIGHT_COLORS[method] for method in available]
                axis.bar(x_values, partition, color=light, edgecolor="black", linewidth=0.35)
                axis.bar(
                    x_values,
                    admm,
                    bottom=partition,
                    color=colors,
                    edgecolor="black",
                    linewidth=0.35,
                )
                axis.set_ylabel("Normalized average time")
                noun = "time"
            axis.set_ylim(0.0, 1.08)
            axis.set_xticks(x_values, labels, rotation=35, ha="right")
            axis.grid(axis="y", alpha=0.25, linewidth=0.6)
            axis.set_title(f"({letter}) {solver_title} {noun}, |V| = {number_nodes}")

    method_handles = [
        Patch(facecolor=METHOD_COLORS["basic"], label="Basic"),
        Patch(facecolor=METHOD_COLORS["bfs"], label="BFS"),
        Patch(facecolor=METHOD_COLORS["milp_0.01"], label="MILP"),
        Patch(facecolor=METHOD_COLORS["gnn"], label="GNN"),
        Patch(facecolor="#d9d9d9", edgecolor="black", label="lighter segment: partition time"),
    ]
    fig.legend(handles=method_handles, loc="lower center", ncol=5, frameon=False)
    fig.suptitle(f"Figure {figure_number}: Section 3.4 {solver_title}", fontsize=15)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.97))
    png = output / "figures" / f"figure_{figure_number}.png"
    pdf = output / "figures" / f"figure_{figure_number}.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    pyplot.close(fig)
    return [png, pdf]


def _comparison_check(
    checks: list[dict[str, object]],
    *,
    key: str,
    metric: str,
    observed: float,
    expected: float,
    tolerance: float,
    enforced: bool,
) -> None:
    error = abs(observed - expected)
    checks.append(
        {
            "key": key,
            "metric": metric,
            "observed": observed,
            "expected": expected,
            "absolute_error": error,
            "tolerance": tolerance,
            "passed": error <= tolerance,
            "enforced": enforced,
        }
    )


def build_reference_comparison(
    records: Sequence[Mapping[str, object]],
    aggregate: Sequence[Mapping[str, object]],
    table_rows: Sequence[Mapping[str, object]],
    mode: str,
) -> dict[str, object]:
    checks: list[dict[str, object]] = []
    missing: list[str] = []
    table_index = {
        (int(row["number_nodes"]), str(row["method"])): row
        for row in table_rows
        if int(row.get("n_seeds") or 0) >= len(PAPER_SEEDS)
    }
    for key, expected in REFERENCE_TABLE.items():
        row = table_index.get(key)
        text_key = f"table_1/N={key[0]}/{key[1]}"
        if row is None:
            if mode != "smoke":
                missing.append(text_key)
            continue
        observed_values = (
            optional_float(row.get("average_original_edges")),
            optional_float(row.get("mean_average_degree")),
            optional_float(row.get("mean_balance")),
            optional_float(row.get("mean_partition_time_seconds")),
        )
        metrics = ("average_edges", "average_degree", "balance", "partition_time_seconds")
        tolerances = (0.51, 0.015, 0.015, 0.015)
        for metric, observed, expected_value, tolerance in zip(
            metrics, observed_values, expected, tolerances
        ):
            if observed is not None:
                _comparison_check(
                    checks,
                    key=text_key,
                    metric=metric,
                    observed=observed,
                    expected=expected_value,
                    tolerance=tolerance,
                    enforced=mode == "archived",
                )

    aggregate_index = {
        (str(row["solver"]), int(row["number_nodes"]), str(row["method"])): row
        for row in aggregate
        if int(row.get("n_runs") or 0) >= 5
    }
    for key, expected in REFERENCE_FIGURES.items():
        row = aggregate_index.get(key)
        text_key = f"figures/{key[0]}/N={key[1]}/{key[2]}"
        if row is None:
            if mode != "smoke":
                missing.append(text_key)
            continue
        tolerance = 0.002 if mode == "archived" else 0.05
        _comparison_check(
            checks,
            key=text_key,
            metric="normalized_iterations",
            observed=float(row["normalized_iterations"]),
            expected=expected[0],
            tolerance=tolerance,
            enforced=mode == "archived",
        )
        _comparison_check(
            checks,
            key=text_key,
            metric="normalized_total_time",
            observed=float(row["normalized_total_time"]),
            expected=expected[1],
            tolerance=tolerance,
            enforced=mode == "archived",
        )

    if mode == "smoke":
        smoke_index = {
            str(row["method"]): row
            for row in records
            if row.get("solver") == "original"
            and optional_int(row.get("number_nodes")) == 50
            and optional_int(row.get("seed")) == 111
        }
        for method, expected_values in SMOKE_REFERENCE.items():
            if method not in smoke_index:
                continue
            row = smoke_index[method]
            for metric in ("iterations", "graph_nodes", "graph_edges"):
                observed = optional_float(row.get(metric))
                if observed is not None:
                    _comparison_check(
                        checks,
                        key=f"smoke/original/N=50/seed=111/{method}",
                        metric=metric,
                        observed=observed,
                        expected=float(expected_values[metric]),
                        tolerance=0.0,
                        enforced=True,
                    )

    compared = len(checks)
    passed = sum(bool(check["passed"]) for check in checks)
    enforced_checks = [check for check in checks if bool(check["enforced"])]
    enforced_passed = sum(bool(check["passed"]) for check in enforced_checks)
    return {
        "reference": "Automating Reformulation for Parallel ADMM, Table 1 and Figures 15--16",
        "mode": mode,
        "checks": checks,
        "missing_reference_keys": missing,
        "summary": {
            "compared": compared,
            "passed": passed,
            "failed": compared - passed,
            "all_compared_checks_passed": compared == passed,
            "enforced": len(enforced_checks),
            "enforced_passed": enforced_passed,
            "enforced_failed": len(enforced_checks) - enforced_passed,
            "all_enforced_checks_passed": len(enforced_checks) == enforced_passed,
            "note": "Partial smoke/parse profiles are not required to contain every paper key.",
        },
    }


def reference_validation_errors(
    comparison: Mapping[str, object],
    *,
    mode: str,
    selected_methods: Sequence[str],
) -> list[str]:
    checks = comparison.get("checks")
    if not isinstance(checks, list):
        return ["Reference comparison did not produce a check list"]
    failed = [
        check
        for check in checks
        if isinstance(check, Mapping)
        and bool(check.get("enforced"))
        and not bool(check.get("passed"))
    ]
    errors = [
        "Reference mismatch "
        f"{check.get('key')}/{check.get('metric')}: expected {check.get('expected')}, "
        f"got {check.get('observed')}"
        for check in failed
    ]
    missing = comparison.get("missing_reference_keys")
    if mode in {"archived", "full"} and isinstance(missing, list) and missing:
        errors.append(f"Missing {len(missing)} full-grid paper reference keys")
    if mode == "smoke":
        expected = {
            (f"smoke/original/N=50/seed=111/{method}", metric)
            for method in selected_methods
            for metric in ("iterations", "graph_nodes", "graph_edges")
        }
        observed = {
            (str(check.get("key")), str(check.get("metric")))
            for check in checks
            if isinstance(check, Mapping) and bool(check.get("enforced"))
        }
        missing_smoke = sorted(expected.difference(observed))
        if missing_smoke:
            errors.append(f"Missing smoke reference checks: {missing_smoke}")
    return errors


def validate_records(
    records: Sequence[Mapping[str, object]],
    *,
    mode: str,
    selected_methods: Sequence[str],
    failed_job_names: Sequence[str],
) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    keys: list[tuple[str, int, int, str]] = []
    allowed_statuses = {"ADMM_TERMINATION_OPTIMAL", "ADMM_TERMINATION_TIME_LIMIT"}
    time_limit_rows: list[dict[str, object]] = []
    configuration_errors: set[str] = set()

    for record in records:
        solver = str(record.get("solver") or "")
        number_nodes = optional_int(record.get("number_nodes"))
        seed = optional_int(record.get("seed"))
        method = str(record.get("method") or "")
        if number_nodes is None or seed is None:
            errors.append(f"Missing metadata in {record.get('source_log')}")
            continue
        key = (solver, number_nodes, seed, method)
        keys.append(key)
        expected_configuration = (
            ("n", optional_int(record.get("n")), PAPER_DECISION_DIMENSION),
            (
                "m",
                optional_int(record.get("m")),
                PAPER_MEASUREMENTS_PER_AGENT,
            ),
            ("initial_rho", optional_float(record.get("initial_rho")), PAPER_RHO),
            ("max_iter", optional_int(record.get("max_iter")), PAPER_MAX_ITER),
            (
                "log_interval",
                optional_int(record.get("log_interval")),
                PAPER_LOG_INTERVAL,
            ),
        )
        for field, observed, expected_value in expected_configuration:
            if observed != expected_value:
                configuration_errors.add(
                    f"{solver}/N={number_nodes}/seed={seed}: {field}={observed!r}, "
                    f"expected {expected_value!r}"
                )
        missing_fields = [
            field
            for field in (
                "partition_time_seconds",
                "graph_nodes",
                "partition_left",
                "partition_right",
                "graph_edges",
                "iterations",
                "admm_time_seconds",
                "admm_status",
            )
            if record.get(field) is None
        ]
        if missing_fields:
            errors.append(f"{key} missing fields: {', '.join(missing_fields)}")
        status = str(record.get("admm_status") or "")
        if status not in allowed_statuses:
            errors.append(f"{key} has unsupported ADMM status {status!r}")
        elif status == "ADMM_TERMINATION_TIME_LIMIT":
            valid = (
                optional_float(record.get("iterations")) is not None
                and optional_float(record.get("admm_time_seconds")) is not None
            )
            time_limit_rows.append(
                {
                    "solver": solver,
                    "number_nodes": number_nodes,
                    "seed": seed,
                    "method": method,
                    "valid_and_included": valid,
                }
            )
            if not valid:
                errors.append(f"{key} time-limit row lacks usable iteration/time data")

    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        errors.append(f"Duplicate run-method keys: {duplicates}")
    errors.extend(sorted(configuration_errors))

    if not records:
        errors.append("No paper run-method records were parsed")

    if mode in {"archived", "full"}:
        expected = {
            (solver, number_nodes, seed, method)
            for solver in PAPER_SOLVERS
            for number_nodes in PAPER_NODE_COUNTS
            for seed in PAPER_SEEDS
            for method in PAPER_METHODS
        }
        missing = sorted(expected.difference(keys))
        extra = sorted(set(keys).difference(expected))
        if missing:
            errors.append(f"Missing {len(missing)} paper run-method combinations; first={missing[:5]}")
        if extra:
            warnings.append(f"Ignored/unexpected paper-grid rows: {extra[:5]}")
    elif mode == "smoke":
        expected = {("original", 50, 111, method) for method in selected_methods}
        missing = sorted(expected.difference(keys))
        if missing:
            errors.append(f"Missing smoke run-method combinations: {missing}")

    if failed_job_names:
        errors.append(f"Fresh Julia jobs failed: {', '.join(failed_job_names)}")

    if time_limit_rows:
        warnings.append(
            "ADMM time-limit rows are valid observations and are retained in averages when "
            "iteration and elapsed-time fields are present."
        )
    return {
        "status": "passed" if not errors else "failed",
        "mode": mode,
        "record_count": len(records),
        "expected_full_record_count": 210,
        "errors": errors,
        "warnings": warnings,
        "time_limit_policy": (
            "ADMM_TERMINATION_TIME_LIMIT is valid and included when iterations and "
            "ADMM time are finite."
        ),
        "time_limit_rows": time_limit_rows,
        "all_time_limit_rows_valid": all(row["valid_and_included"] for row in time_limit_rows),
    }


def build_jobs(args: argparse.Namespace, methods: Sequence[str]) -> list[Job]:
    configurations = (
        [("original", 50, 111)]
        if args.mode == "smoke"
        else [
            (solver, number_nodes, seed)
            for solver in PAPER_SOLVERS
            for number_nodes in PAPER_NODE_COUNTS
            for seed in PAPER_SEEDS
        ]
    )
    method_argument = ",".join(JULIA_METHOD_TOKEN[method] for method in methods)
    return [
        Job(
            f"{solver}/nodes_{number_nodes}/seed_{seed}",
            julia_command(
                args.julia,
                args.threads,
                JULIA_DRIVER,
                number_nodes,
                args.n,
                args.m,
                solver,
                args.rho,
                args.max_iter,
                args.log_interval,
                seed,
                args.mip_heuristic_effort,
                args.mip_time_limit,
                method_argument,
            ),
        )
        for solver, number_nodes, seed in configurations
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce Section 3.4 Table 1 and Figures 15--16.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_arguments(parser, default_output=DEFAULT_OUTPUT)
    parser.add_argument("--n", type=int, default=500, help="Consensus decision dimension.")
    parser.add_argument("--m", type=int, default=250, help="Measurements per agent.")
    parser.add_argument("--rho", type=float, default=10.0, help="Initial ADMM penalty.")
    parser.add_argument("--max-iter", type=int, default=100000, help="Maximum ADMM iterations.")
    parser.add_argument("--log-interval", type=int, default=1000, help="ADMM logging interval.")
    parser.add_argument(
        "--mip-heuristic-effort",
        type=float,
        default=0.2,
        help="HiGHS mip_heuristic_effort used for all four paper gaps.",
    )
    parser.add_argument(
        "--mip-time-limit",
        type=float,
        default=60.0,
        help="HiGHS time limit in seconds for each bipartization MILP.",
    )
    parser.add_argument(
        "--smoke-methods",
        default="basic,bfs",
        help=(
            "Comma-separated smoke subset. Use 'paper' for Basic, BFS, four MILP gaps, "
            "and GNN. The default avoids Python/GNN setup and takes about 2.5 archived minutes."
        ),
    )
    parser.add_argument(
        "--python",
        type=Path,
        dest="pdmo_python",
        help=(
            "Python 3.9-3.11 executable with numpy, torch, and torch_geometric. "
            "Required when the selected method set includes GNN."
        ),
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> tuple[str, ...]:
    validate_common_arguments(args)
    if args.n < 1 or args.m < 1:
        raise SystemExit("--n and --m must be positive")
    if args.rho <= 0 or args.max_iter < 1 or args.log_interval < 1:
        raise SystemExit("--rho, --max-iter, and --log-interval must be positive")
    if not 0.0 <= args.mip_heuristic_effort <= 1.0:
        raise SystemExit("--mip-heuristic-effort must lie in [0, 1]")
    if args.mip_time_limit <= 0:
        raise SystemExit("--mip-time-limit must be positive")
    if args.mode in {"smoke", "full"}:
        expected = (
            ("--n", args.n, PAPER_DECISION_DIMENSION),
            ("--m", args.m, PAPER_MEASUREMENTS_PER_AGENT),
            ("--rho", args.rho, PAPER_RHO),
            ("--max-iter", args.max_iter, PAPER_MAX_ITER),
            ("--log-interval", args.log_interval, PAPER_LOG_INTERVAL),
            (
                "--mip-heuristic-effort",
                args.mip_heuristic_effort,
                PAPER_MIP_HEURISTIC_EFFORT,
            ),
            ("--mip-time-limit", args.mip_time_limit, PAPER_MIP_TIME_LIMIT),
        )
        mismatches = [
            f"{option}={observed!r} (paper value {paper_value!r})"
            for option, observed, paper_value in expected
            if observed != paper_value
        ]
        if mismatches:
            raise SystemExit(
                "Fresh Section 3.4 modes require the paper configuration: "
                + ", ".join(mismatches)
            )
    return parse_method_spec(args.smoke_methods) if args.mode == "smoke" else PAPER_METHODS


def publish_validated_artifacts(
    output: Path,
    aggregate: Sequence[Mapping[str, object]],
    table_rows: Sequence[Mapping[str, object]],
    validation: Mapping[str, object],
    *,
    no_plots: bool,
) -> tuple[list[Path], list[Path]]:
    """Publish Table 1 and Figures 15-16 only after validation passes."""

    if validation.get("status") != "passed":
        return [], []
    table_paths = write_table_1(output, table_rows)
    figure_paths: list[Path] = []
    if not no_plots:
        figure_paths.extend(make_figure(aggregate, "original", 15, output))
        figure_paths.extend(make_figure(aggregate, "doubly", 16, output))
    return table_paths, figure_paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    selected_methods = validate_arguments(args)
    output = prepare_mode_output(args.output, args.mode)

    print(OBJECTIVE_FIDELITY_WARNING, flush=True)
    if args.mode == "full":
        print(FULL_RUNTIME_NOTE, flush=True)
    if args.mode == "smoke":
        print(f"Smoke methods: {', '.join(METHOD_DISPLAY[method] for method in selected_methods)}")

    jobs: list[Job] = []
    job_results = []
    fresh_log_root = output / "raw"
    if args.mode in {"smoke", "full"}:
        environment = None
        if "gnn" in selected_methods:
            if not GNN_MODEL.is_file():
                raise SystemExit(f"Bundled GNN weights not found: {GNN_MODEL}")
            python_executable = _resolve_python(args.pdmo_python)
            _check_python_dependencies(python_executable)
            _check_pycall_dependencies(args.julia, python_executable)
            environment = {"PDMO_PYTHON": python_executable}
        jobs = build_jobs(args, selected_methods)
        job_results = run_jobs(
            jobs,
            log_root=fresh_log_root,
            cwd=legacy_workdir(),
            workers=args.jobs,
            resume=not args.no_resume,
            environment=environment,
        )
        write_json(output / "job_results.json", job_results_as_dicts(job_results))

    with source_logs(args, SECTION_ARCHIVE_PATH, fresh_log_root) as source:
        log_paths = collect_logs(source)
        records = parse_logs(log_paths, args.mode)

    if args.mode == "archived":
        archive_path = args.archive.expanduser().resolve()
        for record in records:
            member = (
                Path("experiments_logs")
                / SECTION_ARCHIVE_PATH
                / str(record["archive_batch"])
                / str(record["run_id"])
                / "stdout.log"
            )
            record["source_log"] = f"{archive_path}::{member.as_posix()}"

    write_csv(output / "runs.csv", RUN_FIELDS, records)
    aggregate = aggregate_runs(records)
    write_csv(output / "aggregate.csv", AGGREGATE_FIELDS, aggregate)
    table_rows = build_table_1(records)

    comparison = build_reference_comparison(records, aggregate, table_rows, args.mode)
    write_json(output / "reference_comparison.json", comparison)
    failed = failed_jobs(job_results)
    validation = validate_records(
        records,
        mode=args.mode,
        selected_methods=selected_methods,
        failed_job_names=[result.name for result in failed],
    )
    validation["errors"].extend(
        reference_validation_errors(
            comparison,
            mode=args.mode,
            selected_methods=selected_methods,
        )
    )
    validation["status"] = "passed" if not validation["errors"] else "failed"
    validation["reference_summary"] = comparison["summary"]
    write_json(output / "validation.json", validation)
    _, figure_paths = publish_validated_artifacts(
        output,
        aggregate,
        table_rows,
        validation,
        no_plots=args.no_plots,
    )

    inputs = [args.archive] if args.mode == "archived" else []
    if args.mode == "parse" and args.logs is not None:
        inputs.append(args.logs)
    if args.mode in {"smoke", "full"}:
        inputs.extend((JULIA_DRIVER, GNN_MODEL))
    write_provenance(
        output,
        section="3.4 decentralized consensus optimization",
        args=args,
        jobs=jobs,
        inputs=inputs,
        notes=(
            OBJECTIVE_FIDELITY_WARNING,
            FULL_RUNTIME_NOTE,
            "Paper grid: N={50,100,200}, seeds={111,222,333,444,555}, "
            "solvers={original,doubly}, MILP gaps={1%,5%,10%,20%}.",
            "Figure time is partitionAlgorithmTime + ADMM iteration time; ADMM "
            "initialization and process setup are excluded, matching the archive plot script.",
            "A valid FLiP N=200 seed=333 Basic observation terminates at the 7200 s "
            "ADMM time limit and remains in the paper average.",
        ),
    )

    print(
        f"Wrote {len(records)} run rows, {len(aggregate)} aggregate rows, and "
        f"{len(table_rows)} Table 1 rows to {output}"
    )
    if figure_paths:
        print("Figures: " + ", ".join(str(path) for path in figure_paths))
    if validation["errors"]:
        for error in validation["errors"]:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


__all__ = [
    "AGGREGATE_FIELDS",
    "TABLE_FIELDS",
    "aggregate_runs",
    "build_reference_comparison",
    "build_table_1",
    "main",
    "parse_logs",
    "publish_validated_artifacts",
    "reference_validation_errors",
    "validate_records",
]
