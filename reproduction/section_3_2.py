#!/usr/bin/env python3
"""Reproduce Section 3.2 and Figure 11 (network-flow experiments).

The fresh modes call the branch's existing Julia driver without changing it.
The archived and parse modes rebuild the paper artifacts from combined Julia
stdout/stderr logs.  Only matplotlib is needed beyond the Python standard
library.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence

try:  # Support both ``python reproduction/section_3_2.py`` and module imports.
    from .common import (
        REPO_ROOT,
        Job,
        add_common_arguments,
        arithmetic_mean,
        collect_logs,
        failed_jobs,
        get_pyplot,
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
except ImportError:
    from common import (  # type: ignore[no-redef]
        REPO_ROOT,
        Job,
        add_common_arguments,
        arithmetic_mean,
        collect_logs,
        failed_jobs,
        get_pyplot,
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


ARCHIVE_SECTION = "3-2-networkflow"
JULIA_RUNNER = REPO_ROOT / "advanced" / "src" / "NetworkFlow" / "runNetworkFlowProblem.jl"
PAPER_NODES = tuple(range(200, 601, 100))
PAPER_SEEDS = tuple(range(1, 11))
PAPER_ARCS = 2000
METHODS = ("basic", "bfs", "milp")
METHOD_LABELS = {"basic": "Basic", "bfs": "BFS", "milp": "MILP"}
METHOD_COLORS = {"basic": "#2980b9", "bfs": "#f39c12", "milp": "#27ae60"}

VALID_STATUSES = {
    "ADMM_TERMINATION_OPTIMAL",
    "ADMM_TERMINATION_ITERATION_LIMIT",
    "ADMM_TERMINATION_TIME_LIMIT",
    "ADMM_TERMINATION_INFEASIBLE",
    "ADMM_TERMINATION_UNBOUNDED",
    "ADMM_TERMINATION_UNKNOWN",
    "ADMM_TERMINATION_ILLPOSED_CASE_B",
    "ADMM_TERMINATION_ILLPOSED_CASE_C",
    "ADMM_TERMINATION_ILLPOSED_CASE_D",
}

# Means parsed from experiments_logs/3-2-networkflow/admm_summary/
# admm_folder_summary.csv.  Timing comparisons are informational because they
# are hardware-dependent; archived iteration means are an exact parser check.
REFERENCE_MEANS: dict[tuple[int, str], dict[str, object]] = {
    (200, "basic"): dict(partition_time_mean=0.0, initialization_time_mean=3.854, admm_time_mean=186.159, iterations_mean=104.8, status_counts="ADMM_TERMINATION_OPTIMAL:10"),
    (200, "bfs"): dict(partition_time_mean=0.0, initialization_time_mean=1.816, admm_time_mean=305.787, iterations_mean=19919.5, status_counts="ADMM_TERMINATION_OPTIMAL:10"),
    (200, "milp"): dict(partition_time_mean=17.904, initialization_time_mean=0.059, admm_time_mean=298.597, iterations_mean=17663.6, status_counts="ADMM_TERMINATION_OPTIMAL:10"),
    (300, "basic"): dict(partition_time_mean=0.0, initialization_time_mean=3.965, admm_time_mean=726.421, iterations_mean=467.9, status_counts="ADMM_TERMINATION_OPTIMAL:10"),
    (300, "bfs"): dict(partition_time_mean=0.001, initialization_time_mean=1.827, admm_time_mean=295.806, iterations_mean=18837.0, status_counts="ADMM_TERMINATION_OPTIMAL:10"),
    (300, "milp"): dict(partition_time_mean=52.712, initialization_time_mean=0.064, admm_time_mean=245.459, iterations_mean=14284.9, status_counts="ADMM_TERMINATION_OPTIMAL:10"),
    (400, "basic"): dict(partition_time_mean=0.0, initialization_time_mean=3.897, admm_time_mean=210.946, iterations_mean=159.6, status_counts="ADMM_TERMINATION_OPTIMAL:10"),
    (400, "bfs"): dict(partition_time_mean=0.0, initialization_time_mean=1.817, admm_time_mean=255.386, iterations_mean=16272.8, status_counts="ADMM_TERMINATION_OPTIMAL:10"),
    (400, "milp"): dict(partition_time_mean=63.718, initialization_time_mean=0.06, admm_time_mean=196.503, iterations_mean=11953.4, status_counts="ADMM_TERMINATION_OPTIMAL:10"),
    (500, "basic"): dict(partition_time_mean=0.0, initialization_time_mean=3.978, admm_time_mean=922.571, iterations_mean=800.0, status_counts="ADMM_TERMINATION_OPTIMAL:9; ADMM_TERMINATION_TIME_LIMIT:1"),
    (500, "bfs"): dict(partition_time_mean=0.0, initialization_time_mean=1.83, admm_time_mean=360.991, iterations_mean=22616.4, status_counts="ADMM_TERMINATION_OPTIMAL:10"),
    (500, "milp"): dict(partition_time_mean=61.459, initialization_time_mean=0.062, admm_time_mean=295.386, iterations_mean=17250.9, status_counts="ADMM_TERMINATION_OPTIMAL:10"),
    (600, "basic"): dict(partition_time_mean=0.0, initialization_time_mean=3.971, admm_time_mean=538.566, iterations_mean=509.0, status_counts="ADMM_TERMINATION_OPTIMAL:10"),
    (600, "bfs"): dict(partition_time_mean=0.001, initialization_time_mean=1.824, admm_time_mean=358.318, iterations_mean=21844.5, status_counts="ADMM_TERMINATION_OPTIMAL:10"),
    (600, "milp"): dict(partition_time_mean=61.482, initialization_time_mean=0.063, admm_time_mean=264.534, iterations_mean=14925.5, status_counts="ADMM_TERMINATION_OPTIMAL:10"),
}
for _reference in REFERENCE_MEANS.values():
    _reference["total_time_mean"] = float(_reference["partition_time_mean"]) + float(
        _reference["admm_time_mean"]
    )

# Exact raw result selected by ``--mode smoke``. These iteration counts and
# statuses are seed-sensitive deterministic checks; timing remains
# informational because it changes with hardware and package versions.
REFERENCE_SMOKE_N200_SEED1: dict[str, dict[str, object]] = {
    "basic": {
        "iterations": 142,
        "termination_status": "ADMM_TERMINATION_OPTIMAL",
        "partition_time_sec": 0.0,
        "initialization_time_sec": 3.82,
        "admm_time_sec": 255.50,
    },
    "bfs": {
        "iterations": 11_001,
        "termination_status": "ADMM_TERMINATION_OPTIMAL",
        "partition_time_sec": 0.0,
        "initialization_time_sec": 1.82,
        "admm_time_sec": 167.02,
    },
    "milp": {
        "iterations": 8_883,
        "termination_status": "ADMM_TERMINATION_OPTIMAL",
        "partition_time_sec": 21.54,
        "initialization_time_sec": 0.06,
        "admm_time_sec": 144.94,
    },
}


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
RUN_STARTS = (
    (re.compile(r"Running Bipartite ADMM with LP formulation\.\.\."), "basic"),
    (re.compile(r"Running Bipartite ADMM with BFS bipartization\.\.\."), "bfs"),
    (re.compile(r"Running Bipartite ADMM with MILP bipartization\.\.\."), "milp"),
)
RE_SEED = re.compile(r"^\s*seed\s*=\s*(\d+)\s*$")
RE_INSTANCE = re.compile(r"Generating random instance(?: \(positional\))?:\s*nodes=(\d+)\s+arcs=(\d+)")
RE_PARTITION = re.compile(
    rf"ADMMBipartiteGraph:\s*(BFS_BIPARTIZATION|MILP_BIPARTIZATION)\s+took\s+({NUMBER})\s+seconds"
)
RE_ALREADY_BIPARTITE = re.compile(
    r"ADMMBipartiteGraph: The graph is already bipartite; skip bipartization algorithm\."
)
RE_INITIALIZATION = re.compile(rf"ADMM:\s*initialization took\s+({NUMBER})\s+seconds")
RE_STATUS = re.compile(r"^\s*Solver Status\s*=\s*(ADMM_TERMINATION_[A-Z_]+)\s*$")
RE_TOTAL_TIME = re.compile(rf"^\s*Total Time\s*=\s*({NUMBER})\s*$")
RE_STOP_ITER = re.compile(r"^\s*Stop\.\s*Iter\s*=\s*(\d+)\s*$")
RE_PROGRESS = re.compile(
    r"^\s*(\d+)"
    r"(?:\s+(?:[-+0-9.eE]+|Inf|NaN)){7}"
    r"(?:\s+\(rho\s+<-\s+(?:[-+0-9.eE]+|Inf|NaN)\))?\s*$"
)
RE_DRIVER_FAILURE = re.compile(
    r"Failed to (?:generate|solve) the network flow problem|MILP bipartization failed"
)


RUN_FIELDS = (
    "nodes",
    "arcs",
    "seed",
    "method",
    "method_label",
    "partition_time_sec",
    "partition_kind",
    "initialization_time_sec",
    "admm_time_sec",
    "total_time_sec",
    "termination_status",
    "iterations",
    "iteration_source",
    "stop_iter",
    "latest_progress_iter",
    "source_log",
    "parse_warnings",
)

SUMMARY_FIELDS = (
    "nodes",
    "arcs",
    "method",
    "method_label",
    "n_runs",
    "partition_time_mean",
    "partition_time_std",
    "initialization_time_mean",
    "initialization_time_std",
    "admm_time_mean",
    "admm_time_std",
    "total_time_mean",
    "total_time_std",
    "iterations_mean",
    "iterations_std",
    "status_counts",
)


def _new_method(method: str) -> dict[str, object]:
    basic = method == "basic"
    return {
        "method": method,
        "partition_time_sec": 0.0 if basic else None,
        "partition_kind": "NONE_OR_ALREADY_BIPARTITE" if basic else "",
        "initialization_time_sec": None,
        "admm_time_sec": None,
        "termination_status": None,
        "stop_iter": None,
        "latest_progress_iter": None,
        "parse_warnings": [],
    }


def parse_log(path: Path, source_log: str) -> tuple[list[dict[str, object]], dict[str, object], list[str]]:
    """Parse one combined Julia stdout/stderr log into three long rows."""

    seed: int | None = None
    nodes: int | None = None
    arcs: int | None = None
    methods: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    errors: list[str] = []
    timing_by_method: dict[str, dict[str, object]] = {
        method: {} for method in METHODS
    }
    timing_method: str | None = None

    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.rstrip("\n")

            match = RE_SEED.match(line)
            if match:
                seed = int(match.group(1))
            match = RE_INSTANCE.search(line)
            if match:
                nodes, arcs = int(match.group(1)), int(match.group(2))

            if RE_DRIVER_FAILURE.search(line):
                errors.append(f"{source_log}:{line_number}: driver reported: {line.strip()}")

            started: str | None = None
            for pattern, method in RUN_STARTS:
                if pattern.search(line):
                    started = method
                    break
            if started is not None:
                if current is not None:
                    methods.append(current)
                current = _new_method(started)
                continue

            match = RE_PARTITION.search(line)
            if match:
                partition_kind = match.group(1)
                classified = partition_kind.partition("_")[0].lower()
                if classified in timing_by_method:
                    timing_method = classified
                    timing_by_method[classified]["partition_kind"] = partition_kind
                    timing_by_method[classified]["partition_time_sec"] = float(match.group(2))
                continue
            if RE_ALREADY_BIPARTITE.search(line):
                current_method = str(current["method"]) if current is not None else None
                if (
                    current_method is not None
                    and "partition_time_sec" not in timing_by_method[current_method]
                ):
                    timing_method = current_method
                else:
                    timing_method = next(
                        (
                            method
                            for method in METHODS
                            if "partition_time_sec" not in timing_by_method[method]
                        ),
                        "basic",
                    )
                timing_by_method[timing_method]["partition_time_sec"] = 0.0
                timing_by_method[timing_method][
                    "partition_kind"
                ] = "SKIPPED_ALREADY_BIPARTITE"
                continue
            match = RE_INITIALIZATION.search(line)
            if match:
                target = timing_method
                if target is None and current is not None:
                    target = str(current["method"])
                if target in timing_by_method:
                    timing_by_method[target]["initialization_time_sec"] = float(match.group(1))
                continue

            if current is None:
                continue

            match = RE_STATUS.match(line)
            if match:
                current["termination_status"] = match.group(1)
                continue
            match = RE_TOTAL_TIME.match(line)
            if match:
                current["admm_time_sec"] = float(match.group(1))
                continue
            match = RE_STOP_ITER.match(line)
            if match:
                current["stop_iter"] = int(match.group(1))
                continue
            match = RE_PROGRESS.match(line)
            if match:
                # Unlike the historical parser, retain the latest progress row.
                current["latest_progress_iter"] = int(match.group(1))

    if current is not None:
        methods.append(current)

    method_order = [str(item["method"]) for item in methods]
    if method_order != list(METHODS):
        errors.append(
            f"{source_log}: expected method order {list(METHODS)}, parsed {method_order}"
        )

    rows: list[dict[str, object]] = []
    for item in methods:
        method = str(item["method"])
        timing = timing_by_method.get(method, {})
        if item["partition_time_sec"] is None and "partition_time_sec" in timing:
            item["partition_time_sec"] = timing["partition_time_sec"]
        if not item["partition_kind"] and "partition_kind" in timing:
            item["partition_kind"] = timing["partition_kind"]
        if item["initialization_time_sec"] is None and "initialization_time_sec" in timing:
            item["initialization_time_sec"] = timing["initialization_time_sec"]

        warnings = list(item["parse_warnings"])
        stop_iter = item["stop_iter"]
        latest = item["latest_progress_iter"]
        if stop_iter is not None:
            iterations = stop_iter
            iteration_source = "stop_iter"
        else:
            iterations = latest
            iteration_source = "latest_progress"
            if latest is not None:
                warnings.append("Stop. Iter missing; used latest progress-table iteration")

        required = {
            "partition_time_sec": item["partition_time_sec"],
            "initialization_time_sec": item["initialization_time_sec"],
            "admm_time_sec": item["admm_time_sec"],
            "termination_status": item["termination_status"],
            "iterations": iterations,
        }
        for field, value in required.items():
            if value is None:
                errors.append(f"{source_log}/{method}: missing {field}")

        status = item["termination_status"]
        if status is not None and status not in VALID_STATUSES:
            errors.append(f"{source_log}/{method}: unrecognized status {status}")

        partition = item["partition_time_sec"]
        admm_time = item["admm_time_sec"]
        total_time = None
        if partition is not None and admm_time is not None:
            total_time = float(partition) + float(admm_time)

        rows.append(
            {
                "nodes": nodes,
                "arcs": arcs,
                "seed": seed,
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "partition_time_sec": partition,
                "partition_kind": item["partition_kind"],
                "initialization_time_sec": item["initialization_time_sec"],
                "admm_time_sec": admm_time,
                "total_time_sec": total_time,
                "termination_status": status,
                "iterations": iterations,
                "iteration_source": iteration_source,
                "stop_iter": stop_iter,
                "latest_progress_iter": latest,
                "source_log": source_log,
                "parse_warnings": "; ".join(warnings),
            }
        )

    metadata = {"source_log": source_log, "nodes": nodes, "arcs": arcs, "seed": seed}
    if seed is None:
        errors.append(f"{source_log}: missing seed metadata")
    if nodes is None or arcs is None:
        errors.append(f"{source_log}: missing instance metadata")
    return rows, metadata, errors


def _source_name(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def parse_logs(
    paths: Sequence[Path], root: Path
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    metadata: list[dict[str, object]] = []
    errors: list[str] = []
    for path in sorted(paths):
        parsed, record, log_errors = parse_log(path, _source_name(path, root))
        rows.extend(parsed)
        metadata.append(record)
        errors.extend(log_errors)
    rows.sort(
        key=lambda row: (
            int(row["nodes"]) if row["nodes"] is not None else -1,
            int(row["seed"]) if row["seed"] is not None else -1,
            METHODS.index(str(row["method"])) if row["method"] in METHODS else len(METHODS),
        )
    )
    return rows, metadata, errors


def expected_pairs(mode: str) -> set[tuple[int, int]] | None:
    if mode in ("archived", "full"):
        return {(nodes, seed) for nodes in PAPER_NODES for seed in PAPER_SEEDS}
    if mode == "smoke":
        return {(200, 1)}
    return None


def validate_grid(
    rows: Sequence[Mapping[str, object]],
    metadata: Sequence[Mapping[str, object]],
    mode: str,
) -> tuple[dict[str, object], list[str]]:
    errors: list[str] = []
    pair_logs: defaultdict[tuple[int, int], list[str]] = defaultdict(list)
    for record in metadata:
        if record["nodes"] is None or record["seed"] is None:
            continue
        pair = (int(record["nodes"]), int(record["seed"]))
        pair_logs[pair].append(str(record["source_log"]))
        if record["arcs"] != PAPER_ARCS:
            errors.append(
                f"{record['source_log']}: expected {PAPER_ARCS} arcs, parsed {record['arcs']}"
            )

    duplicates = {pair: logs for pair, logs in pair_logs.items() if len(logs) != 1}
    for pair, logs in sorted(duplicates.items()):
        errors.append(f"duplicate logs for nodes={pair[0]}, seed={pair[1]}: {logs}")

    rows_by_pair: defaultdict[tuple[int, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        if row["nodes"] is None or row["seed"] is None:
            continue
        rows_by_pair[(int(row["nodes"]), int(row["seed"]))].append(row)
    for pair, pair_rows in sorted(rows_by_pair.items()):
        methods = [str(row["method"]) for row in pair_rows]
        if methods != list(METHODS):
            errors.append(
                f"nodes={pair[0]}, seed={pair[1]}: expected methods {list(METHODS)}, got {methods}"
            )

    expected = expected_pairs(mode)
    observed = set(pair_logs)
    paper_pairs = {(nodes, seed) for nodes in PAPER_NODES for seed in PAPER_SEEDS}
    nonpaper = sorted(observed.difference(paper_pairs))
    if mode == "parse" and nonpaper:
        errors.append(
            "non-paper jobs in parse input: "
            + ", ".join(f"nodes={nodes}/seed={seed}" for nodes, seed in nonpaper)
        )
    missing = sorted(expected - observed) if expected is not None else []
    unexpected = sorted(observed - expected) if expected is not None else []
    if missing:
        errors.append(
            "missing jobs: " + ", ".join(f"nodes={nodes}/seed={seed}" for nodes, seed in missing)
        )
    if unexpected:
        errors.append(
            "unexpected jobs: "
            + ", ".join(f"nodes={nodes}/seed={seed}" for nodes, seed in unexpected)
        )

    report = {
        "mode": mode,
        "expected_job_count": len(expected) if expected is not None else None,
        "observed_log_count": len(metadata),
        "observed_job_count": len(observed),
        "parsed_method_rows": len(rows),
        "observed_pairs": [
            {"nodes": nodes, "seed": seed} for nodes, seed in sorted(observed)
        ],
        "missing_pairs": [{"nodes": nodes, "seed": seed} for nodes, seed in missing],
        "unexpected_pairs": [
            {"nodes": nodes, "seed": seed} for nodes, seed in unexpected
        ],
        "nonpaper_pairs": [
            {"nodes": nodes, "seed": seed} for nodes, seed in nonpaper
        ],
        "duplicate_pairs": [
            {"nodes": nodes, "seed": seed, "logs": logs}
            for (nodes, seed), logs in sorted(duplicates.items())
        ],
    }
    return report, errors


def _numeric(rows: Sequence[Mapping[str, object]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row[field]
        if value is None:
            raise ValueError(f"Cannot aggregate missing {field}")
        values.append(float(value))
    return values


def summarize(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: defaultdict[tuple[int, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        if row["nodes"] is None:
            raise ValueError("Cannot aggregate a row with missing nodes")
        grouped[(int(row["nodes"]), str(row["method"]))].append(row)

    summaries: list[dict[str, object]] = []
    for (nodes, method), group in sorted(
        grouped.items(), key=lambda item: (item[0][0], METHODS.index(item[0][1]))
    ):
        partition = _numeric(group, "partition_time_sec")
        initialization = _numeric(group, "initialization_time_sec")
        admm = _numeric(group, "admm_time_sec")
        total = _numeric(group, "total_time_sec")
        iterations = _numeric(group, "iterations")
        counts = Counter(str(row["termination_status"]) for row in group)
        statuses = "; ".join(f"{status}:{counts[status]}" for status in sorted(counts))
        arcs = {int(row["arcs"]) for row in group if row["arcs"] is not None}
        summaries.append(
            {
                "nodes": nodes,
                "arcs": next(iter(arcs)) if len(arcs) == 1 else "",
                "method": method,
                "method_label": METHOD_LABELS[method],
                "n_runs": len(group),
                "partition_time_mean": arithmetic_mean(partition),
                "partition_time_std": sample_std(partition),
                "initialization_time_mean": arithmetic_mean(initialization),
                "initialization_time_std": sample_std(initialization),
                "admm_time_mean": arithmetic_mean(admm),
                "admm_time_std": sample_std(admm),
                "total_time_mean": arithmetic_mean(total),
                "total_time_std": sample_std(total),
                "iterations_mean": arithmetic_mean(iterations),
                "iterations_std": sample_std(iterations),
                "status_counts": statuses,
            }
        )
    return summaries


def compare_reference(
    summaries: Sequence[Mapping[str, object]],
    mode: str,
    run_rows: Sequence[Mapping[str, object]] = (),
) -> tuple[dict[str, object], list[str]]:
    observed = {(int(row["nodes"]), str(row["method"])): row for row in summaries}
    comparison_rows: list[dict[str, object]] = []
    errors: list[str] = []
    timing_fields = (
        "partition_time_mean",
        "initialization_time_mean",
        "admm_time_mean",
        "total_time_mean",
    )

    for key in sorted(REFERENCE_MEANS, key=lambda item: (item[0], METHODS.index(item[1]))):
        reference = REFERENCE_MEANS[key]
        actual = observed.get(key)
        if actual is None:
            comparison_rows.append(
                {
                    "nodes": key[0],
                    "method": key[1],
                    "available": False,
                    "reference": reference,
                }
            )
            continue

        timing: dict[str, object] = {}
        for field in timing_fields:
            actual_value = float(actual[field])
            reference_value = float(reference[field])
            delta = actual_value - reference_value
            timing[field] = {
                "observed": actual_value,
                "reference": reference_value,
                "delta": delta,
                "relative_delta": delta / reference_value if reference_value else None,
                "enforced": False,
            }

        observed_iterations = float(actual["iterations_mean"])
        reference_iterations = float(reference["iterations_mean"])
        iteration_exact = observed_iterations == reference_iterations
        statuses_exact = str(actual["status_counts"]) == str(reference["status_counts"])
        runs_exact = int(actual["n_runs"]) == 10
        enforce_aggregate = mode == "archived"
        comparison_rows.append(
            {
                "nodes": key[0],
                "method": key[1],
                "available": True,
                "n_runs": {"observed": int(actual["n_runs"]), "reference": 10, "exact": runs_exact},
                "iterations_mean": {
                    "observed": observed_iterations,
                    "reference": reference_iterations,
                    "delta": observed_iterations - reference_iterations,
                    "exact": iteration_exact,
                    "enforced": enforce_aggregate,
                },
                "status_counts": {
                    "observed": actual["status_counts"],
                    "reference": reference["status_counts"],
                    "exact": statuses_exact,
                    "enforced": enforce_aggregate,
                },
                "timing": timing,
            }
        )
        if enforce_aggregate and not iteration_exact:
            errors.append(
                f"archived iteration mean mismatch for nodes={key[0]}/{key[1]}: "
                f"expected {reference_iterations}, got {observed_iterations}"
            )
        if enforce_aggregate and not statuses_exact:
            errors.append(
                f"archived status-count mismatch for nodes={key[0]}/{key[1]}: "
                f"expected {reference['status_counts']}, got {actual['status_counts']}"
            )

    missing_groups = [
        {"nodes": nodes, "method": method}
        for nodes, method in sorted(
            set(REFERENCE_MEANS) - set(observed), key=lambda item: (item[0], METHODS.index(item[1]))
        )
    ]
    smoke_entries: list[dict[str, object]] = []
    if mode == "smoke":
        raw_index = {
            (int(row["nodes"]), int(row["seed"]), str(row["method"])): row
            for row in run_rows
            if row.get("nodes") is not None and row.get("seed") is not None
        }
        for method in METHODS:
            actual = raw_index.get((200, 1, method))
            reference = REFERENCE_SMOKE_N200_SEED1[method]
            if actual is None:
                errors.append(f"missing smoke reference row nodes=200/seed=1/{method}")
                continue
            observed_iterations = int(actual["iterations"])
            observed_status = str(actual["termination_status"])
            iterations_exact = observed_iterations == int(reference["iterations"])
            status_exact = observed_status == str(reference["termination_status"])
            smoke_entries.append(
                {
                    "nodes": 200,
                    "seed": 1,
                    "method": method,
                    "observed_iterations": observed_iterations,
                    "reference_iterations": reference["iterations"],
                    "iterations_exact": iterations_exact,
                    "observed_status": observed_status,
                    "reference_status": reference["termination_status"],
                    "status_exact": status_exact,
                    "observed_timing_seconds": {
                        field: actual.get(field)
                        for field in (
                            "partition_time_sec",
                            "initialization_time_sec",
                            "admm_time_sec",
                        )
                    },
                    "reference_timing_seconds": {
                        field: reference[field]
                        for field in (
                            "partition_time_sec",
                            "initialization_time_sec",
                            "admm_time_sec",
                        )
                    },
                    "timing_enforced": False,
                }
            )
            if not iterations_exact:
                errors.append(
                    f"smoke iteration mismatch for nodes=200/seed=1/{method}: "
                    f"expected {reference['iterations']}, got {observed_iterations}"
                )
            if not status_exact:
                errors.append(
                    f"smoke status mismatch for nodes=200/seed=1/{method}: "
                    f"expected {reference['termination_status']}, got {observed_status}"
                )

    report = {
        "reference_source": (
            "experiments_logs/3-2-networkflow/admm_summary/admm_folder_summary.csv"
        ),
        "mode": mode,
        "timing_policy": "informational only; wall time is hardware and environment dependent",
        "iteration_policy": (
            "exact archived aggregate check"
            if mode == "archived"
            else "exact raw seed-1 check in smoke; aggregate values otherwise informational because the full archive contains a hardware-censored time-limit run"
        ),
        "missing_reference_groups": missing_groups,
        "complete_reference_grid": not missing_groups,
        "rows": comparison_rows,
        "smoke_reference": {
            "source": (
                "experiments_logs.zip Section 3.2 raw job 40148228/50464369 "
                "(nodes=200, seed=1)"
            ),
            "entries": smoke_entries,
            "all_iteration_and_status_values_match": bool(smoke_entries)
            and len(smoke_entries) == len(METHODS)
            and all(
                entry["iterations_exact"] and entry["status_exact"]
                for entry in smoke_entries
            ),
        }
        if mode == "smoke"
        else None,
    }
    return report, errors


def _complete_plot_nodes(summaries: Sequence[Mapping[str, object]]) -> list[int]:
    methods_by_node: defaultdict[int, set[str]] = defaultdict(set)
    for row in summaries:
        methods_by_node[int(row["nodes"])].add(str(row["method"]))
    return sorted(nodes for nodes, methods in methods_by_node.items() if methods == set(METHODS))


def _configure_axis(axis, nodes: Sequence[int], ylabel: str, *, title: str | None = None) -> None:
    axis.set_xticks(list(range(len(nodes))), [str(node) for node in nodes])
    axis.set_xlabel("Number of nodes")
    axis.set_ylabel(ylabel)
    axis.set_ylim(0.0, 1.08)
    if title:
        axis.set_title(title)


def _draw_iterations(axis, summaries: Sequence[Mapping[str, object]], nodes: Sequence[int]) -> None:
    lookup = {(int(row["nodes"]), str(row["method"])): row for row in summaries}
    maxima = {
        node: max(float(lookup[(node, method)]["iterations_mean"]) for method in METHODS)
        for node in nodes
    }
    width = 0.25
    for index, method in enumerate(METHODS):
        positions = [x + (index - 1) * width for x in range(len(nodes))]
        values = [float(lookup[(node, method)]["iterations_mean"]) / maxima[node] for node in nodes]
        axis.bar(
            positions,
            values,
            width=width,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
        )
    _configure_axis(axis, nodes, "Normalized ADMM Iteration")
    axis.legend(ncol=3, frameon=True)


def _draw_time(axis, summaries: Sequence[Mapping[str, object]], nodes: Sequence[int]) -> None:
    lookup = {(int(row["nodes"]), str(row["method"])): row for row in summaries}
    maxima = {
        node: max(float(lookup[(node, method)]["total_time_mean"]) for method in METHODS)
        for node in nodes
    }
    width = 0.25
    for index, method in enumerate(METHODS):
        positions = [x + (index - 1) * width for x in range(len(nodes))]
        partition = [
            float(lookup[(node, method)]["partition_time_mean"]) / maxima[node] for node in nodes
        ]
        admm = [float(lookup[(node, method)]["admm_time_mean"]) / maxima[node] for node in nodes]
        axis.bar(
            positions,
            partition,
            width=width,
            color=METHOD_COLORS[method],
            alpha=0.35,
        )
        axis.bar(
            positions,
            admm,
            width=width,
            bottom=partition,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            alpha=0.9,
        )
    _configure_axis(axis, nodes, "Normalized Time")
    axis.legend(ncol=3, frameon=True)


def make_plots(summaries: Sequence[Mapping[str, object]], figure_dir: Path) -> list[Path]:
    nodes = _complete_plot_nodes(summaries)
    if not nodes:
        raise ValueError("No node size has all three methods; cannot plot Figure 11")
    pyplot = get_pyplot()
    try:
        pyplot.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pyplot.style.use("default")

    outputs: list[Path] = []

    figure, axis = pyplot.subplots(figsize=(12, 5))
    _draw_iterations(axis, summaries, nodes)
    figure.tight_layout()
    for extension in ("png", "pdf"):
        path = figure_dir / f"figure_11a_iterations.{extension}"
        figure.savefig(path, dpi=200)
        outputs.append(path)
    pyplot.close(figure)

    figure, axis = pyplot.subplots(figsize=(12, 5))
    _draw_time(axis, summaries, nodes)
    figure.tight_layout()
    for extension in ("png", "pdf"):
        path = figure_dir / f"figure_11b_total_time.{extension}"
        figure.savefig(path, dpi=200)
        outputs.append(path)
    pyplot.close(figure)

    figure, axes = pyplot.subplots(1, 2, figsize=(14, 5))
    _draw_iterations(axes[0], summaries, nodes)
    axes[0].set_title("(a) Iteration")
    _draw_time(axes[1], summaries, nodes)
    axes[1].set_title("(b) Total Time")
    figure.tight_layout()
    for extension in ("png", "pdf"):
        path = figure_dir / f"figure_11.{extension}"
        figure.savefig(path, dpi=200)
        outputs.append(path)
    pyplot.close(figure)
    return outputs


def build_jobs(args: argparse.Namespace) -> list[Job]:
    if not JULIA_RUNNER.is_file():
        raise SystemExit(f"Existing network-flow runner not found: {JULIA_RUNNER}")
    pairs = [(200, 1)] if args.mode == "smoke" else [
        (nodes, seed) for nodes in PAPER_NODES for seed in PAPER_SEEDS
    ]
    jobs: list[Job] = []
    for nodes, seed in pairs:
        command = julia_command(
            args.julia,
            args.threads,
            JULIA_RUNNER,
            "--solver",
            "original",
            "--maxIter",
            100000,
            "--initialRho",
            1.0,
            "--timeLimit",
            3600.0,
            "--seed",
            seed,
            "--logInterval",
            1000,
            "--random",
            nodes,
            PAPER_ARCS,
        )
        jobs.append(Job(f"nodes_{nodes:04d}/seed_{seed:02d}", command))
    return jobs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce Section 3.2 / Figure 11 from archived logs, existing logs, "
            "or a fresh network-flow sweep."
        )
    )
    add_common_arguments(
        parser,
        default_output=REPO_ROOT / "reproduction" / "output" / "section_3_2",
    )
    return parser


def _job_result_payload(results) -> list[dict[str, object]]:
    return [
        {
            "name": result.name,
            "command": list(result.command),
            "log_path": result.log_path,
            "returncode": result.returncode,
            "elapsed_seconds": result.elapsed_seconds,
            "skipped": result.skipped,
        }
        for result in results
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_common_arguments(args)
    output = prepare_mode_output(args.output, args.mode)
    jobs: list[Job] = []
    results = []
    errors: list[str] = []
    warnings: list[str] = []

    if args.mode in ("smoke", "full"):
        jobs = build_jobs(args)
        results = run_jobs(
            jobs,
            log_root=output / "raw",
            cwd=legacy_workdir(),
            workers=args.jobs,
            resume=not args.no_resume,
        )
        write_json(output / "job_results.json", _job_result_payload(results))
        for result in failed_jobs(results):
            errors.append(f"job {result.name} exited with status {result.returncode}")

    fresh_log_root = output / "raw"
    with source_logs(args, ARCHIVE_SECTION, fresh_log_root) as log_root:
        logs = collect_logs(log_root)
        if args.mode in ("smoke", "full"):
            selected = {Path(result.log_path).resolve() for result in results}
            logs = [path for path in logs if path.resolve() in selected]
        if not logs:
            errors.append(f"no logs found below {log_root}")
        rows, metadata, parse_errors = parse_logs(logs, log_root)
        errors.extend(parse_errors)
        grid_report, grid_errors = validate_grid(rows, metadata, args.mode)
        errors.extend(grid_errors)

    write_csv(output / "runs.csv", RUN_FIELDS, rows)

    summaries: list[dict[str, object]] = []
    if rows and not parse_errors:
        try:
            summaries = summarize(rows)
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"could not aggregate parsed rows: {error}")
    write_csv(output / "summary.csv", SUMMARY_FIELDS, summaries)

    comparison, comparison_errors = compare_reference(summaries, args.mode, rows)
    errors.extend(comparison_errors)
    write_json(output / "reference_comparison.json", comparison)

    figures: list[Path] = []
    if not args.no_plots and summaries and not errors:
        try:
            figures = make_plots(summaries, output / "figures")
        except ValueError as error:
            errors.append(str(error))

    grid_report["errors"] = errors
    grid_report["warnings"] = warnings
    grid_report["valid"] = not errors
    write_json(output / "validation.json", grid_report)

    inputs: list[Path]
    if args.mode == "archived":
        inputs = [args.archive]
    elif args.mode == "parse":
        inputs = [args.logs]
    else:
        inputs = [JULIA_RUNNER]
    write_provenance(
        output,
        section="3.2 Linear Program: The Network Flow Problem / Figure 11",
        args=args,
        jobs=jobs,
        inputs=inputs,
        notes=(
            "Full mode is the exact 50-job paper sweep: nodes 200:100:600, 2000 arcs, seeds 1:10.",
            "Each job runs Basic, BFS, and MILP in that order on one shared generated instance.",
            "Paper settings: original ADMM, rho=1, maxIter=100000, LInf tolerances 1e-4, and a 3600-second per-method limit.",
            "Figure total time is partition time plus ADMM iteration-loop time; initialization is reported but excluded.",
            "The archive contains one Basic time-limit result at nodes=500, seed=6, and includes it in the mean.",
            "Archived measured work totals about 15.9 hours sequentially on warm 16-thread jobs; allow roughly 16 hours plus setup on comparable hardware.",
            "Wall-clock comparisons are informational. Julia/package versions and hardware can materially change timing.",
        ),
    )

    print(f"Wrote {output / 'runs.csv'} ({len(rows)} method rows)")
    print(f"Wrote {output / 'summary.csv'} ({len(summaries)} aggregate rows)")
    if figures:
        print(f"Wrote {len(figures)} Figure 11 files below {output / 'figures'}")
    if warnings:
        for warning in warnings:
            print(f"warning: {warning}", file=sys.stderr)
    if errors:
        print("Section 3.2 reproduction failed validation:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
