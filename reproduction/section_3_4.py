#!/usr/bin/env python3
"""Reproduce Section 3.4, Table 1, Figure 15, and Figure 16.

The default ``archived`` mode rebuilds every artifact from
``experiments_logs.zip`` without running Julia.  ``smoke`` and ``full`` launch
the add-only Julia driver in ``reproduction/julia/section_3_4.jl``.  The latter
runs only the seven methods reported in the paper; unlike the historical
entry point, it does not spend time on the unreported 30%, 40%, and 50% MILP
gaps.

For numerical fidelity this reproduction intentionally retains the archived
``+2*A'*b`` quadratic linear term.  Under PDMO's ``x'Qx + q'x + r`` convention
that is ``||A*x+b||^2`` even though the paper writes ``||A*x-b||^2``.  The
discrepancy is recorded in the console output and provenance file.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

try:
    from .common import REPO_ROOT, REPRODUCTION_ROOT
except ImportError:
    from common import REPO_ROOT, REPRODUCTION_ROOT  # type: ignore


SECTION_ARCHIVE_PATH = "3-4-distributed"
JULIA_DRIVER = REPRODUCTION_ROOT / "julia" / "section_3_4.jl"
DEFAULT_OUTPUT = REPRODUCTION_ROOT / "output" / "section_3_4"
GNN_MODEL = REPO_ROOT / "advanced" / "src" / "gnn" / "GNN" / "model_weights_10.pth"

PAPER_NODE_COUNTS = (50, 100, 200)
PAPER_SOLVERS = ("original", "doubly")
PAPER_GAPS = (0.01, 0.05, 0.1, 0.2)
PAPER_METHODS = (
    "basic",
    "bfs",
    "milp_0.01",
    "milp_0.05",
    "milp_0.1",
    "milp_0.2",
    "gnn",
)
METHOD_DISPLAY = {
    "basic": "Basic",
    "bfs": "BFS",
    "milp_0.01": "MILP(1%)",
    "milp_0.05": "MILP(5%)",
    "milp_0.1": "MILP(10%)",
    "milp_0.2": "MILP(20%)",
    "gnn": "GNN",
}
METHOD_GAP = {
    "milp_0.01": 0.01,
    "milp_0.05": 0.05,
    "milp_0.1": 0.1,
    "milp_0.2": 0.2,
}
JULIA_METHOD_TOKEN = {
    "basic": "basic",
    "bfs": "bfs",
    "milp_0.01": "milp-0.01",
    "milp_0.05": "milp-0.05",
    "milp_0.1": "milp-0.1",
    "milp_0.2": "milp-0.2",
    "gnn": "gnn",
}

METHOD_COLORS = {
    "basic": "#2980b9",
    "bfs": "#f39c12",
    "milp_0.01": "#27ae60",
    "milp_0.05": "#27ae60",
    "milp_0.1": "#27ae60",
    "milp_0.2": "#27ae60",
    "gnn": "#9467bd",
}
METHOD_LIGHT_COLORS = {
    "basic": "#7fb3d5",
    "bfs": "#f8c471",
    "milp_0.01": "#7ddc98",
    "milp_0.05": "#7ddc98",
    "milp_0.1": "#7ddc98",
    "milp_0.2": "#7ddc98",
    "gnn": "#c5b0d5",
}

OBJECTIVE_FIDELITY_WARNING = (
    "Fidelity warning: the archived/current generator uses QuadraticFunction(A'A, "
    "+2*A'b, b'b), which is ||A*x+b||^2 under PDMO's convention; the paper "
    "states ||A*x-b||^2. This runner intentionally preserves the archived sign."
)
FULL_RUNTIME_NOTE = (
    "Archive-based estimate: the exact seven-method, five-seed sweep requires "
    "about 64.8 algorithm-hours when run sequentially with 16 Julia threads per "
    "process (plus compilation/setup); the slowest isolated job was about 10.2 h."
)

# Paper-displayed Table 1 values.  The comparison tolerance accounts for the
# table's two-decimal display and integer rounding of average edge counts.
REFERENCE_TABLE = {
    (50, "basic"): (139.0, 2.91, 0.38, 0.00),
    (50, "bfs"): (139.0, 3.73, 0.83, 0.00),
    (50, "milp_0.01"): (139.0, 4.16, 0.89, 16.42),
    (50, "milp_0.05"): (139.0, 4.14, 0.89, 12.87),
    (50, "milp_0.1"): (139.0, 4.10, 0.88, 9.37),
    (50, "milp_0.2"): (139.0, 4.02, 0.88, 3.68),
    (50, "gnn"): (139.0, 4.06, 0.90, 6.27),
    (100, "basic"): (282.0, 2.92, 0.37, 0.00),
    (100, "bfs"): (282.0, 3.77, 0.74, 0.00),
    (100, "milp_0.01"): (282.0, 4.21, 0.93, 54.57),
    (100, "milp_0.05"): (282.0, 4.20, 0.90, 45.52),
    (100, "milp_0.1"): (282.0, 4.17, 0.93, 41.06),
    (100, "milp_0.2"): (282.0, 4.12, 0.92, 22.26),
    (100, "gnn"): (282.0, 4.14, 0.93, 7.51),
    (200, "basic"): (567.0, 2.93, 0.37, 0.00),
    (200, "bfs"): (567.0, 3.79, 0.71, 0.00),
    (200, "milp_0.01"): (567.0, 4.17, 0.94, 62.39),
    (200, "milp_0.05"): (567.0, 4.17, 0.95, 59.65),
    (200, "milp_0.1"): (567.0, 4.18, 0.97, 54.49),
    (200, "milp_0.2"): (567.0, 4.15, 0.91, 48.27),
    (200, "gnn"): (567.0, 4.18, 0.97, 10.78),
}

# Normalized bar heights extracted from the five archived paper seeds.  These
# are the direct numerical reference for Figures 15 and 16.
REFERENCE_FIGURES = {
    ("original", 50, "basic"): (1.000000, 1.000000),
    ("original", 50, "bfs"): (0.604077, 0.572547),
    ("original", 50, "milp_0.01"): (0.580711, 0.752787),
    ("original", 50, "milp_0.05"): (0.577418, 0.707913),
    ("original", 50, "milp_0.1"): (0.574072, 0.658723),
    ("original", 50, "milp_0.2"): (0.588866, 0.606473),
    ("original", 50, "gnn"): (0.590016, 0.635204),
    ("original", 100, "basic"): (1.000000, 1.000000),
    ("original", 100, "bfs"): (0.612944, 0.581008),
    ("original", 100, "milp_0.01"): (0.591162, 0.957934),
    ("original", 100, "milp_0.05"): (0.605551, 0.904835),
    ("original", 100, "milp_0.1"): (0.603048, 0.868058),
    ("original", 100, "milp_0.2"): (0.591788, 0.720964),
    ("original", 100, "gnn"): (0.600318, 0.640003),
    ("original", 200, "basic"): (1.000000, 1.000000),
    ("original", 200, "bfs"): (0.588900, 0.562937),
    ("original", 200, "milp_0.01"): (0.568608, 0.698754),
    ("original", 200, "milp_0.05"): (0.568685, 0.685389),
    ("original", 200, "milp_0.1"): (0.568023, 0.677586),
    ("original", 200, "milp_0.2"): (0.625278, 0.733421),
    ("original", 200, "gnn"): (0.565803, 0.655928),
    ("doubly", 50, "basic"): (1.000000, 1.000000),
    ("doubly", 50, "bfs"): (0.687358, 0.729744),
    ("doubly", 50, "milp_0.01"): (0.687406, 0.716931),
    ("doubly", 50, "milp_0.05"): (0.648577, 0.680494),
    ("doubly", 50, "milp_0.1"): (0.663592, 0.694802),
    ("doubly", 50, "milp_0.2"): (0.660547, 0.694922),
    ("doubly", 50, "gnn"): (0.700024, 0.725360),
    ("doubly", 100, "basic"): (1.000000, 1.000000),
    ("doubly", 100, "bfs"): (0.593121, 0.605332),
    ("doubly", 100, "milp_0.01"): (0.624589, 0.654689),
    ("doubly", 100, "milp_0.05"): (0.636885, 0.671651),
    ("doubly", 100, "milp_0.1"): (0.634600, 0.669330),
    ("doubly", 100, "milp_0.2"): (0.599637, 0.621822),
    ("doubly", 100, "gnn"): (0.623337, 0.653282),
    ("doubly", 200, "basic"): (1.000000, 1.000000),
    ("doubly", 200, "bfs"): (0.633315, 0.652665),
    ("doubly", 200, "milp_0.01"): (0.735293, 0.768564),
    ("doubly", 200, "milp_0.05"): (0.716950, 0.753618),
    ("doubly", 200, "milp_0.1"): (0.729209, 0.752472),
    ("doubly", 200, "milp_0.2"): (0.744286, 0.765391),
    ("doubly", 200, "gnn"): (0.730231, 0.810293),
}

SMOKE_REFERENCE = {
    "basic": {
        "iterations": 4350.0,
        "partition_time_seconds": 0.0,
        "admm_time_seconds": 92.17,
        "graph_nodes": 182.0,
        "graph_edges": 264.0,
    },
    "bfs": {
        "iterations": 2658.0,
        "partition_time_seconds": 0.0,
        "admm_time_seconds": 53.40,
        "graph_nodes": 105.0,
        "graph_edges": 187.0,
    },
    "milp_0.01": {
        "iterations": 2443.0,
        "partition_time_seconds": 15.518,
        "admm_time_seconds": 48.39,
        "graph_nodes": 79.0,
        "graph_edges": 161.0,
    },
    "milp_0.05": {
        "iterations": 2442.0,
        "partition_time_seconds": 12.029,
        "admm_time_seconds": 48.69,
        "graph_nodes": 80.0,
        "graph_edges": 162.0,
    },
    "milp_0.1": {
        "iterations": 2444.0,
        "partition_time_seconds": 7.452,
        "admm_time_seconds": 48.26,
        "graph_nodes": 81.0,
        "graph_edges": 163.0,
    },
    "milp_0.2": {
        "iterations": 2449.0,
        "partition_time_seconds": 3.472,
        "admm_time_seconds": 49.37,
        "graph_nodes": 88.0,
        "graph_edges": 170.0,
    },
    "gnn": {
        "iterations": 2443.0,
        "partition_time_seconds": 6.479,
        "admm_time_seconds": 48.69,
        "graph_nodes": 81.0,
        "graph_edges": 163.0,
    },
}

NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
METHOD_MARKER_RE = re.compile(
    rf"Solving\s+classic\s+distributed\s+opt\s+problem|"
    rf"Solving\s+distributed\s+opt\s+problem\s+with\s+BFS\s+bipartization|"
    rf"Solving\s+distributed\s+opt\s+problem\s+with\s+MILP\s+bipartization\s+gap\s*=\s*({NUMBER_PATTERN})|"
    rf"Solving\s+distributed\s+opt\s+problem\s+with\s+(GNN(?:-[A-Za-z0-9_]+)?)\s+bipartization",
    re.IGNORECASE,
)

RUN_FIELDS = (
    "source_log",
    "archive_batch",
    "run_id",
    "number_nodes",
    "n",
    "m",
    "solver",
    "seed",
    "kappa",
    "initial_rho",
    "max_iter",
    "log_interval",
    "threads",
    "method",
    "method_display",
    "mip_gap",
    "mip_status",
    "mip_achieved_gap_percent",
    "mip_solver_time_seconds",
    "partition_time_seconds",
    "graph_partition_time_seconds",
    "graph_nodes",
    "partition_left",
    "partition_right",
    "graph_edges",
    "average_degree",
    "balance",
    "admm_status",
    "iterations",
    "admm_time_seconds",
    "admm_initialization_seconds",
    "objective",
    "primal_residual_l2",
    "primal_residual_linf",
    "dual_residual_l2",
    "dual_residual_linf",
    "complete",
    "fidelity_warning",
)


def optional_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"unknown", "nan"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def optional_int(value: object) -> int | None:
    number = optional_float(value)
    return int(number) if number is not None else None


def first_match(text: str, pattern: str, *, flags: int = 0) -> str | None:
    match = re.search(pattern, text, flags)
    return match.group(1).strip() if match else None


def canonical_gap(value: float) -> tuple[str, float] | None:
    for gap in PAPER_GAPS:
        if abs(value - gap) <= 1e-9:
            return f"milp_{gap}", gap
    return None


def canonical_method(label: str) -> tuple[str, float | None] | None:
    compact = re.sub(r"\s+", "", label).lower()
    if compact in {"classic", "basic"}:
        return "basic", None
    if compact == "bfs":
        return "bfs", None
    if compact.startswith("gnn"):
        return "gnn", None
    match = re.search(r"milp\(([^)]+)\)", compact)
    if match:
        numeric = re.match(NUMBER_PATTERN, match.group(1))
        if numeric:
            parsed = canonical_gap(float(numeric.group(0)))
            if parsed:
                return parsed
    return None


def marker_method(match: re.Match[str]) -> tuple[str, float | None] | None:
    text = match.group(0).lower()
    if "classic" in text:
        return "basic", None
    if " bfs " in f" {text} ":
        return "bfs", None
    if "milp" in text:
        gap_text = match.group(1)
        return canonical_gap(float(gap_text)) if gap_text else None
    if "gnn" in text:
        return "gnn", None
    return None


def parse_metadata(text: str, path: Path) -> dict[str, object]:
    archive_batch = ""
    for parent in path.parents:
        if re.fullmatch(r"(?:admm|flip)_\d+_\d+", parent.name):
            archive_batch = parent.name
            break
    run_id = path.parent.name if path.name == "stdout.log" else path.stem

    solver = first_match(text, r"^\s*solver\s*=\s*([^\s]+)\s*$", flags=re.MULTILINE)
    if solver is None and archive_batch:
        solver = "doubly" if archive_batch.startswith("flip_") else "original"

    threads = first_match(text, r"Run Bipartite ADMM with threads\s*=\s*(\d+)")
    return {
        "source_log": str(path.resolve()),
        "archive_batch": archive_batch,
        "run_id": run_id,
        "number_nodes": optional_int(
            first_match(text, r"^\s*numberNodes\s*=\s*(\d+)\s*$", flags=re.MULTILINE)
        ),
        "n": optional_int(first_match(text, r"^\s*n\s*=\s*(\d+)\s*$", flags=re.MULTILINE)),
        "m": optional_int(first_match(text, r"^\s*m\s*=\s*(\d+)\s*$", flags=re.MULTILINE)),
        "solver": solver.lower() if solver else None,
        "seed": optional_int(first_match(text, r"^\s*seed\s*=\s*(\d+)\s*$", flags=re.MULTILINE)),
        "kappa": optional_float(
            first_match(text, rf"^\s*kappa\s*=\s*({NUMBER_PATTERN})\s*$", flags=re.MULTILINE)
        ),
        "initial_rho": optional_float(
            first_match(text, rf"^\s*initialRho\s*=\s*({NUMBER_PATTERN})\s*$", flags=re.MULTILINE)
        ),
        "max_iter": optional_int(
            first_match(text, r"^\s*maxIter\s*=\s*(\d+)\s*$", flags=re.MULTILINE)
        ),
        "log_interval": optional_int(
            first_match(text, r"^\s*logInterval\s*=\s*(\d+)\s*$", flags=re.MULTILINE)
        ),
        "threads": optional_int(threads),
    }


def parse_summary_table(text: str) -> dict[str, dict[str, float]]:
    marker = text.rfind("SUMMARY OF RESULTS")
    if marker < 0:
        return {}
    output: dict[str, dict[str, float]] = {}
    in_table = False
    for line in text[marker:].splitlines():
        if "Method" in line and "BipT" in line and "Iters" in line and "ADMM Time" in line:
            in_table = True
            continue
        if not in_table or "|" not in line:
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 5:
            continue
        canonical = canonical_method(parts[0])
        if canonical is None:
            continue
        method, _ = canonical
        bip_time = optional_float(parts[1])
        iterations = optional_float(parts[2])
        admm_time = optional_float(parts[3])
        objective = optional_float(parts[4])
        if bip_time is None or iterations is None or admm_time is None:
            continue
        output[method] = {
            "partition_time_seconds": bip_time,
            "iterations": iterations,
            "admm_time_seconds": admm_time,
            "objective": objective,
        }
    return output


def parse_graph_block(block: str) -> dict[str, object]:
    marker = block.rfind("Summary of ADMM Bipartitie Graph")
    if marker < 0:
        node_matches = list(re.finditer(r"Number of nodes\s*=\s*(\d+)", block))
        edge_matches = list(re.finditer(r"Number of edges\s*=\s*(\d+)", block))
        partition_matches = list(
            re.finditer(
                r"Par(?:t)?ition size[^=]*=\s*\((\d+)\s*,\s*(\d+)\)",
                block,
            )
        )
        if not node_matches and not edge_matches and not partition_matches:
            return {}
        # Julia's logger writes to stderr while the detailed summaries use
        # stdout. In a merged fresh log all logger headers may precede the
        # method banners, so the final graph values are the last occurrences
        # in the method's stdout block.
        nodes = int(node_matches[-1].group(1)) if node_matches else None
        edges = int(edge_matches[-1].group(1)) if edge_matches else None
        partition = partition_matches[-1] if partition_matches else None
    else:
        graph_text = block[marker:]
        nodes = optional_int(first_match(graph_text, r"Number of nodes\s*=\s*(\d+)"))
        edges = optional_int(first_match(graph_text, r"Number of edges\s*=\s*(\d+)"))
        partition = re.search(
            r"Par(?:t)?ition size[^=]*=\s*\((\d+)\s*,\s*(\d+)\)",
            graph_text,
        )
        # One archived merged stdout/stderr stream prints the graph values just
        # before the summary header. Backfill from the immediately preceding
        # bipartization block without disturbing the normal post-header format.
        if nodes is None or edges is None or partition is None:
            fallback_start = block.rfind("ADMMBipartiteGraph:", 0, marker)
            fallback = block[fallback_start:marker] if fallback_start >= 0 else ""
            if nodes is None:
                nodes = optional_int(first_match(fallback, r"Number of nodes\s*=\s*(\d+)"))
            if edges is None:
                edges = optional_int(first_match(fallback, r"Number of edges\s*=\s*(\d+)"))
            if partition is None:
                partition = re.search(
                    r"Par(?:t)?ition size[^=]*=\s*\((\d+)\s*,\s*(\d+)\)",
                    fallback,
                )

    left = int(partition.group(1)) if partition else None
    right = int(partition.group(2)) if partition else None
    average_degree = 2.0 * edges / nodes if nodes and edges is not None else None
    balance = min(left, right) / max(left, right) if left and right else None
    return {
        "graph_nodes": nodes,
        "partition_left": left,
        "partition_right": right,
        "graph_edges": edges,
        "average_degree": average_degree,
        "balance": balance,
    }


def parse_method_block(block: str, method: str, gap: float | None) -> dict[str, object]:
    partition_time = optional_float(
        first_match(block, rf"ADMMBipartiteGraph:.*?took\s+({NUMBER_PATTERN})\s+seconds")
    )
    if partition_time is None and "skip bipartization" in block.lower():
        partition_time = 0.0

    record: dict[str, object] = {
        "method": method,
        "method_display": METHOD_DISPLAY[method],
        "mip_gap": gap,
        "graph_partition_time_seconds": partition_time,
        "partition_time_seconds": partition_time,
        "admm_initialization_seconds": optional_float(
            first_match(block, rf"ADMM: initialization took\s+({NUMBER_PATTERN})\s+seconds")
        ),
        "admm_status": first_match(block, r"Solver Status\s*=\s*([^\s]+)"),
        "iterations": optional_int(first_match(block, r"Stop\. Iter\s*=\s*(\d+)")),
        "admm_time_seconds": optional_float(
            first_match(block, rf"Total Time\s*=\s*({NUMBER_PATTERN})")
        ),
        "objective": optional_float(first_match(block, rf"Objective\s*=\s*({NUMBER_PATTERN})")),
        "primal_residual_l2": optional_float(
            first_match(block, rf"Pres \(L2\)\s*=\s*({NUMBER_PATTERN})")
        ),
        "primal_residual_linf": optional_float(
            first_match(block, rf"Pres \(LInf\)\s*=\s*({NUMBER_PATTERN})")
        ),
        "dual_residual_l2": optional_float(
            first_match(block, rf"Dres \(L2\)\s*=\s*({NUMBER_PATTERN})")
        ),
        "dual_residual_linf": optional_float(
            first_match(block, rf"Dres \(LInf\)\s*=\s*({NUMBER_PATTERN})")
        ),
    }
    record.update(parse_graph_block(block))

    if method.startswith("milp_"):
        report_marker = block.rfind("Solving report")
        report = block[report_marker:] if report_marker >= 0 else block
        record["mip_status"] = first_match(report, r"Status\s+([^\n]+)")
        record["mip_achieved_gap_percent"] = optional_float(
            first_match(report, rf"Gap\s+({NUMBER_PATTERN})%\s*\(tolerance")
        )
        record["mip_solver_time_seconds"] = optional_float(
            first_match(report, rf"Timing\s+({NUMBER_PATTERN})\s+\(total\)")
        )
    else:
        record["mip_status"] = None
        record["mip_achieved_gap_percent"] = None
        record["mip_solver_time_seconds"] = None
    return record


def parse_log(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    metadata = parse_metadata(text, path)
    summary = parse_summary_table(text)
    marker_matches = list(METHOD_MARKER_RE.finditer(text))
    records: dict[str, dict[str, object]] = {}

    for index, match in enumerate(marker_matches):
        canonical = marker_method(match)
        if canonical is None:
            continue
        method, gap = canonical
        if method not in PAPER_METHODS:
            continue
        block_end = marker_matches[index + 1].start() if index + 1 < len(marker_matches) else len(text)
        block = text[match.end() : block_end]
        parsed = parse_method_block(block, method, gap)
        parsed.update(metadata)
        records[method] = parsed

    # The final table is the most precise archived representation of BipT and
    # is the source used by the original plotting script.  Use it to backfill
    # or refine performance values while retaining the two-decimal graph-log
    # time separately for Table 1.
    for method, table_values in summary.items():
        if method not in PAPER_METHODS:
            continue
        if method not in records:
            records[method] = {
                **metadata,
                "method": method,
                "method_display": METHOD_DISPLAY[method],
                "mip_gap": METHOD_GAP.get(method),
                "graph_partition_time_seconds": None,
            }
        records[method].update(table_values)

    output = []
    for method in PAPER_METHODS:
        record = records.get(method)
        if record is None:
            continue
        graph_complete = all(record.get(field) is not None for field in ("graph_nodes", "graph_edges"))
        performance_complete = all(
            record.get(field) is not None
            for field in ("partition_time_seconds", "iterations", "admm_time_seconds")
        )
        record["complete"] = bool(graph_complete and performance_complete and record.get("admm_status"))
        record["fidelity_warning"] = OBJECTIVE_FIDELITY_WARNING
        output.append(record)
    return output


def parse_method_spec(text: str) -> tuple[str, ...]:
    aliases = {
        "classic": "basic",
        "milp-0.01": "milp_0.01",
        "milp0.01": "milp_0.01",
        "milp-1%": "milp_0.01",
        "milp-0.05": "milp_0.05",
        "milp0.05": "milp_0.05",
        "milp-5%": "milp_0.05",
        "milp-0.1": "milp_0.1",
        "milp0.1": "milp_0.1",
        "milp-10%": "milp_0.1",
        "milp-0.2": "milp_0.2",
        "milp0.2": "milp_0.2",
        "milp-20%": "milp_0.2",
    }
    selected: set[str] = set()
    for raw in text.lower().split(","):
        token = raw.strip()
        if not token:
            continue
        if token in {"all", "paper"}:
            selected.update(PAPER_METHODS)
        elif token in {"milp", "milps"}:
            selected.update(method for method in PAPER_METHODS if method.startswith("milp_"))
        else:
            selected.add(aliases.get(token, token))
    unknown = selected.difference(PAPER_METHODS)
    if unknown:
        raise SystemExit(f"Unknown --smoke-methods token(s): {', '.join(sorted(unknown))}")
    if not selected:
        raise SystemExit("--smoke-methods must select at least one method")
    return tuple(method for method in PAPER_METHODS if method in selected)


if __name__ == "__main__":
    try:
        from .section_3_4_impl import main
    except ImportError:
        from section_3_4_impl import main
    raise SystemExit(main())
