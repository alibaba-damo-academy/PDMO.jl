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
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence
import zipfile

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
PAPER_THREADS = 16
ARCHIVE_OPTIONAL_SKIP_REASON = "archive_not_found_optional_for_full"
METHODS = ("basic", "bfs", "milp")
METHOD_LABELS = {"basic": "Basic", "bfs": "BFS", "milp": "MILP"}
METHOD_COLORS = {"basic": "#2980b9", "bfs": "#f39c12", "milp": "#27ae60"}
PAPER_SETTINGS: dict[str, object] = {
    "solver": "original",
    "maxIter": 100_000,
    "initialRho": 1.0,
    "timeLimit": 3600.0,
    "logInterval": 1_000,
}
PAPER_RUN_KEYS = frozenset(
    (nodes, seed, method)
    for nodes in PAPER_NODES
    for seed in PAPER_SEEDS
    for method in METHODS
)
PAPER_CENSORED_KEY = (500, 6, "basic")
ARCHIVE_RUNS_MEMBER = (
    f"experiments_logs/{ARCHIVE_SECTION}/admm_summary/admm_runs.csv"
)

# Exact scheduler provenance of the five scale batches retained for Figure 11.
# Tuple position is the paper seed (position 0 is seed 1).
ARCHIVE_SCALE_JOBS: dict[int, tuple[str, tuple[int, ...]]] = {
    200: ("40148228", tuple(range(50_464_369, 50_464_379))),
    300: ("40148239", tuple(range(50_464_389, 50_464_399))),
    400: ("40148240", tuple(range(50_464_399, 50_464_409))),
    500: ("40148242", tuple(range(50_464_410, 50_464_420))),
    600: ("40148264", tuple(range(50_464_441, 50_464_451))),
}
ARCHIVE_PAPER_JOBS: dict[tuple[int, int], tuple[str, int]] = {
    (nodes, seed): (scale_folder, run_ids[seed - 1])
    for nodes, (scale_folder, run_ids) in ARCHIVE_SCALE_JOBS.items()
    for seed in PAPER_SEEDS
}

# Independent raw-result fingerprint from the retained Figure 11 jobs. Values
# are ordered Basic/BFS/MILP. All statuses are optimal except PAPER_CENSORED_KEY.
ARCHIVE_ITERATIONS: dict[tuple[int, int], tuple[int, int, int]] = {
    (200, 1): (142, 11001, 8883),
    (200, 2): (215, 15695, 15044),
    (200, 3): (326, 22256, 17569),
    (200, 4): (62, 13381, 22850),
    (200, 5): (38, 14243, 11234),
    (200, 6): (51, 41581, 41486),
    (200, 7): (64, 11669, 10896),
    (200, 8): (25, 20304, 17041),
    (200, 9): (91, 19302, 12313),
    (200, 10): (34, 29763, 19320),
    (300, 1): (1380, 24467, 13672),
    (300, 2): (59, 11881, 9569),
    (300, 3): (2013, 16950, 14638),
    (300, 4): (68, 26048, 17022),
    (300, 5): (105, 13631, 11789),
    (300, 6): (137, 12295, 12378),
    (300, 7): (52, 21034, 14971),
    (300, 8): (652, 27147, 15226),
    (300, 9): (106, 10737, 11396),
    (300, 10): (107, 24180, 22188),
    (400, 1): (186, 28165, 14907),
    (400, 2): (106, 9651, 6596),
    (400, 3): (470, 10702, 7887),
    (400, 4): (292, 14326, 8749),
    (400, 5): (117, 19877, 13515),
    (400, 6): (58, 15237, 13107),
    (400, 7): (99, 13227, 9094),
    (400, 8): (78, 16485, 12904),
    (400, 9): (114, 20181, 21302),
    (400, 10): (76, 14877, 11473),
    (500, 1): (333, 28026, 20446),
    (500, 2): (1364, 15251, 18706),
    (500, 3): (248, 28405, 19291),
    (500, 4): (196, 22751, 20601),
    (500, 5): (86, 24194, 21659),
    (500, 6): (3112, 18479, 14404),
    (500, 7): (1772, 21387, 14961),
    (500, 8): (282, 26350, 17159),
    (500, 9): (465, 16473, 11463),
    (500, 10): (142, 24848, 13819),
    (600, 1): (68, 20834, 17374),
    (600, 2): (244, 17458, 12771),
    (600, 3): (1689, 46539, 14496),
    (600, 4): (188, 17728, 16303),
    (600, 5): (249, 12241, 11454),
    (600, 6): (254, 20297, 14627),
    (600, 7): (279, 17966, 17912),
    (600, 8): (68, 24768, 12339),
    (600, 9): (283, 16615, 12665),
    (600, 10): (1768, 23999, 19314),
}
ARCHIVE_EXPECTED_OUTCOMES: dict[tuple[int, int, str], tuple[int, str]] = {
    (nodes, seed, method): (
        iterations[METHODS.index(method)],
        (
            "ADMM_TERMINATION_TIME_LIMIT"
            if (nodes, seed, method) == PAPER_CENSORED_KEY
            else "ADMM_TERMINATION_OPTIMAL"
        ),
    )
    for (nodes, seed), iterations in ARCHIVE_ITERATIONS.items()
    for method in METHODS
}

# HiGHS returned a feasible incumbent at its 60-second MILP partition limit for
# these 34 retained jobs. Their downstream ADMM result depends on which feasible
# incumbent existed at the cutoff, so it is not a deterministic seed fingerprint.
ARCHIVE_MILP_TIME_LIMIT_PAIRS = frozenset(
    {
        (300, 3),
        (300, 4),
        (300, 9),
        (300, 10),
        *((nodes, seed) for nodes in (400, 500, 600) for seed in PAPER_SEEDS),
    }
)
ARCHIVE_MILP_CENSORED_KEYS = frozenset(
    (nodes, seed, "milp")
    for nodes, seed in ARCHIVE_MILP_TIME_LIMIT_PAIRS
)

# Post-bipartization count fingerprints parsed from the retained stdout logs.
# Each tuple is (nodes, left, right, edges), ordered by paper seed.
ARCHIVE_MILP_PARTITION_FINGERPRINTS: dict[
    tuple[int, int], tuple[int, int, int, int]
] = {
    (nodes, seed): fingerprints[seed - 1]
    for nodes, fingerprints in {
        200: (
            (2172, 186, 1986, 3972),
            (2170, 185, 1985, 3970),
            (2178, 189, 1989, 3978),
            (2176, 188, 1988, 3976),
            (2170, 185, 1985, 3970),
            (2172, 186, 1986, 3972),
            (2166, 183, 1983, 3966),
            (2174, 187, 1987, 3974),
            (2176, 188, 1988, 3976),
            (2166, 183, 1983, 3966),
        ),
        300: (
            (2256, 278, 1978, 3956),
            (2252, 276, 1976, 3952),
            (2260, 280, 1980, 3960),
            (2254, 277, 1977, 3954),
            (2276, 288, 1988, 3976),
            (2260, 280, 1980, 3960),
            (2252, 276, 1976, 3952),
            (2258, 279, 1979, 3958),
            (2260, 280, 1980, 3960),
            (2573, 557, 2016, 4273),
        ),
        400: (
            (2342, 371, 1971, 3942),
            (2348, 374, 1974, 3948),
            (2340, 370, 1970, 3940),
            (2354, 377, 1977, 3954),
            (2356, 378, 1978, 3956),
            (2352, 376, 1976, 3952),
            (2342, 371, 1971, 3942),
            (2342, 371, 1971, 3942),
            (2340, 1970, 370, 3940),
            (2354, 377, 1977, 3954),
        ),
        500: (
            (2438, 469, 1969, 3938),
            (2432, 1966, 466, 3932),
            (2424, 462, 1962, 3924),
            (3181, 1313, 1868, 4681),
            (2434, 467, 1967, 3934),
            (2438, 469, 1969, 3938),
            (2418, 459, 1959, 3918),
            (2430, 465, 1965, 3930),
            (2438, 469, 1969, 3938),
            (2448, 474, 1974, 3948),
        ),
        600: (
            (2721, 766, 1955, 4121),
            (2516, 558, 1958, 3916),
            (2512, 556, 1956, 3912),
            (2516, 558, 1958, 3916),
            (2701, 1927, 774, 4101),
            (2530, 565, 1965, 3930),
            (2508, 554, 1954, 3908),
            (2518, 559, 1959, 3918),
            (2516, 558, 1958, 3916),
            (2534, 567, 1967, 3934),
        ),
    }.items()
    for seed in PAPER_SEEDS
}

# SHA256 of the normalized 150-row semantic manifest produced by
# _archive_semantic_manifest. This is intentionally a literal independent of
# the supplied ZIP; changing a job mapping, command setting, outcome, or censor
# identity changes the digest and rejects full mode before any output.
ARCHIVE_SEMANTIC_SHA256 = (
    "9a49e8104e188c290f5edc83809806573cffd1c0bc8914d60ee3b835f34b50b0"
)

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
RE_SOLVER_SETTING = re.compile(r"^\s*solver\s*=\s*([^\s]+)\s*$")
RE_MAX_ITER_SETTING = re.compile(r"^\s*maxIter\s*=\s*(\d+)\s*$")
RE_INITIAL_RHO_SETTING = re.compile(rf"^\s*initialRho\s*=\s*({NUMBER})\s*$")
RE_TIME_LIMIT_SETTING = re.compile(rf"^\s*timeLimit\s*=\s*({NUMBER})\s*$")
RE_LOG_INTERVAL_SETTING = re.compile(r"^\s*logInterval\s*=\s*(\d+)\s*$")
SETTING_PATTERNS = (
    ("solver", RE_SOLVER_SETTING, str),
    ("maxIter", RE_MAX_ITER_SETTING, int),
    ("initialRho", RE_INITIAL_RHO_SETTING, float),
    ("timeLimit", RE_TIME_LIMIT_SETTING, float),
    ("logInterval", RE_LOG_INTERVAL_SETTING, int),
)
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
RE_HIGHS_STATUS = re.compile(r"^\s*Status\s{2,}(.+?)\s*$", re.MULTILINE)
RE_HIGHS_PRIMAL_BOUND = re.compile(
    rf"^\s*Primal bound\s+({NUMBER}|[-+]?inf)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
RE_HIGHS_DUAL_BOUND = re.compile(
    rf"^\s*Dual bound\s+({NUMBER}|[-+]?inf)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
RE_HIGHS_GAP = re.compile(
    rf"^\s*Gap\s+({NUMBER}|[-+]?inf)%", re.IGNORECASE | re.MULTILINE
)
RE_HIGHS_SOLUTION_STATUS = re.compile(
    r"^\s*Solution status\s+(\S+)\s*$", re.MULTILINE
)
RE_MILP_PARTITION_SUMMARY = re.compile(
    r"Number of nodes\s*=\s*(\d+)\s*"
    r"Parition size \(left, right\)\s*=\s*\((\d+),\s*(\d+)\)\s*"
    r"Number of edges\s*=\s*(\d+)",
    re.DOTALL,
)

MILP_METADATA_FIELDS = (
    "milp_metadata_available",
    "milp_source_nodes",
    "milp_source_arcs",
    "milp_source_seed",
    "milp_solve_status",
    "milp_time_limit",
    "milp_primal_bound",
    "milp_dual_bound",
    "milp_gap_percent",
    "milp_solution_status",
    "milp_solution_feasible",
    "milp_partition_nodes",
    "milp_partition_left",
    "milp_partition_right",
    "milp_partition_edges",
    "milp_partition_fingerprint",
    "milp_partition_valid",
)


def parse_milp_metadata(
    text: str, source_log: str
) -> tuple[dict[str, object | None], list[str]]:
    """Parse HiGHS and post-bipartization structure from one combined log."""

    metadata: dict[str, object | None] = {
        field: None for field in MILP_METADATA_FIELDS
    }
    errors: list[str] = []
    banner = "Running Bipartite ADMM with MILP bipartization..."
    banner_index = text.find(banner)
    report_matches = list(
        re.finditer(r"^\s*Solving report\s*$", text, re.MULTILINE)
    )
    report_index = report_matches[0].start() if report_matches else -1
    if banner_index < 0 and report_index < 0:
        return metadata, errors
    highs_present = "Running HiGHS" in text or report_index >= 0
    if not highs_present:
        return metadata, errors
    if report_index < 0:
        errors.append(f"{source_log}/milp: HiGHS report is missing Solving report")
        return metadata, errors
    if len(report_matches) != 1:
        errors.append(
            f"{source_log}/milp: expected one HiGHS Solving report, "
            f"found {len(report_matches)}"
        )

    # stderr can precede the buffered stdout method banner in merged fresh logs,
    # so parse the unique HiGHS report and MILP graph summary from the full text.
    report = text[report_index : report_index + 4_000]
    instance_match = RE_INSTANCE.search(text)
    seed_match = re.search(r"^\s*seed\s*=\s*(\d+)\s*$", text, re.MULTILINE)
    if instance_match is None:
        errors.append(f"{source_log}/milp: missing source nodes/arcs identity")
    else:
        metadata["milp_source_nodes"] = int(instance_match.group(1))
        metadata["milp_source_arcs"] = int(instance_match.group(2))
    if seed_match is None:
        errors.append(f"{source_log}/milp: missing source seed identity")
    else:
        metadata["milp_source_seed"] = int(seed_match.group(1))
    report_patterns = {
        "milp_solve_status": (RE_HIGHS_STATUS, str),
        "milp_primal_bound": (RE_HIGHS_PRIMAL_BOUND, float),
        "milp_dual_bound": (RE_HIGHS_DUAL_BOUND, float),
        "milp_gap_percent": (RE_HIGHS_GAP, float),
        "milp_solution_status": (RE_HIGHS_SOLUTION_STATUS, str),
    }
    for field, (pattern, convert) in report_patterns.items():
        match = pattern.search(report)
        if match is None:
            errors.append(f"{source_log}/milp: HiGHS report is missing {field}")
            continue
        try:
            metadata[field] = convert(match.group(1).strip())
        except ValueError:
            errors.append(
                f"{source_log}/milp: invalid {field} value {match.group(1)!r}"
            )

    status = metadata["milp_solve_status"]
    if status is not None:
        metadata["milp_time_limit"] = status == "Time limit reached"
        if status not in {"Optimal", "Time limit reached"}:
            errors.append(
                f"{source_log}/milp: unsupported HiGHS status {status!r}"
            )
    solution_status = metadata["milp_solution_status"]
    if solution_status is not None:
        metadata["milp_solution_feasible"] = (
            str(solution_status).lower() == "feasible"
        )
        if metadata["milp_solution_feasible"] is not True:
            errors.append(
                f"{source_log}/milp: HiGHS partition solution is not feasible: "
                f"{solution_status!r}"
            )

    partition_section = text[banner_index:] if banner_index >= 0 else text
    partition_match = RE_MILP_PARTITION_SUMMARY.search(partition_section)
    if partition_match is None:
        errors.append(
            f"{source_log}/milp: missing post-bipartization count fingerprint"
        )
    else:
        partition_nodes, left, right, edges = (
            int(partition_match.group(index)) for index in range(1, 5)
        )
        fingerprint = (partition_nodes, left, right, edges)
        metadata.update(
            {
                "milp_partition_nodes": partition_nodes,
                "milp_partition_left": left,
                "milp_partition_right": right,
                "milp_partition_edges": edges,
                "milp_partition_fingerprint": ":".join(
                    str(value) for value in fingerprint
                ),
                "milp_partition_valid": (
                    partition_nodes > 0
                    and left > 0
                    and right > 0
                    and edges > 0
                    and partition_nodes == left + right
                ),
            }
        )
    metadata["milp_metadata_available"] = (
        all(
            metadata[field] is not None
            for field in MILP_METADATA_FIELDS
            if field != "milp_metadata_available"
        )
        and not errors
    )
    return metadata, errors


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
    *MILP_METADATA_FIELDS,
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
    settings: dict[str, object | None] = {
        setting: None for setting in PAPER_SETTINGS
    }
    methods: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    errors: list[str] = []
    timing_by_method: dict[str, dict[str, object]] = {
        method: {} for method in METHODS
    }
    timing_method: str | None = None
    text = path.read_text(encoding="utf-8", errors="ignore")
    milp_metadata, milp_metadata_errors = parse_milp_metadata(text, source_log)
    errors.extend(milp_metadata_errors)

    for line_number, raw_line in enumerate(io.StringIO(text), 1):
        line = raw_line.rstrip("\n")

        match = RE_SEED.match(line)
        if match:
            seed = int(match.group(1))
        match = RE_INSTANCE.search(line)
        if match:
            nodes, arcs = int(match.group(1)), int(match.group(2))
        for setting, pattern, convert in SETTING_PATTERNS:
            match = pattern.match(line)
            if match is None:
                continue
            value = convert(match.group(1))
            previous = settings[setting]
            if previous is not None and previous != value:
                errors.append(
                    f"{source_log}:{line_number}: conflicting {setting} echoes: "
                    f"{previous!r} then {value!r}"
                )
            settings[setting] = value
            break

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
                **(
                    milp_metadata
                    if method == "milp"
                    else {field: None for field in MILP_METADATA_FIELDS}
                ),
                "source_log": source_log,
                "parse_warnings": "; ".join(warnings),
            }
        )

    metadata = {
        "source_log": source_log,
        "nodes": nodes,
        "arcs": arcs,
        "seed": seed,
        **settings,
    }
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
    setting_logs: list[dict[str, object]] = []
    setting_mismatch_count = 0
    for record in metadata:
        source_log = str(record.get("source_log", "<unknown log>"))
        observed_settings = {
            setting: record.get(setting) for setting in PAPER_SETTINGS
        }
        setting_checks: list[dict[str, object]] = []
        for setting, expected_value in PAPER_SETTINGS.items():
            observed_value = observed_settings[setting]
            exact = observed_value == expected_value
            setting_checks.append(
                {
                    "setting": setting,
                    "observed": observed_value,
                    "expected": expected_value,
                    "exact": exact,
                }
            )
            if not exact:
                setting_mismatch_count += 1
                errors.append(
                    f"{source_log}: expected echoed paper setting "
                    f"{setting}={expected_value!r}, parsed {observed_value!r}"
                )
        setting_logs.append(
            {
                "source_log": source_log,
                "observed": observed_settings,
                "all_match": all(bool(check["exact"]) for check in setting_checks),
                "checks": setting_checks,
            }
        )

        if record["nodes"] is None or record["seed"] is None:
            continue
        pair = (int(record["nodes"]), int(record["seed"]))
        pair_logs[pair].append(source_log)
        if record["arcs"] != PAPER_ARCS:
            errors.append(
                f"{source_log}: expected {PAPER_ARCS} arcs, parsed {record['arcs']}"
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
        "paper_settings": {
            "expected": dict(PAPER_SETTINGS),
            "checked_log_count": len(metadata),
            "checked_value_count": len(metadata) * len(PAPER_SETTINGS),
            "mismatch_count": setting_mismatch_count,
            "all_match": bool(metadata) and setting_mismatch_count == 0,
            "logs": setting_logs,
        },
    }
    return report, errors


def load_archive_runs(archive: Path) -> list[dict[str, object]]:
    """Load the exact per-run Section 3.2 reference table from the archive."""

    archive = archive.expanduser().resolve()
    text: str
    if archive.is_dir():
        relative = Path(ARCHIVE_SECTION) / "admm_summary" / "admm_runs.csv"
        candidates = (
            archive / ARCHIVE_RUNS_MEMBER,
            archive / relative,
            archive / "experiments_logs" / relative,
        )
        csv_path = next((path for path in candidates if path.is_file()), None)
        if csv_path is None:
            matches = [
                path
                for path in archive.rglob("admm_runs.csv")
                if tuple(path.parts[-len(relative.parts) :]) == tuple(relative.parts)
                and "__MACOSX" not in path.parts
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"expected one {relative.as_posix()} below {archive}, found {len(matches)}"
                )
            csv_path = matches[0]
        text = csv_path.read_text(encoding="utf-8-sig")
    elif archive.is_file():
        try:
            with zipfile.ZipFile(archive) as zipped:
                suffix = tuple(Path(ARCHIVE_RUNS_MEMBER).parts)
                matches = [
                    name
                    for name in zipped.namelist()
                    if tuple(Path(name).parts[-len(suffix) :]) == suffix
                    and "__MACOSX" not in Path(name).parts
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"expected one {ARCHIVE_RUNS_MEMBER} in {archive}, found {len(matches)}"
                    )
                text = zipped.read(matches[0]).decode("utf-8-sig")
        except zipfile.BadZipFile as error:
            raise ValueError(f"invalid archive ZIP {archive}: {error}") from error
    else:
        raise ValueError(f"archive reference does not exist: {archive}")

    reader = csv.DictReader(io.StringIO(text))
    required = {
        "scale_folder",
        "run_id",
        "nodes",
        "arcs",
        "seed",
        "method",
        "partition_time_sec",
        "init_time_sec",
        "algorithm_time_sec",
        "termination_status",
        "iterations",
        "stdout_log",
    }
    missing_columns = sorted(required.difference(reader.fieldnames or ()))
    if missing_columns:
        raise ValueError(
            f"{ARCHIVE_RUNS_MEMBER} is missing columns: {missing_columns}"
        )

    rows: list[dict[str, object]] = []
    for line_number, row in enumerate(reader, 2):
        try:
            partition = float(row["partition_time_sec"])
            admm_time = float(row["algorithm_time_sec"])
            scale_folder = row["scale_folder"].strip()
            if not scale_folder:
                raise ValueError("empty scale_folder")
            method = row["method"].strip().lower()
            if method not in METHODS:
                raise ValueError(f"unknown method {method!r}")
            rows.append(
                {
                    "scale_folder": scale_folder,
                    "run_id": int(row["run_id"]),
                    "nodes": int(row["nodes"]),
                    "arcs": int(row["arcs"]),
                    "seed": int(row["seed"]),
                    "method": method,
                    "partition_time_sec": partition,
                    "initialization_time_sec": float(row["init_time_sec"]),
                    "admm_time_sec": admm_time,
                    "total_time_sec": partition + admm_time,
                    "termination_status": row["termination_status"].strip(),
                    "iterations": int(row["iterations"]),
                    "stdout_log": row["stdout_log"].strip(),
                }
            )
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"invalid {ARCHIVE_RUNS_MEMBER} row {line_number}: {error}"
            ) from error
    if not rows:
        raise ValueError(f"{ARCHIVE_RUNS_MEMBER} contains no data rows")
    return rows


def _raw_key(row: Mapping[str, object]) -> tuple[int, int, str] | None:
    try:
        nodes = int(row["nodes"])
        seed = int(row["seed"])
        method = str(row["method"])
    except (KeyError, TypeError, ValueError):
        return None
    if not method:
        return None
    return nodes, seed, method


def _raw_sort_key(key: tuple[int, int, str]) -> tuple[int, int, int, str]:
    method = key[2]
    method_index = METHODS.index(method) if method in METHODS else len(METHODS)
    return key[0], key[1], method_index, method


def _raw_text_key(key: tuple[int, int, str]) -> str:
    return f"nodes={key[0]}/seed={key[1]}/{key[2]}"


def _raw_index(
    rows: Sequence[Mapping[str, object]],
) -> tuple[
    dict[tuple[int, int, str], Mapping[str, object]],
    list[tuple[int, int, str]],
    int,
]:
    index: dict[tuple[int, int, str], Mapping[str, object]] = {}
    duplicates: list[tuple[int, int, str]] = []
    missing_metadata = 0
    for row in rows:
        key = _raw_key(row)
        if key is None:
            missing_metadata += 1
            continue
        if key in index:
            duplicates.append(key)
            continue
        index[key] = row
    return index, sorted(set(duplicates), key=_raw_sort_key), missing_metadata


def _raw_check(
    checks: list[dict[str, object]],
    *,
    key: tuple[int, int, str],
    field: str,
    observed: object,
    expected: object,
    enforced: bool,
    category: str,
    reason: str,
) -> None:
    check: dict[str, object] = {
        "key": _raw_text_key(key),
        "nodes": key[0],
        "seed": key[1],
        "method": key[2],
        "field": field,
        "observed": observed,
        "expected": expected,
        "passed": observed == expected,
        "enforced": enforced,
        "category": category,
        "reason": reason,
    }
    if field.endswith("_time_sec"):
        try:
            check["delta"] = float(observed) - float(expected)
        except (TypeError, ValueError):
            check["delta"] = None
    checks.append(check)


def build_raw_archive_comparison(
    fresh_rows: Sequence[Mapping[str, object]],
    archive_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Compare a fresh grid with field-aware ADMM and MILP censoring."""

    fresh_index, fresh_duplicates, fresh_missing_metadata = _raw_index(fresh_rows)
    archive_index, archive_duplicates, archive_missing_metadata = _raw_index(
        archive_rows
    )
    fresh_keys = set(fresh_index)
    archive_keys = set(archive_index)
    shared_keys = fresh_keys.intersection(archive_keys)
    missing_fresh_keys = sorted(archive_keys.difference(fresh_keys), key=_raw_sort_key)
    extra_fresh_keys = sorted(fresh_keys.difference(archive_keys), key=_raw_sort_key)
    archive_missing_paper_keys = sorted(
        PAPER_RUN_KEYS.difference(archive_keys), key=_raw_sort_key
    )
    archive_extra_keys = sorted(
        archive_keys.difference(PAPER_RUN_KEYS), key=_raw_sort_key
    )

    checks: list[dict[str, object]] = []
    censored_rows: list[dict[str, object]] = []
    for key in sorted(shared_keys, key=_raw_sort_key):
        fresh = fresh_index[key]
        archived = archive_index[key]
        admm_censored = key == PAPER_CENSORED_KEY
        milp_partition_censored = key in ARCHIVE_MILP_CENSORED_KEYS
        outcome_censored = admm_censored or milp_partition_censored
        _raw_check(
            checks,
            key=key,
            field="arcs",
            observed=fresh.get("arcs"),
            expected=archived.get("arcs"),
            enforced=True,
            category="identity",
            reason="the paper network-flow grid uses exactly 2000 arcs",
        )

        if key[2] == "milp":
            for field in (
                "milp_metadata_available",
                "milp_solution_feasible",
                "milp_partition_valid",
            ):
                _raw_check(
                    checks,
                    key=key,
                    field=field,
                    observed=fresh.get(field) is True,
                    expected=True,
                    enforced=True,
                    category="partition_validity",
                    reason=(
                        "every MILP job must return a feasible, structurally "
                        "valid partition before downstream ADMM starts"
                    ),
                )
            _raw_check(
                checks,
                key=key,
                field="milp_partition_fingerprint",
                observed=fresh.get("milp_partition_fingerprint"),
                expected=archived.get("milp_partition_fingerprint"),
                enforced=not milp_partition_censored,
                category=(
                    "partition_structure_censored"
                    if milp_partition_censored
                    else "partition_structure"
                ),
                reason=(
                    "the feasible-incumbent cutoff can change partition counts "
                    "and membership; the observed count fingerprint is reported "
                    "but only internal structural validity is enforced"
                    if milp_partition_censored
                    else "archive-optimal post-bipartization "
                    "node/left/right/edge counts must match"
                ),
            )
            partition_status_reason = (
                "the archive HiGHS solve stopped at 60 seconds with a feasible "
                "incumbent, so status and incumbent bounds are non-exact"
                if milp_partition_censored
                else "archive-optimal HiGHS partition status"
            )
            for field in ("milp_solve_status", "milp_time_limit"):
                _raw_check(
                    checks,
                    key=key,
                    field=field,
                    observed=fresh.get(field),
                    expected=archived.get(field),
                    enforced=not milp_partition_censored,
                    category=(
                        "partition_solver_censored"
                        if milp_partition_censored
                        else "partition_solver"
                    ),
                    reason=partition_status_reason,
                )
            for field in (
                "milp_primal_bound",
                "milp_dual_bound",
                "milp_gap_percent",
            ):
                _raw_check(
                    checks,
                    key=key,
                    field=field,
                    observed=fresh.get(field),
                    expected=archived.get(field),
                    enforced=False,
                    category="partition_bounds_informational",
                    reason=(
                        "HiGHS incumbent/bound progress depends on wall-clock "
                        "throughput and is not an exact reproduction field"
                    ),
                )

        outcome_reason = (
            "the archived paper run hit its 3600-second wall-time limit"
            if admm_censored
            else (
                "the archive MILP partition solve stopped at 60 seconds with a "
                "feasible incumbent; downstream ADMM depends on its membership"
                if milp_partition_censored
                else "deterministic seeded ADMM outcome"
            )
        )
        for field in ("iterations", "termination_status"):
            _raw_check(
                checks,
                key=key,
                field=field,
                observed=fresh.get(field),
                expected=archived.get(field),
                enforced=not outcome_censored,
                category="outcome_censored" if outcome_censored else "outcome",
                reason=outcome_reason,
            )
        if milp_partition_censored:
            iterations = fresh.get("iterations")
            iterations_valid = (
                isinstance(iterations, int)
                and not isinstance(iterations, bool)
                and iterations >= 0
            )
            _raw_check(
                checks,
                key=key,
                field="downstream_iterations_valid",
                observed=iterations_valid,
                expected=True,
                enforced=True,
                category="downstream_validity",
                reason=(
                    "a time-censored feasible partition must still produce a "
                    "valid downstream ADMM result"
                ),
            )
            _raw_check(
                checks,
                key=key,
                field="downstream_status_valid",
                observed=fresh.get("termination_status") in VALID_STATUSES,
                expected=True,
                enforced=True,
                category="downstream_validity",
                reason=(
                    "a time-censored feasible partition must still produce a "
                    "recognized downstream ADMM terminal status"
                ),
            )
        for field in (
            "partition_time_sec",
            "initialization_time_sec",
            "admm_time_sec",
            "total_time_sec",
        ):
            _raw_check(
                checks,
                key=key,
                field=field,
                observed=fresh.get(field),
                expected=archived.get(field),
                enforced=False,
                category="timing_informational",
                reason="wall-clock measurement is hardware-dependent",
            )
        if admm_censored:
            censored_rows.append(
                {
                    "key": _raw_text_key(key),
                    "nodes": key[0],
                    "seed": key[1],
                    "method": key[2],
                    "archive_status": archived.get("termination_status"),
                    "observed_status": fresh.get("termination_status"),
                    "censor_type": "admm_wall_time",
                    "reason": "archive_admm_time_limit",
                    "censor_reason": "archive_admm_time_limit",
                    "outcome_enforced": False,
                    "non_exact_fields": ["iterations", "termination_status"],
                }
            )
        elif milp_partition_censored:
            censored_rows.append(
                {
                    "key": _raw_text_key(key),
                    "nodes": key[0],
                    "seed": key[1],
                    "method": key[2],
                    "censor_type": "milp_partition_time_limit",
                    "censor_reason": "archive_milp_partition_time_limit",
                    "reason": (
                        "archive HiGHS reached its 60-second limit with a "
                        "feasible incumbent"
                    ),
                    "archive_partition_status": archived.get(
                        "milp_solve_status"
                    ),
                    "observed_partition_status": fresh.get(
                        "milp_solve_status"
                    ),
                    "archive_partition_fingerprint": archived.get(
                        "milp_partition_fingerprint"
                    ),
                    "observed_partition_fingerprint": fresh.get(
                        "milp_partition_fingerprint"
                    ),
                    "archive_status": archived.get("termination_status"),
                    "observed_status": fresh.get("termination_status"),
                    "archive_iterations": archived.get("iterations"),
                    "observed_iterations": fresh.get("iterations"),
                    "outcome_enforced": False,
                    "feasibility_and_structural_validity_enforced": True,
                    "count_fingerprint_exact_enforced": False,
                    "non_exact_fields": [
                        "milp_solve_status",
                        "milp_time_limit",
                        "milp_primal_bound",
                        "milp_dual_bound",
                        "milp_gap_percent",
                        "partition_membership_not_serialized",
                        "iterations",
                        "termination_status",
                    ],
                }
            )

    archive_time_limit_keys = sorted(
        (
            key
            for key, row in archive_index.items()
            if row.get("termination_status") == "ADMM_TERMINATION_TIME_LIMIT"
        ),
        key=_raw_sort_key,
    )
    unexpected_archive_time_limit_keys = [
        key for key in archive_time_limit_keys if key != PAPER_CENSORED_KEY
    ]
    archive_milp_time_limit_keys = sorted(
        (
            key
            for key, row in archive_index.items()
            if key[2] == "milp" and row.get("milp_time_limit") is True
        ),
        key=_raw_sort_key,
    )
    archive_milp_time_limit_key_set = set(archive_milp_time_limit_keys)
    missing_archive_milp_censor_keys = sorted(
        ARCHIVE_MILP_CENSORED_KEYS.difference(
            archive_milp_time_limit_key_set
        ),
        key=_raw_sort_key,
    )
    unexpected_archive_milp_censor_keys = sorted(
        archive_milp_time_limit_key_set.difference(
            ARCHIVE_MILP_CENSORED_KEYS
        ),
        key=_raw_sort_key,
    )
    archive_invalid_milp_rows = [
        {
            "key": _raw_text_key(key),
            "metadata_available": row.get("milp_metadata_available"),
            "solution_feasible": row.get("milp_solution_feasible"),
            "partition_valid": row.get("milp_partition_valid"),
        }
        for key, row in sorted(
            archive_index.items(), key=lambda item: _raw_sort_key(item[0])
        )
        if key[2] == "milp"
        and (
            row.get("milp_metadata_available") is not True
            or row.get("milp_solution_feasible") is not True
            or row.get("milp_partition_valid") is not True
        )
    ]
    archive_wrong_arcs = [
        {
            "key": _raw_text_key(key),
            "observed": row.get("arcs"),
            "expected": PAPER_ARCS,
        }
        for key, row in sorted(archive_index.items(), key=lambda item: _raw_sort_key(item[0]))
        if row.get("arcs") != PAPER_ARCS
    ]
    archived_censored = archive_index.get(PAPER_CENSORED_KEY)
    censored_archive_status_valid = bool(archived_censored) and (
        archived_censored.get("termination_status")
        == "ADMM_TERMINATION_TIME_LIMIT"
    )

    enforced_checks = [check for check in checks if bool(check["enforced"])]
    enforced_failed = sum(not bool(check["passed"]) for check in enforced_checks)
    key_integrity_valid = not any(
        (
            fresh_duplicates,
            archive_duplicates,
            fresh_missing_metadata,
            archive_missing_metadata,
            missing_fresh_keys,
            extra_fresh_keys,
            archive_missing_paper_keys,
            archive_extra_keys,
            archive_wrong_arcs,
            archive_invalid_milp_rows,
            missing_archive_milp_censor_keys,
            unexpected_archive_milp_censor_keys,
        )
    )
    enforced_outcome_checks = [
        check for check in enforced_checks if check["category"] == "outcome"
    ]
    enforced_identity_checks = [
        check for check in enforced_checks if check["category"] == "identity"
    ]
    return {
        "reference": f"experiments_logs.zip/{ARCHIVE_RUNS_MEMBER}",
        "policy": {
            "keys": "exact 150-row (nodes, seed, method) paper grid",
            "iterations_and_status": (
                "exact on 115 deterministic rows; non-exact on the 34 MILP "
                "rows whose archive partition solve reached its feasible "
                "60-second cutoff, and nodes=500/seed=6/basic whose archive "
                "ADMM solve reached its 3600-second cutoff"
            ),
            "milp_partition": (
                "all 50 jobs require complete metadata, a feasible partition, "
                "and internally valid (nodes,left,right,edges) counts; for the "
                "34 archive time-limit jobs, count identity, HiGHS "
                "status/bounds, and unrecorded membership are non-exact"
            ),
            "timings": "informational only",
        },
        "fresh_record_count": len(fresh_rows),
        "archive_record_count": len(archive_rows),
        "expected_paper_record_count": len(PAPER_RUN_KEYS),
        "compared_key_count": len(shared_keys),
        "fresh_missing_metadata_count": fresh_missing_metadata,
        "archive_missing_metadata_count": archive_missing_metadata,
        "fresh_duplicates": [list(key) for key in fresh_duplicates],
        "archive_duplicates": [list(key) for key in archive_duplicates],
        "missing_fresh_keys": [list(key) for key in missing_fresh_keys],
        "extra_fresh_keys": [list(key) for key in extra_fresh_keys],
        "archive_missing_paper_keys": [
            list(key) for key in archive_missing_paper_keys
        ],
        "archive_extra_keys": [list(key) for key in archive_extra_keys],
        "archive_wrong_arcs": archive_wrong_arcs,
        "archive_time_limit_keys": [list(key) for key in archive_time_limit_keys],
        "unexpected_archive_time_limit_keys": [
            list(key) for key in unexpected_archive_time_limit_keys
        ],
        "archive_milp_time_limit_keys": [
            list(key) for key in archive_milp_time_limit_keys
        ],
        "expected_archive_milp_time_limit_key_count": len(
            ARCHIVE_MILP_CENSORED_KEYS
        ),
        "missing_archive_milp_censor_keys": [
            list(key) for key in missing_archive_milp_censor_keys
        ],
        "unexpected_archive_milp_censor_keys": [
            list(key) for key in unexpected_archive_milp_censor_keys
        ],
        "archive_milp_censor_manifest_valid": (
            not missing_archive_milp_censor_keys
            and not unexpected_archive_milp_censor_keys
        ),
        "archive_invalid_milp_rows": archive_invalid_milp_rows,
        "censored_archive_status_valid": censored_archive_status_valid,
        "censored_rows": censored_rows,
        "checks": checks,
        "summary": {
            "expected_keys": len(PAPER_RUN_KEYS),
            "compared_keys": len(shared_keys),
            "stable_outcome_rows": sum(
                key != PAPER_CENSORED_KEY
                and key not in ARCHIVE_MILP_CENSORED_KEYS
                for key in shared_keys
            ),
            "censored_row_count": len(censored_rows),
            "admm_censored_row_count": sum(
                row["censor_type"] == "admm_wall_time"
                for row in censored_rows
            ),
            "milp_partition_censored_row_count": sum(
                row["censor_type"] == "milp_partition_time_limit"
                for row in censored_rows
            ),
            "partition_censored_row_count": sum(
                row["censor_type"] == "milp_partition_time_limit"
                for row in censored_rows
            ),
            "censored_outcome_row_count": len(censored_rows),
            "enforced": len(enforced_checks),
            "enforced_identity_checks": len(enforced_identity_checks),
            "enforced_outcome_checks": len(enforced_outcome_checks),
            "enforced_passed": len(enforced_checks) - enforced_failed,
            "enforced_failed": enforced_failed,
            "all_enforced_checks_passed": enforced_failed == 0,
            "key_integrity_valid": key_integrity_valid,
            "all_required_checks_passed": (
                key_integrity_valid
                and censored_archive_status_valid
                and not unexpected_archive_time_limit_keys
                and not missing_archive_milp_censor_keys
                and not unexpected_archive_milp_censor_keys
                and enforced_failed == 0
            ),
        },
    }


def raw_archive_validation_errors(comparison: Mapping[str, object]) -> list[str]:
    errors: list[str] = []
    for count_field, label in (
        ("fresh_missing_metadata_count", "fresh rows missing key metadata"),
        ("archive_missing_metadata_count", "archive rows missing key metadata"),
    ):
        count = int(comparison.get(count_field) or 0)
        if count:
            errors.append(f"Raw archive comparison has {count} {label}")
    for list_field, label in (
        ("fresh_duplicates", "duplicate fresh keys"),
        ("archive_duplicates", "duplicate archive keys"),
        ("missing_fresh_keys", "archive keys missing from fresh full grid"),
        ("extra_fresh_keys", "fresh keys absent from the archive"),
        ("archive_missing_paper_keys", "paper keys missing from archive reference"),
        ("archive_extra_keys", "non-paper keys in archive reference"),
        ("archive_wrong_arcs", "archive rows whose arcs value is not 2000"),
        ("unexpected_archive_time_limit_keys", "unexpected archived time-limit rows"),
        (
            "missing_archive_milp_censor_keys",
            "expected archived MILP partition time-limit rows that are missing",
        ),
        (
            "unexpected_archive_milp_censor_keys",
            "unexpected archived MILP partition time-limit rows",
        ),
        (
            "archive_invalid_milp_rows",
            "archived MILP rows without a valid feasible partition",
        ),
    ):
        values = comparison.get(list_field)
        if isinstance(values, list) and values:
            errors.append(
                f"Raw archive comparison found {len(values)} {label}; first={values[:5]}"
            )
    if not bool(comparison.get("censored_archive_status_valid")):
        errors.append(
            "Raw archive comparison expected nodes=500/seed=6/basic to be the "
            "archived ADMM time-limit row"
        )
    if not bool(comparison.get("archive_milp_censor_manifest_valid")):
        errors.append(
            "Raw archive comparison expected the exact 34-row feasible MILP "
            "partition time-limit censor manifest"
        )

    checks = comparison.get("checks")
    if not isinstance(checks, list):
        return [*errors, "Raw archive comparison did not produce a check list"]
    failed = [
        check
        for check in checks
        if isinstance(check, Mapping)
        and bool(check.get("enforced"))
        and not bool(check.get("passed"))
    ]
    errors.extend(
        "Raw archive mismatch "
        f"{check.get('key')}/{check.get('field')}: expected "
        f"{check.get('expected')!r}, got {check.get('observed')!r}"
        for check in failed
    )
    return errors


def _expected_archive_command_tokens(nodes: int, seed: int) -> tuple[str, ...]:
    return (
        "./julia/run.sh",
        "-t",
        str(PAPER_THREADS),
        "PDMO.jl/advanced/src/NetworkFlow/runNetworkFlowProblem.jl",
        "--solver",
        str(PAPER_SETTINGS["solver"]),
        "--maxIter",
        str(PAPER_SETTINGS["maxIter"]),
        "--initialRho",
        str(PAPER_SETTINGS["initialRho"]),
        "--timeLimit",
        str(PAPER_SETTINGS["timeLimit"]),
        "--seed",
        str(seed),
        "--logInterval",
        str(PAPER_SETTINGS["logInterval"]),
        "--random",
        str(nodes),
        str(PAPER_ARCS),
    )


def _archive_section_root(archive: Path) -> Path:
    candidates = (
        archive / "experiments_logs" / ARCHIVE_SECTION,
        archive / ARCHIVE_SECTION,
        archive,
    )
    for candidate in candidates:
        if (
            candidate.name == ARCHIVE_SECTION
            and (candidate / "admm_summary" / "admm_runs.csv").is_file()
        ):
            return candidate
    matches = sorted(
        {
            path.parent.parent.resolve()
            for path in archive.rglob("admm_runs.csv")
            if path.parent.name == "admm_summary"
            and path.parent.parent.name == ARCHIVE_SECTION
            and "__MACOSX" not in path.parts
        }
    )
    if len(matches) != 1:
        raise ValueError(
            f"expected one extracted {ARCHIVE_SECTION} directory below "
            f"{archive}, found {len(matches)}"
        )
    return matches[0]


def _archive_job_sources(
    archive: Path,
) -> tuple[
    dict[tuple[str, int], str],
    dict[tuple[str, int], str],
    set[tuple[str, int]],
    set[tuple[str, int]],
    list[str],
]:
    """Read exact scheduler command membership without extracting the ZIP."""

    archive = archive.expanduser().resolve()
    commands: dict[tuple[str, int], str] = {}
    stdout_logs: dict[tuple[str, int], str] = {}
    command_jobs: set[tuple[str, int]] = set()
    stdout_jobs: set[tuple[str, int]] = set()
    errors: list[str] = []

    def record(
        *,
        parts: tuple[str, ...],
        leaf: str,
        content: str,
    ) -> None:
        try:
            section_index = parts.index(ARCHIVE_SECTION)
        except ValueError:
            return
        remainder = parts[section_index + 1 :]
        if len(remainder) != 3 or remainder[2] != leaf:
            return
        scale_folder, run_text, _ = remainder
        if not scale_folder.isdigit() or not run_text.isdigit():
            return
        job = (scale_folder, int(run_text))
        target = command_jobs if leaf == "cmd" else stdout_jobs
        if job in target:
            errors.append(f"duplicate archive member for {job[0]}/{job[1]}/{leaf}")
            return
        target.add(job)
        if leaf == "cmd":
            commands[job] = content
        else:
            stdout_logs[job] = content

    if archive.is_dir():
        section_root = _archive_section_root(archive)
        for scale_dir in sorted(section_root.iterdir()):
            if not scale_dir.is_dir() or not scale_dir.name.isdigit():
                continue
            for run_dir in sorted(scale_dir.iterdir()):
                if not run_dir.is_dir() or not run_dir.name.isdigit():
                    continue
                for leaf in ("cmd", "stdout.log"):
                    path = run_dir / leaf
                    if not path.is_file():
                        continue
                    record(
                        parts=tuple(path.parts),
                        leaf=leaf,
                        content=path.read_text(
                            encoding="utf-8", errors="ignore"
                        ),
                    )
    elif archive.is_file():
        try:
            with zipfile.ZipFile(archive) as zipped:
                for name in zipped.namelist():
                    parts = tuple(Path(name).parts)
                    if "__MACOSX" in parts or not parts:
                        continue
                    leaf = parts[-1]
                    if leaf not in {"cmd", "stdout.log"}:
                        continue
                    record(
                        parts=parts,
                        leaf=leaf,
                        content=zipped.read(name).decode(
                            "utf-8", errors="ignore"
                        ),
                    )
        except zipfile.BadZipFile as error:
            raise ValueError(f"invalid archive ZIP {archive}: {error}") from error
    else:
        raise ValueError(f"archive reference does not exist: {archive}")
    return commands, stdout_logs, command_jobs, stdout_jobs, errors


def _bind_archive_milp_metadata(
    rows: Sequence[dict[str, object]],
    stdout_logs: Mapping[tuple[str, int], str],
) -> list[str]:
    """Attach source-parsed MILP metadata to the retained per-method rows."""

    errors: list[str] = []
    rows_by_key = {
        (int(row["nodes"]), int(row["seed"]), str(row["method"])): row
        for row in rows
    }
    for (nodes, seed), job in sorted(ARCHIVE_PAPER_JOBS.items()):
        stdout = stdout_logs.get(job)
        if stdout is None:
            continue
        metadata, parse_errors = parse_milp_metadata(
            stdout, f"{job[0]}/{job[1]}/stdout.log"
        )
        errors.extend(parse_errors)
        row = rows_by_key.get((nodes, seed, "milp"))
        if row is None:
            continue
        row.update(metadata)
    return errors


def _archive_command_config(
    command: str, *, nodes: int, seed: int, source: str
) -> tuple[dict[str, object] | None, list[str]]:
    tokens = tuple(command.split())
    expected = _expected_archive_command_tokens(nodes, seed)
    if tokens != expected:
        return None, [
            f"{source}: archived command mismatch for nodes={nodes}, seed={seed}; "
            f"expected {' '.join(expected)!r}, got {' '.join(tokens)!r}"
        ]
    return (
        {
            "runner": tokens[3],
            "threads": int(tokens[2]),
            "solver": tokens[5],
            "maxIter": int(tokens[7]),
            "initialRho": float(tokens[9]),
            "timeLimit": float(tokens[11]),
            "logInterval": int(tokens[15]),
        },
        [],
    )


def _archive_semantic_manifest(
    rows: Sequence[Mapping[str, object]],
    command_configs: Mapping[tuple[str, int], Mapping[str, object]],
) -> list[dict[str, object]]:
    manifest: list[dict[str, object]] = []
    for row in sorted(
        rows,
        key=lambda item: _raw_sort_key(
            (int(item["nodes"]), int(item["seed"]), str(item["method"]))
        ),
    ):
        nodes = int(row["nodes"])
        seed = int(row["seed"])
        method = str(row["method"])
        scale_folder = str(row["scale_folder"])
        run_id = int(row["run_id"])
        config = command_configs[(scale_folder, run_id)]
        manifest.append(
            {
                "scale_folder": scale_folder,
                "run_id": run_id,
                "nodes": nodes,
                "arcs": int(row["arcs"]),
                "seed": seed,
                "method": method,
                "runner": str(config["runner"]),
                "threads": int(config["threads"]),
                "solver": str(config["solver"]),
                "maxIter": int(config["maxIter"]),
                "initialRho": float(config["initialRho"]),
                "timeLimit": float(config["timeLimit"]),
                "logInterval": int(config["logInterval"]),
                "iterations": int(row["iterations"]),
                "termination_status": str(row["termination_status"]),
                "admm_wall_time_censored": (
                    (nodes, seed, method) == PAPER_CENSORED_KEY
                ),
                "milp_partition_time_censored": (
                    (nodes, seed, method) in ARCHIVE_MILP_CENSORED_KEYS
                ),
                "milp_source_nodes": row.get("milp_source_nodes"),
                "milp_source_arcs": row.get("milp_source_arcs"),
                "milp_source_seed": row.get("milp_source_seed"),
                "milp_solve_status": row.get("milp_solve_status"),
                "milp_time_limit": row.get("milp_time_limit"),
                "milp_solution_status": row.get("milp_solution_status"),
                "milp_solution_feasible": row.get(
                    "milp_solution_feasible"
                ),
                "milp_partition_fingerprint": row.get(
                    "milp_partition_fingerprint"
                ),
                "milp_partition_valid": row.get("milp_partition_valid"),
            }
        )
    return manifest


def _archive_semantic_digest(manifest: Sequence[Mapping[str, object]]) -> str:
    encoded = json.dumps(
        list(manifest), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _archive_reference_errors(
    rows: Sequence[dict[str, object]], archive: Path
) -> list[str]:
    errors: list[str] = []
    (
        commands,
        stdout_logs,
        command_jobs,
        stdout_jobs,
        source_errors,
    ) = _archive_job_sources(archive)
    errors.extend(source_errors)
    errors.extend(_bind_archive_milp_metadata(rows, stdout_logs))

    index, duplicates, missing_metadata = _raw_index(rows)
    if missing_metadata:
        errors.append(f"{missing_metadata} archive rows lack nodes/seed/method metadata")
    if duplicates:
        errors.append(f"duplicate archive result keys: {duplicates[:5]}")
    missing_keys = sorted(PAPER_RUN_KEYS.difference(index), key=_raw_sort_key)
    extra_keys = sorted(set(index).difference(PAPER_RUN_KEYS), key=_raw_sort_key)
    if missing_keys:
        errors.append(f"archive is missing paper result keys: {missing_keys[:5]}")
    if extra_keys:
        errors.append(f"archive contains non-paper result keys: {extra_keys[:5]}")

    for key, row in sorted(index.items(), key=lambda item: _raw_sort_key(item[0])):
        nodes, seed, method = key
        expected_job = ARCHIVE_PAPER_JOBS.get((nodes, seed))
        observed_job = (str(row.get("scale_folder")), row.get("run_id"))
        if expected_job is None or observed_job != expected_job:
            errors.append(
                f"archive job mapping mismatch for nodes={nodes}, seed={seed}: "
                f"expected {expected_job}, got {observed_job}"
            )
        if row.get("arcs") != PAPER_ARCS:
            errors.append(
                f"archive arcs mismatch for nodes={nodes}, seed={seed}, "
                f"method={method}: expected {PAPER_ARCS}, got {row.get('arcs')!r}"
            )
        expected_outcome = ARCHIVE_EXPECTED_OUTCOMES.get(key)
        observed_outcome = (row.get("iterations"), row.get("termination_status"))
        if expected_outcome is None or observed_outcome != expected_outcome:
            errors.append(
                f"archive outcome mismatch for nodes={nodes}, seed={seed}, "
                f"method={method}: expected {expected_outcome}, "
                f"got {observed_outcome}"
            )
        if method == "milp":
            expected_time_limit = (nodes, seed) in ARCHIVE_MILP_TIME_LIMIT_PAIRS
            expected_status = (
                "Time limit reached" if expected_time_limit else "Optimal"
            )
            expected_source_identity = (nodes, PAPER_ARCS, seed)
            observed_source_identity = (
                row.get("milp_source_nodes"),
                row.get("milp_source_arcs"),
                row.get("milp_source_seed"),
            )
            expected_fingerprint = ":".join(
                str(value)
                for value in ARCHIVE_MILP_PARTITION_FINGERPRINTS[(nodes, seed)]
            )
            if row.get("milp_metadata_available") is not True:
                errors.append(
                    f"archive MILP metadata missing for nodes={nodes}, seed={seed}"
                )
            if observed_source_identity != expected_source_identity:
                errors.append(
                    f"archive stdout identity mismatch for nodes={nodes}, "
                    f"seed={seed}: expected {expected_source_identity}, "
                    f"got {observed_source_identity}"
                )
            if row.get("milp_solve_status") != expected_status:
                errors.append(
                    f"archive MILP censor manifest mismatch for nodes={nodes}, "
                    f"seed={seed}: expected {expected_status!r}, "
                    f"got {row.get('milp_solve_status')!r}"
                )
            if row.get("milp_time_limit") is not expected_time_limit:
                errors.append(
                    f"archive MILP time-limit flag mismatch for nodes={nodes}, "
                    f"seed={seed}: expected {expected_time_limit!r}, "
                    f"got {row.get('milp_time_limit')!r}"
                )
            if row.get("milp_solution_feasible") is not True:
                errors.append(
                    f"archive MILP partition is not feasible for nodes={nodes}, "
                    f"seed={seed}"
                )
            if row.get("milp_partition_valid") is not True:
                errors.append(
                    f"archive MILP partition fingerprint is invalid for "
                    f"nodes={nodes}, seed={seed}"
                )
            if row.get("milp_partition_fingerprint") != expected_fingerprint:
                errors.append(
                    f"archive MILP partition fingerprint mismatch for "
                    f"nodes={nodes}, seed={seed}: expected "
                    f"{expected_fingerprint!r}, got "
                    f"{row.get('milp_partition_fingerprint')!r}"
                )
        if expected_job is not None:
            expected_suffix = (
                expected_job[0],
                str(expected_job[1]),
                "stdout.log",
            )
            observed_path = tuple(Path(str(row.get("stdout_log", ""))).parts[-3:])
            if observed_path != expected_suffix:
                errors.append(
                    f"archive stdout_log mismatch for nodes={nodes}, seed={seed}, "
                    f"method={method}: expected suffix {expected_suffix}, "
                    f"got {observed_path}"
                )

    observed_milp_time_limit_keys = {
        key
        for key, row in index.items()
        if key[2] == "milp" and row.get("milp_time_limit") is True
    }
    missing_milp_censors = sorted(
        ARCHIVE_MILP_CENSORED_KEYS.difference(observed_milp_time_limit_keys),
        key=_raw_sort_key,
    )
    unexpected_milp_censors = sorted(
        observed_milp_time_limit_keys.difference(ARCHIVE_MILP_CENSORED_KEYS),
        key=_raw_sort_key,
    )
    if missing_milp_censors or unexpected_milp_censors:
        errors.append(
            "archive MILP censor manifest mismatch: "
            f"missing={missing_milp_censors[:5]}, "
            f"unexpected={unexpected_milp_censors[:5]}"
        )

    expected_jobs = set(ARCHIVE_PAPER_JOBS.values())
    for label, observed_jobs in (
        ("cmd", command_jobs),
        ("stdout.log", stdout_jobs),
    ):
        missing_jobs = sorted(expected_jobs.difference(observed_jobs))
        extra_jobs = sorted(observed_jobs.difference(expected_jobs))
        if missing_jobs:
            errors.append(f"archive is missing {label} jobs: {missing_jobs[:5]}")
        if extra_jobs:
            errors.append(f"archive has unexpected {label} jobs: {extra_jobs[:5]}")

    command_configs: dict[tuple[str, int], Mapping[str, object]] = {}
    for (nodes, seed), job in sorted(ARCHIVE_PAPER_JOBS.items()):
        command = commands.get(job)
        if command is None:
            continue
        config, command_errors = _archive_command_config(
            command,
            nodes=nodes,
            seed=seed,
            source=f"{job[0]}/{job[1]}/cmd",
        )
        errors.extend(command_errors)
        if config is not None:
            command_configs[job] = config

    if (
        not missing_metadata
        and not duplicates
        and not missing_keys
        and not extra_keys
        and len(command_configs) == len(ARCHIVE_PAPER_JOBS)
    ):
        manifest = _archive_semantic_manifest(rows, command_configs)
        digest = _archive_semantic_digest(manifest)
        if digest != ARCHIVE_SEMANTIC_SHA256:
            errors.append(
                "archive semantic manifest SHA256 mismatch: "
                f"expected {ARCHIVE_SEMANTIC_SHA256}, got {digest}"
            )
    return errors


def load_validated_archive_runs(archive: Path) -> list[dict[str, object]]:
    """Load and independently validate the exact Section 3.2 paper archive."""

    rows = load_archive_runs(archive)
    errors = _archive_reference_errors(rows, archive)
    if errors:
        details = "\n".join(f"  - {error}" for error in errors)
        raise ValueError("invalid Section 3.2 raw archive reference:\n" + details)
    return rows


def requires_raw_archive_comparison(
    mode: str,
    rows: Sequence[Mapping[str, object]],
    metadata: Sequence[Mapping[str, object]] = (),
) -> bool:
    """Identify runs to which the raw archive comparison applies.

    Full mode can execute without the optional archive; main converts this
    applicability result into a non-fatal, machine-readable skip in that case.
    A complete-grid parse continues to require the archive comparison.
    """

    if mode == "full":
        return True
    if mode != "parse":
        return False
    pair_source = metadata if metadata else rows
    observed_pairs: set[tuple[int, int]] = set()
    for record in pair_source:
        try:
            observed_pairs.add((int(record["nodes"]), int(record["seed"])))
        except (KeyError, TypeError, ValueError):
            continue
    paper_pairs = {
        (nodes, seed) for nodes in PAPER_NODES for seed in PAPER_SEEDS
    }
    return observed_pairs == paper_pairs


def provenance_inputs(
    args: argparse.Namespace, raw_reference_required: bool
) -> list[Path]:
    """Return every input whose content contributed to the output artifacts."""

    if args.mode == "archived":
        return [args.archive]
    if args.mode == "parse":
        inputs = [args.logs]
        if raw_reference_required:
            inputs.append(args.archive)
        return inputs
    if args.mode == "full":
        inputs = [JULIA_RUNNER]
        if raw_reference_required:
            inputs.append(args.archive)
        return inputs
    return [JULIA_RUNNER]


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


def evaluate_conclusion_consistency(
    summaries: Sequence[Mapping[str, object]], mode: str
) -> tuple[dict[str, object], list[str]]:
    """Check the qualitative Figure 11 conclusions on a complete fresh grid.

    Archived mode records the same checks without enforcing them.  This is
    intentional: the published archive itself has an N=400 timing exception,
    and wall-clock crossover behavior is machine dependent.  Full mode and a
    complete-grid parse enforce the robust iteration and eventual-scalability
    conclusions.
    """

    observed = {
        (int(row["nodes"]), str(row["method"])): row for row in summaries
    }
    expected_keys = {
        (nodes, method) for nodes in PAPER_NODES for method in METHODS
    }
    missing_groups = sorted(
        expected_keys.difference(observed),
        key=lambda key: (key[0], METHODS.index(key[1])),
    )
    wrong_run_counts = [
        {
            "nodes": nodes,
            "method": method,
            "observed": int(observed[(nodes, method)]["n_runs"]),
            "expected": len(PAPER_SEEDS),
        }
        for nodes, method in sorted(
            expected_keys.intersection(observed),
            key=lambda key: (key[0], METHODS.index(key[1])),
        )
        if int(observed[(nodes, method)]["n_runs"]) != len(PAPER_SEEDS)
    ]
    complete_paper_grid = not missing_groups and not wrong_run_counts
    enforce = complete_paper_grid and mode in {"full", "parse"}
    checks: list[dict[str, object]] = []
    errors: list[str] = []

    def add_check(
        *,
        check_id: str,
        nodes: int,
        category: str,
        metric: str,
        left_method: str,
        right_method: str,
        informational: bool = False,
    ) -> None:
        left_value = float(observed[(nodes, left_method)][metric])
        right_value = float(observed[(nodes, right_method)][metric])
        passed = left_value < right_value
        enforced = enforce and not informational
        checks.append(
            {
                "id": check_id,
                "nodes": nodes,
                "category": category,
                "metric": metric,
                "left_method": left_method,
                "left_value": left_value,
                "relation": "<",
                "right_method": right_method,
                "right_value": right_value,
                "passed": passed,
                "enforced": enforced,
                "informational": informational,
            }
        )
        if enforced and not passed:
            errors.append(
                "Section 3.2 conclusion mismatch at "
                f"nodes={nodes}: expected {left_method} {metric} "
                f"({left_value}) < {right_method} {metric} ({right_value})"
            )

    if complete_paper_grid:
        for nodes in PAPER_NODES:
            for method in ("bfs", "milp"):
                add_check(
                    check_id=f"basic_iterations_less_than_{method}",
                    nodes=nodes,
                    category="iteration_pattern",
                    metric="iterations_mean",
                    left_method="basic",
                    right_method=method,
                )
            add_check(
                check_id="milp_iterations_less_than_bfs",
                nodes=nodes,
                category="iteration_pattern",
                metric="iterations_mean",
                left_method="milp",
                right_method="bfs",
            )

        for nodes in PAPER_NODES[1:]:
            for method in ("bfs", "milp"):
                add_check(
                    check_id=f"{method}_total_time_less_than_basic",
                    nodes=nodes,
                    category="scalability_pattern",
                    metric="total_time_mean",
                    left_method=method,
                    right_method="basic",
                    informational=nodes not in PAPER_NODES[-2:],
                )

        for nodes in PAPER_NODES[-2:]:
            add_check(
                check_id="milp_total_time_less_than_bfs",
                nodes=nodes,
                category="timing_crossover",
                metric="total_time_mean",
                left_method="milp",
                right_method="bfs",
                informational=True,
            )

    enforced_checks = [check for check in checks if bool(check["enforced"])]
    informational_checks = [
        check for check in checks if bool(check["informational"])
    ]
    required_pattern_checks = [
        check for check in checks if not bool(check["informational"])
    ]
    enforced_failed = sum(not bool(check["passed"]) for check in enforced_checks)
    required_pattern_failed = sum(
        not bool(check["passed"]) for check in required_pattern_checks
    )
    return (
        {
            "mode": mode,
            "complete_paper_grid": complete_paper_grid,
            "missing_groups": [
                {"nodes": nodes, "method": method}
                for nodes, method in missing_groups
            ],
            "wrong_run_counts": wrong_run_counts,
            "enforced_for_this_mode": enforce,
            "enforcement_policy": (
                "Enforce robust Figure 11 iteration patterns and the eventual "
                "N=500/N=600 reformulation-versus-Basic scalability pattern "
                "for fresh full grids and complete-grid parse mode. N=300/N=400 "
                "timing comparisons and archived reconstruction are recorded "
                "informationally."
            ),
            "aggregation_scope": (
                "Arithmetic means over all 10 paper rows per method, preserving "
                "the existing ADMM and MILP censor policy."
            ),
            "timing_crossover_policy": (
                "MILP total time below BFS at N=500 and N=600 is recorded but "
                "never enforced because wall time is machine dependent."
            ),
            "checks": checks,
            "summary": {
                "check_count": len(checks),
                "enforced_count": len(enforced_checks),
                "enforced_passed": len(enforced_checks) - enforced_failed,
                "enforced_failed": enforced_failed,
                "all_enforced_checks_passed": enforced_failed == 0,
                "required_pattern_count": len(required_pattern_checks),
                "required_pattern_passed": (
                    len(required_pattern_checks) - required_pattern_failed
                ),
                "required_pattern_failed": required_pattern_failed,
                "informational_count": len(informational_checks),
                "informational_passed": sum(
                    bool(check["passed"]) for check in informational_checks
                ),
            },
        },
        errors,
    )


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

    conclusion_consistency, conclusion_errors = evaluate_conclusion_consistency(
        summaries, mode
    )
    errors.extend(conclusion_errors)
    report = {
        "reference_source": (
            "experiments_logs/3-2-networkflow/admm_summary/admm_folder_summary.csv"
        ),
        "mode": mode,
        "timing_policy": (
            "Absolute wall-clock values and the MILP-vs-BFS large-N crossover "
            "are informational. Complete fresh grids enforce only the paper's "
            "eventual relative scalability pattern: BFS and MILP mean total "
            "time below Basic at N=500 and N=600. The corresponding N=300 and "
            "N=400 comparisons are recorded informationally."
        ),
        "iteration_policy": (
            "exact archived aggregate check"
            if mode == "archived"
            else (
                "exact raw seed-1 check in smoke; complete fresh grids enforce "
                "qualitative mean-iteration ordering, while exact aggregate "
                "values remain informational because the full archive contains "
                "ADMM- and MILP-cutoff-censored rows"
            )
        ),
        "missing_reference_groups": missing_groups,
        "complete_reference_grid": not missing_groups,
        "conclusion_consistency": conclusion_consistency,
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
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
    archive_path = args.archive.expanduser()
    archive_available = archive_path.exists()
    preflight_archive_rows: list[dict[str, object]] | None = None
    if args.mode == "full":
        if args.threads != PAPER_THREADS:
            raise SystemExit(
                f"Section 3.2 full mode requires --threads {PAPER_THREADS} "
                "to match the paper experiment."
            )
        if archive_available:
            try:
                preflight_archive_rows = load_validated_archive_runs(args.archive)
            except (OSError, ValueError) as error:
                raise SystemExit(
                    "When the optional full-mode archive exists, it must be a "
                    f"valid experiments_logs.zip raw reference: {error}"
                ) from error
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
    raw_reference_applicable = requires_raw_archive_comparison(
        args.mode, rows, metadata
    )
    optional_archive_skip = (
        args.mode == "full"
        and raw_reference_applicable
        and not archive_available
    )
    raw_reference_required = raw_reference_applicable and not optional_archive_skip
    raw_reference_errors: list[str] = []
    raw_reference_report: dict[str, object]
    if raw_reference_required:
        try:
            archive_rows = (
                preflight_archive_rows
                if preflight_archive_rows is not None
                else load_validated_archive_runs(args.archive)
            )
            raw_reference_report = build_raw_archive_comparison(rows, archive_rows)
            raw_reference_report["status"] = "applied"
            raw_reference_report["applied"] = True
            raw_reference_report["required"] = True
            raw_reference_report["archive_available"] = True
            raw_reference_report["archive_path"] = str(
                args.archive.expanduser().resolve()
            )
            raw_reference_errors = raw_archive_validation_errors(raw_reference_report)
            raw_reference_report["validation_errors"] = list(raw_reference_errors)
            errors.extend(raw_reference_errors)
            comparison["iteration_policy"] = (
                "exact raw archive iteration/status check on 115 stable rows; "
                "34 feasible-incumbent MILP partition-limit rows and "
                "nodes=500/seed=6/basic are explicitly censored; qualitative "
                "mean-iteration ordering is enforced over all ten-row "
                "arithmetic means"
            )
        except (OSError, ValueError) as error:
            message = f"could not load Section 3.2 raw archive reference: {error}"
            errors.append(message)
            raw_reference_errors.append(message)
            raw_reference_report = {
                "status": "error",
                "applied": False,
                "required": True,
                "archive_available": archive_available,
                "reference": f"experiments_logs.zip/{ARCHIVE_RUNS_MEMBER}",
                "archive_path": str(args.archive.expanduser().resolve()),
                "load_error": str(error),
                "validation_errors": [message],
            }
    elif optional_archive_skip:
        reason = (
            "Section 3.2 full mode used the embedded 50-job paper grid and "
            "paper settings without an archive-backed raw comparison because "
            f"the archive path does not exist: {archive_path.resolve()}"
        )
        warnings.append(reason)
        raw_reference_report = {
            "status": "skipped",
            "applied": False,
            "required": False,
            "archive_available": False,
            "reference": f"experiments_logs.zip/{ARCHIVE_RUNS_MEMBER}",
            "archive_path": str(archive_path.resolve()),
            "reason_code": ARCHIVE_OPTIONAL_SKIP_REASON,
            "reason": reason,
            "validation_errors": [],
        }
        comparison["iteration_policy"] = (
            "archive comparison skipped because the optional archive was absent; "
            "the embedded paper seed/configuration grid and qualitative paper "
            "conclusions remain enforced"
        )
    else:
        raw_reference_report = {
            "status": "not_applicable",
            "applied": False,
            "required": False,
            "reference": f"experiments_logs.zip/{ARCHIVE_RUNS_MEMBER}",
            "reason": (
                "raw comparison applies only to full mode or parse mode containing "
                "the exact 50-job paper grid"
            ),
            "validation_errors": [],
        }
    if raw_reference_applicable:
        write_json(output / "raw_archive_comparison.json", raw_reference_report)
    comparison["paper_settings_validation"] = grid_report["paper_settings"]
    comparison["raw_archive_comparison"] = raw_reference_report
    write_json(output / "reference_comparison.json", comparison)

    grid_report["conclusion_consistency"] = comparison[
        "conclusion_consistency"
    ]
    grid_report["raw_archive_reference"] = {
        "applicable": raw_reference_applicable,
        "required": raw_reference_required,
        "applied": bool(raw_reference_report.get("applied")),
        "status": raw_reference_report.get("status"),
        "archive_available": raw_reference_report.get("archive_available"),
        "archive_path": raw_reference_report.get("archive_path"),
        "reason_code": raw_reference_report.get("reason_code"),
        "reason": raw_reference_report.get("reason"),
        "reference": raw_reference_report.get("reference"),
        "summary": raw_reference_report.get("summary"),
        "error_count": len(raw_reference_errors),
        "errors": raw_reference_errors,
    }

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

    inputs = provenance_inputs(args, raw_reference_required)
    provenance_notes = [
        "Full mode is the exact 50-job paper sweep: nodes 200:100:600, 2000 arcs, seeds 1:10.",
        "Each job runs Basic, BFS, and MILP in that order on one shared generated instance.",
        "Paper settings: original ADMM, rho=1, maxIter=100000, LInf tolerances 1e-4, and a 3600-second per-method limit.",
        "Figure total time is partition time plus ADMM iteration-loop time; initialization is reported but excluded.",
        "The archive contains one Basic time-limit result at nodes=500, seed=6, and includes it in the mean.",
        "The archive contains 34 feasible HiGHS MILP partitions stopped at their 60-second limit; their partition-solver state and downstream ADMM outcome are cutoff-censored, while feasibility and count fingerprints remain strict.",
        "Archived measured work totals about 15.9 hours sequentially on warm 16-thread jobs; allow roughly 16 hours plus setup on comparable hardware.",
        "Wall-clock comparisons are informational. Julia/package versions and hardware can materially change timing.",
    ]
    if optional_archive_skip:
        provenance_notes.append(
            f"{ARCHIVE_OPTIONAL_SKIP_REASON}: full mode proceeded from the "
            "embedded paper seed/configuration grid; no archive content "
            f"contributed because {archive_path.resolve()} was absent."
        )
    write_provenance(
        output,
        section="3.2 Linear Program: The Network Flow Problem / Figure 11",
        args=args,
        jobs=jobs,
        inputs=inputs,
        notes=provenance_notes,
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
