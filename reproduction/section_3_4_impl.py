"""Execution and artifact layer for :mod:`reproduction.section_3_4`."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
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
        materialize_archive_section,
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
        SMOKE_ENFORCED_METHODS,
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
        materialize_archive_section,
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
        SMOKE_ENFORCED_METHODS,
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
PAPER_ADMM_TIME_LIMIT = 7_200.0
PAPER_THREADS = 16

# Threaded floating-point termination and fixed wall-clock limits can shift a
# fresh result slightly across Julia/runtime/hardware combinations. Keep those
# allowances explicit and bounded rather than treating machine variance as an
# open-ended waiver.
NON_CENSORED_ITERATION_RELATIVE_TOLERANCE = 0.015
NON_CENSORED_ITERATION_ABSOLUTE_TOLERANCE = 5
FRESH_ONLY_TIME_LIMIT_NEAR_CAP_FRACTION = 0.95
FRESH_ONLY_TIME_LIMIT_ITERATION_RELATIVE_TOLERANCE = 0.10
ARCHIVE_PAPER_CANONICAL_SHA256 = (
    "cac4f9997cb23bf4208fd71de57434bf75ac3ab4bb91ce79e8d334abc61a5359"
)
ARCHIVE_COMMAND_RUNNER = "./julia/run.sh"
ARCHIVE_COMMAND_DRIVER = (
    "PDMO.jl/advanced/src/DistributedOpt/runDistributedOpt.jl"
)

# Exact archive provenance for the 30 runs used by Table 1 and Figures 15--16.
# Each tuple is (archive batch, scheduler job id, solver, N, seed).  The ZIP
# also contains seeds 666--1000; those jobs were not inputs to the paper plots.
ARCHIVE_PAPER_JOBS = (
    ("admm_50_40959569", "51362366", "original", 50, 111),
    ("admm_50_40959569", "51362367", "original", 50, 222),
    ("admm_50_40959569", "51362368", "original", 50, 333),
    ("admm_50_40959569", "51362369", "original", 50, 444),
    ("admm_50_40959569", "51362370", "original", 50, 555),
    ("admm_100_40959570", "51362376", "original", 100, 111),
    ("admm_100_40959570", "51362377", "original", 100, 222),
    ("admm_100_40959570", "51362378", "original", 100, 333),
    ("admm_100_40959570", "51362379", "original", 100, 444),
    ("admm_100_40959570", "51362380", "original", 100, 555),
    ("admm_200_40959571", "51362386", "original", 200, 111),
    ("admm_200_40959571", "51362387", "original", 200, 222),
    ("admm_200_40959571", "51362388", "original", 200, 333),
    ("admm_200_40959571", "51362389", "original", 200, 444),
    ("admm_200_40959571", "51362390", "original", 200, 555),
    ("flip_50_40959572", "51362396", "doubly", 50, 111),
    ("flip_50_40959572", "51362397", "doubly", 50, 222),
    ("flip_50_40959572", "51362398", "doubly", 50, 333),
    ("flip_50_40959572", "51362399", "doubly", 50, 444),
    ("flip_50_40959572", "51362400", "doubly", 50, 555),
    ("flip_100_40959573", "51362406", "doubly", 100, 111),
    ("flip_100_40959573", "51362407", "doubly", 100, 222),
    ("flip_100_40959573", "51362408", "doubly", 100, 333),
    ("flip_100_40959573", "51362409", "doubly", 100, 444),
    ("flip_100_40959573", "51362410", "doubly", 100, 555),
    ("flip_200_41015625", "51425195", "doubly", 200, 111),
    ("flip_200_41015625", "51425196", "doubly", 200, 222),
    ("flip_200_41015625", "51425197", "doubly", 200, 333),
    ("flip_200_41015625", "51425198", "doubly", 200, 444),
    ("flip_200_41015625", "51425199", "doubly", 200, 555),
)

RAW_IDENTITY_FIELDS = (
    "n",
    "m",
    "kappa",
    "initial_rho",
    "max_iter",
    "log_interval",
    "threads",
    "mip_gap",
)
RAW_STRUCTURE_FIELDS = (
    "graph_nodes",
    "partition_left",
    "partition_right",
    "graph_edges",
)
RAW_OUTCOME_FIELDS = ("admm_status", "iterations")
RAW_NUMERICAL_FIELDS = (
    "objective",
    "primal_residual_l2",
    "primal_residual_linf",
    "dual_residual_l2",
    "dual_residual_linf",
    "mip_achieved_gap_percent",
)
RAW_TIMING_FIELDS = (
    "partition_time_seconds",
    "graph_partition_time_seconds",
    "mip_solver_time_seconds",
    "admm_time_seconds",
    "admm_initialization_seconds",
)
ARCHIVE_CANONICAL_FIELDS = (
    "source_member",
    "archive_batch",
    "run_id",
    "solver",
    "number_nodes",
    "seed",
    "method",
    "n",
    "m",
    "kappa",
    "initial_rho",
    "max_iter",
    "log_interval",
    "threads",
    "mip_gap",
    "graph_nodes",
    "partition_left",
    "partition_right",
    "graph_edges",
    "mip_status",
    "admm_status",
    "iterations",
)
ARCHIVE_CANONICAL_COMMAND_FIELDS = (
    "source_member",
    "archive_batch",
    "run_id",
    "tokens",
)
MIP_TIME_LIMIT_STATUS = "Time limit reached"
ADMM_TIME_LIMIT_STATUS = "ADMM_TERMINATION_TIME_LIMIT"
PAPER_ADMM_CENSORED_KEY = ("doubly", 200, 333, "basic")

# The archived HiGHS 60-second censoring pattern is itself part of the raw
# provenance contract.  The pattern is identical for original and doubly.
_ARCHIVE_MIP_TIME_CENSORED_CONFIGS = (
    (100, 111, ("milp_0.01",)),
    (100, 333, ("milp_0.01", "milp_0.05", "milp_0.1")),
    (100, 444, ("milp_0.01", "milp_0.05", "milp_0.1")),
    (200, 111, ("milp_0.01", "milp_0.05", "milp_0.1", "milp_0.2")),
    (200, 222, ("milp_0.01",)),
    (200, 333, ("milp_0.01", "milp_0.05", "milp_0.1", "milp_0.2")),
    (200, 444, ("milp_0.01", "milp_0.05", "milp_0.1", "milp_0.2")),
    (200, 555, ("milp_0.01", "milp_0.05", "milp_0.1", "milp_0.2")),
)
ARCHIVE_MIP_TIME_CENSORED_KEYS = frozenset(
    (solver, number_nodes, seed, method)
    for solver in PAPER_SOLVERS
    for number_nodes, seed, methods in _ARCHIVE_MIP_TIME_CENSORED_CONFIGS
    for method in methods
)
ARCHIVE_ADMM_TIME_CENSORED_KEYS = frozenset((PAPER_ADMM_CENSORED_KEY,))

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


def _expected_archive_command_tokens(
    solver: str,
    number_nodes: int,
    seed: int,
) -> tuple[str, ...]:
    """Return the normalized scheduler invocation used by one paper job."""

    return (
        "bash",
        ARCHIVE_COMMAND_RUNNER,
        "-t",
        str(PAPER_THREADS),
        ARCHIVE_COMMAND_DRIVER,
        str(number_nodes),
        str(PAPER_DECISION_DIMENSION),
        str(PAPER_MEASUREMENTS_PER_AGENT),
        solver,
        str(PAPER_RHO),
        str(PAPER_MAX_ITER),
        str(PAPER_LOG_INTERVAL),
        str(seed),
    )


def _normalize_archive_command(
    text: str,
    *,
    source: str,
) -> tuple[tuple[str, ...] | None, list[str]]:
    """Extract the exact Julia invocation from a retained shell command."""

    try:
        tokens = tuple(shlex.split(text, posix=True))
    except ValueError as error:
        return None, [f"Archive command {source} is not valid shell syntax: {error}"]
    driver_indices = [
        index for index, token in enumerate(tokens) if token == ARCHIVE_COMMAND_DRIVER
    ]
    if len(driver_indices) != 1:
        return None, [
            f"Archive command {source} must contain exactly one "
            f"{ARCHIVE_COMMAND_DRIVER!r} token; found {len(driver_indices)}"
        ]
    start = driver_indices[0] - 4
    if start < 0:
        return None, [
            f"Archive command {source} is missing the bash/run.sh/-t/thread prefix"
        ]
    return tokens[start:], []


def read_archive_paper_commands(
    section_root: Path,
) -> tuple[dict[tuple[str, str], tuple[str, ...]], list[str]]:
    """Read normalized command tokens for all 30 selected scheduler jobs."""

    root = section_root.expanduser().resolve()
    commands: dict[tuple[str, str], tuple[str, ...]] = {}
    errors: list[str] = []
    for batch, job_id, *_metadata in ARCHIVE_PAPER_JOBS:
        path = root / batch / job_id / "cmd"
        source = f"{batch}/{job_id}/cmd"
        if not path.is_file():
            errors.append(f"Section 3.4 archive is missing exact paper command: {source}")
            continue
        try:
            command_text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            errors.append(
                f"Archive command {source} is not valid UTF-8: {error}"
            )
            continue
        except OSError as error:
            errors.append(f"Archive command {source} could not be read: {error}")
            continue
        normalized, parse_errors = _normalize_archive_command(
            command_text,
            source=source,
        )
        errors.extend(parse_errors)
        if normalized is not None:
            commands[(batch, job_id)] = normalized
    return commands, errors


def archived_paper_log_paths(section_root: Path) -> list[Path]:
    """Resolve only the 30 scheduler jobs that fed the paper artifacts."""

    root = section_root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Section 3.4 archive directory not found: {root}")
    paths = [root / batch / job_id / "stdout.log" for batch, job_id, *_ in ARCHIVE_PAPER_JOBS]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        relative = [
            path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
            for path in missing
        ]
        raise SystemExit(
            "Section 3.4 archive is missing exact paper job log(s): "
            + ", ".join(relative[:5])
        )
    if len(paths) != 30 or len(set(paths)) != 30:
        raise RuntimeError("Section 3.4 archive paper-job manifest must contain 30 unique logs")
    return paths


def validate_archive_job_manifest(
    records: Sequence[Mapping[str, object]],
    commands: Mapping[tuple[str, str], Sequence[str]],
    command_source_errors: Sequence[str] = (),
) -> dict[str, object]:
    """Tie each parsed seed/configuration and command to its scheduler job."""

    expected = {
        (batch, job_id): (solver, number_nodes, seed)
        for batch, job_id, solver, number_nodes, seed in ARCHIVE_PAPER_JOBS
    }
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        grouped[
            (str(record.get("archive_batch") or ""), str(record.get("run_id") or ""))
        ].append(record)

    observed_jobs = set(grouped)
    expected_jobs = set(expected)
    missing_jobs = sorted(expected_jobs.difference(observed_jobs))
    unexpected_jobs = sorted(observed_jobs.difference(expected_jobs))
    observed_command_jobs = set(commands)
    missing_command_jobs = sorted(expected_jobs.difference(observed_command_jobs))
    unexpected_command_jobs = sorted(observed_command_jobs.difference(expected_jobs))
    errors = list(command_source_errors)
    if missing_jobs:
        errors.append(
            f"Archive paper-job manifest is missing {len(missing_jobs)} job(s); "
            f"first={missing_jobs[:5]}"
        )
    if unexpected_jobs:
        errors.append(
            f"Archive paper-job manifest has {len(unexpected_jobs)} unexpected job(s); "
            f"first={unexpected_jobs[:5]}"
        )
    if missing_command_jobs:
        errors.append(
            "Archive paper-job manifest is missing "
            f"{len(missing_command_jobs)} command(s); first={missing_command_jobs[:5]}"
        )
    if unexpected_command_jobs:
        errors.append(
            "Archive paper-job manifest has "
            f"{len(unexpected_command_jobs)} unexpected command(s); "
            f"first={unexpected_command_jobs[:5]}"
        )

    job_checks: list[dict[str, object]] = []
    for batch, job_id, solver, number_nodes, seed in ARCHIVE_PAPER_JOBS:
        job_records = grouped.get((batch, job_id), [])
        expected_command = _expected_archive_command_tokens(
            solver,
            number_nodes,
            seed,
        )
        observed_command = commands.get((batch, job_id))
        command_match = (
            observed_command is not None
            and tuple(observed_command) == expected_command
        )
        observed_methods = [str(record.get("method") or "") for record in job_records]
        methods_match = sorted(observed_methods) == sorted(PAPER_METHODS)
        metadata_match = all(
            str(record.get("solver") or "") == solver
            and optional_int(record.get("number_nodes")) == number_nodes
            and optional_int(record.get("seed")) == seed
            for record in job_records
        )
        passed = (
            len(job_records) == len(PAPER_METHODS)
            and methods_match
            and metadata_match
            and command_match
        )
        job_checks.append(
            {
                "archive_batch": batch,
                "run_id": job_id,
                "expected_solver": solver,
                "expected_number_nodes": number_nodes,
                "expected_seed": seed,
                "observed_record_count": len(job_records),
                "observed_methods": observed_methods,
                "methods_match": methods_match,
                "metadata_match": metadata_match,
                "expected_command_tokens": list(expected_command),
                "observed_command_tokens": (
                    list(observed_command) if observed_command is not None else None
                ),
                "command_match": command_match,
                "passed": passed,
            }
        )
        if job_records and not (
            len(job_records) == len(PAPER_METHODS)
            and methods_match
            and metadata_match
        ):
            errors.append(
                f"Archive job {batch}/{job_id} does not match "
                f"{solver}/N={number_nodes}/seed={seed} with all seven paper methods"
            )
        if not command_match:
            errors.append(
                f"Archive command mismatch for {batch}/{job_id}: "
                f"expected {list(expected_command)!r}, got "
                f"{list(observed_command) if observed_command is not None else None!r}"
            )

    return {
        "status": "passed" if not errors else "failed",
        "expected_job_count": len(ARCHIVE_PAPER_JOBS),
        "observed_job_count": len(observed_jobs),
        "expected_command_count": len(ARCHIVE_PAPER_JOBS),
        "observed_command_count": len(observed_command_jobs),
        "missing_jobs": [list(key) for key in missing_jobs],
        "unexpected_jobs": [list(key) for key in unexpected_jobs],
        "missing_command_jobs": [list(key) for key in missing_command_jobs],
        "unexpected_command_jobs": [list(key) for key in unexpected_command_jobs],
        "checks": job_checks,
        "errors": errors,
    }


def validate_archive_censor_status_manifest(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Enforce the exact archived MILP and ADMM time-limit pattern."""

    index, duplicates, missing_metadata = _raw_index(records)
    errors: list[str] = []
    if duplicates:
        errors.append(f"Archive censor manifest has duplicate keys: {duplicates[:5]}")
    if missing_metadata:
        errors.append(
            f"Archive censor manifest has {missing_metadata} row(s) missing key metadata"
        )

    observed_mip_time_limits: set[tuple[str, int, int, str]] = set()
    observed_admm_time_limits: set[tuple[str, int, int, str]] = set()
    for key, record in index.items():
        if key[3].startswith("milp_"):
            expected_mip_status = (
                MIP_TIME_LIMIT_STATUS
                if key in ARCHIVE_MIP_TIME_CENSORED_KEYS
                else "Optimal"
            )
            observed_mip_status = str(record.get("mip_status") or "")
            if observed_mip_status == MIP_TIME_LIMIT_STATUS:
                observed_mip_time_limits.add(key)
            if observed_mip_status != expected_mip_status:
                errors.append(
                    f"Archive MIP status mismatch {_raw_text_key(key)}: "
                    f"expected {expected_mip_status!r}, got {observed_mip_status!r}"
                )

        expected_admm_status = (
            ADMM_TIME_LIMIT_STATUS
            if key in ARCHIVE_ADMM_TIME_CENSORED_KEYS
            else "ADMM_TERMINATION_OPTIMAL"
        )
        observed_admm_status = str(record.get("admm_status") or "")
        if observed_admm_status == ADMM_TIME_LIMIT_STATUS:
            observed_admm_time_limits.add(key)
        if observed_admm_status != expected_admm_status:
            errors.append(
                f"Archive ADMM status mismatch {_raw_text_key(key)}: "
                f"expected {expected_admm_status!r}, got {observed_admm_status!r}"
            )

    missing_mip_time_limits = sorted(
        ARCHIVE_MIP_TIME_CENSORED_KEYS.difference(observed_mip_time_limits)
    )
    unexpected_mip_time_limits = sorted(
        observed_mip_time_limits.difference(ARCHIVE_MIP_TIME_CENSORED_KEYS)
    )
    missing_admm_time_limits = sorted(
        ARCHIVE_ADMM_TIME_CENSORED_KEYS.difference(observed_admm_time_limits)
    )
    unexpected_admm_time_limits = sorted(
        observed_admm_time_limits.difference(ARCHIVE_ADMM_TIME_CENSORED_KEYS)
    )
    for values, label in (
        (missing_mip_time_limits, "missing expected MIP time-limit keys"),
        (unexpected_mip_time_limits, "unexpected MIP time-limit keys"),
        (missing_admm_time_limits, "missing expected ADMM time-limit keys"),
        (unexpected_admm_time_limits, "unexpected ADMM time-limit keys"),
    ):
        if values:
            errors.append(
                f"Archive censor manifest has {len(values)} {label}; first={values[:5]}"
            )
    return {
        "status": "passed" if not errors else "failed",
        "expected_mip_time_limit_count": len(ARCHIVE_MIP_TIME_CENSORED_KEYS),
        "observed_mip_time_limit_count": len(observed_mip_time_limits),
        "expected_admm_time_limit_count": len(ARCHIVE_ADMM_TIME_CENSORED_KEYS),
        "observed_admm_time_limit_count": len(observed_admm_time_limits),
        "missing_mip_time_limit_keys": [list(key) for key in missing_mip_time_limits],
        "unexpected_mip_time_limit_keys": [list(key) for key in unexpected_mip_time_limits],
        "missing_admm_time_limit_keys": [list(key) for key in missing_admm_time_limits],
        "unexpected_admm_time_limit_keys": [list(key) for key in unexpected_admm_time_limits],
        "errors": errors,
    }


def build_archive_canonical_manifest(
    records: Sequence[Mapping[str, object]],
    commands: Mapping[tuple[str, str], Sequence[str]],
) -> dict[str, object]:
    """Hash the exact non-timing rows and normalized commands used by the paper."""

    canonical_records: list[dict[str, object]] = []
    for record in records:
        archive_batch = str(record.get("archive_batch") or "")
        run_id = str(record.get("run_id") or "")
        canonical_records.append(
            {
                "source_member": (
                    f"{SECTION_ARCHIVE_PATH}/{archive_batch}/{run_id}/stdout.log"
                ),
                "archive_batch": archive_batch,
                "run_id": run_id,
                "solver": str(record.get("solver") or ""),
                "number_nodes": optional_int(record.get("number_nodes")),
                "seed": optional_int(record.get("seed")),
                "method": str(record.get("method") or ""),
                "n": optional_int(record.get("n")),
                "m": optional_int(record.get("m")),
                "kappa": optional_float(record.get("kappa")),
                "initial_rho": optional_float(record.get("initial_rho")),
                "max_iter": optional_int(record.get("max_iter")),
                "log_interval": optional_int(record.get("log_interval")),
                "threads": optional_int(record.get("threads")),
                "mip_gap": optional_float(record.get("mip_gap")),
                "graph_nodes": optional_int(record.get("graph_nodes")),
                "partition_left": optional_int(record.get("partition_left")),
                "partition_right": optional_int(record.get("partition_right")),
                "graph_edges": optional_int(record.get("graph_edges")),
                "mip_status": (
                    str(record.get("mip_status"))
                    if record.get("mip_status") is not None
                    else None
                ),
                "admm_status": (
                    str(record.get("admm_status"))
                    if record.get("admm_status") is not None
                    else None
                ),
                "iterations": optional_int(record.get("iterations")),
            }
        )
    canonical_records.sort(
        key=lambda record: (
            str(record["source_member"]),
            str(record["method"]),
        )
    )
    canonical_commands = [
        {
            "source_member": f"{SECTION_ARCHIVE_PATH}/{batch}/{job_id}/cmd",
            "archive_batch": batch,
            "run_id": job_id,
            "tokens": [str(token) for token in tokens],
        }
        for (batch, job_id), tokens in commands.items()
    ]
    canonical_commands.sort(key=lambda command: str(command["source_member"]))
    digest_payload = {
        "schema": "section_3_4_archive_paper_rows_and_commands_v2",
        "fields": list(ARCHIVE_CANONICAL_FIELDS),
        "records": canonical_records,
        "command_fields": list(ARCHIVE_CANONICAL_COMMAND_FIELDS),
        "commands": canonical_commands,
    }
    digest = hashlib.sha256(
        json.dumps(
            digest_payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    errors: list[str] = []
    if len(canonical_records) != 210:
        errors.append(
            "Archive canonical manifest has "
            f"{len(canonical_records)} rows; expected 210"
        )
    expected_command_jobs = {
        (batch, job_id) for batch, job_id, *_metadata in ARCHIVE_PAPER_JOBS
    }
    observed_command_jobs = set(commands)
    if len(canonical_commands) != len(ARCHIVE_PAPER_JOBS):
        errors.append(
            "Archive canonical manifest has "
            f"{len(canonical_commands)} commands; expected {len(ARCHIVE_PAPER_JOBS)}"
        )
    if observed_command_jobs != expected_command_jobs:
        missing = sorted(expected_command_jobs.difference(observed_command_jobs))
        unexpected = sorted(observed_command_jobs.difference(expected_command_jobs))
        errors.append(
            "Archive canonical command members do not match the paper jobs: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    if digest != ARCHIVE_PAPER_CANONICAL_SHA256:
        errors.append(
            "Archive canonical row digest mismatch (rows and commands): "
            f"expected {ARCHIVE_PAPER_CANONICAL_SHA256}, got {digest}"
        )
    return {
        **digest_payload,
        "expected_record_count": 210,
        "record_count": len(canonical_records),
        "expected_command_count": len(ARCHIVE_PAPER_JOBS),
        "command_count": len(canonical_commands),
        "expected_sha256": ARCHIVE_PAPER_CANONICAL_SHA256,
        "observed_sha256": digest,
        "matches_expected": digest == ARCHIVE_PAPER_CANONICAL_SHA256,
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "timing_policy": (
            "Wall-clock fields are excluded from this archive-integrity digest; "
            "fresh timing comparisons remain informational."
        ),
    }


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


def evaluate_conclusion_consistency(
    aggregate: Sequence[Mapping[str, object]],
    mode: str,
) -> tuple[dict[str, object], list[str]]:
    """Report the robust and hardware-dependent Figures 15--16 patterns."""

    index = {
        (str(row["solver"]), int(row["number_nodes"]), str(row["method"])): row
        for row in aggregate
    }
    expected_keys = {
        (solver, number_nodes, method)
        for solver in PAPER_SOLVERS
        for number_nodes in PAPER_NODE_COUNTS
        for method in PAPER_METHODS
    }
    missing_groups = sorted(expected_keys.difference(index))
    wrong_run_counts = [
        {
            "solver": solver,
            "number_nodes": number_nodes,
            "method": method,
            "observed": int(index[(solver, number_nodes, method)].get("n_runs") or 0),
            "expected": len(PAPER_SEEDS),
        }
        for solver, number_nodes, method in sorted(expected_keys.intersection(index))
        if int(index[(solver, number_nodes, method)].get("n_runs") or 0)
        != len(PAPER_SEEDS)
    ]
    complete_paper_grid = not missing_groups and not wrong_run_counts
    enforce = complete_paper_grid and mode in {"full", "parse"}
    checks: list[dict[str, object]] = []
    errors: list[str] = []

    def add_check(
        *,
        check_id: str,
        solver: str,
        number_nodes: int,
        category: str,
        metric: str,
        left_method: str,
        right_method: str,
        informational: bool,
    ) -> None:
        left_value = float(index[(solver, number_nodes, left_method)][metric])
        right_value = float(index[(solver, number_nodes, right_method)][metric])
        passed = left_value < right_value
        enforced = enforce and not informational
        check = {
            "id": check_id,
            "solver": solver,
            "number_nodes": number_nodes,
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
        checks.append(check)
        if enforced and not passed:
            errors.append(
                "Section 3.4 conclusion mismatch at "
                f"solver={solver}, nodes={number_nodes}: expected {left_method} "
                f"{metric} ({left_value}) < {right_method} {metric} ({right_value})"
            )

    if complete_paper_grid:
        for solver in PAPER_SOLVERS:
            for number_nodes in PAPER_NODE_COUNTS:
                for method in PAPER_METHODS:
                    if method == "basic":
                        continue
                    add_check(
                        check_id=f"{method}_iterations_below_basic",
                        solver=solver,
                        number_nodes=number_nodes,
                        category="iteration_pattern",
                        metric="mean_iterations",
                        left_method=method,
                        right_method="basic",
                        informational=method.startswith("milp_"),
                    )
                    add_check(
                        check_id=f"{method}_total_time_below_basic",
                        solver=solver,
                        number_nodes=number_nodes,
                        category="timing_pattern",
                        metric="mean_total_time_seconds",
                        left_method=method,
                        right_method="basic",
                        informational=True,
                    )

                milp_methods = ("milp_0.01", "milp_0.05", "milp_0.1", "milp_0.2")
                for tighter, looser in zip(milp_methods, milp_methods[1:]):
                    add_check(
                        check_id="milp_partition_time_decreases_as_gap_loosens",
                        solver=solver,
                        number_nodes=number_nodes,
                        category="milp_gap_partition_time_trend",
                        metric="mean_partition_time_seconds",
                        left_method=looser,
                        right_method=tighter,
                        informational=True,
                    )

    enforced_checks = [check for check in checks if bool(check["enforced"])]
    informational_checks = [check for check in checks if bool(check["informational"])]
    required_checks = [check for check in checks if not bool(check["informational"])]
    enforced_failed = sum(not bool(check["passed"]) for check in enforced_checks)
    required_failed = sum(not bool(check["passed"]) for check in required_checks)
    return (
        {
            "mode": mode,
            "complete_paper_grid": complete_paper_grid,
            "missing_groups": [
                {"solver": solver, "number_nodes": number_nodes, "method": method}
                for solver, number_nodes, method in missing_groups
            ],
            "wrong_run_counts": wrong_run_counts,
            "enforced_for_this_mode": enforce,
            "enforcement_policy": (
                "On a complete fresh full grid (or complete parse), enforce that BFS "
                "and GNN have fewer mean iterations than Basic in every solver/node "
                "panel. MILP iteration comparisons are informational because their "
                "aggregates contain 60-second partition solves; all wall-time and "
                "MILP-gap trend comparisons are informational."
            ),
            "checks": checks,
            "summary": {
                "check_count": len(checks),
                "enforced_count": len(enforced_checks),
                "enforced_passed": len(enforced_checks) - enforced_failed,
                "enforced_failed": enforced_failed,
                "all_enforced_checks_passed": enforced_failed == 0,
                "required_pattern_count": len(required_checks),
                "required_pattern_passed": len(required_checks) - required_failed,
                "required_pattern_failed": required_failed,
                "all_required_patterns_passed": required_failed == 0,
                "informational_count": len(informational_checks),
                "informational_passed": sum(
                    bool(check["passed"]) for check in informational_checks
                ),
            },
        },
        errors,
    )


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


def _raw_record_key(record: Mapping[str, object]) -> tuple[str, int, int, str] | None:
    solver = str(record.get("solver") or "")
    number_nodes = optional_int(record.get("number_nodes"))
    seed = optional_int(record.get("seed"))
    method = str(record.get("method") or "")
    if not solver or number_nodes is None or seed is None or not method:
        return None
    return solver, number_nodes, seed, method


def _raw_text_key(key: tuple[str, int, int, str]) -> str:
    solver, number_nodes, seed, method = key
    return f"{solver}/N={number_nodes}/seed={seed}/{method}"


def _raw_check(
    checks: list[dict[str, object]],
    *,
    key: str,
    field: str,
    observed: object,
    expected: object,
    enforced: bool,
    category: str,
    reason: str,
    passed: bool | None = None,
    details: Mapping[str, object] | None = None,
) -> None:
    check = {
        "key": key,
        "field": field,
        "observed": observed,
        "expected": expected,
        "passed": observed == expected if passed is None else passed,
        "enforced": enforced,
        "category": category,
        "reason": reason,
    }
    if details:
        check.update(details)
    checks.append(check)


def _raw_index(
    records: Sequence[Mapping[str, object]],
) -> tuple[
    dict[tuple[str, int, int, str], Mapping[str, object]],
    list[tuple[str, int, int, str]],
    int,
]:
    index: dict[tuple[str, int, int, str], Mapping[str, object]] = {}
    duplicates: set[tuple[str, int, int, str]] = set()
    missing_metadata = 0
    for record in records:
        key = _raw_record_key(record)
        if key is None:
            missing_metadata += 1
            continue
        if key in index:
            duplicates.add(key)
        index[key] = record
    return index, sorted(duplicates), missing_metadata


def build_raw_archive_comparison(
    fresh_records: Sequence[Mapping[str, object]],
    archive_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Compare a fresh full grid with the exact paper rows retained in the ZIP.

    Identity and non-censored structure are strict. Stable ADMM statuses stay
    exact, while their iterations use a small explicit machine-variance bound.
    ADMM time censoring is classified dynamically from both rows; a fresh-only
    cutoff must satisfy near-cap timing and iteration-progress integrity gates.
    Timings and floating residual/objective values otherwise remain informational
    because they depend on hardware throughput and parallel floating-point
    execution.
    """

    fresh_index, fresh_duplicates, fresh_missing_metadata = _raw_index(fresh_records)
    archive_index, archive_duplicates, archive_missing_metadata = _raw_index(
        archive_records
    )
    paper_keys = {
        (solver, number_nodes, seed, method)
        for solver in PAPER_SOLVERS
        for number_nodes in PAPER_NODE_COUNTS
        for seed in PAPER_SEEDS
        for method in PAPER_METHODS
    }
    fresh_keys = set(fresh_index)
    archive_keys = set(archive_index)
    missing_fresh_keys = sorted(archive_keys.difference(fresh_keys))
    extra_fresh_keys = sorted(fresh_keys.difference(archive_keys))
    archive_missing_paper_keys = sorted(paper_keys.difference(archive_keys))
    archive_extra_keys = sorted(archive_keys.difference(paper_keys))

    checks: list[dict[str, object]] = []
    warnings: list[str] = []
    censored_rows: list[dict[str, object]] = []
    stable_outcome_row_count = 0
    stable_exact_iteration_match_count = 0
    stable_iteration_within_tolerance_count = 0
    stable_status_match_count = 0
    bounded_iteration_variance_row_count = 0
    archive_admm_time_censored_row_count = 0
    fresh_admm_time_censored_row_count = 0
    both_admm_time_censored_row_count = 0
    fresh_only_admm_time_censored_row_count = 0
    fresh_only_integrity_checked_count = 0
    fresh_only_integrity_passed_count = 0
    fresh_only_integrity_failed_count = 0
    for key in sorted(fresh_keys.intersection(archive_keys)):
        fresh = fresh_index[key]
        archived = archive_index[key]
        text_key = _raw_text_key(key)
        method = key[3]
        mip_censor_reasons: list[str] = []
        if method.startswith("milp_"):
            if str(archived.get("mip_status") or "") == MIP_TIME_LIMIT_STATUS:
                mip_censor_reasons.append("archive_mip_time_limit")
            if str(fresh.get("mip_status") or "") == MIP_TIME_LIMIT_STATUS:
                mip_censor_reasons.append("fresh_mip_time_limit")
        archive_admm_status = str(archived.get("admm_status") or "")
        fresh_admm_status = str(fresh.get("admm_status") or "")
        archive_admm_time_censored = archive_admm_status == ADMM_TIME_LIMIT_STATUS
        fresh_admm_time_censored = fresh_admm_status == ADMM_TIME_LIMIT_STATUS
        admm_censor_reasons: list[str] = []
        if archive_admm_time_censored:
            archive_admm_time_censored_row_count += 1
            admm_censor_reasons.append("archive_admm_time_limit")
        if fresh_admm_time_censored:
            fresh_admm_time_censored_row_count += 1
            admm_censor_reasons.append("fresh_admm_time_limit")
        if archive_admm_time_censored and fresh_admm_time_censored:
            both_admm_time_censored_row_count += 1
        elif fresh_admm_time_censored:
            fresh_only_admm_time_censored_row_count += 1

        for field in RAW_IDENTITY_FIELDS:
            _raw_check(
                checks,
                key=text_key,
                field=field,
                observed=fresh.get(field),
                expected=archived.get(field),
                enforced=True,
                category="identity",
                reason="same archived paper configuration and seeded instance",
            )

        structure_enforced = not mip_censor_reasons
        structure_reason = (
            "logged graph/partition count fingerprint"
            if structure_enforced
            else "MILP partition is censored by a 60-second wall-time limit"
        )
        for field in RAW_STRUCTURE_FIELDS:
            _raw_check(
                checks,
                key=text_key,
                field=field,
                observed=fresh.get(field),
                expected=archived.get(field),
                enforced=structure_enforced,
                category="structure" if structure_enforced else "mip_time_limit_censored",
                reason=structure_reason,
            )

        archive_iterations = optional_int(archived.get("iterations"))
        fresh_iterations = optional_int(fresh.get("iterations"))
        iteration_difference = (
            abs(fresh_iterations - archive_iterations)
            if fresh_iterations is not None and archive_iterations is not None
            else None
        )
        allowed_iteration_difference = (
            max(
                NON_CENSORED_ITERATION_ABSOLUTE_TOLERANCE,
                math.ceil(
                    NON_CENSORED_ITERATION_RELATIVE_TOLERANCE
                    * max(archive_iterations, 1)
                ),
            )
            if archive_iterations is not None
            else None
        )
        iteration_within_tolerance = (
            iteration_difference is not None
            and allowed_iteration_difference is not None
            and iteration_difference <= allowed_iteration_difference
        )
        iteration_exact = (
            iteration_difference == 0 if iteration_difference is not None else False
        )
        status_exact = fresh_admm_status == archive_admm_status

        outcome_censor_reasons = mip_censor_reasons + admm_censor_reasons
        stable_outcome = not outcome_censor_reasons
        outcome_reason = (
            "exact status and explicitly bounded non-censored ADMM iterations"
            if stable_outcome
            else "downstream outcome is censored by a MILP or ADMM wall-time limit"
        )
        _raw_check(
            checks,
            key=text_key,
            field="admm_status",
            observed=fresh_admm_status,
            expected=archive_admm_status,
            enforced=stable_outcome,
            category="outcome" if stable_outcome else "outcome_censored",
            reason=outcome_reason,
        )
        _raw_check(
            checks,
            key=text_key,
            field="iterations",
            observed=fresh_iterations,
            expected=archive_iterations,
            enforced=stable_outcome,
            category="outcome" if stable_outcome else "outcome_censored",
            reason=outcome_reason,
            passed=(
                iteration_within_tolerance
                if stable_outcome
                else fresh_iterations == archive_iterations
            ),
            details={
                "exact_match": iteration_exact,
                "absolute_difference": iteration_difference,
                "allowed_absolute_difference": (
                    allowed_iteration_difference if stable_outcome else None
                ),
                "relative_tolerance": (
                    NON_CENSORED_ITERATION_RELATIVE_TOLERANCE
                    if stable_outcome
                    else None
                ),
                "absolute_tolerance": (
                    NON_CENSORED_ITERATION_ABSOLUTE_TOLERANCE
                    if stable_outcome
                    else None
                ),
            },
        )

        if stable_outcome:
            stable_outcome_row_count += 1
            if status_exact:
                stable_status_match_count += 1
            if iteration_exact:
                stable_exact_iteration_match_count += 1
            if iteration_within_tolerance:
                stable_iteration_within_tolerance_count += 1
            if iteration_within_tolerance and not iteration_exact:
                bounded_iteration_variance_row_count += 1
                warnings.append(
                    f"{text_key}: fresh/archive iterations differ by "
                    f"{iteration_difference} within the allowed "
                    f"{allowed_iteration_difference}-iteration machine-variance bound"
                )

        fresh_only_integrity_checked = (
            fresh_admm_time_censored
            and not archive_admm_time_censored
            and not mip_censor_reasons
        )
        fresh_only_integrity_passed: bool | None = None
        archive_admm_time_near_cap: bool | None = None
        fresh_admm_time_near_cap: bool | None = None
        relative_iteration_difference: float | None = None
        if fresh_only_integrity_checked:
            fresh_only_integrity_checked_count += 1
            near_cap_seconds = (
                FRESH_ONLY_TIME_LIMIT_NEAR_CAP_FRACTION * PAPER_ADMM_TIME_LIMIT
            )
            archive_admm_time = optional_float(archived.get("admm_time_seconds"))
            fresh_admm_time = optional_float(fresh.get("admm_time_seconds"))
            archive_admm_time_near_cap = (
                archive_admm_time is not None and archive_admm_time >= near_cap_seconds
            )
            fresh_admm_time_near_cap = (
                fresh_admm_time is not None and fresh_admm_time >= near_cap_seconds
            )
            relative_iteration_difference = (
                iteration_difference / max(abs(archive_iterations), 1)
                if iteration_difference is not None and archive_iterations is not None
                else None
            )
            iteration_progress_near_archive = (
                relative_iteration_difference is not None
                and relative_iteration_difference
                <= FRESH_ONLY_TIME_LIMIT_ITERATION_RELATIVE_TOLERANCE
            )
            _raw_check(
                checks,
                key=text_key,
                field="archive_admm_time_near_cap",
                observed=archive_admm_time,
                expected=f">= {near_cap_seconds:.1f}",
                enforced=True,
                category="fresh_only_admm_time_limit_integrity",
                reason=(
                    "a fresh-only ADMM cutoff is machine-compatible only when "
                    "the archived run also finished near the same 7200-second cap"
                ),
                passed=archive_admm_time_near_cap,
                details={
                    "minimum_seconds": near_cap_seconds,
                    "cap_seconds": PAPER_ADMM_TIME_LIMIT,
                },
            )
            _raw_check(
                checks,
                key=text_key,
                field="fresh_admm_time_near_cap",
                observed=fresh_admm_time,
                expected=f">= {near_cap_seconds:.1f}",
                enforced=True,
                category="fresh_only_admm_time_limit_integrity",
                reason=(
                    "a time-limit status must carry elapsed ADMM time near the "
                    "configured 7200-second cap"
                ),
                passed=fresh_admm_time_near_cap,
                details={
                    "minimum_seconds": near_cap_seconds,
                    "cap_seconds": PAPER_ADMM_TIME_LIMIT,
                },
            )
            _raw_check(
                checks,
                key=text_key,
                field="fresh_only_admm_iteration_relative_drift",
                observed=relative_iteration_difference,
                expected=(
                    f"<= {FRESH_ONLY_TIME_LIMIT_ITERATION_RELATIVE_TOLERANCE:.2f}"
                ),
                enforced=True,
                category="fresh_only_admm_time_limit_integrity",
                reason=(
                    "a fresh-only cutoff must retain comparable iteration progress "
                    "to the near-cap archived run"
                ),
                passed=iteration_progress_near_archive,
                details={
                    "maximum_relative_difference": (
                        FRESH_ONLY_TIME_LIMIT_ITERATION_RELATIVE_TOLERANCE
                    ),
                    "absolute_iteration_difference": iteration_difference,
                },
            )
            fresh_only_integrity_passed = bool(
                archive_admm_time_near_cap
                and fresh_admm_time_near_cap
                and iteration_progress_near_archive
            )
            if fresh_only_integrity_passed:
                fresh_only_integrity_passed_count += 1
                warnings.append(
                    f"{text_key}: fresh-only ADMM cutoff accepted because both "
                    f"runs are within the {FRESH_ONLY_TIME_LIMIT_NEAR_CAP_FRACTION:.0%} "
                    f"cap boundary and iteration drift is "
                    f"{relative_iteration_difference:.2%}"
                )
            else:
                fresh_only_integrity_failed_count += 1

        if method.startswith("milp_"):
            _raw_check(
                checks,
                key=text_key,
                field="mip_status",
                observed=fresh.get("mip_status"),
                expected=archived.get("mip_status"),
                enforced=not mip_censor_reasons,
                category="outcome" if not mip_censor_reasons else "mip_time_limit_censored",
                reason=(
                    "deterministic non-censored MILP status"
                    if not mip_censor_reasons
                    else "MILP status is censored by a 60-second wall-time limit"
                ),
            )

        for field in RAW_NUMERICAL_FIELDS:
            if fresh.get(field) is None and archived.get(field) is None:
                continue
            _raw_check(
                checks,
                key=text_key,
                field=field,
                observed=fresh.get(field),
                expected=archived.get(field),
                enforced=False,
                category="numerical_informational",
                reason="floating-point or time-limited numerical diagnostic; not a plot input",
            )
        for field in RAW_TIMING_FIELDS:
            if fresh.get(field) is None and archived.get(field) is None:
                continue
            _raw_check(
                checks,
                key=text_key,
                field=field,
                observed=fresh.get(field),
                expected=archived.get(field),
                enforced=False,
                category="timing_informational",
                reason="wall-clock measurement is hardware-dependent",
            )

        if outcome_censor_reasons:
            if mip_censor_reasons and admm_censor_reasons:
                classification = "milp_and_admm_time_censored"
            elif mip_censor_reasons:
                classification = "milp_time_censored"
            elif archive_admm_time_censored and fresh_admm_time_censored:
                classification = "both_admm_time_censored"
            elif archive_admm_time_censored:
                classification = "archive_admm_time_censored"
            elif fresh_only_integrity_passed:
                classification = "fresh_only_admm_time_censored_near_cap"
            else:
                classification = "fresh_only_admm_time_censored_integrity_failed"
            censored_rows.append(
                {
                    "key": text_key,
                    "classification": classification,
                    "mip_censored": bool(mip_censor_reasons),
                    "admm_censored": bool(admm_censor_reasons),
                    "archive_admm_time_censored": archive_admm_time_censored,
                    "fresh_admm_time_censored": fresh_admm_time_censored,
                    "fresh_only_integrity_checked": fresh_only_integrity_checked,
                    "fresh_only_integrity_passed": fresh_only_integrity_passed,
                    "archive_admm_time_near_cap": archive_admm_time_near_cap,
                    "fresh_admm_time_near_cap": fresh_admm_time_near_cap,
                    "relative_iteration_difference": relative_iteration_difference,
                    "structure_equality_enforced": structure_enforced,
                    "outcome_equality_enforced": False,
                    "reasons": sorted(set(outcome_censor_reasons)),
                }
            )

    same_instance_missing: list[str] = []
    for number_nodes in PAPER_NODE_COUNTS:
        for seed in PAPER_SEEDS:
            original_key = ("original", number_nodes, seed, "basic")
            doubly_key = ("doubly", number_nodes, seed, "basic")
            original = fresh_index.get(original_key)
            doubly = fresh_index.get(doubly_key)
            instance_key = f"same_instance/N={number_nodes}/seed={seed}/basic"
            if original is None or doubly is None:
                same_instance_missing.append(instance_key)
                continue
            for field in ("kappa", *RAW_STRUCTURE_FIELDS):
                _raw_check(
                    checks,
                    key=instance_key,
                    field=field,
                    observed=doubly.get(field),
                    expected=original.get(field),
                    enforced=True,
                    category="same_instance",
                    reason=(
                        "original and doubly solvers must match the available "
                        "seeded-instance count fingerprint"
                    ),
                )

    enforced_checks = [check for check in checks if bool(check["enforced"])]
    enforced_failed = sum(not bool(check["passed"]) for check in enforced_checks)
    categories: dict[str, int] = defaultdict(int)
    for check in checks:
        categories[str(check["category"])] += 1
    mip_censored_row_count = sum(
        bool(row["mip_censored"]) for row in censored_rows
    )
    admm_time_censored_row_count = sum(
        bool(row["admm_censored"]) for row in censored_rows
    )
    return {
        "reference": "experiments_logs.zip/3-4-distributed selected paper rows",
        "policy": {
            "identity": "exact for every row",
            "structure": (
                "logged node/edge and partition-cardinality counts are exact except when "
                "either archived or fresh MILP hit its 60 s limit; ADMM censoring never "
                "relaxes structure, and the archive has no edge lists or vertex-to-side "
                "memberships"
            ),
            "stable_admm_status": "exact for every non-censored outcome row",
            "stable_admm_iterations": (
                "bounded by max(5 iterations, 1.5% of the archived iteration count); "
                "exact equality is retained as a diagnostic"
            ),
            "admm_time_censoring": (
                "classified dynamically when either archived or fresh status reaches "
                "the 7200-second ADMM time limit"
            ),
            "fresh_only_admm_time_limit": (
                "accepted only when both archived and fresh ADMM elapsed times are at "
                "least 95% of the 7200-second cap and iteration drift is at most 10%; "
                "the downstream outcome of a MILP-censored row is already non-exact"
            ),
            "milp_time_censoring": (
                "partition structure and downstream outcome are non-exact when either "
                "side reaches the 60-second MILP limit"
            ),
            "timings": "informational only except fresh-only cutoff integrity gates",
            "numerical_diagnostics": "informational only",
        },
        "fresh_record_count": len(fresh_records),
        "archive_record_count": len(archive_records),
        "expected_paper_record_count": 210,
        "fresh_duplicates": [list(key) for key in fresh_duplicates],
        "archive_duplicates": [list(key) for key in archive_duplicates],
        "fresh_missing_metadata_count": fresh_missing_metadata,
        "archive_missing_metadata_count": archive_missing_metadata,
        "missing_fresh_keys": [list(key) for key in missing_fresh_keys],
        "extra_fresh_keys": [list(key) for key in extra_fresh_keys],
        "archive_missing_paper_keys": [list(key) for key in archive_missing_paper_keys],
        "archive_extra_keys": [list(key) for key in archive_extra_keys],
        "same_instance_missing": same_instance_missing,
        "censored_rows": censored_rows,
        "checks": checks,
        "warnings": warnings,
        "summary": {
            "compared": len(checks),
            "enforced": len(enforced_checks),
            "enforced_passed": len(enforced_checks) - enforced_failed,
            "enforced_failed": enforced_failed,
            "all_enforced_checks_passed": enforced_failed == 0,
            "stable_outcome_row_count": stable_outcome_row_count,
            "stable_exact_iteration_match_count": (
                stable_exact_iteration_match_count
            ),
            "stable_iteration_within_tolerance_count": (
                stable_iteration_within_tolerance_count
            ),
            "stable_status_match_count": stable_status_match_count,
            "all_stable_iterations_match": (
                stable_outcome_row_count == stable_exact_iteration_match_count
            ),
            "all_stable_iterations_within_tolerance": (
                stable_outcome_row_count
                == stable_iteration_within_tolerance_count
            ),
            "all_stable_statuses_match": (
                stable_outcome_row_count == stable_status_match_count
            ),
            "bounded_iteration_variance_row_count": (
                bounded_iteration_variance_row_count
            ),
            "censored_row_count": len(censored_rows),
            "mip_censored_row_count": mip_censored_row_count,
            "admm_time_censored_row_count": admm_time_censored_row_count,
            "archive_admm_time_censored_row_count": (
                archive_admm_time_censored_row_count
            ),
            "fresh_admm_time_censored_row_count": (
                fresh_admm_time_censored_row_count
            ),
            "both_admm_time_censored_row_count": (
                both_admm_time_censored_row_count
            ),
            "fresh_only_admm_time_censored_row_count": (
                fresh_only_admm_time_censored_row_count
            ),
            "fresh_only_integrity_checked_count": (
                fresh_only_integrity_checked_count
            ),
            "fresh_only_integrity_passed_count": (
                fresh_only_integrity_passed_count
            ),
            "fresh_only_integrity_failed_count": (
                fresh_only_integrity_failed_count
            ),
            "fresh_only_integrity_not_applicable_due_to_mip_censor_count": (
                fresh_only_admm_time_censored_row_count
                - fresh_only_integrity_checked_count
            ),
            "warning_count": len(warnings),
            "timings_are_informational_except_fresh_only_integrity": True,
            "check_categories": dict(sorted(categories.items())),
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
        ("missing_fresh_keys", "archive keys missing from fresh full run"),
        ("extra_fresh_keys", "fresh keys absent from selected archive rows"),
        ("archive_missing_paper_keys", "paper keys missing from archive reference"),
        ("archive_extra_keys", "non-paper keys in archive reference"),
        ("same_instance_missing", "same-instance Basic comparisons missing"),
    ):
        values = comparison.get(list_field)
        if isinstance(values, list) and values:
            errors.append(f"Raw archive comparison found {len(values)} {label}; first={values[:5]}")

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
    for check in failed:
        key = check.get("key")
        field = check.get("field")
        allowed_difference = check.get("allowed_absolute_difference")
        minimum_seconds = check.get("minimum_seconds")
        maximum_relative_difference = check.get("maximum_relative_difference")
        if field == "iterations" and allowed_difference is not None:
            errors.append(
                f"Raw archive mismatch {key}/iterations: absolute drift "
                f"{check.get('absolute_difference')!r} exceeds the allowed "
                f"{allowed_difference!r}-iteration machine-variance bound"
            )
        elif minimum_seconds is not None:
            errors.append(
                f"Raw archive mismatch {key}/{field}: observed "
                f"{check.get('observed')!r}s is below the required "
                f"{FRESH_ONLY_TIME_LIMIT_NEAR_CAP_FRACTION:.0%} near-cap "
                f"boundary ({minimum_seconds!r}s)"
            )
        elif maximum_relative_difference is not None:
            observed_relative = optional_float(check.get("observed"))
            observed_text = (
                f"{observed_relative:.2%}"
                if observed_relative is not None
                else repr(check.get("observed"))
            )
            errors.append(
                f"Raw archive mismatch {key}/{field}: iteration drift "
                f"{observed_text} exceeds "
                f"{float(maximum_relative_difference):.0%}"
            )
        else:
            errors.append(
                "Raw archive mismatch "
                f"{key}/{field}: expected {check.get('expected')!r}, "
                f"got {check.get('observed')!r}"
            )
    return errors


def _merge_raw_archive_validation(
    validation: dict[str, object],
    comparison: Mapping[str, object],
) -> None:
    """Attach raw-reference errors, warnings, and summary to final validation."""

    validation_errors = validation.get("errors")
    validation_warnings = validation.get("warnings")
    if not isinstance(validation_errors, list) or not isinstance(
        validation_warnings, list
    ):
        raise TypeError("Validation payload must contain mutable errors and warnings lists")
    validation_errors.extend(raw_archive_validation_errors(comparison))
    raw_warnings = comparison.get("warnings")
    if isinstance(raw_warnings, list):
        validation_warnings.extend(str(warning) for warning in raw_warnings)
    validation["raw_archive_summary"] = comparison["summary"]


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
                        enforced=method in SMOKE_ENFORCED_METHODS,
                    )

    compared = len(checks)
    passed = sum(bool(check["passed"]) for check in checks)
    enforced_checks = [check for check in checks if bool(check["enforced"])]
    enforced_passed = sum(bool(check["passed"]) for check in enforced_checks)
    conclusion_consistency, _ = evaluate_conclusion_consistency(aggregate, mode)
    return {
        "reference": "Automating Reformulation for Parallel ADMM, Table 1 and Figures 15--16",
        "mode": mode,
        "checks": checks,
        "conclusion_consistency": conclusion_consistency,
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
    conclusion = comparison.get("conclusion_consistency")
    if not isinstance(conclusion, Mapping):
        errors.append("Reference comparison did not produce conclusion-consistency checks")
    else:
        conclusion_checks = conclusion.get("checks")
        if not isinstance(conclusion_checks, list):
            errors.append("Conclusion consistency did not produce a check list")
        else:
            for check in conclusion_checks:
                if (
                    isinstance(check, Mapping)
                    and bool(check.get("enforced"))
                    and not bool(check.get("passed"))
                ):
                    errors.append(
                        "Conclusion pattern mismatch "
                        f"{check.get('solver')}/N={check.get('number_nodes')}/"
                        f"{check.get('id')}: expected {check.get('left_method')} "
                        f"{check.get('metric')} < {check.get('right_method')}"
                    )
    missing = comparison.get("missing_reference_keys")
    if mode in {"archived", "full"} and isinstance(missing, list) and missing:
        errors.append(f"Missing {len(missing)} full-grid paper reference keys")
    if mode == "smoke":
        expected = {
            (f"smoke/original/N=50/seed=111/{method}", metric)
            for method in selected_methods
            if method in SMOKE_ENFORCED_METHODS
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
    allowed_mip_statuses = {"Optimal", "Time limit reached"}
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
        if mode in {"archived", "full"}:
            expected_configuration = (
                *expected_configuration,
                ("threads", optional_int(record.get("threads")), PAPER_THREADS),
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
        graph_nodes = optional_int(record.get("graph_nodes"))
        partition_left = optional_int(record.get("partition_left"))
        partition_right = optional_int(record.get("partition_right"))
        if (
            graph_nodes is not None
            and partition_left is not None
            and partition_right is not None
            and graph_nodes != partition_left + partition_right
        ):
            errors.append(f"{key} graph_nodes does not equal partition_left + partition_right")
        if method.startswith("milp_"):
            missing_fields.extend(
                field
                for field in (
                    "mip_status",
                    "mip_achieved_gap_percent",
                    "mip_solver_time_seconds",
                )
                if record.get(field) is None
                or (
                    field == "mip_status" and not str(record.get(field)).strip()
                )
            )
            mip_status = str(record.get("mip_status") or "")
            if mip_status and mip_status not in allowed_mip_statuses:
                errors.append(f"{key} has unsupported MIP status {mip_status!r}")
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


def build_archive_reference_validation(
    records: Sequence[Mapping[str, object]],
    commands: Mapping[tuple[str, str], Sequence[str]],
    command_source_errors: Sequence[str] = (),
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
    dict[str, object],
]:
    """Build and strictly validate every archived paper aggregate."""

    aggregate = aggregate_runs(records)
    table_rows = build_table_1(records)
    comparison = build_reference_comparison(records, aggregate, table_rows, "archived")
    validation = validate_records(
        records,
        mode="archived",
        selected_methods=PAPER_METHODS,
        failed_job_names=(),
    )
    job_manifest = validate_archive_job_manifest(
        records,
        commands,
        command_source_errors,
    )
    censor_manifest = validate_archive_censor_status_manifest(records)
    canonical_manifest = build_archive_canonical_manifest(records, commands)
    validation["errors"].extend(job_manifest["errors"])
    validation["errors"].extend(censor_manifest["errors"])
    validation["errors"].extend(canonical_manifest["errors"])
    validation["errors"].extend(
        reference_validation_errors(
            comparison,
            mode="archived",
            selected_methods=PAPER_METHODS,
        )
    )
    validation["archive_job_manifest"] = job_manifest
    validation["archive_censor_status_manifest"] = censor_manifest
    validation["archive_canonical_manifest"] = canonical_manifest
    validation["reference_summary"] = comparison["summary"]
    validation["conclusion_consistency"] = comparison["conclusion_consistency"]
    validation["status"] = "passed" if not validation["errors"] else "failed"
    return aggregate, table_rows, comparison, validation


def _set_archive_source_identifiers(
    records: Sequence[dict[str, object]], archive: Path
) -> None:
    archive_path = archive.expanduser().resolve()
    for record in records:
        member = (
            Path("experiments_logs")
            / SECTION_ARCHIVE_PATH
            / str(record.get("archive_batch") or "")
            / str(record.get("run_id") or "")
            / "stdout.log"
        )
        record["source_log"] = f"{archive_path}::{member.as_posix()}"


def load_archive_reference(
    archive: Path,
    *,
    output: Path | None = None,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
    dict[str, object],
]:
    """Load the exact 30 paper jobs, validate them, and optionally persist proof."""

    with materialize_archive_section(archive, SECTION_ARCHIVE_PATH) as archive_source:
        archive_commands, command_errors = read_archive_paper_commands(archive_source)
        archive_records = parse_logs(
            archived_paper_log_paths(archive_source), "archived"
        )
    _set_archive_source_identifiers(archive_records, archive)
    aggregate, table_rows, comparison, validation = build_archive_reference_validation(
        archive_records,
        archive_commands,
        command_errors,
    )
    if output is not None:
        _write_archive_reference_artifacts(
            output,
            archive_records,
            aggregate,
            table_rows,
            comparison,
            validation,
        )
    return archive_records, aggregate, table_rows, comparison, validation


def _write_archive_reference_artifacts(
    output: Path,
    records: Sequence[Mapping[str, object]],
    aggregate: Sequence[Mapping[str, object]],
    table_rows: Sequence[Mapping[str, object]],
    comparison: Mapping[str, object],
    validation: Mapping[str, object],
) -> None:
    """Persist an already validated archive preflight after output preparation."""

    write_csv(output / "archive_reference_runs.csv", RUN_FIELDS, records)
    write_csv(output / "archive_reference_aggregate.csv", AGGREGATE_FIELDS, aggregate)
    write_csv(output / "archive_reference_table_1.csv", TABLE_FIELDS, table_rows)
    write_json(output / "archive_reference_comparison.json", comparison)
    write_json(
        output / "archive_reference_manifest.json",
        validation["archive_canonical_manifest"],
    )
    write_json(output / "archive_reference_validation.json", validation)


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
    parser.add_argument(
        "--n",
        type=int,
        default=500,
        help="Paper-pinned consensus decision dimension for fresh modes.",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=250,
        help="Paper-pinned measurements per agent for fresh modes.",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=10.0,
        help="Paper-pinned initial ADMM penalty for fresh modes.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100000,
        help="Paper-pinned maximum ADMM iterations for fresh modes.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1000,
        help="Paper-pinned ADMM logging interval for fresh modes.",
    )
    parser.add_argument(
        "--mip-heuristic-effort",
        type=float,
        default=0.2,
        help=(
            "Paper-pinned HiGHS mip_heuristic_effort used for all four fresh "
            "MILP gaps."
        ),
    )
    parser.add_argument(
        "--mip-time-limit",
        type=float,
        default=60.0,
        help=(
            "Paper-pinned HiGHS time limit for each fresh bipartization MILP, "
            "in seconds."
        ),
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
        if args.mode == "full":
            expected = (*expected, ("--threads", args.threads, PAPER_THREADS))
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

    archive_records: list[dict[str, object]] = []
    archive_reference_aggregate: list[dict[str, object]] = []
    archive_reference_table_rows: list[dict[str, object]] = []
    archive_reference_comparison: dict[str, object] = {}
    archive_reference_validation: dict[str, object] | None = None
    archive_path = args.archive.expanduser()
    archive_available = archive_path.exists()
    archive_skip_message: str | None = None
    if args.mode == "full" and archive_available:
        (
            archive_records,
            archive_reference_aggregate,
            archive_reference_table_rows,
            archive_reference_comparison,
            archive_reference_validation,
        ) = load_archive_reference(args.archive)
        if archive_reference_validation["errors"]:
            raise SystemExit(
                "Section 3.4 full-mode archive preflight failed: "
                + "; ".join(
                    str(error) for error in archive_reference_validation["errors"]
                )
            )
    elif args.mode == "full":
        archive_skip_message = (
            "Section 3.4 full mode did not find the optional archive at "
            f"{archive_path}; archive preflight and the field-wise raw archive "
            "comparison are skipped. Fresh paper-grid, configuration, terminal-result, "
            "aggregation, and conclusion checks remain active."
        )
        print(f"WARNING: {archive_skip_message}", file=sys.stderr, flush=True)

    output = prepare_mode_output(args.output, args.mode)
    if archive_reference_validation is not None:
        _write_archive_reference_artifacts(
            output,
            archive_records,
            archive_reference_aggregate,
            archive_reference_table_rows,
            archive_reference_comparison,
            archive_reference_validation,
        )

    print(OBJECTIVE_FIDELITY_WARNING, flush=True)
    if args.mode == "full":
        print(FULL_RUNTIME_NOTE, flush=True)
    if args.mode == "smoke":
        print(f"Smoke methods: {', '.join(METHOD_DISPLAY[method] for method in selected_methods)}")

    jobs: list[Job] = []
    job_results = []
    fresh_log_root = output / "raw"
    archive_commands: dict[tuple[str, str], tuple[str, ...]] = {}
    archive_command_errors: list[str] = []
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
        if args.mode == "archived":
            archive_commands, archive_command_errors = read_archive_paper_commands(
                source
            )
        log_paths = (
            archived_paper_log_paths(source)
            if args.mode == "archived"
            else collect_logs(source)
        )
        records = parse_logs(log_paths, args.mode)

    raw_archive_comparison: dict[str, object] | None = None
    if args.mode == "full" and archive_reference_validation is not None:
        raw_archive_comparison = build_raw_archive_comparison(records, archive_records)
        write_json(output / "raw_archive_comparison.json", raw_archive_comparison)
    elif args.mode == "full":
        raw_archive_comparison = {
            "status": "skipped",
            "required": False,
            "archive_available": False,
            "archive_path": str(archive_path),
            "reason": archive_skip_message,
            "errors": [],
        }
        write_json(output / "raw_archive_comparison.json", raw_archive_comparison)

    if args.mode == "archived":
        _set_archive_source_identifiers(records, args.archive)

    write_csv(output / "runs.csv", RUN_FIELDS, records)
    if args.mode == "archived":
        aggregate, table_rows, comparison, validation = (
            build_archive_reference_validation(
                records,
                archive_commands,
                archive_command_errors,
            )
        )
    else:
        aggregate = aggregate_runs(records)
        table_rows = build_table_1(records)
        comparison = build_reference_comparison(records, aggregate, table_rows, args.mode)
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

    write_csv(output / "aggregate.csv", AGGREGATE_FIELDS, aggregate)
    write_json(output / "reference_comparison.json", comparison)
    if (
        raw_archive_comparison is not None
        and raw_archive_comparison.get("status") != "skipped"
    ):
        _merge_raw_archive_validation(validation, raw_archive_comparison)
    if archive_reference_validation is not None:
        validation["archive_reference_status"] = archive_reference_validation["status"]
        validation["archive_reference_summary"] = archive_reference_validation[
            "reference_summary"
        ]
    elif args.mode == "full":
        validation["archive_reference_status"] = "skipped_missing_archive"
        validation["raw_archive_comparison_status"] = "skipped_missing_archive"
        validation["archive_reference_path"] = str(archive_path)
        validation["warnings"].append(archive_skip_message)
    validation["status"] = "passed" if not validation["errors"] else "failed"
    validation["reference_summary"] = comparison["summary"]
    validation["conclusion_consistency"] = comparison["conclusion_consistency"]
    write_json(output / "validation.json", validation)
    _, figure_paths = publish_validated_artifacts(
        output,
        aggregate,
        table_rows,
        validation,
        no_plots=args.no_plots,
    )

    inputs = (
        [args.archive]
        if args.mode == "archived" or (args.mode == "full" and archive_available)
        else []
    )
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
            (
                "Fresh full archive-backed validation keeps raw identity/configuration "
                "and non-MILP-censored graph structure exact; stable ADMM status is exact "
                "and iterations use the documented max(5, 1.5%) bound. Dynamic "
                "time-censored outcomes use explicit near-cap/progress integrity gates; "
                "edge memberships were not retained and absolute timing measurements are "
                "hardware-informational."
                if args.mode != "full" or archive_reference_validation is not None
                else archive_skip_message
            ),
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
    "ARCHIVE_ADMM_TIME_CENSORED_KEYS",
    "ARCHIVE_MIP_TIME_CENSORED_KEYS",
    "ARCHIVE_PAPER_JOBS",
    "ARCHIVE_PAPER_CANONICAL_SHA256",
    "TABLE_FIELDS",
    "aggregate_runs",
    "archived_paper_log_paths",
    "build_archive_reference_validation",
    "build_archive_canonical_manifest",
    "build_raw_archive_comparison",
    "build_reference_comparison",
    "build_table_1",
    "evaluate_conclusion_consistency",
    "load_archive_reference",
    "main",
    "parse_logs",
    "publish_validated_artifacts",
    "read_archive_paper_commands",
    "raw_archive_validation_errors",
    "reference_validation_errors",
    "validate_archive_censor_status_manifest",
    "validate_archive_job_manifest",
    "validate_records",
]
