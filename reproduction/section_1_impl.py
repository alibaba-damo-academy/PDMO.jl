"""Strict execution/validation layer for the Figure 2 reviewer entry point."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

try:
    from .common import (
        REPO_ROOT,
        REPRODUCTION_ROOT,
        Job,
        add_common_arguments,
        failed_jobs,
        job_results_as_dicts,
        julia_command,
        prepare_mode_output,
        run_jobs,
        validate_common_arguments,
        write_csv,
        write_json,
        write_provenance,
    )
    from .section_1 import _plot, _tag
except ImportError:
    from common import (  # type: ignore
        REPO_ROOT,
        REPRODUCTION_ROOT,
        Job,
        add_common_arguments,
        failed_jobs,
        job_results_as_dicts,
        julia_command,
        prepare_mode_output,
        run_jobs,
        validate_common_arguments,
        write_csv,
        write_json,
        write_provenance,
    )
    from section_1 import _plot, _tag  # type: ignore


SOLVE_MAX_ITER = 10_000
PAPER_SEED = 126
PAPER_RUNS = ((1.0, 10_000), (10.0, 1_000), (100.0, 100))
PAPER_CUTOFF = dict(PAPER_RUNS)
FORMULATIONS = ("12", "23", "31")
FORMULATION_LABEL = {
    "12": "breaking 1st constraint",
    "23": "breaking 2nd constraint",
    "31": "breaking 3rd constraint",
}
EXPECTED_STATUS = "ADMM_TERMINATION_ITERATION_LIMIT"
EXPECTED_SOLVER = "OriginalADMMSubproblemSolver"
JULIA_DRIVER = REPRODUCTION_ROOT / "julia" / "section_1_exact.jl"
DEMO_SOURCE = REPO_ROOT / "applications" / "Demo" / "demo.jl"
DEFAULT_OUTPUT = REPRODUCTION_ROOT / "output" / "section_1"

ROW_FIELDS = (
    "rho",
    "formulation",
    "label",
    "iteration",
    "actual_iteration",
    "pres_l2",
    "dres_l2",
    "residual_sum",
    "status",
    "stop_iter",
    "solve_max_iter",
    "plot_cutoff",
    "solver",
    "seed",
    "source_csv",
)
REQUIRED_INPUT_FIELDS = set(ROW_FIELDS).difference({"source_csv"})


def _float(value: object, field: str, path: Path) -> float:
    try:
        return float(str(value))
    except ValueError as error:
        raise ValueError(f"{path}: invalid {field} value {value!r}") from error


def _integer(value: object, field: str, path: Path) -> int:
    number = _float(value, field, path)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{path}: invalid integer {field} value {value!r}")
    return int(number)


def read_rows(paths: Sequence[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            missing = REQUIRED_INPUT_FIELDS.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path}: missing CSV fields {sorted(missing)}")
            for raw in reader:
                row: dict[str, object] = dict(raw)
                for field in ("rho", "pres_l2", "dres_l2", "residual_sum"):
                    row[field] = _float(raw[field], field, path)
                for field in (
                    "iteration",
                    "actual_iteration",
                    "stop_iter",
                    "solve_max_iter",
                    "plot_cutoff",
                    "seed",
                ):
                    row[field] = _integer(raw[field], field, path)
                row["source_csv"] = str(path.resolve())
                rows.append(row)
    if not rows:
        raise ValueError("No Figure 2 trajectory rows were found")
    return rows


def _same_sum(left: float, right: float, observed: float) -> bool:
    expected = left + right
    if math.isinf(expected) or math.isnan(expected):
        return (math.isinf(expected) and math.isinf(observed)) or (
            math.isnan(expected) and math.isnan(observed)
        )
    return math.isclose(expected, observed, rel_tol=1e-12, abs_tol=1e-14)


def validate_rows(
    rows: Sequence[Mapping[str, object]], expected_runs: Sequence[tuple[float, int]]
) -> dict[str, object]:
    errors: list[str] = []
    expected_rhos = {rho: cutoff for rho, cutoff in expected_runs}
    groups: dict[tuple[float, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        rho = float(row["rho"])
        formulation = str(row["formulation"])
        groups[(rho, formulation)].append(row)

    expected_keys = {
        (rho, formulation) for rho in expected_rhos for formulation in FORMULATIONS
    }
    observed_keys = set(groups)
    missing = sorted(expected_keys.difference(observed_keys))
    extra = sorted(observed_keys.difference(expected_keys))
    if missing:
        errors.append(f"Missing rho/formulation trajectories: {missing}")
    if extra:
        errors.append(f"Unexpected rho/formulation trajectories: {extra}")

    for key in sorted(expected_keys.intersection(observed_keys)):
        rho, formulation = key
        cutoff = expected_rhos[rho]
        selected = sorted(groups[key], key=lambda row: int(row["iteration"]))
        iterations = [int(row["iteration"]) for row in selected]
        actual_iterations = [int(row["actual_iteration"]) for row in selected]
        if iterations != list(range(1, cutoff + 1)):
            errors.append(
                f"{key}: expected display samples 1:{cutoff}, got {len(iterations)} rows "
                f"spanning {iterations[:1]}..{iterations[-1:]}."
            )
        if actual_iterations != list(range(cutoff)):
            errors.append(f"{key}: actual iterations must be 0:{cutoff - 1}")

        for field, expected in (
            ("label", FORMULATION_LABEL[formulation]),
            ("status", EXPECTED_STATUS),
            ("stop_iter", SOLVE_MAX_ITER),
            ("solve_max_iter", SOLVE_MAX_ITER),
            ("plot_cutoff", cutoff),
            ("solver", EXPECTED_SOLVER),
            ("seed", PAPER_SEED),
        ):
            values = {row[field] for row in selected}
            if values != {expected}:
                errors.append(f"{key}: {field}={sorted(map(str, values))}, expected {expected!r}")

        for row in selected:
            sample = int(row["iteration"])
            pres = float(row["pres_l2"])
            dres = float(row["dres_l2"])
            residual_sum = float(row["residual_sum"])
            if pres < 0 or (sample > 1 and not math.isfinite(pres)):
                errors.append(f"{key} sample {sample}: invalid primal residual {pres}")
            if dres < 0 or (sample > 1 and not math.isfinite(dres)):
                errors.append(f"{key} sample {sample}: invalid dual residual {dres}")
            if not _same_sum(pres, dres, residual_sum):
                errors.append(f"{key} sample {sample}: residual_sum is inconsistent")

    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "row_count": len(rows),
        "trajectory_count": len(groups),
        "expected_runs": [
            {"rho": rho, "plot_cutoff": cutoff, "solve_max_iter": SOLVE_MAX_ITER}
            for rho, cutoff in expected_runs
        ],
        "seed": PAPER_SEED,
        "semantics": (
            "Every rho is solved for 10,000 ADMM iterations; plot_cutoff selects the "
            "first K legacy history samples, whose display indices are 1:K and actual "
            "ADMM indices are 0:K-1."
        ),
    }


def _artifact_schema_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            fields = set(csv.DictReader(stream).fieldnames or ())
    except OSError:
        return False
    return REQUIRED_INPUT_FIELDS.issubset(fields)


def discover_parse_inputs(path: Path) -> list[Path]:
    root = path.expanduser().resolve()
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise SystemExit(f"Figure 2 log directory not found: {root}")
    merged = root / "residuals.csv"
    if merged.is_file():
        return [merged]
    candidates = sorted(root.rglob("residuals.csv"))
    if not candidates:
        raise SystemExit(f"No residuals.csv files found below {root}")
    return candidates


def _expected_runs(mode: str, rows: Sequence[Mapping[str, object]] = ()) -> tuple[tuple[float, int], ...]:
    if mode == "full":
        return PAPER_RUNS
    if mode == "smoke":
        return ((100.0, 100),)
    present = {float(row["rho"]) for row in rows}
    unknown = sorted(present.difference(PAPER_CUTOFF))
    if unknown:
        raise SystemExit(f"Parse input contains non-paper rho values: {unknown}")
    return tuple((rho, cutoff) for rho, cutoff in PAPER_RUNS if rho in present)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce the three residual panels in paper Figure 2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_arguments(parser, default_output=DEFAULT_OUTPUT)
    parser.set_defaults(mode="full")
    for action in parser._actions:
        if action.dest == "mode":
            action.help = (
                "full: exact three-panel run (default); smoke: rho=100 panel; "
                "parse: validate/plot --logs; archived: unavailable because Figure 2 "
                "logs are absent from the supplied archive"
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_common_arguments(args)
    if args.mode == "archived":
        raise SystemExit(
            "The supplied experiments_logs.zip starts at Section 3.1 and contains no "
            "Figure 2 data. Use --mode full (default) or --mode parse."
        )
    output = prepare_mode_output(args.output, args.mode)

    jobs: list[Job] = []
    job_results = []
    input_paths: list[Path]
    if args.mode in {"smoke", "full"}:
        expected_runs = _expected_runs(args.mode)
        artifact_paths: list[Path] = []
        for rho, cutoff in expected_runs:
            artifact = output / "raw" / f"rho_{_tag(rho)}" / "residuals.csv"
            artifact_paths.append(artifact)
            jobs.append(
                Job(
                    f"rho_{_tag(rho)}/run",
                    julia_command(
                        args.julia,
                        args.threads,
                        JULIA_DRIVER,
                        artifact,
                        rho,
                        cutoff,
                    ),
                )
            )
        # A successful subprocess marker is resumable only when every required
        # structured artifact still has the expected schema.
        resume = not args.no_resume and all(_artifact_schema_valid(path) for path in artifact_paths)
        job_results = run_jobs(
            jobs,
            log_root=output / "raw",
            cwd=REPO_ROOT,
            workers=args.jobs,
            resume=resume,
        )
        write_json(output / "job_results.json", job_results_as_dicts(job_results))
        failures = failed_jobs(job_results)
        if failures:
            names = ", ".join(result.name for result in failures)
            raise SystemExit(f"Figure 2 Julia jobs failed: {names}. Inspect {output / 'raw'}")
        missing_artifacts = [path for path in artifact_paths if not _artifact_schema_valid(path)]
        if missing_artifacts:
            raise SystemExit(f"Figure 2 jobs did not produce valid CSVs: {missing_artifacts}")
        input_paths = artifact_paths
    else:
        input_paths = discover_parse_inputs(args.logs)
        expected_runs = ()

    try:
        rows = read_rows(input_paths)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    if args.mode == "parse":
        expected_runs = _expected_runs(args.mode, rows)
    validation = validate_rows(rows, expected_runs)

    write_csv(output / "residuals.csv", ROW_FIELDS, rows)
    write_json(output / "validation.json", validation)
    write_json(
        output / "reference_comparison.json",
        {
            "reference": "applications/Demo/demo.jl legacy Figure 2 execution semantics",
            "numeric_archive_available": False,
            "configuration_checks_passed": validation["status"] == "passed",
            "note": (
                "The supplied archive contains no Figure 2 trajectories, so this file records "
                "strict configuration/history checks rather than a numeric archived comparison."
            ),
        },
    )
    if not args.no_plots and validation["status"] == "passed":
        _plot(list(rows), output)
    write_provenance(
        output,
        section="Figure 2 / circuit demo",
        args=args,
        jobs=jobs,
        inputs=input_paths if args.mode == "parse" else (JULIA_DRIVER, DEMO_SOURCE),
        notes=(
            "The original demo and this reviewer driver explicitly use global RNG seed 126.",
            "All rho values are solved for maxIter=10000 with original ADMM and logInterval=1.",
            "Paper panel cutoffs are rho=1:10000, rho=10:1000, rho=100:100 legacy history samples.",
            "The first displayed sample is ADMM initialization (actual iteration zero), matching demo.jl.",
        ),
    )

    if validation["errors"]:
        for error in validation["errors"]:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Wrote validated Figure 2 artifacts to {output}")
    return 0


__all__ = (
    "ROW_FIELDS",
    "discover_parse_inputs",
    "main",
    "read_rows",
    "validate_rows",
)
