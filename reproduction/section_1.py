#!/usr/bin/env python3
"""Reproduce the three residual panels in paper Figure 2."""

from __future__ import annotations

import argparse
import csv
import math
import sys

if __package__:
    from . import common as _common
    sys.modules.setdefault("common", _common)
from pathlib import Path

from common import (
    REPRODUCTION_ROOT,
    REPO_ROOT,
    Job,
    add_common_arguments,
    failed_jobs,
    get_pyplot,
    julia_command,
    prepare_output,
    resolve_julia,
    run_jobs,
    validate_common_arguments,
    write_csv,
    write_provenance,
)


PAPER_RUNS = ((1.0, 10_000), (10.0, 1_000), (100.0, 100))
ROW_FIELDS = (
    "rho",
    "formulation",
    "label",
    "iteration",
    "pres_l2",
    "dres_l2",
    "residual_sum",
    "status",
    "stop_iter",
)


def _tag(rho: float) -> str:
    return str(int(rho)) if rho.is_integer() else str(rho).replace(".", "p")


def _read_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("residuals.csv")):
        with path.open(newline="", encoding="utf-8") as stream:
            for raw in csv.DictReader(stream):
                row: dict[str, object] = dict(raw)
                for field in ("rho", "pres_l2", "dres_l2", "residual_sum"):
                    row[field] = float(raw[field])
                for field in ("iteration", "stop_iter"):
                    row[field] = int(raw[field])
                rows.append(row)
    if not rows:
        raise SystemExit(f"No residuals.csv files found below {root}")
    return rows


def _plot(rows: list[dict[str, object]], output: Path) -> None:
    pyplot = get_pyplot()
    styles = {"12": "-", "23": "--", "31": ":"}
    labels = {
        "12": "breaking 1st constraint",
        "23": "breaking 2nd constraint",
        "31": "breaking 3rd constraint",
    }
    rhos = sorted({float(row["rho"]) for row in rows})
    figure, axes = pyplot.subplots(1, len(rhos), figsize=(5.2 * len(rhos), 4.0), squeeze=False)
    for panel, rho in enumerate(rhos):
        axis = axes[0][panel]
        for formulation in ("12", "23", "31"):
            selected = sorted(
                (
                    row
                    for row in rows
                    if float(row["rho"]) == rho and row["formulation"] == formulation
                ),
                key=lambda row: int(row["iteration"]),
            )
            x_values = [int(row["iteration"]) for row in selected]
            y_values = [
                math.log10(float(row["residual_sum"]))
                if float(row["residual_sum"]) > 0 and math.isfinite(float(row["residual_sum"]))
                else math.nan
                for row in selected
            ]
            axis.plot(x_values, y_values, styles[formulation], linewidth=2.0, label=labels[formulation])
        axis.set_title(rf"$\rho={rho:g}$")
        axis.set_xlabel("Iteration")
        axis.grid(True, alpha=0.35)
        if panel == 0:
            axis.set_ylabel(r"$\log_{10}(\|Pres\|_2 + \|Dres\|_2)$")
            axis.legend(fontsize=9)
    figure.tight_layout()
    figure.savefig(output / "figures" / "figure_2.png", dpi=200)
    figure.savefig(output / "figures" / "figure_2.pdf")
    pyplot.close(figure)

    for rho in rhos:
        figure, axis = pyplot.subplots(figsize=(6.4, 4.0))
        for formulation in ("12", "23", "31"):
            selected = sorted(
                (
                    row
                    for row in rows
                    if float(row["rho"]) == rho and row["formulation"] == formulation
                ),
                key=lambda row: int(row["iteration"]),
            )
            axis.plot(
                [int(row["iteration"]) for row in selected],
                [
                    math.log10(float(row["residual_sum"]))
                    if float(row["residual_sum"]) > 0 and math.isfinite(float(row["residual_sum"]))
                else math.nan
                    for row in selected
                ],
                styles[formulation],
                linewidth=2.2,
                label=labels[formulation],
            )
        axis.set_xlabel("Iteration")
        axis.set_ylabel(r"$\log_{10}(\|Pres\|_2 + \|Dres\|_2)$")
        axis.grid(True, alpha=0.35)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / "figures" / f"figure_2_rho_{_tag(rho)}.png", dpi=200)
        pyplot.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, default_output=REPRODUCTION_ROOT / "output" / "section_1")
    parser.set_defaults(mode="full")
    args = parser.parse_args()
    validate_common_arguments(args)
    output = prepare_output(args.output)

    if args.mode == "archived":
        raise SystemExit(
            "The supplied experiments_logs.zip starts at Section 3.1 and contains no Figure 2 data. "
            "Use --mode full (default) or --mode parse."
        )

    jobs: list[Job] = []
    if args.mode in ("smoke", "full"):
        resolve_julia(args.julia)
        runs = ((100.0, 100),) if args.mode == "smoke" else PAPER_RUNS
        driver = REPRODUCTION_ROOT / "julia" / "section_1_exact.jl"
        for rho, horizon in runs:
            job_dir = output / "raw" / f"rho_{_tag(rho)}"
            jobs.append(
                Job(
                    f"rho_{_tag(rho)}/run",
                    julia_command(
                        args.julia,
                        args.threads,
                        driver,
                        job_dir / "residuals.csv",
                        rho,
                        horizon,
                    ),
                )
            )
        results = run_jobs(
            jobs,
            log_root=output / "raw",
            cwd=REPO_ROOT,
            workers=args.jobs,
            resume=not args.no_resume,
        )
        failures = failed_jobs(results)
        if failures:
            names = ", ".join(result.name for result in failures)
            raise SystemExit(f"Figure 2 Julia jobs failed: {names}. See logs below {output / 'raw'}")
        source = output / "raw"
    else:
        source = args.logs.expanduser().resolve()

    rows = _read_rows(source)
    write_csv(output / "residuals.csv", ROW_FIELDS, rows)
    if not args.no_plots:
        _plot(rows, output)
    write_provenance(
        output,
        section="Figure 2 / circuit demo",
        args=args,
        jobs=jobs,
        notes=(
            "Paper horizons are rho=1:10000, rho=10:1000, rho=100:100.",
            "The reviewer Julia driver writes data outside applications/ and does not modify experiment code.",
        ),
    )
    print(f"Wrote Figure 2 artifacts to {output}")
    return 0


if __name__ == "__main__":
    try:
        from .section_1_impl import main as strict_main
    except ImportError:
        from section_1_impl import main as strict_main
    raise SystemExit(strict_main())

