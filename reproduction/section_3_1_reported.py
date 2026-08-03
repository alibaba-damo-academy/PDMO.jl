"""Run and verify the historical workflow that produced paper Figures 7--10.

The manuscript describes a five-pass co-clustering.  The final published panels
were produced later by the public applications/GenericLP/enlight_hard_demo.jl
workflow, which uses ten passes and promotes pairwise row groups.  This module
keeps that reported workflow separate from the manuscript-literal smoke/full
path in section_3_1_impl.py.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import re
import struct
import sys
from pathlib import Path
from typing import Mapping

try:
    from .common import (
        REPO_ROOT,
        Job,
        failed_jobs,
        job_results_as_dicts,
        julia_command,
        run_jobs,
        sha256_file,
        write_json,
        write_provenance,
    )
except ImportError:
    from common import (  # type: ignore
        REPO_ROOT,
        Job,
        failed_jobs,
        job_results_as_dicts,
        julia_command,
        run_jobs,
        sha256_file,
        write_json,
        write_provenance,
    )


REPORTED_DRIVER = REPO_ROOT / "applications" / "GenericLP" / "enlight_hard_demo.jl"
REPORTED_INSPECT_SOURCE = REPO_ROOT / "applications" / "GenericLP" / "inspect_cocluster.jl"
REPORTED_GENERIC_LP_SOURCE = REPO_ROOT / "applications" / "GenericLP" / "GenericLP.jl"
REPORTED_SOURCE_FILES = {
    "enlight_hard_demo.jl": REPORTED_DRIVER,
    "inspect_cocluster.jl": REPORTED_INSPECT_SOURCE,
    "GenericLP.jl": REPORTED_GENERIC_LP_SOURCE,
}
REPORTED_SOURCE_SHA256 = {
    "enlight_hard_demo.jl": "c35c33d3ee5fae25cd6824eb0ff2b3fdecf7a2f2adfa6b99d0bb8615f4996384",
    "inspect_cocluster.jl": "af43735b56e7c635af9608c320e544deb8481932a900ca06d1bf48fb1edeae38",
    "GenericLP.jl": "199ec11f9fa2f69184f62971090a860ab2dae3cc7262e9cc42437a536c5cf5c1",
}

OFFICIAL_MPS_GZIP_SHA256 = "942168c2126a2a91ae3ec1ededea59bc1af0cad55f94223edf4c03d20e831f66"
OFFICIAL_MPS_CONTENT_SHA256 = "572ca23c17d0ad734895e8338af458525a753ee76bdb117d2917e4069c6b65b0"

REPORTED_CONFIGURATION = {
    "classification": "fresh historical/published workflow",
    "global_seed": 126,
    "co_clustering_local_seed": 42,
    "column_clusters": 4,
    "alternating_passes": 10,
    "force_split_single_block": True,
    "promote_pairwise_rows": True,
    "initial_rho": 1000.0,
    "maximum_iterations": 100_000,
    "log_interval": 1000,
    "solver": "DoublyLinearizedSolver",
    "apply_scaling": False,
}

# Hash the decoded RGBA pixels and dimensions, not PNG container bytes.  This
# ignores harmless compression and metadata differences while remaining an
# exact check of every rendered pixel.
PUBLISHED_PANEL_PIXEL_SHA256 = {
    "matrix_original.png": "8567bfe123fd63174bbca301247cfd8063eaad6e5148cefae784675c56287046",
    "matrix_coclustered.png": "6386b27ebb9c521aeacfec6c41f415b3dfb960bae71c791a88a240ca26229c0d",
    "matrix_block_couplings.png": "5893676c06f1a25ba8b977cdd9e4ee2c581fad82fcc23fc36f10141db54f40c0",
    "graph_original_coclustered.png": "e505c7e932b92ca7d0b5a7890ea5b8a69a027e63e70a79ae1216525d1a81cad9",
    "graph_bipartization_milp.png": "177d8b318952a8a0b932bf6e43d3630e3d40c48b2605f85955b33e94805e7828",
    "graph_bipartization_bfs.png": "06f0eafb62860e2723310237510a4a164f28921c11327cddf93276bdcd88c908",
    "primal_residuals.png": "6cae69ba009587dff104d34eb64820a1340f1139d7b657619713d37f0a9dd2b1",
    "dual_residuals.png": "63ccd36a691a86244cd25507825f817d797682e03b2c78375ecf9823c3268e9e",
}
PANEL_TO_FIGURE = {
    "matrix_original.png": 7,
    "matrix_coclustered.png": 7,
    "matrix_block_couplings.png": 8,
    "graph_original_coclustered.png": 8,
    "graph_bipartization_milp.png": 9,
    "graph_bipartization_bfs.png": 9,
    "primal_residuals.png": 10,
    "dual_residuals.png": 10,
}

_SOURCE_CHECKS = (
    (
        "global seed",
        "enlight_hard_demo.jl",
        r"Random\.seed!\(\s*126\s*\)",
    ),
    (
        "four column clusters",
        "enlight_hard_demo.jl",
        r"cocluster_k\s*=\s*4\b",
    ),
    (
        "ten alternating passes",
        "enlight_hard_demo.jl",
        r"cocluster_iters\s*=\s*10\b",
    ),
    (
        "force split",
        "enlight_hard_demo.jl",
        r"cocluster_force_split\s*=\s*true\b",
    ),
    (
        "pairwise-row promotion",
        "enlight_hard_demo.jl",
        r"cocluster_promote_pairwise_rows\s*=\s*true\b",
    ),
    (
        "rho",
        "enlight_hard_demo.jl",
        r"initialRho\s*=\s*1000\.0\b",
    ),
    (
        "iteration limit",
        "enlight_hard_demo.jl",
        r"maxIter\s*=\s*100000\b",
    ),
    (
        "log interval",
        "enlight_hard_demo.jl",
        r"logInterval\s*=\s*1000\b",
    ),
    (
        "doubly linearized solver",
        "enlight_hard_demo.jl",
        r"solver\s*=\s*DoublyLinearizedSolver\(\)",
    ),
    (
        "scaling disabled",
        "enlight_hard_demo.jl",
        r"applyScaling\s*=\s*false\b",
    ),
    (
        "co-clustering local seed",
        "inspect_cocluster.jl",
        r"MersenneTwister\(\s*42\s*\)",
    ),
)


def _stream_sha256(stream) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def validate_official_mps(path: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve()
    errors: list[str] = []
    raw_sha256 = None
    content_sha256 = None
    compression = "plain"
    if not resolved.is_file():
        errors.append(f"Official MIPLIB input is missing: {resolved}")
    else:
        raw_sha256 = sha256_file(resolved)
        try:
            with resolved.open("rb") as stream:
                gzip_encoded = stream.read(2) == b"\x1f\x8b"
            if gzip_encoded:
                compression = "gzip"
                with gzip.open(resolved, "rb") as stream:
                    content_sha256 = _stream_sha256(stream)
            else:
                content_sha256 = raw_sha256
        except OSError as error:
            errors.append(f"Could not read MIPLIB input {resolved}: {error}")
        if content_sha256 is not None and content_sha256 != OFFICIAL_MPS_CONTENT_SHA256:
            errors.append(
                "The supplied MPS does not match the official enlight_hard content "
                f"SHA256 {OFFICIAL_MPS_CONTENT_SHA256}"
            )

    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "path": str(resolved),
        "compression": compression,
        "raw_sha256": raw_sha256,
        "official_gzip_sha256": OFFICIAL_MPS_GZIP_SHA256,
        "raw_hash_matches_official_download": raw_sha256 == OFFICIAL_MPS_GZIP_SHA256,
        "content_sha256": content_sha256,
        "official_content_sha256": OFFICIAL_MPS_CONTENT_SHA256,
        "content_hash_matches": content_sha256 == OFFICIAL_MPS_CONTENT_SHA256,
    }


def validate_reported_driver_source(
    source_files: Mapping[str, Path] | None = None,
    expected_hashes: Mapping[str, str] | None = None,
) -> dict[str, object]:
    selected = dict(source_files or REPORTED_SOURCE_FILES)
    pinned = dict(REPORTED_SOURCE_SHA256 if expected_hashes is None else expected_hashes)
    errors: list[str] = []
    records: dict[str, dict[str, object]] = {}
    texts: dict[str, str] = {}

    for name, path in selected.items():
        resolved = path.expanduser().resolve()
        record: dict[str, object] = {
            "path": str(resolved),
            "exists": resolved.is_file(),
        }
        if resolved.is_file():
            digest = sha256_file(resolved)
            expected = pinned.get(name)
            record.update(
                {
                    "sha256": digest,
                    "expected_sha256": expected,
                    "hash_matches": expected is None or digest == expected,
                }
            )
            if expected is not None and digest != expected:
                errors.append(
                    f"{name} SHA256 {digest} differs from the reported-workflow source {expected}"
                )
            try:
                texts[name] = resolved.read_text(encoding="utf-8")
            except OSError as error:
                errors.append(f"Could not read {resolved}: {error}")
        else:
            errors.append(f"Reported-workflow source is missing: {resolved}")
        records[name] = record

    checks = []
    for label, source_name, pattern in _SOURCE_CHECKS:
        matched = bool(re.search(pattern, texts.get(source_name, "")))
        checks.append(
            {
                "label": label,
                "source": source_name,
                "pattern": pattern,
                "matched": matched,
            }
        )
        if not matched:
            errors.append(f"Reported driver does not expose the expected {label}")

    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "expected_configuration": REPORTED_CONFIGURATION,
        "source_files": records,
        "configuration_checks": checks,
        "manuscript_distinction": (
            "The manuscript states five passes. The final published panels were generated "
            "by this pinned ten-pass workflow with pairwise-row promotion."
        ),
    }


def pixel_sha256(path: Path) -> tuple[str, tuple[int, int]]:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "Published-panel validation requires Pillow; install reproduction/requirements.txt"
        ) from error

    try:
        with Image.open(path) as image:
            rgba = image.convert("RGBA")
            rgba.load()
            size = rgba.size
            payload = struct.pack(">II", *size) + rgba.tobytes()
    except OSError as error:
        raise ValueError(f"{path} is not a readable image: {error}") from error
    return hashlib.sha256(payload).hexdigest(), size


def validate_reported_panels(
    panel_dir: Path,
    expected_pixel_hashes: Mapping[str, str] | None = None,
) -> dict[str, object]:
    expected = dict(expected_pixel_hashes or PUBLISHED_PANEL_PIXEL_SHA256)
    root = panel_dir.expanduser().resolve()
    errors: list[str] = []
    panels: list[dict[str, object]] = []

    for filename, expected_digest in expected.items():
        path = root / filename
        record: dict[str, object] = {
            "filename": filename,
            "paper_figure": PANEL_TO_FIGURE.get(filename),
            "path": str(path),
            "exists": path.is_file(),
            "expected_pixel_sha256": expected_digest,
        }
        if not path.is_file():
            errors.append(f"Missing reported source panel: {path}")
            record["pixel_hash_matches"] = False
        else:
            record["size_bytes"] = path.stat().st_size
            record["png_sha256"] = sha256_file(path)
            try:
                digest, dimensions = pixel_sha256(path)
            except (RuntimeError, ValueError) as error:
                errors.append(str(error))
                record["pixel_hash_matches"] = False
            else:
                matched = digest == expected_digest
                record.update(
                    {
                        "dimensions": list(dimensions),
                        "pixel_sha256": digest,
                        "pixel_hash_matches": matched,
                    }
                )
                if not matched:
                    errors.append(
                        f"{filename} decoded pixels SHA256 {digest} do not match "
                        f"the published panel {expected_digest}"
                    )
        panels.append(record)

    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "panel_count": len(panels),
        "expected_panel_count": len(expected),
        "all_published_pixel_hashes_match": not errors,
        "pixel_hash_definition": "sha256(big-endian width,height + decoded RGBA bytes)",
        "panels": panels,
    }


def _reported_artifacts_present(panel_dir: Path) -> bool:
    expected = [panel_dir / name for name in PUBLISHED_PANEL_PIXEL_SHA256]
    expected.append(panel_dir / "matrix_cocluster_info.txt")
    return all(path.is_file() and path.stat().st_size > 0 for path in expected)


def run_reported_mode(args: argparse.Namespace, output: Path) -> int:
    if args.no_plots:
        raise SystemExit("--mode reported always renders the eight published source panels")
    if args.mps is None:
        raise SystemExit(
            "Reported mode did not receive an MPS path; the CLI should supply its bundled default."
        )

    mps = args.mps.expanduser().resolve()
    mps_validation = validate_official_mps(mps)
    source_validation = validate_reported_driver_source()
    preflight_errors = [
        *mps_validation["errors"],
        *source_validation["errors"],
    ]
    configuration = {
        "mode": "reported",
        "configuration": REPORTED_CONFIGURATION,
        "mps_validation": mps_validation,
        "source_validation": source_validation,
    }
    write_json(output / "reported_configuration.json", configuration)

    source_inputs = [mps, *REPORTED_SOURCE_FILES.values()]
    if preflight_errors:
        validation = {
            "status": "failed",
            "mode": "reported",
            "errors": preflight_errors,
            "configuration": configuration,
            "source_panel_validation": {"status": "not_run"},
        }
        write_json(output / "validation.json", validation)
        write_provenance(
            output,
            section="Section 3.1 / reported Figures 7--10",
            args=args,
            inputs=source_inputs,
            notes=(
                "Reported-mode preflight failed; Julia was not launched.",
                source_validation["manuscript_distinction"],
            ),
        )
        for error in preflight_errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    panel_dir = output / "figures"
    job = Job(
        "reported/enlight_hard",
        julia_command(
            args.julia,
            args.threads,
            REPORTED_DRIVER,
            mps,
            panel_dir,
        ),
    )
    results = run_jobs(
        (job,),
        log_root=output / "raw",
        cwd=REPO_ROOT,
        workers=1,
        resume=not args.no_resume and _reported_artifacts_present(panel_dir),
    )
    write_json(output / "job_results.json", job_results_as_dicts(results))

    job_errors = [
        f"Historical reported-workflow Julia job failed; inspect {result.log_path}"
        for result in failed_jobs(results)
    ]
    panel_validation = validate_reported_panels(panel_dir)
    write_json(output / "source_panel_validation.json", panel_validation)
    errors = [*job_errors, *panel_validation["errors"]]
    validation = {
        "status": "passed" if not errors else "failed",
        "mode": "reported",
        "classification": "fresh_historical_reported_rerun",
        "errors": errors,
        "configuration_status": source_validation["status"],
        "mps_status": mps_validation["status"],
        "source_panel_status": panel_validation["status"],
        "all_eight_published_pixel_hashes_match": panel_validation[
            "all_published_pixel_hashes_match"
        ],
        "manuscript_distinction": source_validation["manuscript_distinction"],
    }
    write_json(output / "validation.json", validation)
    write_json(
        output / "manifest.json",
        {
            "classification": "fresh_historical_reported_rerun",
            "paper_artifacts": {
                "Figure 7": [
                    "figures/matrix_original.png",
                    "figures/matrix_coclustered.png",
                ],
                "Figure 8": [
                    "figures/matrix_block_couplings.png",
                    "figures/graph_original_coclustered.png",
                ],
                "Figure 9": [
                    "figures/graph_bipartization_milp.png",
                    "figures/graph_bipartization_bfs.png",
                ],
                "Figure 10": [
                    "figures/primal_residuals.png",
                    "figures/dual_residuals.png",
                ],
            },
            "configuration": REPORTED_CONFIGURATION,
            "published_pixel_hashes": PUBLISHED_PANEL_PIXEL_SHA256,
            "manuscript_distinction": source_validation["manuscript_distinction"],
        },
    )
    write_provenance(
        output,
        section="Section 3.1 / reported Figures 7--10",
        args=args,
        jobs=(job,),
        inputs=source_inputs,
        notes=(
            "This mode invokes the existing public historical driver without modifying it.",
            "All generated files are redirected to the mode-specific reproduction output.",
            "Decoded RGBA pixel hashes are compared with all eight final arXiv source panels.",
            source_validation["manuscript_distinction"],
            "The manuscript-literal five-pass implementation remains available as smoke/full.",
        ),
    )

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Wrote exact reported Section 3.1 panels to {output}")
    return 0


__all__ = (
    "OFFICIAL_MPS_CONTENT_SHA256",
    "PUBLISHED_PANEL_PIXEL_SHA256",
    "REPORTED_CONFIGURATION",
    "REPORTED_DRIVER",
    "pixel_sha256",
    "run_reported_mode",
    "validate_official_mps",
    "validate_reported_driver_source",
    "validate_reported_panels",
)

