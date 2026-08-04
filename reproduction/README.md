# Reviewer reproduction guide

This directory provides reviewer-facing entry points for every numerical or
tabular result in *Automating Reformulation for Parallel ADMM*
(arXiv:2603.19417) without modifying `src/`, `applications/`, or `advanced/`.
It distinguishes fresh reruns from regeneration using retained logs, and it
does not label a copied paper image as a rerun. Raw logs, machine-readable
results, figures, validation reports, and provenance are written below the path
given by `--output`, in a mode-specific subdirectory (for example,
`reproduction/output/section_3_2/archived/`).

## Complete paper artifact map

| Paper artifact | Reviewer entry point | What can be reproduced |
|---|---|---|
| Figure 2 | `section_1.py` | Fresh `full` run; no numerical archive was retained |
| Figures 7--9 | `section_3_1.py` | Fresh historical `reported` run from the public MIPLIB input, with all six submitted panels validated pixel-for-pixel |
| Figure 10 | `section_3_1.py` | Fresh historical `reported` run, with both submitted panels validated pixel-for-pixel; `archived` retains a distinct later experiment |
| Figure 11 | `section_3_2.py` | Exact reported aggregates from `archived`, or the complete fresh 50-job grid with `full` |
| Figure 13 | `section_3_3.py` | Exact reported aggregates from `archived`, or the complete fresh OPF grid with `full` |
| Figure 14 | `section_3_3.py` | Published `rho=2000` profile from `archived`, or an exact fresh rerun of that same selected profile with `full`; the caption's `rho=1000` is retained as typo metadata only |
| Table 1 and Figures 15--16 | `section_3_4.py` | Validated reconstruction from retained logs with a documented 0.01-second Table 1 display discrepancy, or the complete fresh consensus grid with `full` |
| Table 2 | `appendix_a.py` | Deterministic CSV, Markdown, and LaTeX export with `table` |
| Figure 18 | `appendix_a.py` | Byte-verified reported raster with `archived-source`, or a validated plot from the original history with `parse` |

Figures 1, 3--6, 12, and 17 are explanatory illustrations rather than
numerical results. They are defined by inline TikZ/circuitikz or the included
`plots/GNN_Structure.pdf` in the public arXiv source; the source-build command is
given below.

One source limitation is material: neither this repository nor the internal
experiment repository contains the final Figure 18 training program, the
10,400-graph corpus, its exact split/seeds, or the epoch-by-epoch accuracy
history. The committed `.pth` file is a final inference state dictionary and
cannot reconstruct earlier accuracies. Consequently, `appendix_a.py --mode
full` fails with a machine-readable inventory of the missing assets rather than
fabricating a curve. Recovering either the original training project and data,
or the original accuracy-history CSV, is required to turn that row into a fresh
retraining result.

Accordingly, every submitted figure and table has a reviewer command for a
fresh rerun, deterministic export, retained-data reconstruction, or verified
source build. Figure 18 is the sole artifact without a genuine fresh numerical
rerun; its exact submitted raster remains reproducible as source preservation.

## Prerequisites and dependency initialization

Run every command in this guide from the repository root on Linux or another
POSIX-like system. The reviewer wrappers require Python 3.10 or newer (tested
with Python 3.12.4). `requirements.txt` installs Matplotlib for plots and Pillow
for decoded-image validation; all other wrapper imports use the Python standard
library. Initialize the isolated environment with:

```bash
python3 -m venv reproduction/.venv-reproduction
. reproduction/.venv-reproduction/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r reproduction/requirements.txt
```

The archived commands require `experiments_logs.zip` at the repository root,
unless `--archive PATH` is supplied. A reviewer distribution must include this
retained input. Verify it before use:

```bash
echo "3a1a5e7a5e9f1c2996426b5cf41ae7b7672f5f9dd03fbbf166a322487f083138  experiments_logs.zip" \
  | sha256sum --check -
```

Fresh MIPLIB and MATPOWER problem inputs are bundled below
`reproduction/instances/`. Verify all seven files before a fresh run:

```bash
sha256sum --check reproduction/instances/SHA256SUMS
```

Fresh numerical runs require Julia 1.11.5. The repository's ordinary manifests
are ignored, so reproducibility snapshots of both tested environments are
included under `reproduction/julia_manifests/`. Install those locks before the
warmups described below:

```bash
cp reproduction/julia_manifests/PDMO.Manifest.toml Manifest.toml
cp reproduction/julia_manifests/advanced.Manifest.toml advanced/Manifest.toml
```

The locks include the Julia solvers and packages used by the workflows,
including HiGHS, Ipopt, PowerModels, and PyCall. No MATLAB, commercial optimizer,
or licensed HSL installation is required: the warmups use the bundled
`HSL_jll_placeholder` and Ipopt's default linear solver. HSL remains an optional
performance enhancement.

Initial dependency setup and public arXiv-source retrieval require network
access plus `git`, `curl`, `tar`, and GNU `sha256sum` (use `shasum -a 256` instead on macOS and
compare the printed digest). Fresh GNN methods additionally require a separate
Python 3.9--3.11 interpreter with the pinned packages configured below; the
experiment drivers force CPU inference, so CUDA and a GPU are not required. Only
the explanatory-figure paper build requires TeX: use TeX Live 2025 with
`pdflatex`, BibTeX, and the packages loaded by `main.tex`; a full TeX Live
installation is the simplest option. TeX is not required for the numerical
CSVs, tables, or plots.

## Fastest review path: rebuild from the supplied archive

After the initialization above, regenerate every retained numerical result:

```bash
python3 reproduction/section_3_1.py --mode archived
python3 reproduction/section_3_2.py --mode archived
python3 reproduction/section_3_3.py --mode archived
python3 reproduction/section_3_4.py --mode archived

curl -L --fail https://arxiv.org/e-print/2603.19417v1 \
  -o /tmp/pdmo-paper-source.tar.gz
echo "f370667d2f464fea5f00df1f22682bd390fdead70c19cda97929b6b7a1dcf107  /tmp/pdmo-paper-source.tar.gz" \
  | sha256sum --check -
python3 reproduction/appendix_a.py --mode archived-source \
  --arxiv-source /tmp/pdmo-paper-source.tar.gz
```

These commands parse `experiments_logs.zip`, validate the expected experiment
grids, rebuild the retained non-paper Section 3.1 comparison, Figure 11,
Figures 13--16, and Table 1, preserve the
reported Figure 18 raster, and write provenance. Override the log archive with
`--archive PATH`. These modes perform no Julia solve and do not require MIPLIB,
MATPOWER, or the GNN Python environment. The Figure 18 source copy must have
SHA256 `b93e66ce848b2801e567da74e63edc265e448d18217ef2cf63fea546775dc476`.

Figure 2 is absent from the supplied archive and is inexpensive to regenerate:

```bash
python3 reproduction/section_1.py --mode full
```

Figures 7--10 use the bundled MIPLIB input with the Section 3.1 `reported`
command shown below. Figure 18 fresh retraining has the explicit source gap
described in the artifact map.

## Common modes

The five main-section entry points use the following modes where applicable:

- `archived`: parse the supplied archive and rebuild artifacts (the default for
  Sections 3.1--3.4).
- `parse --logs DIR`: rebuild artifacts from a previous raw-log directory
  without launching Julia.
- `reported` (Section 3.1 only): rerun the pinned historical workflow that
  actually generated the submitted Figures 7--10 and validate every decoded
  panel pixel.
- `smoke`: run a documented subset locally, then parse and plot it.
- `full`: run the paper-result parameter grid, then parse, validate, and plot it.

Figure 2 has no `archived` input and rejects that mode. Appendix A instead uses
`table`, `parse --accuracy-csv FILE`, `archived-source --arxiv-source FILE`, and
`full`; its `full` mode is deliberately a failing asset audit until the original
training inputs are recovered.

The `--output` value is a base directory. Each command writes to
`--output/<mode>/`, so a smoke run cannot replace archived or full-grid
artifacts. If the supplied output path already ends in the selected mode, it is
used directly. Every mode directory includes `artifact_profile.json`; smoke
outputs are marked `validation_only_subset` because their tables and figures
are useful checks, not complete paper artifacts.

Fresh jobs are isolated into separate subprocesses and logs.  Successful jobs
are resumed by default; pass `--no-resume` to rerun them.  The default is one
Julia process at a time because each process already uses 16 threads.  Change
this with `--threads` and `--jobs` only when the machine has sufficient cores
and memory.  `--no-plots` skips matplotlib while retaining CSV outputs.

Run the lightweight parser, grid, seed, command-construction, and reference
regression tests with:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s reproduction/tests -v
```

The suite includes deliberate wrong-seed and wrong-result mutations, so a
passing test verifies rejection behavior as well as the expected paths.

## Julia and GNN setup for fresh experiments

The branch's advanced entry points expect the checkout directory to be named
`PDMO.jl` and their warmup scripts to be run from its parent:

```bash
cd /path/to/parent
julia PDMO.jl/warmup.jl
julia PDMO.jl/advanced/warmup.jl
cd PDMO.jl
```

Fresh Julia subprocesses automatically remove only the active Conda
environment's `lib` and `lib64` entries from `LD_LIBRARY_PATH`. This avoids
Conda's GLib libraries overriding Julia's JLL libraries while retaining CUDA,
Gurobi, and other unrelated library paths. The adjustment is local to each
Julia child and is recorded in its `.command.json`.

Section 3.3 and Section 3.4 runs that include GNN also require a Python
interpreter containing `torch`, `torch-geometric`, and `numpy` (the default
Section 3.4 smoke subset is Basic plus BFS and does not require it):

```bash
# Install Python 3.9 with the system package manager or pyenv if it is absent.
python3.9 -m venv reproduction/.venv-gnn
reproduction/.venv-gnn/bin/python -m pip install --upgrade pip
reproduction/.venv-gnn/bin/python -m pip install \
  -r reproduction/requirements.txt
reproduction/.venv-gnn/bin/python -m pip install \
  torch==2.8.0 numpy==2.0.2 torch-geometric==2.6.1
export PDMO_PYTHON="$PWD/reproduction/.venv-gnn/bin/python"
```

The fresh GNN runner accepts Python 3.9 through 3.11; Python 3.9 most closely
matches the archived runs. Python 3.12 is rejected before Julia starts because
the current PyCall/Torch path can crash during interpreter finalization.
Installing the reviewer requirements in this environment is also necessary:
rebuilding PyCall for GNN makes Julia's PyPlot use the same interpreter, and
the fresh Section 3.1 drivers import Matplotlib through PyPlot.
After setting `PDMO_PYTHON`, instantiate the advanced environment so PyCall is
built against that exact interpreter:

```bash
cd ..
julia PDMO.jl/advanced/warmup.jl
cd PDMO.jl
```

Sections 3.3 and 3.4 preflight both the standalone interpreter and Julia's actual
`PyCall.python`, check that their canonical paths match, and import all three
GNN dependencies through PyCall before launching any experiment job.

The original archive did not include lock files. Retained Sections 3.2--3.4
identify HiGHS.jl 1.18.0, HiGHS_jll 1.11.0+1, and HiGHS core 1.11.0 (git
`364c83a51e`), so the included Julia 1.11.5 lock snapshots deliberately pin
that exact pair. The distinct retained Section 3.1 transcript names HiGHS.jl
1.9.1 but does not expose its JLL/core version. Other visible archive versions
include Ipopt.jl 1.10.3 and PowerModels 0.21.3; the GNN jobs used Python 3.9,
torch 2.8.0, torch-geometric 2.6.1, and numpy 2.0.2. The snapshots are the
tested reproduction environment, not a claim that every unrecorded historical
dependency or runtime detail has been reconstructed.

## Section commands

### Random-seed contract

The wrappers pass or reset every experiment seed explicitly and validators
reject non-paper values:

| Experiment | Outer Julia seed(s) | Smoke subset | Separate local RNG |
|---|---|---|---|
| Figure 2 | 126 | rho=100 panel | none |
| Section 3.1 | 126 | 200 iterations | historical helper used local 42; the literal manuscript clustering does not |
| Section 3.2 | 1--10 | N=200, seed 1 | none |
| Section 3.3 | 126 | case30, P=3 | GNN Python inference fixes 42 |
| Section 3.4 | 111, 222, 333, 444, 555 | original/N=50/seed 111 | GNN Python inference fixes 42 |

The GNN-local seed does not replace the outer Julia seed that generates the
optimization instance. Generated CSV and JSON files retain the outer seed.

The default smoke subsets have exact archived fingerprints:

| Experiment | Enforced seed-specific result |
|---|---|
| Section 3.2, N=200/seed 1 | Basic/BFS/MILP iterations = 142/11,001/8,883; all optimal |
| Section 3.3, case30/P=3/seed 126 | BFS/MILP/GNN iterations = 136/136/136 and centralized objective = 565.2059663999221 |
| Section 3.4, original/N=50/seed 111 | Basic = 4,350 iterations, 182 nodes, 264 edges; BFS = 2,658 iterations, 105 nodes, 187 edges |

Archive mode strictly checks every reported aggregate. Smoke mode strictly
checks the deterministic rows above. Section 3.3 full mode additionally parses
the exact 162 archive rows used by Figures 13--14 and enforces configuration,
iteration, and terminal-status identity on all 159 non-censored rows. The three
archived `case1888rte`/P=10 rows are explicitly classified as time-censored;
their fresh stop state and iteration count, and all runtimes, are informational.
Section 3.4 also makes an archive-backed, field-aware raw comparison, with
deterministic identity and result fields strict and explicitly censored fields
and timings informational.

### Figure 2: circuit example

```bash
python3 reproduction/section_1.py --mode smoke
python3 reproduction/section_1.py --mode full
```

Every panel uses the legacy solve limit of 10,000 ADMM iterations.  The original `demo.jl --k` values are plot cutoffs, not shorter solve horizons:
`(rho, cutoff) = (1, 10000), (10, 1000), (100, 100)`.  The exporter records both the legacy
display index and the actual ADMM index, validates all nine trajectories and the
original-ADMM identity, explicitly records the original global seed 126, and
writes only below `--output`.

### Section 3.1: LP case study, Figures 7--10

Fresh execution defaults to the byte-verified
`reproduction/instances/miplib/enlight_hard.mps.gz`:

```bash
# Reproduce the eight panels actually submitted as Figures 7--10.
python3 reproduction/section_3_1.py --mode reported

# Separately exercise the five-pass algorithm stated in the manuscript.
python3 reproduction/section_3_1.py --mode smoke
python3 reproduction/section_3_1.py --mode full
```

The `--mps PATH` option can override the bundled file, but every fresh mode
requires the official decompressed content hash. The bundled gzip is the exact
public MIPLIB download and has compressed SHA256
`942168c2126a2a91ae3ec1ededea59bc1af0cad55f94223edf4c03d20e831f66`.

The authoritative paper command is `reported`. It invokes the public historical
driver that produced the submitted panels: global seed 126, co-clustering-local
seed 42, `k=4`, ten alternating passes, forced splitting, pairwise-row
promotion, doubly linearized ADMM, `rho=1000`, and a 100,000-iteration limit.
It verifies the official MPS content and pinned driver sources, then requires
all eight generated source panels for Figures 7--10 to match the arXiv v1
decoded RGBA pixels exactly. Raw PNG hashes may differ only because encoder
metadata is ignored. Because a bipartition and its global left/right complement
are the same MILP solution, the graph renderer anchors `x_1` on the left before
plotting the MILP result; this removes a solver-version-dependent mirror without
changing the graph or numerical result. The validation run completed all three
solves and matched 8/8 panels.

The `smoke` and `full` modes deliberately remain a separate
manuscript-literal profile: columns start cyclically without shuffling, five
row/column passes use smallest-index tie breaking, and four final row clusters
become four block constraints. They record deterministic graph ordering, fail
visibly if MILP bipartization fails, and use the same global seed 126. Their
Figures 7--10 are diagnostic outputs and are not labeled as the submitted
panels.

The supplied Section 3.1 archive is also a distinct later experiment. Its
terminal iterations (Basic/BFS/MILP = 72,462/53,700/16,049) differ from the
submitted historical run (79,955/49,178/28,111), so `archived` reconstructs
that retained comparison but is not the paper's Figure 10. The generated
profiles and provenance keep all three sources separate.

`enlight_hard` is distributed as a mixed-integer MIPLIB instance. All three
profiles read its matrix, bounds, and objective into `GenericLP` and solve the
continuous LP relaxation, matching the experiment implementation.

### Section 3.2: network flow, Figure 11

```bash
python3 reproduction/section_3_2.py --mode smoke
python3 reproduction/section_3_2.py --mode full
```

The full grid is 50 fresh processes: 2,000 arcs, node counts 200 through 600 in
steps of 100, and seeds 1 through 10. Each process uses the paper's 16 Julia
threads and runs Basic, BFS, and MILP on the same generated instance with
original ADMM, `rho=1`, maximum 100,000 iterations, and the paper's
infinity-norm tolerance `1e-4`.

Full mode requires `experiments_logs.zip` at the repository root (or an explicit
`--archive PATH`). Before creating its output directory or starting Julia, it
validates the five retained scale folders and all 50 scheduler jobs:
`40148228/50464369--78` (200 nodes), `40148239/50464389--98` (300),
`40148240/50464399--50464408` (400), `40148242/50464410--19` (500), and
`40148264/50464441--50` (600), with ascending job IDs mapped to seeds 1--10.
Their command files must encode the exact 16-thread paper configuration, and
their 150 CSV rows must match a pinned semantic SHA256 over job provenance,
configuration, iteration/status results, and the sole censored identity. A
mutated job mapping or stable result is therefore rejected before the long
sweep starts, as is a full run with other than 16 threads.

After the sweep, echoed solver settings and the complete key grid must match.
Iteration count and terminal status must equal the archive on all 149 stable
rows. The archived `nodes=500`, seed 6 Basic result is explicitly
time-limit-censored, so its stop iteration and status are informational; all
wall-clock values are also informational. The field-wise result is written to
`raw_archive_comparison.json`, and the archive ZIP hash is recorded in
`provenance.json`. A complete already-running or caller-supplied grid can receive
the identical post-hoc check without rerunning Julia:

The retained Section 3.2 logs record the node/arc counts, seed, configuration,
and solver outcomes, but not the generated edge list or a graph hash. The
runner therefore proves the exact seed/generator configuration and compares
the stable outcome fingerprints; it cannot retroactively prove byte-identical
graph membership from the ZIP alone.

```bash
python3 reproduction/section_3_2.py --mode parse \
  --logs /path/to/section_3_2/full/raw \
  --archive experiments_logs.zip
```

### Section 3.3: distributed DC-OPF, Figures 13--14

Rebuild the submitted Figure 13 and historical Figure 14 profiles from the
retained data without a Julia solve:

```bash
python3 reproduction/section_3_3.py --mode archived
```

Fresh execution defaults to the six files under
`reproduction/instances/matpower/`. They are byte-exact MATPOWER 8.1 files
from tag `8.1`, commit `1a828c7af590714499284e36ee9c81273388c594`:

```bash
python3 reproduction/section_3_3.py --mode smoke --python "$PDMO_PYTHON"
python3 reproduction/section_3_3.py --mode full --python "$PDMO_PYTHON"
```

Full mode requires `experiments_logs.zip` at the repository root (or an
explicit `--archive PATH`). It extracts only the 54 selected paper jobs, writes
their 162 method rows to `archive_reference_runs.csv`, and writes the field-wise
fresh comparison to `archive_raw_comparison.json`. Before any fresh Julia job,
the preflight binds each parsed log to its actual archive folder and job ID:
`case30/51455118--51455125`, `case57/51455152--51455159`,
`case89pegase/51455204--51455211`, `case118/51455233--51455240`,
`case300/51455276--51455283`, `case1888rte/51455616--51455623`, and
`case57_flip_2000/51523865--51523870`. Swapping contents between those paths is
therefore rejected even if the case-level aggregates remain unchanged.

All 162 selected method rows are also pinned independently by schema
`pdmo-section-3.3-archive-semantic-v1`. Records are sorted by
`(archive_folder, integer archive_job_id, method)` and serialized as UTF-8 JSON
with `sort_keys=True`, `separators=(',', ':')`, and `allow_nan=False`. The
75,278-byte canonical payload has SHA256
`c2a9cd539527ca3ebcf6873573159e4de5b10b425c8e2a1d23ff288bd95b6d35`.
It includes the source folder/job, Figure/case/P/method, every solver setting,
seed and thread count, centralized objective, iterations and status, and both
component times. Thus offsetting raw-row changes that preserve a published mean
still fail preflight. The runner writes `archive_semantic_manifest.json` in that
exact compact form, so hashing the file produces the stated digest, plus a
human-readable `archive_semantic_validation.json`. This is a semantic digest, not the ZIP-byte
digest; the currently supplied `experiments_logs.zip` itself has SHA256
`3a1a5e7a5e9f1c2996426b5cf41ae7b7672f5f9dd03fbbf166a322487f083138`,
which is recorded informationally so byte-neutral ZIP repacking does not
invalidate the scientific manifest.

Use `--matpower-dir DIR` only to override the bundled directory. Preflight
requires the same six SHA256 hashes before starting Julia. Per-file attribution
and licensing notes are preserved in the case headers and summarized in the
[bundled-input notice](instances/README.md).

Figure 13 uses the six cases, 3--10 partitions, original ADMM, and `rho=100`.
The submitted Figure 14 panels use `case57`, 13--18 partitions, doubly
linearized ADMM, and `rho=2000`; the six retained commands and logs establish
that profile and `archived` validates all 18 underlying method values. The
Figure 14 caption instead states `rho=1000`; this is treated as a caption typo.
Fresh `full` therefore runs the exact published `rho=2000` profile. A complete
caller-supplied `rho=1000` log grid remains recognizable only with `--mode
parse`, where it is labeled `caption_typo_parse_only`. Every job uses seed 126.
Fresh mode launches one Julia process for each `(case, partitions)` pair,
matching the archive's partition reuse and timing scope. Fresh full artifacts
are labeled `fresh_published_profile`, never as an archive reconstruction.

The archive did not identify the MATPOWER release or case-file checksums.
The bundled retained MATPOWER 8.1 files reproduce the archived objective
fingerprints. It also did not serialize each job's `bus2Area` partition mapping,
so graph/partition membership cannot be compared byte-for-byte after the fact;
the legacy path records seed 126 but did not explicitly forward a separate seed
to METIS. The runner validates both each bundled input hash before Julia starts
and the centralized DC objective printed by the driver:

| Case | MATPOWER 8.1 SHA256 | Expected centralized objective |
|---|---|---:|
| `case30` | `3d9030311259b553be85d02336b7e1bcb24ec04775bee6671bdb62d18e4e2137` | 565.2059663999221 |
| `case57` | `2218325a6e8fe6c7b8b28202f523670459268075a6fd41b4959d66f17d47d28b` | 41006.73694205554 |
| `case89pegase` | `7eb25c591f04a08dcd99ab433451054eaafd8bb3d7999d9279a8ddb23d8ffe58` | 5733.370870000001 |
| `case118` | `bc2e6f22b4b9e776572885ee4b50e4f4ab2ee0c5577e9126e86d906f14c4b5f7` | 125947.88140239994 |
| `case300` | `69a90280e999ef533d94656e0fbc08311f1347c962dd2753ff2005ff5e3f9ac5` | 706292.3242436099 |
| `case1888rte` | `df675cd826bb300e91596795ee3258a70deb81f0329dbc49ccf24c6048668037` | 59110.49999102508 |

### Section 3.4: decentralized consensus, Table 1 and Figures 15--16

```bash
python3 reproduction/section_3_4.py --mode smoke
python3 reproduction/section_3_4.py --mode full --python "$PDMO_PYTHON"
```

To exercise every paper method on only the smoke instance, including GNN:

```bash
python3 reproduction/section_3_4.py --mode smoke --smoke-methods paper --python "$PDMO_PYTHON"
```

The default smoke command runs Basic and BFS for speed. With
`--smoke-methods paper`, all seven paper methods are compared against and must
match the exact archived smoke fingerprint. This stronger contract is reliable
because the tested locks pin the archive's HiGHS.jl 1.18.0 wrapper and HiGHS
1.11.0 core.

The full grid uses graph sizes 50, 100, and 200; local dimensions `(n,m) =
(500,250)`; seeds 111, 222, 333, 444, and 555; original and doubly linearized
ADMM; `rho=10`; and MILP relative gaps 1%, 5%, 10%, and 20%.  The reviewer-only
Julia driver executes only those four paper gaps.  The older general driver also
runs 30%, 40%, and 50%, which adds roughly 26 hours of sequential work not used
in the paper figures.

Full mode requires the supplied `experiments_logs.zip`, selects its exact 210
paper run-method rows, and writes `raw_archive_comparison.json`. It requires the
fresh and archived keys, seeded configuration, logged graph/partition count
fingerprints (`graph_nodes`, left/right cardinalities, and `graph_edges`), ADMM
status, and iteration count to agree exactly. It also verifies that the original
and doubly linearized jobs regenerate the same Basic count fingerprint for each
`(N, seed)`. The retained logs do not serialize graph edge lists or vertex-to-side
partition membership, so exact graph/partition identity cannot be checked
retroactively. The archive has 48 MILP rows stopped by the 60-second HiGHS limit;
when either side is MILP-time-limited, its partition and downstream outcome
comparisons are explicitly marked censored. The archived doubly
linearized `N=200`, seed 333 Basic row is similarly ADMM-censored at 7200
seconds. All wall-clock fields and floating diagnostic residuals are retained
as informational comparisons. The ZIP hash is recorded in full-mode
`provenance.json`.

Archive selection is an explicit provenance manifest, not a directory scan:
the six plotted batches and their 30 scheduler job IDs are mapped one-to-one to
the five paper seeds. Before any full-mode Julia process starts, the runner
validates those exact jobs, all archived Table 1/Figures 15--16 aggregates, the
48 known MILP time-limit rows, and the sole ADMM-time-limit row. A canonical
210-row manifest additionally pins each relative archive member and scheduler
job ID, seeded configuration, logged structural counts, and iteration/status
outcomes to SHA-256
`e7a707de30a10e9c9d53ed9d16bcae8b615dccbc2a490e4b31872d8cd1836f2b`.
Timing fields are deliberately excluded from this archive-integrity digest. It
persists
`archive_reference_runs.csv`, `archive_reference_aggregate.csv`,
`archive_reference_table_1.csv`, `archive_reference_comparison.json`,
`archive_reference_manifest.json`, and `archive_reference_validation.json`.
Full mode requires every selected method row to echo the original 16-thread
configuration. Any archive-preflight mismatch aborts before the full output
directory is prepared or a Julia job starts.

For fidelity, this workflow reproduces the implementation actually represented
by the archive.  The manuscript writes the local loss as `||A_i x-b_i||^2`,
whereas `DistributedOpt.jl` constructs the linear term with `+2 A_i' b_i` under
PDMO's `x'Qx+q'x+r` convention, corresponding to `||A_i x+b_i||^2`.  This is
flagged in the generated provenance; the reproduction layer does not silently
change the experimental model.

Archived validation enforces 168 paper-value checks across Table 1 and Figures
15--16, and all pass their printed-precision tolerances. One Table 1 display
cell is not textually identical: the retained logs give a mean partition time
of 16.428 seconds for `N=50`, MILP gap 1%, which renders as 16.43, while the
paper prints 16.42. The validator retains the raw mean and records the 0.008
second difference instead of rewriting the value to force a match.

### Appendix A: Table 2 and Figure 18

Export Table 2 without Julia or GNN dependencies:

```bash
python3 reproduction/appendix_a.py --mode table
```

Preserve and byte-verify the reported Figure 18 raster from the public paper
source:

```bash
curl -L --fail https://arxiv.org/e-print/2603.19417v1 \
  -o /tmp/pdmo-paper-source.tar.gz
echo "f370667d2f464fea5f00df1f22682bd390fdead70c19cda97929b6b7a1dcf107  /tmp/pdmo-paper-source.tar.gz" \
  | sha256sum --check -
python3 reproduction/appendix_a.py --mode archived-source \
  --arxiv-source /tmp/pdmo-paper-source.tar.gz
```

The command accepts only the verified arXiv v1 raster by default. Its expected
SHA256 is `b93e66ce848b2801e567da74e63edc265e448d18217ef2cf63fea546775dc476`; it
labels the copied image `reported-source-only`, not a numerical rerun.

If the original epoch/test-accuracy history is recovered, supply a CSV with
`epoch,accuracy` columns (accuracy fractions are the default):

```bash
python3 reproduction/appendix_a.py --mode parse \
  --accuracy-csv /path/to/original_accuracy_history.csv --accuracy-scale fraction
```

Parse mode validates the history and renders Figure 18 as PNG/PDF, but it still
does not rerun training.

To audit the missing inputs for a genuine fresh retraining:

```bash
python3 reproduction/appendix_a.py --mode full
```

This command is expected to exit nonzero while writing a machine-readable
failed validation and complete missing-assets inventory. It can become a true
training entry point only after the original project, data, split, and seeds
are recovered.

### Static explanatory figures

The public arXiv source contains the inline source for Figures 1, 3--6, and 12,
and the included PDF for Figure 17. Build the submitted paper with:

```bash
curl -L --fail https://arxiv.org/e-print/2603.19417v1 \
  -o /tmp/pdmo-paper-source.tar.gz
echo "f370667d2f464fea5f00df1f22682bd390fdead70c19cda97929b6b7a1dcf107  /tmp/pdmo-paper-source.tar.gz" \
  | sha256sum --check -
mkdir -p /tmp/pdmo-paper-source
tar -xzf /tmp/pdmo-paper-source.tar.gz -C /tmp/pdmo-paper-source
cd /tmp/pdmo-paper-source
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The archive metadata specifies TeX Live 2025 and `pdflatex`; the bibliography step also requires BibTeX. This build renders
the explanatory figures in their paper context; it does not convert them into
numerical experiment outputs.

## Measured runtime expectations

These are sums of archived partition plus ADMM times on the original 16-thread
jobs.  They exclude dependency installation, compilation, initialization, data
loading, and plotting, and should be treated as order-of-magnitude estimates.

| Section | Full grid, sequential | Longest single process / smoke evidence |
|---|---:|---:|
| 3.2 network flow | about 16 h | about 10 min for the archived smoke point |
| 3.3 DC-OPF | about 33 h | up to about 6 h |
| 3.4 consensus | about 65 h | full-grid critical process about 10 h |

The paper used cluster nodes with 8 CPU cores and 30 GB RAM while requesting 16
Julia threads.  Wall-clock time is therefore not a strict reproducibility
criterion.  The scripts validate parameter grids, termination states,
iterations, aggregation, and normalized qualitative comparisons; raw and
normalized values are always retained beside the plots.

Two censored results are part of the reported averages and are not discarded:
the 500-node/seed-6 Basic network-flow run, and the 200-node/seed-333 Basic
FLiP consensus run.  In DC-OPF, all three methods for `case1888rte` with ten
partitions reach the time limit and are likewise retained.

## Output contract

Each mode-specific output directory contains, as applicable:

- `artifact_profile.json`: the execution mode and artifact scope, including an
  explicit validation-only warning for smoke output;
- `raw/`: one merged stdout/stderr log, command record, and completion record
  per subprocess;
- `runs.csv` or `residuals.csv`: long-form parsed measurements;
- Section 3.3 full mode also writes `archive_reference_runs.csv`,
  `archive_reference_validation.json`, `archive_semantic_manifest.json`,
  `archive_semantic_validation.json`, and `archive_raw_comparison.json`;
- Section 3.4 full mode writes the six `archive_reference_*` preflight files
  described above before starting fresh jobs;
- `summary.csv` / `aggregate.csv` / `aggregates.csv`: unnormalized
  arithmetic means used by the plots;
- `table_1.csv` and a rendered table for Section 3.4;
- `table_2.csv`, `table_2.md`, and `table_2.tex` for Appendix A;
- `figure_18_accuracy.csv` for a supplied original history, or
  `archived_source_manifest.json` for a source-only Figure 18 copy;
- `figures/`: paper-composed PNG and PDF artifacts plus useful individual
  panels;
- `validation.json`: strict grid, terminal-state, and structured-artifact checks;
- `reference_comparison.json` or Appendix `reference.json`: archived-regression
  checks or an explicit note when no numeric reference exists;
- `raw_archive_comparison.json`: Section 3.2 and Section 3.4 full-grid,
  field-aware comparisons with exact, censored, and hardware-informational
  checks separated;
- `provenance.json`: arguments, exact commands, git state, input hashes where
  available, machine details, and fidelity notes.

Parsers treat a subprocess exit code as necessary but not sufficient: the
existing Julia drivers catch some method errors, so each expected method and
terminal summary must also be present before a full or archived grid is
accepted.
