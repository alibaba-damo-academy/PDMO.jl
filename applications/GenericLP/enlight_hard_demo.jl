# include(joinpath(@__DIR__, "../../warmup.jl"))
import Pkg

const SCRIPT_DIR = @__DIR__
const LEGACY_ENV_DIR = joinpath(SCRIPT_DIR, "legacy_env")
const REPO_ROOT = normpath(joinpath(SCRIPT_DIR, "..", ".."))

function setup_and_activate_legacy_env()
    # Conda-injected LD_LIBRARY_PATH can break JLL runtime loading.
    if haskey(ENV, "LD_LIBRARY_PATH")
        ld_path = lowercase(ENV["LD_LIBRARY_PATH"])
        if occursin("conda", ld_path) || occursin("anaconda", ld_path)
            # Re-exec once with a clean library path when run as a script.
            if abspath(PROGRAM_FILE) == abspath(@__FILE__) && get(ENV, "PDMO_LEGACY_REEXEC", "0") != "1"
                cmd_parts = copy(Base.julia_cmd().exec)
                append!(cmd_parts, [abspath(@__FILE__); ARGS])
                cmd = addenv(Cmd(cmd_parts),
                    "LD_LIBRARY_PATH" => "",
                    "PDMO_LEGACY_REEXEC" => "1")
                proc = run(ignorestatus(cmd))
                exit(proc.exitcode)
            end
            delete!(ENV, "LD_LIBRARY_PATH")
        end
    end

    project_file = joinpath(LEGACY_ENV_DIR, "Project.toml")
    isfile(project_file) || error("Missing legacy environment at $(project_file)")

    Pkg.activate(LEGACY_ENV_DIR)
    # Ensure local PDMO is used in this script-local environment.
    Pkg.develop(path=REPO_ROOT)
    # Always instantiate: works both with and without Manifest.toml.
    Pkg.instantiate()
end

setup_and_activate_legacy_env()

include(joinpath(@__DIR__, "GenericLP.jl"))
include(joinpath(@__DIR__, "inspect_cocluster.jl"))

using PDMO

using LinearAlgebra
using SparseArrays
using Random
using PyPlot
using Printf
using JuMP 
import MathOptInterface 
using Ipopt
using HiGHS

Random.seed!(126)

const DEFAULT_OUTPUT_DIR = joinpath(@__DIR__, "enlight_hard_plots")

function usage()
    println("""
Usage:
    julia applications/GenericLP/enlight_hard_demo.jl <mps_path> [output_dir]

Defaults:
    output_dir = applications/GenericLP/enlight_hard_plots
""")
end

function _plot_residuals(results::AbstractDict, field::Symbol, title_text::String, outfile::String; k::Int=1)
    labels = Dict(
        "Basic" => "Basic",
        "MILP" => "MILP",
        "BFS" => "BFS",
    )
    linestyles = Dict(
        "Basic" => :solid,
        "MILP" => :dash,
        "BFS" => :dot,
    )
    colors = Dict(
        "Basic" => :blue,
        "MILP" => :green,
        "BFS" => :orange,
    )

    fig, ax = subplots(figsize=(6.4, 4.0))
    all_vals = Float64[]
    for key in ["Basic", "MILP", "BFS"]
        res = get(results, key, nothing)
        res === nothing && continue
        series = getfield(res.iterationInfo, field)
        isempty(series) && continue
        stride = max(1, k)
        iters = collect(1:stride:length(series))
        vals = series[iters]
        if field == :dresL2 && !isempty(vals)
            # The first stored dres entry corresponds to iter 0 and may be Inf.
            iters = iters[2:end]
            vals = vals[2:end]
        end
        isempty(vals) && continue
        yvals = [v > 0 ? log10(v) : NaN for v in vals]
        append!(all_vals, yvals)
        style = linestyles[key] == :solid ? "-" : linestyles[key] == :dash ? "--" : ":"
        ax.plot(iters, yvals;
            label=labels[key],
            linestyle=style,
            color=string(colors[key]),
            linewidth=2.5)
    end

    ax.set_xlabel("Iteration")
    if field == :presL2
        ax.set_ylabel("log10(||Pres||_2)")
    else
        ax.set_ylabel("log10(||Dres||_2)")
    end
    ax.set_title(title_text)
    finite_vals = filter(isfinite, all_vals)
    if !isempty(finite_vals)
        ymin = minimum(finite_vals)
        ymax = maximum(finite_vals)
        if ymin == ymax
            # Avoid invalid/singular axis ranges when residuals are constant.
            pad = max(abs(ymin) * 0.05, 1e-6)
            ax.set_ylim(ymin - pad, ymax + pad)
        else
            ax.set_ylim(ymin, ymax)
        end
    end
    ax.grid(true)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    close(fig)
    println("[info] saved plot -> $(outfile)")
end


if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 1
        usage()
        exit(1)
    end

    mps_path   = abspath(ARGS[1])
    outdir = length(ARGS) >= 2 ? ARGS[2] : DEFAULT_OUTPUT_DIR
    isdir(outdir) || mkpath(outdir)

    inspect_cocluster(mps_path; output_dir=outdir, k=6, iters=5, forceSplit=true)

    lp = GenericLP(mps_path)

    initialRho = 1000.0 
    maxIter = 100000
    logInterval = 1000

    results = Dict()
     try 
        mbp = generateGenericLP(lp)
        param = ADMMParam(
            initialRho = initialRho,
            maxIter = maxIter,
            logInterval = logInterval,
            solver = DoublyLinearizedSolver(),
            applyScaling = false
        )
        results["Basic"] = runBipartiteADMM(mbp, param)
    catch e
        @error "Failed to solve the problem with classic bipartization." exception=(e, catch_backtrace())
        return
    end


    try 
        mbp = generateGenericLPWithCoClustering(lp)
        param = ADMMParam(
            initialRho = initialRho,
            maxIter = maxIter,
            logInterval = logInterval,
            solver = DoublyLinearizedSolver(),
            applyScaling = false
        )
        results["BFS"] = runBipartiteADMM(mbp, param; bipartizationAlgorithm = BFS_BIPARTIZATION)
    catch e
        @error "Failed to solve the problem with BFS bipartization." exception=(e, catch_backtrace())
        return
    end

    try 
        mbp = generateGenericLPWithCoClustering(lp)
        param = ADMMParam(
            initialRho = initialRho,
            maxIter = maxIter,
            logInterval = logInterval,
            solver = DoublyLinearizedSolver(),
            applyScaling = false
        )
        results["MILP"] = runBipartiteADMM(mbp, param; bipartizationAlgorithm = MILP_BIPARTIZATION)
    catch e
        @error "Failed to solve the problem with MILP bipartization." exception=(e, catch_backtrace())
        return
    end

    _plot_residuals(
        results,
        :presL2,
        "Primal Residuals",
        joinpath(outdir, "primal_residuals.png"),
        k=500,
    )
    _plot_residuals(
        results,
        :dresL2,
        "Dual Residuals",
        joinpath(outdir, "dual_residuals.png"),
        k=1,
    )

end 