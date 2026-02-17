# include(joinpath(@__DIR__, "../../warmup.jl"))
import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "GenericLP.jl"))

using PDMO

using LinearAlgebra
using SparseArrays
using Random
using Plots
using Printf
using LaTeXStrings
using JuMP 
import MathOptInterface 
using Ipopt
using HiGHS

Random.seed!(126)

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

    plt = plot(size=(640, 400))
    all_vals = Float64[]
    for key in ["Basic", "MILP", "BFS"]
        res = get(results, key, nothing)
        res === nothing && continue
        series = getfield(res.iterationInfo, field)
        isempty(series) && continue
        stride = max(1, k)
        iters = collect(1:stride:length(series))
        vals = series[iters]
        yvals = [v > 0 ? log10(v) : NaN for v in vals]
        append!(all_vals, yvals)
        plot!(
            plt,
            iters,
            yvals,
            label=labels[key],
            linestyle=linestyles[key],
            color=colors[key],
            linewidth=2.5,
        )
    end

    xlabel!(plt, "Iteration")
    if field == :presL2
        ylabel!(plt, L"\log_{10}(||\mathrm{Pres}||_2)")
    else
        ylabel!(plt, L"\log_{10}(||\mathrm{Dres}||_2)")
    end
    # title!(plt, title_text)
    if !isempty(all_vals)
        ymin = minimum(all_vals)
        ymax = maximum(all_vals)
        plot!(plt, ylims=(ymin, ymax))
    end
    plot!(plt, legend=:best, grid=true)
    savefig(plt, outfile)
    println("[info] saved plot -> $(outfile)")
end


if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 1
        usage()
        exit(1)
    end

    mps_path   = abspath(ARGS[1])
    outdir = length(ARGS) >= 2 ? ARGS[2] :  @__DIR__

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