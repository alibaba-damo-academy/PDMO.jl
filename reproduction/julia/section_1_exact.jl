#!/usr/bin/env julia

# Reviewer-only exporter for the circuit experiment behind Figure 2.  The
# legacy plotting script always solved 10,000 ADMM iterations and used `--k`
# only to truncate the plotted history.  Keeping those two concepts separate
# is important: rho=10 and rho=100 must not be solved with shorter horizons.
import Pkg

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const SOLVE_MAX_ITER = 10_000
const PAPER_SEED = 126
Pkg.activate(REPO_ROOT)

# This loads the exact paper problem constructors.  The guarded main block in
# demo.jl does not run when included, so it creates no application artifacts.
include(joinpath(REPO_ROOT, "applications", "Demo", "demo.jl"))

function usage()
    println("Usage: julia section_1_exact.jl OUTPUT_CSV RHO PLOT_CUTOFF")
end

function main(args)
    length(args) == 3 || (usage(); error("Expected three arguments"))
    output_csv = abspath(args[1])
    rho = parse(Float64, args[2])
    plot_cutoff = parse(Int, args[3])
    rho > 0 || error("RHO must be positive")
    1 <= plot_cutoff <= SOLVE_MAX_ITER ||
        error("PLOT_CUTOFF must lie in 1:$(SOLVE_MAX_ITER)")

    # demo.jl currently seeds the global RNG while it is included.  Reset it
    # explicitly here so the reviewer contract does not depend on include-time
    # side effects or on future changes to that source file.
    Random.seed!(PAPER_SEED)

    param = ADMMParam()
    param.logInterval = 1
    param.initialRho = rho
    param.maxIter = SOLVE_MAX_ITER
    param.presTolL2 = 1e-30
    param.dresTolL2 = 1e-30
    param.presTolLInf = 1e-30
    param.dresTolLInf = 1e-30
    param.solver = OriginalADMMSubproblemSolver()
    param.logLevel = 1

    formulations = [
        ("12", "breaking 1st constraint", demo_break12()),
        ("23", "breaking 2nd constraint", demo_break23()),
        ("31", "breaking 3rd constraint", demo_break31()),
    ]
    results = Dict{String, Any}()
    for (key, _, problem) in formulations
        println(
            "Running reformulation breaking-$(key), rho=$(rho), " *
            "maxIter=$(SOLVE_MAX_ITER), plotCutoff=$(plot_cutoff)",
        )
        results[key] = runBipartiteADMM(problem, param)
    end

    mkpath(dirname(output_csv))
    open(output_csv, "w") do io
        println(
            io,
            "rho,formulation,label,iteration,actual_iteration,pres_l2,dres_l2," *
            "residual_sum,status,stop_iter,solve_max_iter,plot_cutoff,solver,seed",
        )
        for (key, label, _) in formulations
            info = results[key].iterationInfo
            available = min(length(info.presL2), length(info.dresL2))
            available >= plot_cutoff || error(
                "$(key) produced $(available) history samples, fewer than cutoff $(plot_cutoff)",
            )
            status = string(info.terminationStatus)
            for sample_index in 1:plot_cutoff
                pres = info.presL2[sample_index]
                dres = info.dresL2[sample_index]
                println(
                    io,
                    join(
                        (
                            rho,
                            key,
                            label,
                            sample_index,
                            sample_index - 1,
                            pres,
                            dres,
                            pres + dres,
                            status,
                            info.stopIter,
                            SOLVE_MAX_ITER,
                            plot_cutoff,
                            "OriginalADMMSubproblemSolver",
                            PAPER_SEED,
                        ),
                        ",",
                    ),
                )
            end
        end
    end
    println("Wrote exact legacy-cutoff trajectory data to $(output_csv)")
end

main(ARGS)
