include(joinpath(@__DIR__, "../../warmup.jl"))
# local test only:
# import Pkg
# Pkg.activate(joinpath(@__DIR__, "..", ".."))

using PDMO

using LinearAlgebra
using SparseArrays
using JuMP 
import MathOptInterface 
using Ipopt
using HiGHS

include(joinpath(@__DIR__, "GenericLP.jl"))


function runLPPipeline(mps_dir::String; 
    admmSolver::AbstractADMMSubproblemSolver = AdaptiveLinearizedSolver(gamma=1.0, r=1000.0, ifSimple=false),
    initialRho::Float64 = 100.0, 
    maxIter::Int = 1000000,
    logInterval::Int = 1000
)
    # println("runLPPipeline: mps_dir = $(mps_dir)")

    lp = GenericLP(mps_dir)
    mbp = generateGenericLP(lp)

    param = ADMMParam(
        initialRho = initialRho,
        maxIter = maxIter,
        logInterval = logInterval,
        solver = admmSolver,
        applyScaling = false
    )
    result = runBipartiteADMM(mbp, param)

    return result
end 


function runLPPipelineCoCluster(mps_dir::String; 
    admmSolver::AbstractADMMSubproblemSolver = AdaptiveLinearizedSolver(gamma=1.0, r=1000.0, ifSimple=false),
    bipartizationAlgorithm::BipartizationAlgorithm = MILP_BIPARTIZATION,
    initialRho::Float64 = 100.0, 
    maxIter::Int = 1000000,
    logInterval::Int = 1000,
    forceSplitSingleBlock::Bool = true
)
    lp = GenericLP(mps_dir)
    mbp = generateGenericLPWithCoClustering(lp; forceSplitSingleBlock=forceSplitSingleBlock)

    param = ADMMParam(
        initialRho = initialRho,
        maxIter = maxIter,
        logInterval = logInterval,
        solver = admmSolver,
        applyScaling = false
    )
    result = runBipartiteADMM(mbp, param; 
        bipartizationAlgorithm = bipartizationAlgorithm)

    return result
end 


# Allow running as a script:
# julia -t 16 applications/GenericLP/runLPPipeline.jl <mps_dir> [pipeline] [solver] [initialRho] [maxIter] [bipartAlg] [forceSplit]
#   mps_dir: absolute path to .mps or .mps.gz
#   pipeline: basic | cocluster (default: basic)
#   solver:   adaptive | simple | doubly  (default: adaptive)
#   initialRho: Float64 (default: 100.0)
#   maxIter:   Int (default: 1000000)
#   bipartAlg (for cocluster only): milp | bfs | dfs | spanning (default: milp)
#   forceSplit (for cocluster only): true | false (default: true)
# helper to parse bool-ish strings
parse_bool(str::AbstractString) = lowercase(str) in ("true","1","yes","y")

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 1
        error("Usage: julia -t 16 applications/GenericLP/runLPPipeline.jl <mps_dir> [pipeline] [solver] [initialRho] [maxIter] [bipartAlg]")
    end
    input_arg   = ARGS[1]  # mps_dir
    pipeline    = length(ARGS) >= 2 ? lowercase(ARGS[2]) : "basic"
    solver_name = length(ARGS) >= 3 ? lowercase(ARGS[3]) : "adaptive"
    initial_rho = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 100.0
    max_iter    = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 1000000
    bip_name    = length(ARGS) >= 6 ? lowercase(ARGS[6]) : "milp"
    force_split = length(ARGS) >= 7 ? parse_bool(ARGS[7]) : true
    log_interval = length(ARGS) >= 8 ? parse(Int, ARGS[8]) : 1000
    admm_solver = begin
        if solver_name == "adaptive"
            AdaptiveLinearizedSolver(gamma=1.0, r=1000.0, ifSimple=false)
        elseif solver_name == "simple"
            AdaptiveLinearizedSolver(gamma=1.0, r=1000.0, ifSimple=true)
        elseif solver_name == "doubly"
            DoublyLinearizedSolver()
        else
            @warn "Unknown solver '$solver_name'. Falling back to 'adaptive'."
            AdaptiveLinearizedSolver(gamma=1.0, r=1000.0, ifSimple=false)
        end
    end

    if pipeline == "cocluster"
        bip_alg = begin
            if bip_name == "milp"
                MILP_BIPARTIZATION
            elseif bip_name == "bfs"
                BFS_BIPARTIZATION
            elseif bip_name == "dfs"
                DFS_BIPARTIZATION
            elseif bip_name == "spanning"
                SPANNING_TREE_BIPARTIZATION
            else
                @warn "Unknown bipartization algorithm '$bip_name'. Falling back to 'milp'."
                MILP_BIPARTIZATION
            end
        end
        mps_dir = input_arg
        println("runLPPipelineCoCluster args:")
        println("  mps_dir    = ", mps_dir)
        println("  solver     = ", solver_name)
        println("  initialRho = ", initial_rho)
        println("  maxIter    = ", max_iter)
        println("  bipartAlg  = ", bip_name)
        println("  forceSplit = ", force_split)
        runLPPipelineCoCluster(mps_dir;
            admmSolver = admm_solver,
            initialRho = initial_rho,
            maxIter    = max_iter,
            logInterval = log_interval,
            bipartizationAlgorithm = bip_alg,
            forceSplitSingleBlock = force_split)
    else
        mps_dir = input_arg
        println("runLPPipeline args:")
        println("  mps_dir    = ", mps_dir)
        println("  solver     = ", solver_name)
        println("  initialRho = ", initial_rho)
        println("  maxIter    = ", max_iter)
        runLPPipeline(mps_dir; admmSolver=admm_solver, initialRho=initial_rho, maxIter=max_iter, logInterval=log_interval)
    end
end