# Optional warmup for the advanced project (skip by default to avoid Pkg on constrained envs)
include(joinpath(@__DIR__, "../../warmup.jl"))

import Metis, LightGraphs
using Random
using Statistics, Printf 

include(joinpath(@__DIR__, "../include.jl"))
include(joinpath(@__DIR__, "DistributedDCOPF.jl"))
include(joinpath(@__DIR__, "..", "gnn", "io.jl"))
include(joinpath(@__DIR__, "..", "gnn", "GnnBipartization.jl"))


# test different bipartization algorithms 

# Allow running as a script:
# julia -t 16 advanced/src/OPF/runDistributedOPF.jl MATPOWER_PATH [numberPartitions] [admmSolver] [initialRho] [tol] [maxIter] [timeLimit] [logInterval] [useCustomizedGeneratorCost] [seed] [r_value] [bipartization]
#   MATPOWER_PATH: path to a Matpower case file (e.g. ".../case30.m") (required)
#   numberPartitions: integer (default: 3)
#   admmSolver: "original" | "adaptive" | "doubly" (default: "original")
#   initialRho: Float64 (default: 10.0)
#   tol: Float64 (default: 1e-4)
#   maxIter: Int (default: 1_000_000)
#   timeLimit: Float64 seconds (default: 7200.0)
#   logInterval: Int (default: 100)
#   useCustomizedGeneratorCost: Bool (default: false)
#   r_value: Float64, used only when admmSolver="adaptive" (default: 1e4)
#   seed: Int (default: 123)
if abspath(PROGRAM_FILE) == @__FILE__
    
    matpower_path = abspath(ARGS[1]) 
    numberPartitions = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
    admmSolver = length(ARGS) >= 3 ? ARGS[3] : "original"

    initialRho = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 10.0
    tol = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 1.0e-4
    maxIter = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 1000000
    timeLimit = length(ARGS) >= 7 ? parse(Float64, ARGS[7]) : 7200.0
    logInterval = length(ARGS) >= 8 ? parse(Int, ARGS[8]) : 100
    r_value = length(ARGS) >= 9 ? parse(Float64, ARGS[9]) : 1.0e4
    seed = length(ARGS) >= 10 ? parse(Int, ARGS[10]) : 126

    gnn_force_cpu = true
    useCustomizedGeneratorCost = false
    mergeConstraints = false
    
    Random.seed!(seed)

    function setSolver()
        if admmSolver == "adaptive"
            return AdaptiveLinearizedSolver(gamma=1.0, r=r_value, ifSimple=false)
        elseif admmSolver == "doubly"
            return DoublyLinearizedSolver()
        else
            return OriginalADMMSubproblemSolver()
        end
    end

    println("Running Distributed DCOPF")
    println("  case file        = ", matpower_path)
    println("  num. Partitions  = ", numberPartitions)
    println("  admmSolver       = ", admmSolver)
    println("  initialRho       = ", initialRho)
    println("  tol              = ", tol)
    println("  maxIter          = ", maxIter)
    println("  timeLimit        = ", timeLimit)
    println("  logInterval      = ", logInterval)
    println("  r_value          = ", r_value)
    println("  seed             = ", seed)

    println("  addtional options:")
    println("  gnn eval. on_cpu = ", gnn_force_cpu)
    println("  customized Cost  = ", useCustomizedGeneratorCost)
    println("  mergeConstraints = ", mergeConstraints)

    networkInfo = PowerNetworkInfo(matpower_path)
    dcObj, jumpTime = DCModel(networkInfo; customizedGeneratorCost=useCustomizedGeneratorCost)

    println("="^120)
    println("True DC objective: $dcObj")
    println("JuMP solution time: $(round(jumpTime, digits=3)) seconds")
    println("="^120)

    # create inital partition of the network 
    bus2Area = generatePartitions(networkInfo, numberPartitions)

    results = Vector{Tuple{String, Any}}()

    bfs_result = try
        testDistributedOPF(networkInfo,
            numberPartitions,
            bus2Area,
            BFS_BIPARTIZATION,
            setSolver();
            dcObj = dcObj,
            tol = tol,
            initialRho = initialRho,
            maxIter = maxIter,
            timeLimit = timeLimit,
            logInterval = logInterval,
            useCustomizedGeneratorCost = useCustomizedGeneratorCost,
            mergeConstraints = mergeConstraints,
            seed = seed)
    catch e
        @error "Failed to solve the distributed DCOPF problem with BFS bipartization."
        return
    end
    push!(results, ("BFS", bfs_result))

    println("="^120)
    milp_result = try
        testDistributedOPF(networkInfo,
            numberPartitions,
            bus2Area,
            MILP_BIPARTIZATION,
            setSolver();
            dcObj = dcObj,
            tol = tol,
            initialRho = initialRho,
            maxIter = maxIter,
            timeLimit = timeLimit,
            logInterval = logInterval,
            useCustomizedGeneratorCost = useCustomizedGeneratorCost,
            mergeConstraints = mergeConstraints,
            seed = seed)
    catch e
        @error "Failed to solve the distributed DCOPF problem with MILP bipartization." exception=(e, catch_backtrace())
        return
    end
    push!(results, ("MILP", milp_result))

    registerGnnBipartizationImpl!(; force_cpu = gnn_force_cpu, model_path = GNN_MODEL_PATH)
    println("="^120)
    gnn_result = try
        testDistributedOPF(networkInfo,
            numberPartitions,
            bus2Area,
            GNN_BIPARTIZATION,
            setSolver();
            dcObj = dcObj,
            tol = tol,
            initialRho = initialRho,
            maxIter = maxIter,
            timeLimit = timeLimit,
            logInterval = logInterval,
            useCustomizedGeneratorCost = useCustomizedGeneratorCost,
            mergeConstraints = mergeConstraints,
            seed = seed)
    catch e
        @error "Failed to solve the distributed DCOPF problem with GNN bipartization." exception=(e, catch_backtrace())
        return
    end
    push!(results, ("GNN", gnn_result))

    println("="^60)
    println("SUMMARY OF RESULTS for partitions: $numberPartitions")
    println("="^60)
    println("Method | BipT | Iters | ADMM Time | ADMM Obj |")
    println("-"^80)
    for (label, res) in results
        println(@sprintf("%6s | %4.3f | %5d | %9.2f | %8.4f |",
            label,
            res.partitionAlgorithmTime,
            res.stopIter,
            res.totalTime,
            res.obj[end]))
    end
end