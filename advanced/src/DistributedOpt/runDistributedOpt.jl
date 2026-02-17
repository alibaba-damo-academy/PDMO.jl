# Optional warmup for the advanced project (skip by default to avoid Pkg on constrained envs)
include(joinpath(@__DIR__, "../../warmup.jl"))

using LightGraphs
using Random
using Printf

include(joinpath(@__DIR__, "../include.jl"))
include(joinpath(@__DIR__, "DistributedOpt.jl"))
include(joinpath(@__DIR__, "..", "gnn", "io.jl"))
include(joinpath(@__DIR__, "..", "gnn", "GnnBipartization.jl"))


function main(args)

    if length(args) < 1
        error("Usage: julia -t 16 advanced/src/DistributedOpt/runDistributedOpt.jl numberNodes kappa n m")
    end

    numberNodes = parse(Int, args[1])
    n = parse(Int, args[2])
    m = parse(Int, args[3])
    solver_name = length(args) >= 4 ? args[4] : "original"
    initialRho = length(args) >= 5 ? parse(Float64, args[5]) : 10.0
    maxIter = length(args) >= 6 ? parse(Int, args[6]) : 100000
    logInterval = length(args) >= 7 ? parse(Int, args[7]) : 1000
    seed = length(args) >= 8 ? parse(Int, args[8]) : 126
    mipRelGap = length(args) >= 9 ? parse(Float64, args[9]) : 0.01
    mipHeuristicEffort = length(args) >= 10 ? parse(Float64, args[10]) : 0.2
    mipTimeLimit = length(args) >= 11 ? parse(Float64, args[11]) : 60.0
    
    gnn_force_cpu = true

    Random.seed!(seed)
    
    kappaMax = 10.0/numberNodes 
    kappaMin = 2.0/numberNodes 
    kappa = kappaMin + rand() * (kappaMax - kappaMin)
    
    function setSolver()
        if solver_name == "original"    
            return OriginalADMMSubproblemSolver()
        elseif solver_name == "doubly"
            return DoublyLinearizedSolver()
        else
            error("Usage: julia -t 16 advanced/src/DistributedOpt/runDistributedOpt.jl numberNodes n m solver_name")
        end
    end 

    println("Running Distributed Opt")
    println("  numberNodes = ", numberNodes)
    println("  kappa = ", kappa)
    println("  n = ", n)
    println("  m = ", m)
    println("  solver = ", solver_name)
    println("  initialRho = ", initialRho)
    println("  maxIter = ", maxIter)
    println("  logInterval = ", logInterval)
    println("  seed = ", seed)
    println("  gnn eval. on_cpu = ", gnn_force_cpu)
    println("="^60)

    g, objFunctions = generateDistributedOptInstance(numberNodes, kappa, n, m)

    
    param = ADMMParam(
        initialRho = initialRho,
        maxIter = maxIter,
        logInterval = logInterval,
        presTolL2 = 1e-4, 
        presTolLInf = 1e-6, 
        dresTolL2 = 1e-4, 
        dresTolLInf = 1e-6, 
        logLevel = 1,
        timeLimit = 7200.0
        # applyScaling = true
    )

    results = Vector{Tuple{String, Any}}()

    try
        println("="^60)
        println("Solving classic distributed opt problem...")
        param.solver = setSolver()
        mbpClassic = generateClassicDistributedOptProblem(g, objFunctions)
        result = runBipartiteADMM(mbpClassic, param;
            saveSolutionInMultiblockProblem = false, 
            tryJuMP = false)
        push!(results, ("Classic", result))
    catch e
        @error "Failed to solve the classic distributed opt problem."
    end 
    println("="^60)

    
    mbp = generateDistributedOptProblem(g, objFunctions)
    try
        println("Solving distributed opt problem with BFS bipartization...")
        param.solver = setSolver()
        result_bfs = runBipartiteADMM(mbp, param;
            bipartizationAlgorithm = BFS_BIPARTIZATION,
            saveSolutionInMultiblockProblem = false, 
            tryJuMP = false)
        push!(results, ("BFS", result_bfs))
    catch e
        @error "Failed to solve the distributed opt problem with BFS bipartization."
    end 
    println("="^60)

    for gap in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        try
            println("Solving distributed opt problem with MILP bipartization gap=$gap...")
            param.solver = setSolver()
            result_milp = runBipartiteADMM(mbp, param;
                bipartizationAlgorithm = MILP_BIPARTIZATION,
                mipRelGap = gap,
                mipTimeLimit = mipTimeLimit,
                mipHeuristicEffort = mipHeuristicEffort, 
                saveSolutionInMultiblockProblem = false, 
                tryJuMP = false)
            push!(results, ("MILP($gap)", result_milp))
        catch e
            @error "Failed to solve the distributed opt problem with MILP bipartization."
        end 
        println("="^60)
    end 

    registerGnnBipartizationImpl!(; force_cpu = gnn_force_cpu, model_path = GNN_MODEL_PATH)
    try
        println("Solving distributed opt problem with GNN bipartization...")
        param.solver = setSolver()
        result_gnn = runBipartiteADMM(mbp, param;
            bipartizationAlgorithm = GNN_BIPARTIZATION,
            saveSolutionInMultiblockProblem = false,
            tryJuMP = false)
        push!(results, ("GNN", result_gnn))
    catch e
        @error "Failed to solve the distributed opt problem with GNN bipartization." exception=(e, catch_backtrace())
    end

    println("="^60)
    println("SUMMARY OF RESULTS")
    println("="^60)
    println("Method | BipT | Iters | ADMM Time | ADMM Obj |")
    println("-"^80)
    for (label, res) in results
        info = hasproperty(res, :iterationInfo) ? res.iterationInfo : res
        bip_t = hasproperty(info, :partitionAlgorithmTime) ? info.partitionAlgorithmTime : 0.0
        iters = hasproperty(info, :stopIter) ? info.stopIter : 0
        total_time = hasproperty(info, :totalTime) ? info.totalTime : 0.0
        obj = hasproperty(info, :obj) ? info.obj[end] : NaN
        println(@sprintf("%15s | %4.3f | %5d | %9.2f | %8.4f |",
            label, bip_t, iters, total_time, obj))
    end
end 


if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
