# Optional warmup for the advanced project (skip by default to avoid Pkg on constrained envs)
include(joinpath(@__DIR__, "../../warmup.jl"))

using Random 

include(joinpath(@__DIR__, "../include.jl"))
include(joinpath(@__DIR__, "NetworkFlow.jl"))


function main(args)
    function usage()
        println("Usage:")
        println("  julia -t <threads> advanced/src/NetworkFlow/runNetworkFlowProblem.jl [options]")
        println()
        println("Input options (choose one):")
        println("  --instance <path>                 Read DIMACS min-cost-flow instance")
        println("  --random <nodes> <arcs>           Generate a random feasible instance")
        println()
        println("ADMM options:")
        println("  --solver <original|doubly|adaptive>   ADMM subproblem solver (default: original)")
        println("  --maxIter <int>                      ADMM max iterations (default: 10000)")
        println("  --initialRho <float>                 Initial rho (default: 10.0)")
        println("  --timeLimit <float>                  Max wall-clock time in seconds (default: 3600.0)")
        println()
        println("Misc:")
        println("  --seed <int>                         RNG seed (default: 123)")
        println("  --logInterval <int>                  ADMM log interval (default: 100)")
        println("  -h, --help                           Show this help")
    end

    # -----------------
    # Parse CLI args
    # -----------------
    if any(a -> a in ("-h", "--help"), args)
        usage()
        return
    end

    opts = Dict{String, Vector{String}}()
    positionals = String[]
    i = 1
    while i <= length(args)
        a = args[i]
        if startswith(a, "--")
            key = a
            vals = String[]
            # Collect following tokens until next flag or end (simple but practical).
            j = i + 1
            while j <= length(args) && !startswith(args[j], "--")
                push!(vals, args[j])
                j += 1
            end
            opts[key] = vals
            i = j
        else
            push!(positionals, a)
            i += 1
        end
    end

    # Helper: fetch a single-value option with validation.
    function get1(opts::Dict{String, Vector{String}}, key::String, default::String)
        if !haskey(opts, key)
            return default
        end
        vals = opts[key]
        length(vals) == 1 || error("$key expects exactly 1 value")
        return vals[1]
    end

    # Defaults
    solver_name = get1(opts, "--solver", "original")
    maxIter = parse(Int, get1(opts, "--maxIter", "10000"))
    initialRho = parse(Float64, get1(opts, "--initialRho", "10.0"))
    timeLimit = parse(Float64, get1(opts, "--timeLimit", "3600.0"))
    seed = parse(Int, get1(opts, "--seed", "123"))
    logInterval = parse(Int, get1(opts, "--logInterval", "100"))

    Random.seed!(seed)

    function setSolver(name::String)
        n = lowercase(strip(name))
        if n in ("original", "orig")
            return OriginalADMMSubproblemSolver()
        elseif n in ("doubly", "doublylinearized", "doubly_linearized")
            return DoublyLinearizedSolver()
        elseif n in ("adaptive", "adaptivelinearized", "adaptive_linearized")
            return AdaptiveLinearizedSolver()
        else
            error("Unknown --solver '$name'. Use one of: original, doubly, adaptive.")
        end
    end

    # -----------------
    # Build problem
    # -----------------
    nfp = nothing
    if haskey(opts, "--instance")
        vals = opts["--instance"]
        length(vals) == 1 || error("--instance expects exactly 1 value")
        path = vals[1]
        println("Reading instance: $path")
        nfp = readDimacsMinCostFlowInstance(path; applyLowerBounds=true)
    elseif haskey(opts, "--random")
        vals = opts["--random"]
        length(vals) == 2 || error("--random expects 2 values: <nodes> <arcs>")
        n = parse(Int, vals[1])
        m = parse(Int, vals[2])
        println("Generating random instance: nodes=$n arcs=$m")
        nfp = generateRandomNetworkFlowProblem(n, m)
    else
        # Convenience fallback: allow positional input as either a path or (n m)
        if length(positionals) == 1
            path = positionals[1]
            println("Reading instance (positional): $path")
            nfp = readDimacsMinCostFlowInstance(path; applyLowerBounds=true)
        elseif length(positionals) == 2
            n = parse(Int, positionals[1])
            m = parse(Int, positionals[2])
            println("Generating random instance (positional): nodes=$n arcs=$m")
            nfp = generateRandomNetworkFlowProblem(n, m)
        else
            println("Missing input.")
            usage()
            return
        end
    end

    @assert nfp !== nothing
    println("NetworkFlowProblem:")
    println("  numberNodes = $(nfp.numberNodes)")
    println("  numberArcs  = $(length(nfp.arcs))")
    println("  sum(supply) = $(sum(nfp.supply))")
    println("  offset      = $(nfp.offset)")
    println()

    println("ADMM:")
    println("  solver     = $solver_name")
    println("  seed       = $seed")
    println("  maxIter    = $maxIter")
    println("  initialRho = $initialRho")
    println("  timeLimit  = $timeLimit")
    println("  logInterval= $logInterval")
    println("="^120)

    println("="^120)
    println("Running Bipartite ADMM with LP formulation...")

    try 
        mbpLP = generateNetworkFlowProblemLP(nfp)
        param = ADMMParam(
            solver = setSolver(solver_name),
            maxIter = maxIter,
            initialRho = initialRho,
            timeLimit = timeLimit,
            logInterval = logInterval,
            presTolL2 = Inf, 
            dresTolL2 = Inf, 
            presTolLInf = 1e-4, 
            dresTolLInf = 1e-4, 
            logLevel = 1,
        )

        runBipartiteADMM(mbpLP, param; saveSolutionInMultiblockProblem=false)
    catch e
        @error "Failed to generate the network flow problem with LP formulation."
    end

    println("="^120)
    println("Running Bipartite ADMM with BFS bipartization...")
    try
        mbp = generateNetworkFlowProblem(nfp)
        param = ADMMParam(
            solver = setSolver(solver_name),
            maxIter = maxIter,
            initialRho = initialRho,
            timeLimit = timeLimit,
            logInterval = logInterval,
            presTolL2 = Inf, 
            dresTolL2 = Inf, 
            presTolLInf = 1e-4, 
            dresTolLInf = 1e-4, 
            logLevel = 1,
        )

        runBipartiteADMM(mbp, param; bipartizationAlgorithm = BFS_BIPARTIZATION, saveSolutionInMultiblockProblem=false)
    catch e
        @error "Failed to solve the network flow problem with BFS bipartization."
    end
    println("="^120)
    println("Running Bipartite ADMM with MILP bipartization...")
    
    try
        mbp = generateNetworkFlowProblem(nfp)
        
        param = ADMMParam(
            solver = setSolver(solver_name),
            maxIter = maxIter,
            initialRho = initialRho,
            timeLimit = timeLimit,
            logInterval = logInterval,
            presTolL2 = Inf, 
            dresTolL2 = Inf, 
            presTolLInf = 1e-4, 
            dresTolLInf = 1e-4, 
            logLevel = 1,
        )

        runBipartiteADMM(mbp, param; bipartizationAlgorithm = MILP_BIPARTIZATION, saveSolutionInMultiblockProblem=false)
    catch e
        @error "Failed to solve the network flow problem with MILP bipartization."
    end
end 


if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
