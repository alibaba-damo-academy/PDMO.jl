#!/usr/bin/env julia

# Reviewer driver for Section 3.4 (decentralized consensus optimization).
#
# This is deliberately an add-only driver.  It reuses the paper implementation
# without changing src/, applications/, or advanced/, and it runs exactly the
# methods reported in Table 1 and Figures 15--16.

const SECTION34_REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

# The branch's advanced environment setup expects the process to be launched
# from the parent of a checkout named PDMO.jl.  The Python wrapper enforces that
# legacy constraint for fresh runs.
include(joinpath(SECTION34_REPO_ROOT, "advanced", "warmup.jl"))

using LightGraphs
using Printf
using Random

include(joinpath(SECTION34_REPO_ROOT, "advanced", "src", "include.jl"))
include(joinpath(SECTION34_REPO_ROOT, "advanced", "src", "DistributedOpt", "DistributedOpt.jl"))
include(joinpath(SECTION34_REPO_ROOT, "advanced", "src", "gnn", "io.jl"))
include(joinpath(SECTION34_REPO_ROOT, "advanced", "src", "gnn", "GnnBipartization.jl"))

const PAPER_GAPS = (0.01, 0.05, 0.1, 0.2)
const PAPER_METHOD_TOKENS = (
    "basic",
    "bfs",
    "milp-0.01",
    "milp-0.05",
    "milp-0.1",
    "milp-0.2",
    "gnn",
)

function usage_error()
    error(
        "Usage: julia -t 16 reproduction/julia/section_3_4.jl " *
        "N n m solver rho maxIter logInterval seed " *
        "[mipHeuristicEffort=0.2] [mipTimeLimit=60.0] " *
        "[methods=basic,bfs,milp-0.01,milp-0.05,milp-0.1,milp-0.2,gnn]",
    )
end

function parse_methods(text::String)
    requested = Set{String}()
    aliases = Dict(
        "classic" => "basic",
        "milp0.01" => "milp-0.01",
        "milp0.05" => "milp-0.05",
        "milp0.1" => "milp-0.1",
        "milp0.2" => "milp-0.2",
        "milp-1%" => "milp-0.01",
        "milp-5%" => "milp-0.05",
        "milp-10%" => "milp-0.1",
        "milp-20%" => "milp-0.2",
    )

    for raw_token in split(lowercase(text), ',')
        token = strip(raw_token)
        isempty(token) && continue
        if token == "all" || token == "paper"
            union!(requested, PAPER_METHOD_TOKENS)
        elseif token == "milp" || token == "milps"
            union!(requested, ("milp-$gap" for gap in PAPER_GAPS))
        else
            push!(requested, get(aliases, token, token))
        end
    end

    isempty(requested) && error("At least one Section 3.4 method must be requested.")
    unknown = setdiff(requested, Set(PAPER_METHOD_TOKENS))
    isempty(unknown) || error("Unknown Section 3.4 method token(s): $(join(sort!(collect(unknown)), ", "))")
    return requested
end

function make_solver(name::String)
    if name == "original"
        return OriginalADMMSubproblemSolver()
    elseif name == "doubly"
        return DoublyLinearizedSolver()
    end
    error("Unknown solver '$name'; expected 'original' or 'doubly'.")
end

function run_and_record!(
    label::String,
    results::Vector{Tuple{String, Any}},
    failures::Vector{String},
    thunk::Function,
)
    try
        push!(results, (label, thunk()))
    catch err
        push!(failures, label)
        @error "Section 3.4 method failed" method=label exception=(err, catch_backtrace())
    end
    println("="^60)
    return nothing
end

function main(args)
    length(args) >= 8 || usage_error()

    number_nodes = parse(Int, args[1])
    n = parse(Int, args[2])
    m = parse(Int, args[3])
    solver_name = lowercase(args[4])
    initial_rho = parse(Float64, args[5])
    max_iter = parse(Int, args[6])
    log_interval = parse(Int, args[7])
    seed = parse(Int, args[8])
    mip_heuristic_effort = length(args) >= 9 ? parse(Float64, args[9]) : 0.2
    mip_time_limit = length(args) >= 10 ? parse(Float64, args[10]) : 60.0
    method_text = length(args) >= 11 ? args[11] : join(PAPER_METHOD_TOKENS, ',')
    methods = parse_methods(method_text)

    number_nodes > 1 || error("N must exceed one.")
    n > 0 || error("n must be positive.")
    m > 0 || error("m must be positive.")
    max_iter > 0 || error("maxIter must be positive.")
    log_interval > 0 || error("logInterval must be positive.")
    mip_time_limit > 0 || error("mipTimeLimit must be positive.")
    make_solver(solver_name) # Validate before generating the large instance.

    Random.seed!(seed)
    kappa_min = 2.0 / number_nodes
    kappa_max = 10.0 / number_nodes
    kappa = kappa_min + rand() * (kappa_max - kappa_min)

    println("Running Distributed Opt")
    println("  reviewer_section = 3.4")
    println("  numberNodes = ", number_nodes)
    println("  kappa = ", kappa)
    println("  n = ", n)
    println("  m = ", m)
    println("  solver = ", solver_name)
    println("  initialRho = ", initial_rho)
    println("  maxIter = ", max_iter)
    println("  logInterval = ", log_interval)
    println("  seed = ", seed)
    println("  mipRelGaps = ", collect(PAPER_GAPS))
    println("  mipHeuristicEffort = ", mip_heuristic_effort)
    println("  mipTimeLimit = ", mip_time_limit)
    println("  admmTimeLimit = 7200.0")
    println("  gnn eval. on_cpu = true")
    println("  requestedMethods = ", join((token for token in PAPER_METHOD_TOKENS if token in methods), ','))
    println(
        "WARNING: reproduction fidelity retains the archived quadratic linear term " *
        "+2*A'*b. Under PDMO's x'Qx + q'x + r convention this is ||A*x+b||^2, " *
        "whereas the paper writes ||A*x-b||^2.",
    )
    println("="^60)

    graph, objective_functions = generateDistributedOptInstance(number_nodes, kappa, n, m)

    param = ADMMParam(
        initialRho=initial_rho,
        maxIter=max_iter,
        logInterval=log_interval,
        presTolL2=1e-4,
        presTolLInf=1e-6,
        dresTolL2=1e-4,
        dresTolLInf=1e-6,
        logLevel=1,
        timeLimit=7200.0,
    )

    results = Vector{Tuple{String, Any}}()
    failures = String[]

    if "basic" in methods
        println("Solving classic distributed opt problem...")
        run_and_record!("Classic", results, failures, () -> begin
            param.solver = make_solver(solver_name)
            classic_problem = generateClassicDistributedOptProblem(graph, objective_functions)
            runBipartiteADMM(
                classic_problem,
                param;
                saveSolutionInMultiblockProblem=false,
                tryJuMP=false,
            )
        end)
    end

    distributed_problem = generateDistributedOptProblem(graph, objective_functions)

    if "bfs" in methods
        println("Solving distributed opt problem with BFS bipartization...")
        run_and_record!("BFS", results, failures, () -> begin
            param.solver = make_solver(solver_name)
            runBipartiteADMM(
                distributed_problem,
                param;
                bipartizationAlgorithm=BFS_BIPARTIZATION,
                saveSolutionInMultiblockProblem=false,
                tryJuMP=false,
            )
        end)
    end

    for gap in PAPER_GAPS
        token = "milp-$gap"
        token in methods || continue
        println("Solving distributed opt problem with MILP bipartization gap=$gap...")
        run_and_record!("MILP($gap)", results, failures, () -> begin
            param.solver = make_solver(solver_name)
            runBipartiteADMM(
                distributed_problem,
                param;
                bipartizationAlgorithm=MILP_BIPARTIZATION,
                mipRelGap=gap,
                mipTimeLimit=mip_time_limit,
                mipHeuristicEffort=mip_heuristic_effort,
                saveSolutionInMultiblockProblem=false,
                tryJuMP=false,
            )
        end)
    end

    if "gnn" in methods
        println("Solving distributed opt problem with GNN-Pycall bipartization...")
        run_and_record!("GNN-Pycall", results, failures, () -> begin
            registerGnnBipartizationImpl!(; force_cpu=true, model_path=GNN_MODEL_PATH)
            param.solver = make_solver(solver_name)
            runBipartiteADMM(
                distributed_problem,
                param;
                bipartizationAlgorithm=GNN_BIPARTIZATION,
                saveSolutionInMultiblockProblem=false,
                tryJuMP=false,
            )
        end)
    end

    println("="^60)
    println("SUMMARY OF RESULTS")
    println("="^60)
    println("Method | BipT | Iters | ADMM Time | ADMM Obj |")
    println("-"^80)
    for (label, result) in results
        info = hasproperty(result, :iterationInfo) ? result.iterationInfo : result
        bip_time = hasproperty(info, :partitionAlgorithmTime) ? info.partitionAlgorithmTime : 0.0
        iterations = hasproperty(info, :stopIter) ? info.stopIter : 0
        admm_time = hasproperty(info, :totalTime) ? info.totalTime : 0.0
        objective = hasproperty(info, :obj) && !isempty(info.obj) ? info.obj[end] : NaN
        println(
            @sprintf(
                "%15s | %4.3f | %5d | %9.2f | %8.4f |",
                label,
                bip_time,
                iterations,
                admm_time,
                objective,
            ),
        )
    end

    if !isempty(failures)
        error("Section 3.4 failed method(s): $(join(failures, ", "))")
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end

