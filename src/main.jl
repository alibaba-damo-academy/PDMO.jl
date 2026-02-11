include("Util/PDMOLogger.jl")
include("Util/DEC.jl")

# basic algorithmic components
include("Components/Functions/AbstractFunction.jl")
include("Components/Mappings/AbstractMapping.jl")


# formulations 
include("Formulations/BlockVariable.jl")
include("Formulations/BlockConstraint.jl")
include("Formulations/MultiblockProblem.jl")
include("Formulations/MultiblockProblemJuMP.jl")
include("Formulations/MultiblockProblemScaling.jl")
include("Formulations/MultiblockGraph.jl")
include("Formulations/BipartizationAlgorithms.jl")
include("Formulations/ADMMBipartiteGraph.jl")

# algorithmic components 
include("Algorithms/ADMM/BipartiteADMM.jl")
include("Algorithms/AdaPDM/AdaPDM.jl") 
include("Algorithms/BCD/BCD.jl") 

# # io 
# include("Util/io.jl")



"""
    runBipartiteADMM(mbp::MultiblockProblem, param::ADMMParam; kwargs...)

Solve a multiblock optimization problem using the bipartite ADMM algorithm.

This function implements a bipartite ADMM approach for solving large-scale multiblock optimization problems.
It includes automatic bipartization of the problem graph, scaling options, and comprehensive solution validation.

# Arguments
- `mbp::MultiblockProblem`: The multiblock optimization problem to solve
- `param::ADMMParam`: ADMM algorithm parameters including penalty parameter, tolerances, max iterations, 
  scaling options (`applyScaling`), and algorithmic components (solver, adapter, accelerator)

# Keyword Arguments
- `saveSolutionInMultiblockProblem::Bool = true`: Whether to save the solution back to the problem structure
- `bipartizationAlgorithm::BipartizationAlgorithm = BFS_BIPARTIZATION`: Algorithm for graph bipartization
  - `BFS_BIPARTIZATION`: Breadth-first search bipartization (default)
  - `DFS_BIPARTIZATION`: Depth-first search bipartization
  - `MILP_BIPARTIZATION`: Mixed-integer linear programming bipartization
  - `SPANNING_TREE_BIPARTIZATION`: Spanning tree bipartization
- `trueObj::Float64 = Inf`: Known true objective value for validation (if available)
- `tryJuMP::Bool = true`: Whether to verify solution using JuMP solver

# Returns
- `NamedTuple`: Contains solution and iteration information
  - `solution::Dict{BlockID, NumericVariable}`: Optimal solution for each block
  - `iterationInfo::ADMMIterationInfo`: Detailed iteration information including convergence history

# Examples
```julia
# Basic usage
mbp = MultiblockProblem()
# ... add blocks and constraints ...
param = ADMMParam()
param.initialRho = 1.0
param.maxIter = 1000
param.presTolL2 = 1e-6
param.dresTolL2 = 1e-6
result = runBipartiteADMM(mbp, param)

# With scaling and custom bipartization
param = ADMMParam()
param.applyScaling = true
param.solver = OriginalADMMSubproblemSolver()
result = runBipartiteADMM(mbp, param; 
    bipartizationAlgorithm=MILP_BIPARTIZATION)

# Access solution
solution = result.solution
info = result.iterationInfo
```

# Algorithm Steps
1. **Validation**: Checks multiblock problem validity
2. **Scaling**: Optionally applies problem scaling for numerical stability (controlled by `param.applyScaling`)
3. **Graph Construction**: Creates multiblock graph representation
4. **Bipartization**: Applies bipartization algorithm to create bipartite graph
5. **ADMM Execution**: Runs bipartite ADMM algorithm
6. **Solution Retrieval**: Extracts and unscales primal solution
7. **Verification**: Validates solution feasibility and objective value

See also: `runAdaPDM`, `MultiblockProblem`, `ADMMParam`
"""
function runBipartiteADMM(mbp::MultiblockProblem, 
    param::ADMMParam;
    saveSolutionInMultiblockProblem::Bool = true,  
    bipartizationAlgorithm::BipartizationAlgorithm = BFS_BIPARTIZATION,
    mipRelGap::Float64 = 0.01, 
    mipTimeLimit::Float64 = 60.0,
    mipHeuristicEffort::Float64 = 0.2,
    trueObj::Float64 = Inf,
    tryJuMP::Bool = true)

    @PDMOInfo param.logLevel "Run Bipartite ADMM with threads = $(Threads.nthreads())."
    
    # 1. sanity check for the problem 
    if mbp.couplingFunction != nothing 
        @PDMOError param.logLevel "MultiblockProblem: The instance has multiblock function;not ready for bipartite ADMM."
        return 
    end 
   
    if checkMultiblockProblemValidity(mbp) == false 
        @PDMOError param.logLevel "MultiblockProblem: The instance is not valid. "
        return 
    end 

    summary(mbp, param.logLevel)

    # 2. scale the problem 
    scalingInfo = param.applyScaling ? scaleMultiblockProblem!(mbp, scalingOptions=moderateScaling()) : nothing

    # 3. create a multblock graph instance from the problem
    graph = MultiblockGraph(mbp)
    summary(graph, param.logLevel)

    # 4. generate ADMM bipartite graph 
    admmGraph = ADMMBipartiteGraph(graph, mbp, bipartizationAlgorithm, param.logLevel; 
        mipRelGap=mipRelGap, 
        mipTimeLimit=mipTimeLimit,
        mipHeuristicEffort=mipHeuristicEffort)
    summary(admmGraph, param.logLevel)

    # 5. execute ADMM
    info = BipartiteADMM(admmGraph, param)
    info.partitionAlgorithmTime = admmGraph.partitionAlgorithmTime
    
    # 6. retrieve primal solution
    primalSolution = Dict{BlockID, NumericVariable}() 
    for block in mbp.blocks 
        admmNodeID = admmGraph.mbpBlockID2admmNodeID[block.id]
        primalSolution[block.id] = similar(info.primalSol[admmNodeID])
        copyto!(primalSolution[block.id], info.primalSol[admmNodeID])
    end 

    # unscale the problem and solution
    if scalingInfo != nothing && scalingInfo.isScaled
        unscaleMultiblockProblem!(mbp, primalSolution, scalingInfo)
    end 

    if saveSolutionInMultiblockProblem 
        for block in mbp.blocks 
            copyto!(block.val, primalSolution[block.id])
        end 
    end 

    # 6. summarize the ADMM result    
    if trueObj < Inf
        ADMMLog(info, param.logLevel, trueObj)
    else 
        objByJuMP = Inf
        if tryJuMP
            try 
                objByJuMP = solveMultiblockProblemByJuMP(mbp, param.logLevel)
            catch e
                @PDMOWarn param.logLevel "MultiblockProblem: Failed to verify objective value using JuMP ($e)"
            end 
        end 
        ADMMLog(info, param.logLevel, objByJuMP)
    end

    presL2, presLinf = checkMultiblockProblemFeasibility(mbp, primalSolution)
    @PDMOInfo param.logLevel Printf.@sprintf("Infeasibility measures of original blocks: Pres (L2) = %.4e, Pres (LInf) = %.4e", presL2, presLinf)
    
    return (solution=primalSolution, iterationInfo=info)
end 


"""
    runAdaPDM(mbp::MultiblockProblem, param::AbstractAdaPDMParam; kwargs...)

Solve a composite multiblock optimization problem using the Adaptive Primal-Dual Method (AdaPDM).

This function implements various adaptive primal-dual methods for solving composite optimization problems
with the structure: minimize f(x) + g(Ax), where some blocks may be proximal-only (g functions).

# Arguments
- `mbp::MultiblockProblem`: The composite multiblock optimization problem to solve
- `param::AbstractAdaPDMParam`: AdaPDM algorithm parameters. Can be one of:
  - `AdaPDMParam`: Standard adaptive primal-dual method parameters
  - `AdaPDMPlusParam`: Enhanced AdaPDM with additional features
  - `MalitskyPockParam`: Malitsky-Pock algorithm parameters
  - `CondatVuParam`: Condat-Vũ algorithm parameters

# Keyword Arguments
- `saveSolutionInMultiblockProblem::Bool = true`: Whether to save the solution back to the problem structure
- `trueObj::Float64 = Inf`: Known true objective value for validation (if available)
- `tryJuMP::Bool = true`: Whether to verify solution using JuMP solver

# Returns
- `NamedTuple`: Contains solution and iteration information
  - `solution::Dict{BlockID, NumericVariable}`: Optimal solution for each block
  - `iterationInfo::AdaPDMIterationInfo`: Detailed iteration information including convergence history

# Examples
```julia
# Basic AdaPDM usage
mbp = MultiblockProblem()
# ... add blocks with composite structure ...
param = AdaPDMParam(mbp; maxIter=1000, primalTol=1e-6, dualTol=1e-6)
result = runAdaPDM(mbp, param)

# Using Condat-Vũ algorithm
param = CondatVuParam(mbp; maxIter=1000)
result = runAdaPDM(mbp, param)

# Access solution
solution = result.solution
info = result.iterationInfo
```

# Problem Structure
The method is designed for composite problems where:
- Some blocks have smooth functions (f) with gradient oracles
- Some blocks have proximal functions (g) with proximal oracles
- The last block typically represents the coupling constraint variable

# Algorithm Variants
- **AdaPDM**: Standard adaptive primal-dual method
- **AdaPDM+**: Enhanced version with additional acceleration
- **Malitsky-Pock**: Malitsky-Pock primal-dual algorithm
- **Condat-Vũ**: Condat-Vũ primal-dual algorithm

See also: `runBipartiteADMM`, `MultiblockProblem`, `AdaPDMParam`
"""
function runAdaPDM(mbp::MultiblockProblem, param::AbstractAdaPDMParam; 
    saveSolutionInMultiblockProblem::Bool = true, 
    trueObj::Float64 = Inf,
    tryJuMP::Bool = true)
    @info "Run AdaPDM with threads = $(Threads.nthreads())."

    # 1. sanity check for the problem 
    if mbp.couplingFunction != nothing 
        @PDMOError param.logLevel "MultiblockProblem: The instance has multiblock function;not ready for adaptive primal-dual method."
        return
    end 

    if checkMultiblockProblemValidity(mbp) == false 
        @PDMOError param.logLevel "runAdaPDM: The instance is not ready for adpative primal-dual method."
        return
    end 

    if checkCompositeProblemValidity!(mbp, param.logLevel) == false 
        @PDMOError param.logLevel "runAdaPDM: The instance is not ready for adpative primal-dual method."
        return
    end 
    
    summary(mbp, param.logLevel)
    
    # 2. run the algorithm 
    info = AdaptivePrimalDualMethod(mbp, param)

    # 3. summarize the result 
    if trueObj < Inf
        AdaPDMLog(info, param.logLevel, trueObj)
    else 
        objByJuMP = Inf
        if tryJuMP
            try 
                objByJuMP = solveMultiblockProblemByJuMP(mbp)
            catch 
                @info "MultiblockProblem: Failed to solve the whole problem using JuMP. "
            end 
        end 
        AdaPDMLog(info, param.logLevel, objByJuMP)
    end
    
    # 4. save the solution 
    primalSolution = Dict{BlockID, NumericVariable}()
    for block in mbp.blocks[1:end-1]
        primalSolution[block.id] = info.primalSol[block.id]
        if saveSolutionInMultiblockProblem
            copyto!(block.val, info.primalSol[block.id])
        end
    end
    primalSolution[mbp.blocks[end].id] = info.bufferAx
    if saveSolutionInMultiblockProblem
        copyto!(mbp.blocks[end].val, info.bufferAx) 
    end
 
    return (solution=primalSolution, iterationInfo=info)
end 



"""
    runBCD(mbp::MultiblockProblem, param::BCDParam; kwargs...)

Solve a multiblock optimization problem using the Block Coordinate Descent (BCD) algorithm.

This function implements the BCD algorithm for solving multiblock optimization problems of the form:

```
min F(x₁,...,xₙ) + ∑ⱼ₌₁ⁿ (fⱼ(xⱼ) + gⱼ(xⱼ))
```

where F is a smooth coupling function, fⱼ are smooth block functions, and gⱼ are proximable functions
(including domain constraints via indicator functions).

# Arguments
- `mbp::MultiblockProblem`: The multiblock optimization problem to solve. Must have:
  - A coupling function (cannot be separable)
  - No constraints (constraint-free formulation)
  - Valid problem structure
- `param::BCDParam`: BCD algorithm parameters including:
  - `blockOrderRule`: Strategy for block update ordering (e.g., CyclicRule)
  - `solver`: Subproblem solver (Original, Proximal, or Linearized BCD)
  - `dresTolL2`, `dresTolLInf`: Dual residual tolerances for convergence
  - `maxIter`: Maximum number of iterations
  - `timeLimit`: Maximum computation time (seconds)
  - `logLevel`: Logging verbosity level

# Keyword Arguments
- `saveSolutionInMultiblockProblem::Bool = true`: Whether to save the solution back to the problem blocks
- `trueObj::Float64 = Inf`: Known true objective value for validation (if available)
- `tryJuMP::Bool = true`: Whether to verify solution using JuMP solver for comparison

# Returns
A named tuple containing:
- `solution::Dict{BlockID, NumericVariable}`: Final solution for each block indexed by block ID
- `iterationInfo::BCDIterationInfo`: Detailed algorithm information including:
  - `obj::Vector{Float64}`: Objective value history
  - `dresL2`, `dresLInf::Vector{Float64}`: Dual residual histories
  - `solution::Vector{NumericVariable}`: Final solution blocks
  - `stopIter::Int64`: Final iteration count
  - `totalTime::Float64`: Total computation time
  - `terminationStatus::BCDTerminationStatus`: Convergence status

# Termination Criteria
The algorithm terminates when any of the following conditions are met:
- Dual residual L2 norm ≤ `param.dresTolL2`
- Dual residual L∞ norm ≤ `param.dresTolLInf`
- Maximum iterations (`param.maxIter`) reached
- Time limit (`param.timeLimit`) exceeded

# Examples
```julia
using PDMO

# Set up BCD parameters
param = BCDParam(
    blockOrderRule = CyclicRule(),
    solver = BCDProximalSubproblemSolver(),
    dresTolL2 = 1e-6,
    dresTolLInf = 1e-6,
    maxIter = 1000,
    timeLimit = 3600.0,
    logLevel = 1
)

# Solve the problem
result = runBCD(mbp, param)

# Access results
solution = result.solution
objective_history = result.iterationInfo.obj
final_iteration = result.iterationInfo.stopIter
```

# Error Handling
The function returns `nothing` and logs errors for:
- Problems with constraints
- Problems without coupling functions
- Invalid problem structures
"""
function runBCD(mbp::MultiblockProblem, param::BCDParam; 
    saveSolutionInMultiblockProblem::Bool = true, 
    trueObj::Float64 = Inf,
    tryJuMP::Bool = true)
    @PDMOInfo param.logLevel "Run BCD with threads = $(Threads.nthreads())."

    # 1. sanity check for the problem 
    if isempty(mbp.constraints) == false 
        @PDMOError param.logLevel "runBCD: The instance has constraints; not ready for BCD."
        return
    end 

    if mbp.couplingFunction == nothing 
        # Todo: if the problem is separable, we should handle it in PDMO. 
        @PDMOError param.logLevel "runBCD: The instance has no coupling function; try decompose the problem first."
        return
    end 

    if checkMultiblockProblemValidity(mbp) == false 
        @PDMOError param.logLevel "runBCD: The instance is not valid."
        return
    end 

    summary(mbp, param.logLevel)

    # 2. run the algorithm 
    info = BCD(mbp, param)
    
    # 3. summarize the result 
    if trueObj < Inf
        BCDLog(info, param.logLevel, trueObj)
    else 
        objByJuMP = Inf
        if tryJuMP
            try 
                objByJuMP = solveMultiblockProblemByJuMP(mbp)
            catch err
                @PDMOError param.logLevel "MultiblockProblem: Failed to solve the whole problem using JuMP. error = $err"
            end 
        end 
        BCDLog(info, param.logLevel, objByJuMP)
    end

    # 3. save the solution 
    primalSolution = Dict{BlockID, NumericVariable}()
    for i in 1:length(mbp.blocks)
        primalSolution[mbp.blocks[i].id] = info.solution[i]
        if saveSolutionInMultiblockProblem
            copyto!(mbp.blocks[i].val, info.solution[i])
        end
    end 

    return (solution=primalSolution, iterationInfo=info)
end 
    