"""
    ADMMTerminationStatus

Enumeration of possible termination statuses for the ADMM algorithm.

- `ADMM_TERMINATION_UNSPECIFIED`: Default status before termination
- `ADMM_TERMINATION_OPTIMAL`: Converged to an optimal solution
- `ADMM_TERMINATION_ITERATION_LIMIT`: Reached maximum iterations
- `ADMM_TERMINATION_TIME_LIMIT`: Reached time limit
- `ADMM_TERMINATION_INFEASIBLE`: Problem determined to be infeasible
- `ADMM_TERMINATION_UNBOUNDED`: Problem determined to be unbounded
- `ADMM_TERMINATION_UNKNOWN`: Terminated with unknown status
"""
@enum ADMMTerminationStatus begin 
    ADMM_TERMINATION_UNSPECIFIED
    ADMM_TERMINATION_OPTIMAL
    ADMM_TERMINATION_ITERATION_LIMIT
    ADMM_TERMINATION_TIME_LIMIT
    ADMM_TERMINATION_INFEASIBLE
    ADMM_TERMINATION_UNBOUNDED
    ADMM_TERMINATION_UNKNOWN

    ADMM_TERMINATION_ILLPOSED_CASE_B
    ADMM_TERMINATION_ILLPOSED_CASE_C
    ADMM_TERMINATION_ILLPOSED_CASE_D
end

"""
    ADMMIterationInfo

Data structure to track the progress and results of ADMM iterations.

# Fields
- `presL2::Vector{Float64}`: Primal residual L2 norms
- `dresL2::Vector{Float64}`: Dual residual L2 norms
- `presLInf::Vector{Float64}`: Primal residual infinity norms
- `dresLInf::Vector{Float64}`: Dual residual infinity norms
- `obj::Vector{Float64}`: Objective values
- `alObj::Vector{Float64}`: Augmented Lagrangian objective values
- `rhoHistory::Vector{Tuple{Float64, Int64}}`: History of penalty parameter updates
- `primalSol::Dict{String, NumericVariable}`: Current primal solution
- `dualSol::Dict{String, NumericVariable}`: Current dual solution
- `dualSolPrev::Dict{String, NumericVariable}`: Previous dual solution
- `primalBuffer::Dict{String, NumericVariable}`: Buffer for primal computations
- `dualBuffer::Dict{String, NumericVariable}`: Buffer for dual computations
- `stopIter::Int64`: Iteration at which the algorithm stopped
- `totalTime::Float64`: Total computation time
- `terminationStatus::ADMMTerminationStatus`: Termination status
"""
mutable struct ADMMIterationInfo
    # history info  
    presL2::Vector{Float64} 
    dresL2::Vector{Float64}
    presLInf::Vector{Float64} 
    dresLInf::Vector{Float64}
    obj::Vector{Float64}
    alObj::Vector{Float64}
    rhoHistory::Vector{Tuple{Float64, Int64}} # (rho, iter being updated)

    # buffer for primal computations 
    primalSol::Dict{String, NumericVariable}
    primalSolPrev::Dict{String, NumericVariable}
    primalBuffer::Dict{String, NumericVariable}

    # buffer for dual computations 
    dualSol::Dict{String, NumericVariable}
    dualSolPrev::Dict{String, NumericVariable}
    dualBuffer::Dict{String, NumericVariable}
    edgeIDs::Vector{String}

    # termination info
    partitionAlgorithmTime::Float64
    stopIter::Int64
    totalTime::Float64
    terminationStatus::ADMMTerminationStatus
    """
        ADMMIterationInfo()

    Construct an empty ADMMIterationInfo structure with default values.
    """
    ADMMIterationInfo() = new(
        Vector{Float64}(), 
        Vector{Float64}(), 
        Vector{Float64}(), 
        Vector{Float64}(), 
        Vector{Float64}(), 
        Vector{Float64}(),
        Vector{Tuple{Float64, Int64}}(),
        Dict{String, NumericVariable}(),
        Dict{String, NumericVariable}(),
        Dict{String, NumericVariable}(), 
        Dict{String, NumericVariable}(),
        Dict{String, NumericVariable}(), 
        Dict{String, NumericVariable}(),
        String[],
        Inf, -1, 0.0,
        ADMM_TERMINATION_UNSPECIFIED)
end 

"""
    updatePrimalResidualsInBuffer!(info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph)

Update primal residuals and objective values in the iteration information structure.

This function computes and stores the primal residuals, objective values, and augmented
Lagrangian values for the current ADMM iteration. It evaluates constraint violations
and updates the iteration history with computed metrics.

# Arguments
- `info::ADMMIterationInfo`: The iteration info structure to update
- `admmGraph::ADMMBipartiteGraph`: The ADMM bipartite graph model containing problem structure

# Computed Values
- **Primal objective**: Computes Σᵢ(fᵢ(xᵢ) + gᵢ(xᵢ)) across all nodes
- **Primal residuals**: Computes constraint violations Ax + By - c for each edge
- **Augmented Lagrangian terms**: Computes AL contributions for each constraint

# Updated Fields
- `info.presL2`: Appends L2 norm of primal residuals ||Ax + By - c||₂
- `info.presLInf`: Appends L∞ norm of primal residuals ||Ax + By - c||∞
- `info.obj`: Appends current primal objective value
- `info.alObj`: Appends current augmented Lagrangian objective value
- `info.dualBuffer`: Stores primal residuals for each constraint edge

# Mathematical Formulation
The function computes:
1. **Primal residuals**: For each edge e with constraint Aₑx + Bₑy = cₑ:
   ```
   rₑ = Aₑx + Bₑy - cₑ
   ```

2. **Primal objective**: 
   ```
   obj = Σᵢ fᵢ(xᵢ) + gᵢ(xᵢ)
   ```

3. **Augmented Lagrangian**: For each constraint:
   ```
   AL_term = yₑᵀrₑ + (ρ/2)||rₑ||²
   ```

4. **Total augmented Lagrangian**:
   ```
   AL_obj = obj + Σₑ AL_term
   ```

# Threading
- Uses `@threads` for parallel computation across constraint edges
- Thread-safe operations on separate edge buffers
- Efficient parallel reduction for norm computations

# Algorithm Details
1. **Objective computation**: Evaluates node functions fᵢ and gᵢ for all nodes
2. **Constraint evaluation**: Uses constraint mappings to compute Aₑx + Bₑy
3. **Residual computation**: Subtracts right-hand side values to get violations
4. **Norm computation**: Computes both L2 and L∞ norms of residual vectors
5. **Augmented Lagrangian**: Combines primal objective with penalty terms

# Performance Notes
- Constraint mappings are evaluated in parallel
- Memory-efficient buffer reuse for residual computations
- Single-pass computation of multiple norms
- Minimal memory allocation during iteration

# Usage Context
This function is typically called after primal variable updates but before
dual variable updates in the ADMM iteration loop. It provides essential
metrics for convergence monitoring and termination criteria.

# Notes
- The penalty parameter ρ is obtained from `info.rhoHistory[end][1]`
- Constraint mappings handle the linear algebra operations efficiently
- Results are automatically appended to iteration history vectors
- Thread-safe design enables parallel execution of constraint evaluations
"""
function updatePrimalResidualsInBuffer!(info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph) 
    
    obj = sum((admmGraph.nodes[nodeID].f)(info.primalSol[nodeID]) + (admmGraph.nodes[nodeID].g)(info.primalSol[nodeID]) for nodeID in keys(info.primalSol))

    edges = info.edgeIDs
    numberEdges = length(edges)

    rho = info.rhoHistory[end][1]
    addToBuffer = true 

    # Thread-local reductions (the loop below runs in parallel).
    #
    # IMPORTANT (Julia >= 1.9 threadpools):
    # `Threads.threadid()` can be larger than `Threads.nthreads()` (default pool size),
    # e.g. when an interactive threadpool exists. Allocate by `Threads.maxthreadid()`
    # to avoid BoundsError on machines where `threadid()` reaches `nthreads()+1`.
    nT = Threads.maxthreadid()
    presL2SquareByThread = zeros(Float64, nT)
    presLInfByThread = fill(0.0, nT)
    alTermByThread = zeros(Float64, nT)
    @threads for idx in 1:numberEdges
        tid = Threads.threadid()
        edgeID = edges[idx]
        nodeID1 = admmGraph.edges[edgeID].nodeID1
        nodeID2 = admmGraph.edges[edgeID].nodeID2

        info.dualBuffer[edgeID] .= -admmGraph.edges[edgeID].rhs
        admmGraph.edges[edgeID].mappings[nodeID1](info.primalSol[nodeID1], info.dualBuffer[edgeID], addToBuffer)
        admmGraph.edges[edgeID].mappings[nodeID2](info.primalSol[nodeID2], info.dualBuffer[edgeID], addToBuffer)

        rnorm2 = dot(info.dualBuffer[edgeID], info.dualBuffer[edgeID])
        rnormInf = norm(info.dualBuffer[edgeID], Inf)
        presL2SquareByThread[tid] += rnorm2
        presLInfByThread[tid] = max(presLInfByThread[tid], rnormInf)
        
        alTermByThread[tid] += dot(info.dualSol[edgeID], info.dualBuffer[edgeID])
        # Augmented Lagrangian penalty term is per-constraint: (ρ/2)||r_e||^2
        alTermByThread[tid] += 0.5 * rho * rnorm2
    end 
    presL2Square = sum(presL2SquareByThread)
    presLInf = maximum(presLInfByThread)

    push!(info.presL2, sqrt(presL2Square))
    push!(info.presLInf, presLInf)
    push!(info.obj, obj)
    push!(info.alObj, obj + sum(alTermByThread))
end 

"""
    ADMMIterationInfo(admmGraph::ADMMBipartiteGraph, rho::Float64) -> ADMMIterationInfo

Construct and initialize an ADMMIterationInfo structure for ADMM algorithm execution.

This constructor creates a fully initialized iteration information structure that tracks
the state and progress of the ADMM algorithm. It initializes all primal and dual variables,
sets up buffer spaces, and computes initial metrics.

# Arguments
- `admmGraph::ADMMBipartiteGraph`: The ADMM bipartite graph model defining the optimization problem
- `rho::Float64`: Initial penalty parameter value for the augmented Lagrangian

# Returns
- `ADMMIterationInfo`: Fully initialized iteration info object ready for ADMM execution

# Initialization Process
1. **Variable Initialization**:
   - Creates primal solution variables for each node using `similar(node.val)`
   - Initializes primal variables using proximal operator: `proxₘ(node.val, 1.0)`
   - Creates dual solution variables for each edge as zero vectors
   - Sets up buffer spaces for computations

2. **Penalty Parameter Setup**:
   - Stores initial penalty parameter in `rhoHistory` with iteration 0
   - Penalty parameter can be adapted during algorithm execution

3. **Initial Metrics Computation**:
   - Computes initial primal residuals using `updatePrimalResidualsInBuffer!`
   - Sets initial dual residuals to infinity (no previous dual solution)
   - Initializes objective and augmented Lagrangian values

# Data Structure Setup
- **Primal variables**: Initialized using proximal mapping of node functions
- **Dual variables**: Initialized as zero vectors matching constraint dimensions
- **History vectors**: Empty vectors ready to collect iteration metrics
- **Buffer spaces**: Allocated for efficient computation during iterations

# Memory Allocation
- Allocates memory for all primal and dual variables
- Creates buffer spaces matching variable dimensions
- Initializes all history tracking vectors
- Memory layout optimized for parallel computation

# Mathematical Initialization
The primal variables are initialized using:
```
x₀ = prox_{g}(x̂, 1.0)
```
where `x̂` is the initial guess and `g` is the node function.

The dual variables are initialized as:
```
y₀ = 0
```

# Initial Metrics
- **Primal residuals**: ||Ax₀ + By₀ - c||₂ and ||Ax₀ + By₀ - c||∞
- **Dual residuals**: Set to infinity (no previous iterate)
- **Objective**: f(x₀) + g(x₀)
- **Augmented Lagrangian**: obj + penalty terms

# Performance Considerations
- Efficient memory allocation using `similar()` for matching dimensions
- Parallel-friendly data layout for multi-threaded execution
- Minimal redundant computations during initialization
- Memory-efficient buffer management

# Usage Context
This constructor is typically called at the beginning of the BipartiteADMM algorithm:
```julia
info = ADMMIterationInfo(admmGraph, initialRho)
```

# Notes
- The proximal operator initialization provides a good starting point
- Initial dual residuals are infinite by convention
- All history vectors start empty and are populated during iterations
- Buffer spaces are pre-allocated for efficient computation
- The structure is ready for immediate use in ADMM iterations
"""
function ADMMIterationInfo(admmGraph::ADMMBipartiteGraph, rho::Float64)
    info = ADMMIterationInfo()
    info.edgeIDs = collect(keys(admmGraph.edges))

    # initialize primal dual buffers
    for (nodeID, node) in admmGraph.nodes 
        info.primalSol[nodeID] = similar(node.val)
        info.primalSolPrev[nodeID] = similar(node.val)
        info.primalBuffer[nodeID] = similar(node.val)
        proximalOracle!(info.primalSol[nodeID], node.g, node.val, 1.0)
        copyto!(info.primalSolPrev[nodeID], info.primalSol[nodeID])
    end 

    for (edgeID, edge) in admmGraph.edges 
        info.dualSol[edgeID] = zero(edge.rhs)
        info.dualSolPrev[edgeID] = zero(edge.rhs)
        info.dualBuffer[edgeID] = similar(edge.rhs)
    end 

    # initial penalty parameter 
    push!(info.rhoHistory, (rho, 0))

    # update initial primal residuals 
    updatePrimalResidualsInBuffer!(info, admmGraph)
    
    # initial dual residuals are Inf
    push!(info.dresL2, Inf)
    push!(info.dresLInf, Inf)

    return info 
end 
