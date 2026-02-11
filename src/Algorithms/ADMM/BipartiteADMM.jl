using Base.Threads 
include("ADMMIterationInfo.jl")
include("ADMMAdapters/AbstractADMMAdapter.jl")
include("ADMMAccelerators/AbstractADMMAccelerator.jl")
include("ADMMSubproblemSolvers/AbstractADMMSubproblemSolver.jl")
include("ADMMParameter.jl")
include("ADMMDualUpdates.jl")
include("ADMMTerminationMetrics.jl")
include("ADMMTerminationCriteria.jl")
include("ADMMUtil.jl")

"""
    BipartiteADMM(admmGraph::ADMMBipartiteGraph, param::ADMMParam) -> ADMMIterationInfo

Solve an optimization problem using the Bipartite Alternating Direction Method of Multipliers (ADMM).

This function implements the bipartite ADMM algorithm for solving optimization problems of the form:
```
minimize    Σᵢ fᵢ(xᵢ) + Σⱼ gⱼ(zⱼ)
subject to  Σᵢ Aᵢxᵢ + Σⱼ Bⱼzⱼ = c
```

where the variables are partitioned into two sets (left and right nodes) in a bipartite graph structure.

# Arguments
- `admmGraph::ADMMBipartiteGraph`: Bipartite graph representation of the optimization problem
- `param::ADMMParam`: ADMM algorithm parameters including tolerances, solver, adapter, and accelerator

# Returns
- `ADMMIterationInfo`: Complete iteration information including solution, residuals, and termination status

# Algorithm Overview
The bipartite ADMM algorithm alternates between updating variables in the left and right nodes:

1. **Initialization**: 
   - Initialize iteration info with primal/dual variables
   - Initialize subproblem solver (with fallback to DoublyLinearizedSolver)
   - Initialize accelerator and adapter
   - Initialize termination criteria

2. **Main Iteration Loop**:
   - **Left node updates**: Solve subproblems for all left nodes
   - **Acceleration between primal updates**: Apply acceleration (e.g., Anderson)
   - **Metrics collection**: Collect termination metrics between primal updates
   - **Right node updates**: Solve subproblems for all right nodes
   - **Dual residual computation**: Update dual residuals (solver-dependent)
   - **Primal residual computation**: Update primal residuals
   - **Dual variable updates**: Update dual variables
   - **Metrics collection**: Collect termination metrics after dual updates
   - **Penalty parameter adaptation**: Update ρ if needed
   - **Acceleration after dual updates**: Apply acceleration (e.g., Halpern)
   - **Termination check**: Check all termination criteria

# Threading
- Uses `Threads.nthreads()` to determine parallelization strategy
- Single-threaded execution for single-node problems
- Multi-threaded execution for multi-node problems using `@threads`

# Solver Initialization
- Attempts to initialize the specified subproblem solver
- Falls back to `DoublyLinearizedSolver()` if initialization fails
- Logs warnings for failed solver initialization

# Termination Criteria
The algorithm terminates when any of the following conditions are met:
- **Optimality**: Both primal and dual residuals satisfy tolerances
- **Iteration limit**: Maximum iterations reached
- **Time limit**: Maximum time exceeded
- **Numerical issues**: NaN or Inf values detected
- **Advanced criteria** (with metrics): Infeasibility, unboundedness, or ill-posed problem detection

# Logging
- Logs initialization information and solver selection
- Logs iteration progress at intervals specified by `param.logInterval`
- Logs final iteration results and termination status

# Performance Considerations
- Initialization time is reported separately from iteration time
- Thread-safe operations for parallel node updates
- Memory-efficient buffer management for large problems
- Adaptive penalty parameter updates can improve convergence

# Mathematical Background
The augmented Lagrangian for the bipartite ADMM problem is:
```
L(x,z,y) = Σᵢ fᵢ(xᵢ) + Σⱼ gⱼ(zⱼ) + yᵀ(Ax + Bz - c) + (ρ/2)||Ax + Bz - c||²
```

The algorithm alternates between:
1. x-update: xᵢ^{k+1} = argmin L(xᵢ, z^k, y^k)
2. z-update: zⱼ^{k+1} = argmin L(x^{k+1}, zⱼ, y^k)
3. y-update: y^{k+1} = y^k + ρ(Ax^{k+1} + Bz^{k+1} - c)

# Notes
- The bipartite structure allows for efficient parallel processing of variables
- Different solvers (exact, linearized, adaptive) can be used for subproblems
- Acceleration techniques can significantly improve convergence rates
- Advanced termination criteria can detect problematic problem instances
"""
function BipartiteADMM(admmGraph::ADMMBipartiteGraph, param::ADMMParam)
    startTime = time()
    nThreads = Threads.nthreads()

    @PDMOInfo param.logLevel "#"^40 * " Bipartite ADMM " * "#"^40
    # initialize ADMM iteration info 
    info = ADMMIterationInfo(admmGraph, param.initialRho) 
    
    # initialize subproblem solver 
    if initialize!(param.solver, admmGraph, info, param.logLevel) == false 
        @warn "BipartiteADMM: failed to initialize $(getADMMSubproblemSolverName(param.solver)); set subproblem solver to DOUBLY_LINEARIZED_SOLVER instead."
        param.solver = DoublyLinearizedSolver()
        initialize!(param.solver, admmGraph, info, param.logLevel)
    end 

    # initialize accelerator 
    initialize!(param.accelerator, info, admmGraph)

    # initialize adapter 
    initialize!(param.adapter, info, admmGraph)
    
    # initialize termination criteria 
    terminationCriteria = ADMMTerminationCriteria(param, info)    
    
    @PDMOInfo param.logLevel "ADMM: subproblem solver = $(getADMMSubproblemSolverName(param.solver))"
    @PDMOInfo param.logLevel "ADMM: accelerator = $(getADMMAcceleratorName(param.accelerator))"
    @PDMOInfo param.logLevel "ADMM: adapter = $(getADMMAdapterName(param.adapter))"
    @PDMOInfo param.logLevel Printf.@sprintf("ADMM: initialization took %.2f seconds \n", time() - startTime)
   
    startTime = time()
    
    ADMMLog(0, info, param, true)

    rhoUpdated = false 

    for iter in 1:param.maxIter
        # update solver-specific information due to changes caused by adapter or accelerator
        update!(param.solver, info, admmGraph, rhoUpdated)
        
        # left nodes update
        updateLeftNodes!(info, param, admmGraph, nThreads)
        
        # accelerate between primal updates, i.e., Anderson acceleration
        accelerateBetweenPrimalUpdates!(param.accelerator, info, admmGraph) 

        # collect termination metrics between primal updates
        collectTerminationMetricsBetweenPrimalUpdates!(terminationCriteria, info, admmGraph)

        # right nodes update
        updateRightNodes!(info, param, admmGraph, nThreads)
    
        # update dual residuals in info.primalBuffer; dual residuals are solver-specific and may be affected by accelerator 
        updateDualResidualsInBuffer!(param.solver, info, admmGraph, param.accelerator)
        
        # update primal residuals in info.dualBuffer  
        updatePrimalResidualsInBuffer!(info, admmGraph)
        
        # update dual variables in info.dualSol; dual updates depend on solver or accelerator
        # assume primal residuals are stored in info.dualBuffer 
        if param.dualDescent 
            updateDualDescent!(info, admmGraph, param)
            # if iter % 1000 == 0 && iter > 1000
            #     rho = info.rhoHistory[end][1]
            #     newRho = min(1e10, 1.5 * rho)
            #     push!(info.rhoHistory, (newRho, iter))
            # end 
        else 
            updateDual!(info, admmGraph, param)
        end 

        # collect termination metrics after dual updates
        collectTerminationMetricsAfterDualUpdates!(terminationCriteria, info, admmGraph)

        # update penalty parameter in info.rhoHistory; return true iff rho is updated 
        rhoUpdated = updatePenalty(param.adapter, info, admmGraph, iter)
    
        
        # log iteration
        info.totalTime = time() - startTime 
        iterLogged = ADMMLog(iter, info, param, rhoUpdated) 

        # stop criteria 
        checkTerminationCriteria(info, terminationCriteria)
        if (terminationCriteria.terminated)
            if (iterLogged == false) # log the last iteration if it hasn't been logged yet 
                ADMMLog(iter, info, param, rhoUpdated; final = true)
            end
            break 
        end 

        # accelerate after dual updates, i.e., Halpern acceleration
        accelerateAfterDualUpdates!(param.accelerator, info)
    end 
    
    return info 
end 

"""
    updateLeftNodes!(info::ADMMIterationInfo, param::ADMMParam, admmGraph::ADMMBipartiteGraph, nThreads::Int64)

Update the primal variables for all left nodes in the bipartite ADMM algorithm.

This function performs the left-side primal variable updates in the bipartite ADMM
iteration. It solves the subproblems for all nodes assigned to the left partition
of the bipartite graph, using the specified subproblem solver.

# Arguments
- `info::ADMMIterationInfo`: Current iteration information including primal/dual solutions
- `param::ADMMParam`: ADMM parameters including solver, accelerator, and tolerances
- `admmGraph::ADMMBipartiteGraph`: Bipartite graph structure defining the optimization problem
- `nThreads::Int64`: Number of available threads for parallel execution

# Algorithm Details
The function updates variables by solving subproblems of the form:
```
xᵢ^{k+1} = argmin_{xᵢ} fᵢ(xᵢ) + yₖᵀAᵢxᵢ + (ρ/2)||Aᵢxᵢ + Bⱼzⱼᵏ - cᵢⱼ||²
```
for each left node i connected to right nodes j.

# Threading Strategy
- **Single left node**: Uses single-threaded execution with thread-safe operations enabled
- **Multiple left nodes**: Uses `@threads` for parallel execution across nodes
- Threading decision based on problem structure and available computational resources

# Subproblem Solving
- Uses `param.solver` to solve each node's subproblem
- Passes accelerator information for acceleration-aware solvers
- Handles both exact and approximate subproblem solutions

# Performance Considerations
- Parallel execution when multiple left nodes are present
- Thread-safe operations for single-node problems
- Efficient memory usage through shared data structures
- Solver-specific optimizations for different subproblem types

# Usage Context
This function is called during the first phase of each ADMM iteration:
1. Update left nodes (this function)
2. Apply acceleration between primal updates
3. Update right nodes
4. Update dual variables

# Notes
- The `isLeft = true` parameter indicates this is a left node update
- Thread safety is enabled for single-node problems
- Multiple threads are used efficiently for multi-node problems
- The function integrates with the acceleration framework
"""
function updateLeftNodes!(info::ADMMIterationInfo, param::ADMMParam, admmGraph::ADMMBipartiteGraph, nThreads::Int64)
    if length(admmGraph.left) == 1 
        solve!(param.solver, admmGraph.left[1], param.accelerator, admmGraph, info, true, nThreads > 1)
    else 
        @threads for nodeID in admmGraph.left
            solve!(param.solver, nodeID, param.accelerator, admmGraph, info, true, false)
        end 
    end 
end 

"""
    updateRightNodes!(info::ADMMIterationInfo, param::ADMMParam, admmGraph::ADMMBipartiteGraph, nThreads::Int64)

Update the primal variables for all right nodes in the bipartite ADMM algorithm.

This function performs the right-side primal variable updates in the bipartite ADMM
iteration. It solves the subproblems for all nodes assigned to the right partition
of the bipartite graph, using the specified subproblem solver.

# Arguments
- `info::ADMMIterationInfo`: Current iteration information including primal/dual solutions
- `param::ADMMParam`: ADMM parameters including solver, accelerator, and tolerances
- `admmGraph::ADMMBipartiteGraph`: Bipartite graph structure defining the optimization problem
- `nThreads::Int64`: Number of available threads for parallel execution

# Algorithm Details
The function updates variables by solving subproblems of the form:
```
zⱼ^{k+1} = argmin_{zⱼ} gⱼ(zⱼ) + yₖᵀBⱼzⱼ + (ρ/2)||Aᵢxᵢ^{k+1} + Bⱼzⱼ - cᵢⱼ||²
```
for each right node j connected to left nodes i.

# Threading Strategy
- **Single right node**: Uses single-threaded execution with thread-safe operations enabled
- **Multiple right nodes**: Uses `@threads` for parallel execution across nodes
- Threading decision based on problem structure and available computational resources

# Subproblem Solving
- Uses `param.solver` to solve each node's subproblem
- Passes accelerator information for acceleration-aware solvers
- Handles both exact and approximate subproblem solutions

# Performance Considerations
- Parallel execution when multiple right nodes are present
- Thread-safe operations for single-node problems
- Efficient memory usage through shared data structures
- Solver-specific optimizations for different subproblem types

# Usage Context
This function is called during the second phase of each ADMM iteration:
1. Update left nodes
2. Apply acceleration between primal updates
3. Update right nodes (this function)
4. Update dual variables

# Notes
- The `isLeft = false` parameter indicates this is a right node update
- Thread safety is enabled for single-node problems
- Multiple threads are used efficiently for multi-node problems
- The function integrates with the acceleration framework
- Right node updates use the most recent left node solutions
"""
function updateRightNodes!(info::ADMMIterationInfo, param::ADMMParam, admmGraph::ADMMBipartiteGraph, nThreads::Int64)
    if length(admmGraph.right) == 1 
        solve!(param.solver, admmGraph.right[1], param.accelerator, admmGraph, info, false, nThreads > 1)
    else 
        @threads for nodeID in admmGraph.right
            solve!(param.solver, nodeID, param.accelerator, admmGraph, info, false, false)
        end 
    end     
end 
