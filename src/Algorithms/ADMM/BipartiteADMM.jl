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

Run the core bipartite ADMM iteration loop.

# Arguments
- `admmGraph::ADMMBipartiteGraph`: Bipartite graph representation of the optimization problem
- `param::ADMMParam`: ADMM algorithm parameters including tolerances, solver, adapter, and accelerator

# Returns
- `ADMMIterationInfo`: Complete iteration information including solution, residuals, and termination status
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

Update all left-partition primal nodes for one ADMM iteration.

# Arguments
- `info::ADMMIterationInfo`: Current iteration information including primal/dual solutions
- `param::ADMMParam`: ADMM parameters including solver, accelerator, and tolerances
- `admmGraph::ADMMBipartiteGraph`: Bipartite graph structure defining the optimization problem
- `nThreads::Int64`: Number of available threads for parallel execution
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

Update all right-partition primal nodes for one ADMM iteration.

# Arguments
- `info::ADMMIterationInfo`: Current iteration information including primal/dual solutions
- `param::ADMMParam`: ADMM parameters including solver, accelerator, and tolerances
- `admmGraph::ADMMBipartiteGraph`: Bipartite graph structure defining the optimization problem
- `nThreads::Int64`: Number of available threads for parallel execution
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
