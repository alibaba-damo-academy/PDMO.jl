"""
    updateDual!(info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph, param::ADMMParam)

Updates the dual variables in the Alternating Direction Method of Multipliers (ADMM) algorithm.

# Arguments
- `info::ADMMIterationInfo`: Structure containing algorithm state information, including primal and dual solutions, and buffers.
- `admmGraph::ADMMBipartiteGraph`: The bipartite graph representation of the ADMM problem structure.
- `param::ADMMParam`: Parameters for the ADMM algorithm, including solver type, accelerator, and step sizes.

# Notes
- This function should be called after `updatePrimalResidualInBuffer!`.
- Different update strategies are used based on the solver and accelerator types:
  - For `AdaptiveLinearizedSolver`, dual updates are performed at the beginning of each iteration.
  - For `OriginalADMMSubproblemSolver` with `AndersonAccelerator`, updates use implicit fixed point variables.
  - Otherwise, regular dual variable updates are performed using the formula: dual += dualStepsize * residual.
"""
# this should be called after updatePrimalResidualInBuffer!
function updateDual!(info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph, param::ADMMParam)
    if isa(param.solver, AdaptiveLinearizedSolver) 
        # dual updates have been performed at the begining of each iteration in update!
        return 
    end 

    @threads for edgeID in info.edgeIDs
        copyto!(info.dualSolPrev[edgeID], info.dualSol[edgeID])
    end 

    rho = info.rhoHistory[end][1] 
    if isa(param.accelerator, AndersonAccelerator) && isa(param.solver, OriginalADMMSubproblemSolver)
        # Anderson acceleration update dual variables via the implicit fixed point variables zeta
        @threads for edgeID in info.edgeIDs
            edge = admmGraph.edges[edgeID]
            rightNodeID = param.accelerator.converter.isLeft[edge.nodeID1] ? edge.nodeID2 : edge.nodeID1
            edge.mappings[rightNodeID](info.primalSol[rightNodeID], info.dualSol[edgeID], false)     # info.dualSol <- Bz 
            axpby!(1.0, param.accelerator.converter.outputBuffer[edgeID], rho, info.dualSol[edgeID]) # info.dualSol <- zeta + rho * Bz
        end 
        return 
    end 
    
    # regular updates dual <- dual + dualStepsize * rho * (Ax+By-b)
    dualStepsize = rho 
    if isa(param.solver, DoublyLinearizedSolver)
        dualStepsize *= param.solver.dualStepsize 
    end 

    @threads for edgeID in info.edgeIDs
        axpy!(dualStepsize, info.dualBuffer[edgeID], info.dualSol[edgeID])
    end 
end


function updateDualDescent!(info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph, param::ADMMParam; 
    tau::Float64=1.0, 
    omega::Float64=4.0)

    @threads for edgeID in info.edgeIDs
        copyto!(info.dualSolPrev[edgeID], info.dualSol[edgeID])
    end 

    rho = info.rhoHistory[end][1] 
    dualStepsize = -1.0 * rho / (tau * omega)
    scaledFactor = tau / (1.0 + tau)
    @threads for edgeID in info.edgeIDs
        axpy!(dualStepsize, info.dualBuffer[edgeID], info.dualSol[edgeID])
        info.dualSol[edgeID] .*= scaledFactor 
    end 
end 