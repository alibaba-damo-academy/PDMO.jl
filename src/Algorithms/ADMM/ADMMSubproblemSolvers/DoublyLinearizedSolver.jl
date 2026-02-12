"""
    DoublyLinearizedSolver <: AbstractADMMSubproblemSolver

Doubly Linearized Subproblem Solver for ADMM optimization.

This solver uses a doubly linearized approach where both the objective function gradients
and the augmented Lagrangian terms are linearized. It maintains separate proximal stepsizes
for left and right nodes (α and β) and uses operator norm estimates for stability.

# Fields
- `dualStepsize::Float64`: Step size for dual variable updates
- `proximalStepsizeAlpha::Float64`: Proximal step size for left nodes  
- `proximalStepsizeBeta::Float64`: Proximal step size for right nodes
- `primalBuffer::Dict{String, NumericVariable}`: Buffer for primal computations
- `dualBuffer::Dict{String, NumericVariable}`: Buffer for dual computations
- `maxLeftLipschitzConstant::Float64`: Maximum Lipschitz constant for left node objectives
- `maxRightLipschitzConstant::Float64`: Maximum Lipschitz constant for right node objectives
- `maxLeftMatrixAdjointSelfOperatorNorm::Float64`: Maximum operator norm for left mappings
- `maxRightMatrixAdjointSelfOperatorNorm::Float64`: Maximum operator norm for right mappings
"""
mutable struct DoublyLinearizedSolver <: AbstractADMMSubproblemSolver
    dualStepsize::Float64
    proximalStepsizeAlpha::Float64
    proximalStepsizeBeta::Float64
    # edgeData::Dict{String, EdgeData}
    primalBuffer::Dict{String, NumericVariable} # buffer for primal computations 
    dualBuffer::Dict{String, NumericVariable}   # buffer for dual computations 
    
    maxLeftLipschitzConstant::Float64
    maxRightLipschitzConstant::Float64
    maxLeftMatrixAdjointSelfOperatorNorm::Float64
    maxRightMatrixAdjointSelfOperatorNorm::Float64
    
    logLevel::Int64
    DoublyLinearizedSolver(; dualStepsize::Float64=1.0) = new(
        dualStepsize,
        1e-3, 1e-3, 
        # Dict{String, EdgeData}(), 
        Dict{String, NumericVariable}(),
        Dict{String, NumericVariable}(),
        0.0, 0.0, 0.0, 0.0, 1)
end

"""
    computeProximalStepsize!(solver::DoublyLinearizedSolver, rho::Float64)

Compute and update the proximal step sizes α and β for the doubly linearized solver.

This function estimates the proximal step sizes based on the penalty parameter ρ,
Lipschitz constants of the objective functions, and operator norms of the constraint
mappings. The step sizes are computed to ensure convergence stability.

# Arguments
- `solver::DoublyLinearizedSolver`: The solver instance to update
- `rho::Float64`: Current penalty parameter value

# Effects
- Updates `solver.proximalStepsizeAlpha` for left nodes
- Updates `solver.proximalStepsizeBeta` for right nodes
- Clamps step sizes to [1e-8, 1e-2] for numerical stability

# Algorithm
- α = 1/(||A'A||ρ + L_left + ε) where L_left is max left Lipschitz constant
- β = 1/(||B'B||ρ + 3L_right + ε) where L_right is max right Lipschitz constant
"""
function computeProximalStepsize!(solver::DoublyLinearizedSolver, rho::Float64)
    # estimate alpha and beta 
    solver.proximalStepsizeAlpha = 1.0 / (solver.maxLeftMatrixAdjointSelfOperatorNorm * rho + solver.maxLeftLipschitzConstant + 100 * FeasTolerance)
    solver.proximalStepsizeBeta = 1.0 / (solver.maxRightMatrixAdjointSelfOperatorNorm * rho + 3 * solver.maxRightLipschitzConstant + 100 * FeasTolerance)
    solver.proximalStepsizeAlpha = clamp(solver.proximalStepsizeAlpha, 1.0e-8, 1.0e-2)
    solver.proximalStepsizeBeta = clamp(solver.proximalStepsizeBeta, 1.0e-8, 1.0e-2)
    msg = Printf.@sprintf("DOUBLY_LINEARIZED_SOLVER: given rho = %.2e, estimated alpha = %.2e, beta = %.2e \n", rho, solver.proximalStepsizeAlpha, solver.proximalStepsizeBeta)
    @PDMOWarn solver.logLevel msg 
end 

"""
    initialize!(solver::DoublyLinearizedSolver, admmGraph::ADMMBipartiteGraph, info::ADMMIterationInfo) -> Bool

Initialize the doubly linearized solver with problem-specific parameters.

This function performs comprehensive initialization including buffer allocation,
Lipschitz constant estimation, operator norm computation, and proximal step size
calculation. It prepares the solver for efficient iterative solving.

# Arguments
- `solver::DoublyLinearizedSolver`: The solver instance to initialize
- `admmGraph::ADMMBipartiteGraph`: The bipartite graph structure of the problem
- `info::ADMMIterationInfo`: Current iteration information containing penalty parameter

# Returns
- `Bool`: Always returns `true` upon successful initialization

# Effects
- Allocates `primalBuffer` and `dualBuffer` for all nodes and edges
- Estimates Lipschitz constants for all objective functions
- Computes operator norms for all constraint mappings
- Calculates initial proximal step sizes α and β
- Creates node assignment mapping for left/right classification

# Performance Notes
- Buffer allocation and operator norm estimation are the most expensive operations
- Lipschitz constant estimation uses sampling-based approximation
"""
function initialize!(solver::DoublyLinearizedSolver, admmGraph::ADMMBipartiteGraph, info::ADMMIterationInfo, logLevel::Int64)
    # Pre-allocate buffers
    # timeStart = time() 
    rho = info.rhoHistory[end][1]
    solver.logLevel = logLevel
    
    for (nodeID, node) in admmGraph.nodes 
        solver.primalBuffer[nodeID] = similar(node.val)
    end  
    # @info "primal buffer allocation took $(time() - timeStart) seconds"
    
    # timeStart = time() 
    for (edgeID, edge) in admmGraph.edges 
        # solver.edgeData[edgeID] = EdgeData(edge)
        solver.dualBuffer[edgeID] = similar(edge.rhs) 
    end 
    # @info "edge data allocation took $(time() - timeStart) seconds"
    
    # Create assignment map
    assignment = Dict{String, Int64}()
    sizehint!(assignment, length(admmGraph.left) + length(admmGraph.right))
    for nodeID in admmGraph.left
        assignment[nodeID] = 0
    end
    for nodeID in admmGraph.right
        assignment[nodeID] = 1  
    end
    
    # timeStart = time() 
    # estimate the Lipschitz constant of the gradient of the objective function
    for nodeID in admmGraph.left 
        blockLip = estimateLipschitzConstant(admmGraph.nodes[nodeID].f, admmGraph.nodes[nodeID].val)
        solver.maxLeftLipschitzConstant = max(solver.maxLeftLipschitzConstant, blockLip)
    end 

    for nodeID in admmGraph.right 
        blockLip = estimateLipschitzConstant(admmGraph.nodes[nodeID].f, admmGraph.nodes[nodeID].val)
        solver.maxRightLipschitzConstant = max(solver.maxRightLipschitzConstant, blockLip)
    end 
    # @info "Lipschitz constant estimation took $(time() - timeStart) seconds"
    
    # Estimate operator norms
    # timeStart = time() 
    for (edgeID, edge) in admmGraph.edges 
        if assignment[edge.nodeID1] == 0 
            solver.maxLeftMatrixAdjointSelfOperatorNorm = max(solver.maxLeftMatrixAdjointSelfOperatorNorm, operatorNorm2(edge.mappings[edge.nodeID1])^2)
            solver.maxRightMatrixAdjointSelfOperatorNorm = max(solver.maxRightMatrixAdjointSelfOperatorNorm, operatorNorm2(edge.mappings[edge.nodeID2])^2)
        else 
            solver.maxLeftMatrixAdjointSelfOperatorNorm = max(solver.maxLeftMatrixAdjointSelfOperatorNorm, operatorNorm2(edge.mappings[edge.nodeID2])^2)
            solver.maxRightMatrixAdjointSelfOperatorNorm = max(solver.maxRightMatrixAdjointSelfOperatorNorm, operatorNorm2(edge.mappings[edge.nodeID1])^2)
        end 
    end 
    # @info "operator norm estimation took $(time() - timeStart) seconds"

    # timeStart = time() 
    computeProximalStepsize!(solver, rho)
    # @info "proximal stepsize computation took $(time() - timeStart) seconds"
    return true 
end 

"""
    solve!(solver::DoublyLinearizedSolver, nodeID::String, accelerator::AbstractADMMAccelerator, 
           admmGraph::ADMMBipartiteGraph, info::ADMMIterationInfo, isLeft::Bool, enableParallel::Bool=false)

Solve the ADMM subproblem for a specific node using the doubly linearized method.

This function performs one iteration of the doubly linearized algorithm for the specified node.
It computes linearized gradients of both the objective function and augmented Lagrangian terms,
then applies a proximal gradient step to update the primal variable.

# Arguments
- `solver::DoublyLinearizedSolver`: The solver instance
- `nodeID::String`: Identifier of the node to solve for
- `accelerator::AbstractADMMAccelerator`: Acceleration scheme (unused in current implementation)
- `admmGraph::ADMMBipartiteGraph`: The bipartite graph structure
- `info::ADMMIterationInfo`: Current iteration information
- `isLeft::Bool`: Whether the node is on the left side of the bipartite graph
- `enableParallel::Bool`: Whether to enable parallel computation (default: false)

# Effects
- Updates `info.primalSol[nodeID]` with the new primal solution
- Stores previous solution in `info.primalSolPrev[nodeID]`
- Uses `solver.primalBuffer` and `solver.dualBuffer` for intermediate computations

# Algorithm
1. Compute ∇f(x^k) for the current iterate
2. For each neighboring edge: compute λ + ρ(Ax + By - b)
3. Apply adjoint operators: A'(λ + ρ(Ax + By - b))
4. Perform proximal gradient step: prox_{αg}(x - α(∇f + A'(λ + ρ(Ax + By - b))))
"""
function solve!(solver::DoublyLinearizedSolver, 
    nodeID::String,
    accelerator::AbstractADMMAccelerator,
    admmGraph::ADMMBipartiteGraph,
    info::ADMMIterationInfo, 
    isLeft::Bool,
    enableParallel::Bool = false)
    
    rho = info.rhoHistory[end][1]
    
    # solver.primalBuffer <- ∇f(x^k)
    gradientOracle!(solver.primalBuffer[nodeID], admmGraph.nodes[nodeID].f, info.primalSol[nodeID])

    function updateLinearizedALGradInBuffer(edgeID::String)
        edge = admmGraph.edges[edgeID]
        otherID = edge.nodeID1 == nodeID ? edge.nodeID2 : edge.nodeID1

        # solver.dualBuffer <- Ax + By -b 
        copyto!(solver.dualBuffer[edgeID], edge.rhs)
        rmul!(solver.dualBuffer[edgeID], -1.0)
        edge.mappings[nodeID](info.primalSol[nodeID], solver.dualBuffer[edgeID], true)
        edge.mappings[otherID](info.primalSol[otherID], solver.dualBuffer[edgeID], true)

        # solver.dualBuffer <- lmd + rho(Ax+By-b)
        axpby!(1.0, info.dualSol[edgeID], rho, solver.dualBuffer[edgeID])
    end 

    if enableParallel
        @threads for edgeID in collect(admmGraph.nodes[nodeID].neighbors)
            updateLinearizedALGradInBuffer(edgeID)
        end 
    else 
        for edgeID in admmGraph.nodes[nodeID].neighbors
            updateLinearizedALGradInBuffer(edgeID)
        end 
    end 

    # Update primal buffer with adjoint terms
    for edgeID in admmGraph.nodes[nodeID].neighbors
        adjoint!(admmGraph.edges[edgeID].mappings[nodeID], solver.dualBuffer[edgeID], solver.primalBuffer[nodeID], true)
    end 

    # primal buffer <- x - proxStepsize * (nabla f_i(x_i) + A'(lmd + rho(Ax+By-b)))
    proxStepsize = isLeft ? solver.proximalStepsizeAlpha : solver.proximalStepsizeBeta 
    axpby!(1.0, info.primalSol[nodeID], -proxStepsize, solver.primalBuffer[nodeID])
    
    # save previous primal solution 
    copyto!(info.primalSolPrev[nodeID], info.primalSol[nodeID])
    
    # Proximal step
    proximalOracle!(info.primalSol[nodeID], admmGraph.nodes[nodeID].g, solver.primalBuffer[nodeID], proxStepsize)
end 

"""
    updateDualResidualsInBuffer!(solver::DoublyLinearizedSolver, info::ADMMIterationInfo, 
                                admmGraph::ADMMBipartiteGraph, accelerator::AbstractADMMAccelerator)

Compute and store dual residuals for convergence monitoring.

This function computes the dual residuals by evaluating the linearization error
of the doubly linearized method. The residuals measure how well the current iterate
satisfies the optimality conditions and are used for convergence assessment.

# Arguments
- `solver::DoublyLinearizedSolver`: The solver instance
- `info::ADMMIterationInfo`: Current iteration information
- `admmGraph::ADMMBipartiteGraph`: The bipartite graph structure
- `accelerator::AbstractADMMAccelerator`: Acceleration scheme (unused)

# Effects
- Updates `info.dresL2` with L2 norm of dual residuals
- Updates `info.dresLInf` with L∞ norm of dual residuals
- Uses `info.primalBuffer` to store computed residuals per node
- Uses solver buffers for intermediate computations

# Algorithm
For each node, compute dual residual as:
- Left nodes: ∇f(x^k) - ∇f(x^{k+1}) - (1/α)(x^{k+1} - x^k) + ρA'(Ax^{k+1} + By^{k+1} - Ax^k - By^k)
- Right nodes: ∇f(y^k) - ∇f(y^{k+1}) - (1/β)(y^{k+1} - y^k) + ρB'(Ax^{k+1} + By^{k+1} - Ay^k - By^k)

If dual step size ≠ 1, add correction term: (γ-1)ρA'(Ax^{k+1} + By^{k+1} - b)

# Performance Notes
- Uses parallel computation with @threads for independent node computations
- Requires gradient evaluations at both current and previous iterates
"""
function updateDualResidualsInBuffer!(solver::DoublyLinearizedSolver, 
    info::ADMMIterationInfo, 
    admmGraph::ADMMBipartiteGraph, 
    accelerator::AbstractADMMAccelerator)
    
    rho = info.rhoHistory[end][1]
    """ 
        info.primalBufferr are filled with x^k
    """
    # Compute dual residuals for left nodes
    @threads for nodeID in admmGraph.left 
        node = admmGraph.nodes[nodeID]
        
        # Compute ∇f(x^{k+1}) and ∇f(x^k)
        gradientOracle!(info.primalBuffer[nodeID], node.f, info.primalSolPrev[nodeID])
        gradientOracle!(solver.primalBuffer[nodeID], node.f, info.primalSol[nodeID])

        # info.primalBuffer <- ∇f(x^k) - ∇f(x^{k+1})
        axpy!(-1.0, solver.primalBuffer[nodeID], info.primalBuffer[nodeID]) 
        
        # Add proximal term: -(1/α)(x^{k+1} - x^k)
        axpy!(-1.0/solver.proximalStepsizeAlpha, info.primalSol[nodeID], info.primalBuffer[nodeID])
        axpy!(1.0/solver.proximalStepsizeAlpha, info.primalSolPrev[nodeID], info.primalBuffer[nodeID])

        for edgeID in node.neighbors
            edge = admmGraph.edges[edgeID]
            otherID = edge.nodeID1 == nodeID ? edge.nodeID2 : edge.nodeID1
            
            # Compute primal residual difference in solver.dualBuffer: (Ax^{k+1} + By^{k+1} - Ax^k - By^k)
            edge.mappings[nodeID](info.primalSolPrev[nodeID], solver.dualBuffer[edgeID], false)
            edge.mappings[otherID](info.primalSolPrev[otherID], solver.dualBuffer[edgeID], true)
            rmul!(solver.dualBuffer[edgeID], -1.0)
            edge.mappings[nodeID](info.primalSol[nodeID], solver.dualBuffer[edgeID], true)
            edge.mappings[otherID](info.primalSol[otherID], solver.dualBuffer[edgeID], true)
            
            # Scale by rho A' and add to solver.primalBuffer: rho A' (Ax^{k+1} + By^{k+1} - Ax^k - By^k)
            rmul!(solver.dualBuffer[edgeID], rho)
            adjoint!(edge.mappings[nodeID], solver.dualBuffer[edgeID], info.primalBuffer[nodeID], true)
        end 
    end 

    # Compute dual residuals for right nodes
    @threads for nodeID in admmGraph.right  
        node = admmGraph.nodes[nodeID]
        
        # Compute ∇f(y^{k+1}) and ∇f(y^k)
        gradientOracle!(info.primalBuffer[nodeID], node.f, info.primalSolPrev[nodeID])
        gradientOracle!(solver.primalBuffer[nodeID], node.f, info.primalSol[nodeID])
        
        # info.primalBuffer <- ∇f(y^k) - ∇f(y^{k+1})
        axpy!(-1.0, solver.primalBuffer[nodeID], info.primalBuffer[nodeID])
        
        # Add proximal term: -(1/β)(y^{k+1} - y^k)
        axpy!(-1.0/solver.proximalStepsizeBeta, info.primalSol[nodeID], info.primalBuffer[nodeID])
        axpy!(1.0/solver.proximalStepsizeBeta, info.primalSolPrev[nodeID], info.primalBuffer[nodeID])

        for edgeID in node.neighbors
            edge = admmGraph.edges[edgeID]
            # Compute residual difference
            edge.mappings[nodeID](info.primalSolPrev[nodeID], solver.dualBuffer[edgeID], false)
            rmul!(solver.dualBuffer[edgeID], -1.0)
            edge.mappings[nodeID](info.primalSol[nodeID], solver.dualBuffer[edgeID], true)
            
            # Scale by rho and update primal buffer
            rmul!(solver.dualBuffer[edgeID], rho)
            adjoint!(edge.mappings[nodeID], solver.dualBuffer[edgeID], info.primalBuffer[nodeID], true)
        end 
    end 

    # if dualStepsize != 1.0, add additional term: 
    # add (dualStepsize - 1) * rho * A'(Ax^{k+1} + By^{k+1} -b) to info.primalBuffer
    if abs(solver.dualStepsize - 1.0) > ZeroTolerance 
        @threads for edgeID in collect(keys(info.dualSol))
            edge = admmGraph.edges[edgeID]
            nodeID1 = edge.nodeID1
            nodeID2 = edge.nodeID2
            solver.dualBuffer[edgeID] .= -admmGraph.edges[edgeID].rhs
            edge.mappings[nodeID1](info.primalSolPrev[nodeID1], solver.dualBuffer[edgeID], true)
            edge.mappings[nodeID2](info.primalSolPrev[nodeID2], solver.dualBuffer[edgeID], true)
        end  

        @threads for nodeID in admmGraph.left 
            solver.primalBuffer[nodeID] .= 0.0
            for edgeID in admmGraph.nodes[nodeID].neighbors
                adjoint!(admmGraph.edges[edgeID].mappings[nodeID], solver.dualBuffer[edgeID], solver.primalBuffer[nodeID], true)
            end 
            axpy!((solver.dualStepsize - 1.0) * rho, solver.primalBuffer[nodeID], info.primalBuffer[nodeID])
        end 

        @threads for nodeID in admmGraph.right 
            solver.primalBuffer[nodeID] .= 0.0
            for edgeID in admmGraph.nodes[nodeID].neighbors
                adjoint!(admmGraph.edges[edgeID].mappings[nodeID], solver.dualBuffer[edgeID], solver.primalBuffer[nodeID], true)
            end 
            axpy!((solver.dualStepsize - 1.0) * rho, solver.primalBuffer[nodeID], info.primalBuffer[nodeID])
        end 
    end 

    dresL2Square = 0.0 
    dresLInf = 0.0 
    for nodeID in keys(info.primalBuffer)
        dresL2Square += dot(info.primalBuffer[nodeID], info.primalBuffer[nodeID])
        dresLInf = max(dresLInf, norm(info.primalBuffer[nodeID], Inf))
    end 

    push!(info.dresL2, sqrt(dresL2Square))
    push!(info.dresLInf, dresLInf)
end 

"""
    update!(solver::DoublyLinearizedSolver, info::ADMMIterationInfo, 
           admmGraph::ADMMBipartiteGraph, rhoUpdated::Bool)

Update solver parameters when the penalty parameter ρ changes.

This function responds to penalty parameter updates by recalculating the proximal
step sizes α and β. The step sizes depend on ρ and must be updated to maintain
convergence guarantees when ρ changes.

# Arguments
- `solver::DoublyLinearizedSolver`: The solver instance to update
- `info::ADMMIterationInfo`: Current iteration information with updated ρ
- `admmGraph::ADMMBipartiteGraph`: The bipartite graph structure (unused)
- `rhoUpdated::Bool`: Flag indicating whether ρ was updated

# Effects
- If `rhoUpdated` is true, recomputes `proximalStepsizeAlpha` and `proximalStepsizeBeta`
- If `rhoUpdated` is false, performs no operations

# Notes
- This function is typically called by ADMM adapters after penalty parameter updates
- The new step sizes maintain the same convergence rate with the updated penalty parameter
"""
function update!(solver::DoublyLinearizedSolver, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph, rhoUpdated::Bool)
    if rhoUpdated == false 
        return 
    end 

    # estimate new alpha and beta 
    newRho = info.rhoHistory[end][1]
    computeProximalStepsize!(solver, newRho)
end 