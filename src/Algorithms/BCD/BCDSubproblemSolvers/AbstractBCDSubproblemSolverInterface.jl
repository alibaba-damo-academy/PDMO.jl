"""
    initialize!(solver::AbstractBCDSubproblemSolver, 
               mbp::MultiblockProblem, 
               param::BCDParam, 
               info::BCDIterationInfo)

Initialize solver state before BCD iterations.

# Arguments
- `solver::AbstractBCDSubproblemSolver`: The solver instance to initialize
- `mbp::MultiblockProblem`: The complete multiblock problem definition
- `param::BCDParam`: Algorithm parameters including tolerances and solver settings
- `info::BCDIterationInfo`: Initial iteration information and solution state

Concrete solver types should implement this method.

See also: `solve!`, `updateDualResidual!`
"""
function initialize!(solver::AbstractBCDSubproblemSolver, 
    mbp::MultiblockProblem, 
    param::BCDParam, 
    info::BCDIterationInfo)

    error("AbstractBCDSubproblemSolver: initialize! is not implemented for $(typeof(solver))")
end 

"""
    solve!(solver::AbstractBCDSubproblemSolver, 
           blockIndex::Int64, 
           mbp::MultiblockProblem, 
           param::BCDParam, 
           info::BCDIterationInfo)

Solve one block subproblem and update `info.solution[blockIndex]`.

# Arguments
- `solver::AbstractBCDSubproblemSolver`: The solver instance
- `blockIndex::Int64`: Index of the block to update (1-based)
- `mbp::MultiblockProblem`: Complete multiblock problem definition
- `param::BCDParam`: Algorithm parameters and solver settings
- `info::BCDIterationInfo`: Current iteration state and solution

Concrete solver types should implement this method.

See also: `initialize!`, `updateDualResidual!`
"""
function solve!(solver::AbstractBCDSubproblemSolver, blockIndex::Int64, 
    mbp::MultiblockProblem, 
    param::BCDParam, 
    info::BCDIterationInfo)

    error("AbstractBCDSubproblemSolver: solve! is not implemented for $(typeof(solver))")
end 


"""
    updateDualResidual!(solver::AbstractBCDSubproblemSolver, mbp::MultiblockProblem, param::BCDParam, info::BCDIterationInfo)

Update BCD dual residual metrics after block updates.
"""
function updateDualResidual!(solver::AbstractBCDSubproblemSolver, 
    mbp::MultiblockProblem, 
    param::BCDParam, 
    info::BCDIterationInfo)
    
    error("AbstractBCDSubproblemSolver: updateDualResidual! is not implemented for $(typeof(solver))")
end 

# """
#     update!(solver::AbstractBCDSubproblemSolver, 
#            mbp::MultiblockProblem, 
#            param::BCDParam, 
#            info::BCDIterationInfo)

# Update the solver state based on algorithm progress and current iteration information.

# This method is called periodically during BCD iterations to allow adaptive solvers
# to modify their behavior based on convergence progress, solution quality, or
# performance metrics. Static solvers may implement this as a no-op.

# **Arguments**
# - `solver::AbstractBCDSubproblemSolver`: The solver instance to update
# - `mbp::MultiblockProblem`: The multiblock problem definition (for reference)
# - `param::BCDParam`: Algorithm parameters (may be modified for adaptive behavior)
# - `info::BCDIterationInfo`: Current iteration information including convergence history

# **Optional Implementation**
# This method is optional for concrete solvers. The default behavior is to do nothing,
# which is appropriate for static solvers that don't adapt their behavior.

# **Adaptive Behavior Examples**
# 1. **Step Size Adaptation**: Adjust solver step sizes based on convergence rate
# 2. **Tolerance Adjustment**: Modify solver tolerances based on dual residuals
# 3. **Strategy Switching**: Change solving strategy based on problem characteristics
# 4. **Regularization Updates**: Adjust regularization parameters
# 5. **Workspace Reallocation**: Resize buffers based on observed problem size

# **Example Implementation**
# ```julia
# function update!(solver::AdaptiveBCDSolver, mbp::MultiblockProblem, param::BCDParam, info::BCDIterationInfo)
#     # Adapt step size based on objective progress
#     if length(info.obj) > 10
#         recentProgress = info.obj[end-9] - info.obj[end]
#         if recentProgress > 0.01
#             solver.stepSize = min(solver.stepSize * 1.1, solver.maxStepSize)
#         else
#             solver.stepSize = max(solver.stepSize * 0.9, solver.minStepSize)
#         end
#     end
    
#     # Adjust solver tolerance based on dual residuals
#     if !isempty(info.dresL2)
#         currentDualRes = info.dresL2[end]
#         if currentDualRes < param.dresTolL2 * 10
#             # Tighten solver tolerance as we approach convergence
#             solver.tolerance = min(solver.tolerance, currentDualRes / 100)
#         end
#     end
    
#     # Switch strategies based on solve success rate
#     if length(info.obj) > 20
#         recentFailures = countRecentSolveFailures(info, 20)
#         if recentFailures > 5
#             solver.strategy = :conservative  # Switch to more robust strategy
#         end
#     end
# end
# ```

# **Update Timing**
# - Called after each complete iteration (all blocks updated)
# - May be called more frequently for highly adaptive solvers
# - Should be computationally lightweight to avoid overhead

# **State Modification Guidelines**
# - Modify solver internal state as needed
# - Do NOT modify `mbp` (problem definition should remain constant)
# - May modify `param` for algorithm-level adaptations (use with caution)
# - May read from `info` but avoid modifying historical data

# **Performance Considerations**
# - Keep update operations lightweight (< 1% of solve time)
# - Avoid expensive computations unless they provide significant benefit
# - Consider update frequency vs. adaptation benefit trade-offs
# - Cache computations when possible

# See also: `initialize!`, `solve!`, `BCDIterationInfo`, `BCDParam`
# """
# function update!(solver::AbstractBCDSubproblemSolver, 
#     mbp::MultiblockProblem, 
#     param::BCDParam, 
#     info::BCDIterationInfo)

#     error("AbstractBCDSubproblemSolver: update! is not implemented for $(typeof(solver))")
# end


# Include concrete solver implementations
include("BCDProximalSubproblemSolver.jl")
include("BCDProximalLinearSubproblemSolver.jl")          # Uncomment when implemented 


"""
    getBCDSubproblemSolverName(solver::AbstractBCDSubproblemSolver)

Get a string identifier for the BCD subproblem solver type.

This function provides consistent naming for different solver types, useful for
logging, debugging, performance tracking, and algorithm configuration.
"""
function getBCDSubproblemSolverName(solver::AbstractBCDSubproblemSolver)
    if typeof(solver) == BCDProximalSubproblemSolver && solver.originalSubproblem == true
        return "BCD_ORIGINAL_SUBPROBLEM_SOLVER"
    elseif typeof(solver) == BCDProximalSubproblemSolver && solver.originalSubproblem == false
        return "BCD_PROXIMAL_SUBPROBLEM_SOLVER"
    elseif typeof(solver) == BCDProximalLinearSubproblemSolver
        return "BCD_PROXIMAL_LINEAR_SUBPROBLEM_SOLVER"
    else 
        return "UNKNOWN_BCD_SUBPROBLEM_SOLVER"
    end 
end
