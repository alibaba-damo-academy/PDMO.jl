include("BCDIterationInfo.jl")
include("BCDBlockUpdateOrders.jl")
include("BCDSubproblemSolvers/AbstractBCDSubproblemSolver.jl")
include("BCDParameter.jl") 
include("BCDSubproblemSolvers/AbstractBCDSubproblemSolverInterface.jl")
include("BCDTerminationCriteria.jl")
include("BCDUtil.jl")

"""
    BCD(couplingFunction::AbstractMultiblockFunction, 
        blockVariables::Vector{BlockVariable},
        param::BCDParam) -> BCDIterationInfo

Solve a multiblock optimization problem using Block Coordinate Descent.

# Arguments
- `couplingFunction::AbstractMultiblockFunction`: The coupling function f(x₁,...,xₙ)
- `blockVariables::Vector{BlockVariable}`: Block variable definitions with initial values
- `param::BCDParam`: Algorithm parameters and configuration

# Returns
- `BCDIterationInfo`: Complete results including solution, convergence status, and metrics

# Algorithm Steps
1. Initialize iteration info and solver
2. Evaluate initial objective value
3. Main iteration loop:
   - Determine block update order according to selection rule
   - For each block in order: solve subproblem and update solution
   - Evaluate full objective after all block updates
   - Check convergence criteria and record metrics
4. Finalize timing and return results

# Throws
- `ArgumentError`: If inputs are invalid (mismatched dimensions, etc.)
- `ErrorException`: If algorithm encounters unrecoverable errors

# Example
```julia
result = BCD(couplingFunction, blockVariables, param)
if result.convergenceStatus != NotConverged
    println("Algorithm converged!")
    println("Solution: ", result.solution)
    println("Iterations: ", result.iterationCount)
end
```
"""
function BCD(mbp::MultiblockProblem, param::BCDParam)
    startTime = time()

    @PDMOInfo param.logLevel "#"^40 * " Block Coordinate Descent " * "#"^40
    
    # Initialize iteration info and termination criteria
    info = BCDIterationInfo(mbp)
    terminationCriteria = BCDTerminationCriteria(param)

    # Initialize solver
    @PDMOInfo param.logLevel "BCD: subproblem solver = $(getBCDSubproblemSolverName(param.solver))"
    initialize!(param.solver, mbp, param, info)
    
    for iter in 1:param.maxIter 
        # update block order 
        updateBlockOrder!(param.blockOrderRule, mbp, info)

        # update blocks in order 
        for blockIndex in param.blockOrderRule.blocksToUpdate 
            solve!(param.solver, blockIndex, mbp, param, info)
        end 

        # update dual residuals 
        updateDualResidual!(param.solver, mbp, param, info)
        
        # log iteration info 
        info.totalTime = time() - startTime 
        iterLogged = BCDLog(iter, info, param)

        # check termination criteria 
        checkTerminationCriteria(info, terminationCriteria)
        if terminationCriteria.terminated 
            if iterLogged == false 
                BCDLog(iter, info, param; final = true)
            end 
            break 
        end 
    end 

    return info 
end 