mutable struct BCDJuMPModel 
    model::JuMP.Model
    var::Vector{JuMP.VariableRef}
    objExpressions::Vector{Union{JuMP.AffExpr, JuMP.QuadExpr}}
    nonlinearObjExpressions::Vector{Any}
    buffer::NumericVariable
    proximalCoefficients::Float64
    originalSubproblem::Bool
    function BCDJuMPModel(model::JuMP.Model,
        var::Vector{JuMP.VariableRef},
        objExpressions::Vector{Union{JuMP.AffExpr, JuMP.QuadExpr}}, 
        nonlinearObjExpressions::Vector{Any},
        buffer::NumericVariable,
        proximalCoefficients::Float64, 
        originalSubproblem::Bool)
        return new(model, 
            var, 
            objExpressions, 
            nonlinearObjExpressions, 
            buffer, 
            proximalCoefficients, 
            originalSubproblem)
    end 
end 


function BCDJuMPModel(blockIndex::Int64, mbp::MultiblockProblem, originalSubproblem::Bool)
    block = mbp.blocks[blockIndex]
    objExpressions = Vector{Union{JuMP.AffExpr, JuMP.QuadExpr}}()
    nonlinearObjExpressions = Vector{Any}()

    model = JuMP.Model(Ipopt.Optimizer)

    # set solver options 
    JuMP.set_silent(model)
    if HSL_FOUND 
        JuMP.set_attribute(model, "hsllib", HSL_jll.libhsl_path)
        JuMP.set_attribute(model, "linear_solver", "ma27")
    end 

    var_dict = Dict{BlockID, Vector{JuMP.VariableRef}}()

    # add block variables to JuMP model 
    addBlockVariableToJuMPModel!(model, 
        block.f, 
        block.g, 
        block.val, 
        block.id, 
        var_dict,
        objExpressions, 
        nonlinearObjExpressions)

    if originalSubproblem == true
        proximalCoefficients = 0.0
    else 
        # Choosing proximal coefficients as a multiple of Lipschitz constant 
        # can ensure convexity of subproblem even when smooth functions are not convex;
        # however, for convex problems, it is better to use a smaller multiple 
        # TODO: better strategies?
        x = NumericVariable[block.val for block in mbp.blocks]
        estimate = estimateLipschitzConstant(mbp.couplingFunction, x)
        blockEstimate = estimate + estimateLipschitzConstant(block.f, block.val)
        proximalCoefficients = 0.1 * blockEstimate
        # @info "BCDProximalSubproblemSolver: proximal coefficient for block $blockIndex = $proximalCoefficients"
    end 

    return BCDJuMPModel(model, 
        var_dict[block.id], 
        objExpressions, 
        nonlinearObjExpressions, 
        similar(block.val, Float64), 
        proximalCoefficients, 
        originalSubproblem)
end 


"""
    BCDProximalSubproblemSolver <: AbstractBCDSubproblemSolver

A JuMP/Ipopt-based BCD subproblem solver.

# Fields
- `models::Vector{BCDJuMPModel}`: JuMP models for each block subproblem
- `originalSubproblem::Bool`: `true` for original subproblems, `false` to add
  proximal quadratic regularization.

See also: `AbstractBCDSubproblemSolver`, `BCDProximalLinearSubproblemSolver`, `initialize!`, `solve!`
"""
mutable struct BCDProximalSubproblemSolver <: AbstractBCDSubproblemSolver
    # Inherit all fields from BCDOriginalSubproblemSolver
    models::Vector{BCDJuMPModel}
    originalSubproblem::Bool
    function BCDProximalSubproblemSolver(;originalSubproblem::Bool = true)
        return new(Vector{BCDJuMPModel}(), originalSubproblem)
    end 
end 

function initialize!(solver::BCDProximalSubproblemSolver, 
    mbp::MultiblockProblem, 
    param::BCDParam, 
    info::BCDIterationInfo)

    numberBlocks = length(mbp.blocks)
    empty!(solver.models)
    
    for k in 1:numberBlocks 
        push!(solver.models, BCDJuMPModel(k, mbp, solver.originalSubproblem))
    end 

end 

function solve!(blockModel::BCDJuMPModel, blockIndex::Int64, mbp::MultiblockProblem, info::BCDIterationInfo)
    couplingObjExpr = JuMPAddPartialBlockFunction(mbp.couplingFunction, 
        blockModel.model, 
        blockIndex, 
        blockModel.var, 
        info.solution)

    # Add proximal term: (L_i/2) ||x_i - x_i^k||^2
    L_i = blockModel.proximalCoefficients
    x_k = info.solution[blockIndex]  # Current solution for this block

    quadraticObj = JuMP.QuadExpr()
    if blockModel.originalSubproblem == false && L_i > 0.0
        # println("Adding proximal term for block $blockIndex")
        for j in 1:length(blockModel.var)
            JuMP.add_to_expression!(quadraticObj, (L_i/2) * (blockModel.var[j] - x_k[j])^2)
        end 
    end 

    isCouplingFunctionQuadratic = isa(couplingObjExpr, JuMP.QuadExpr) || isa(couplingObjExpr, JuMP.AffExpr)
    
    if isCouplingFunctionQuadratic && isempty(blockModel.nonlinearObjExpressions)
    
        JuMP.add_to_expression!(quadraticObj, couplingObjExpr)
        for expr in blockModel.objExpressions 
            JuMP.add_to_expression!(quadraticObj, expr)
        end 
        JuMP.@objective(blockModel.model, Min, quadraticObj)
    else 
        nonlinearObj = blockModel.originalSubproblem == false && L_i > 0.0 ? couplingObjExpr + quadraticObj : couplingObjExpr
        for expr in blockModel.nonlinearObjExpressions 
            nonlinearObj += expr
        end 
        for expr in blockModel.objExpressions 
            nonlinearObj += expr
        end 
        JuMP.@NLobjective(blockModel.model, Min, nonlinearObj)
    end 
    
    JuMP.optimize!(blockModel.model)

     
    # save current and previous solution 
    copyto!(info.solutionPrev[blockIndex], info.solution[blockIndex])
    info.solution[blockIndex] .= JuMP.value.(blockModel.var)

    # compute partial gradient of the coupling function after the update 
    partialGradientOracle!(blockModel.buffer, 
        mbp.couplingFunction, 
        info.solution,
        blockIndex)

end 

function solve!(solver::BCDProximalSubproblemSolver, 
    blockIndex::Int64, 
    mbp::MultiblockProblem, 
    param::BCDParam, 
    info::BCDIterationInfo)

    solve!(solver.models[blockIndex], blockIndex, mbp, info)
end 


function updateDualResidual!(blockModel::BCDJuMPModel, info::BCDIterationInfo, blockIndex::Int64)

    L_i = blockModel.proximalCoefficients
    if blockModel.originalSubproblem == false && L_i > 0.0
        axpy!(L_i, info.solution[blockIndex], blockModel.buffer)
        axpy!(-L_i, info.solutionPrev[blockIndex], blockModel.buffer)
    end 
    axpy!(-1.0, blockModel.buffer, info.blockDres[blockIndex])
    info.blockDresL2[blockIndex] = norm(info.blockDres[blockIndex])
    info.blockDresLInf[blockIndex] = norm(info.blockDres[blockIndex], Inf)
end 

function updateDualResidual!(solver::BCDProximalSubproblemSolver, 
    mbp::MultiblockProblem, 
    param::BCDParam, 
    info::BCDIterationInfo)
    
    # record latest objective value
    recordBCDObjectiveValue(info, mbp)

    # get the latest gradient of the coupling function in info.blockDres
    gradientOracle!(info.blockDres, mbp.couplingFunction, info.solution)
    
    # compute dual residuals for each block 
    numberBlocks = length(mbp.blocks)
    for i in 1:numberBlocks 
        updateDualResidual!(solver.models[i], info, i)
    end 

    push!(info.dresL2, norm(info.blockDresL2))
    push!(info.dresLInf, norm(info.blockDresLInf, Inf))
end 
