
mutable struct BCDProximalLinearModel
    proximalCoefficients::Float64
    extrapolationWeights::Float64
    extrapolationPoints::NumericVariable
    proximalCenter::NumericVariable
    buffer1::NumericVariable
    buffer2::NumericVariable

    function BCDProximalLinearModel(
        proximalCoefficients::Float64,
        extrapolationWeights::Float64,
        extrapolationPoints::NumericVariable,
        proximalCenter::NumericVariable,
        buffer1::NumericVariable,
        buffer2::NumericVariable)
        return new(proximalCoefficients, extrapolationWeights, extrapolationPoints, proximalCenter, buffer1, buffer2)
    end 
end

function BCDProximalLinearModel(blockIndex::Int64, mbp::MultiblockProblem)
    block = mbp.blocks[blockIndex]
    
    # Estimate proximal coefficients
    x = NumericVariable[block.val for block in mbp.blocks]
    estimate = estimateLipschitzConstant(mbp.couplingFunction, x)
    blockEstimate = estimate + estimateLipschitzConstant(block.f, block.val)
    proximalCoefficients = 1.0 * blockEstimate
    # @info "BCDProximalLinearSubproblemSolver: proximal coefficient for block $blockIndex = $proximalCoefficients"
    
    # Initialize extrapolation weights (start with 0)
    extrapolationWeights = 0.0
    
    # Initialize buffers with same structure as block value
    extrapolationPoints = similar(block.val, Float64)
    proximalCenter = similar(block.val, Float64)
    buffer1 = similar(block.val, Float64)
    buffer2 = similar(block.val, Float64)
    
    return BCDProximalLinearModel(
        proximalCoefficients,
        extrapolationWeights,
        extrapolationPoints,
        proximalCenter,
        buffer1,
        buffer2
    )
end 

"""
    BCDProximalLinearSubproblemSolver <: AbstractBCDSubproblemSolver

A proximal-gradient BCD subproblem solver using linearized coupling terms.

# Fields
- `models::Vector{BCDProximalLinearModel}`: Proximal linear models for each block
"""
mutable struct BCDProximalLinearSubproblemSolver <: AbstractBCDSubproblemSolver
    models::Vector{BCDProximalLinearModel}
    
    function BCDProximalLinearSubproblemSolver()
        return new(Vector{BCDProximalLinearModel}())
    end 
end 

function initialize!(solver::BCDProximalLinearSubproblemSolver, 
    mbp::MultiblockProblem, 
    param::BCDParam, 
    info::BCDIterationInfo)
    
    numberBlocks = length(mbp.blocks)
    empty!(solver.models)
    
    # Create model for each block
    for k in 1:numberBlocks 
        push!(solver.models, BCDProximalLinearModel(k, mbp))
    end 
    
end 

function solve!(blockModel::BCDProximalLinearModel, blockIndex::Int64, mbp::MultiblockProblem, info::BCDIterationInfo)
    # prepare gradient vectors 
    partialGradientOracle!(blockModel.buffer1, 
        mbp.couplingFunction, 
        info.solution,
        blockIndex)
    
    gradientOracle!(blockModel.buffer2, mbp.blocks[blockIndex].f, info.solution[blockIndex])

    # prepare extrapolation points 
    copyto!(blockModel.extrapolationPoints, info.solution[blockIndex])
    w_i = blockModel.extrapolationWeights
    if (w_i > 0.0)
        axpy!(w_i, info.solution[blockIndex], blockModel.extrapolationPoints)
        axpy!(-w_i, info.solutionPrev[blockIndex], blockModel.extrapolationPoints)
    end 

    # prepare proximal center 
    copyto!(blockModel.proximalCenter, blockModel.extrapolationPoints)
    L_i = blockModel.proximalCoefficients
    gamma_i = 1.0/L_i
    axpy!(-gamma_i, blockModel.buffer1, blockModel.proximalCenter)
    axpy!(-gamma_i, blockModel.buffer2, blockModel.proximalCenter)
    
    # save previous solution 
    copyto!(info.solutionPrev[blockIndex], info.solution[blockIndex]) 
    
    # solve 
    proximalOracle!(info.solution[blockIndex], 
        mbp.blocks[blockIndex].g,
        blockModel.proximalCenter,
        gamma_i)
end

function solve!(solver::BCDProximalLinearSubproblemSolver, 
    blockIndex::Int64, 
    mbp::MultiblockProblem, 
    param::BCDParam, 
    info::BCDIterationInfo)
    
    solve!(solver.models[blockIndex], blockIndex, mbp, info)
end 


function updateDualResidual!(blockModel::BCDProximalLinearModel, blockIndex::Int64, mbp::MultiblockProblem, info::BCDIterationInfo)
    # compute old coupling gradient + old block gradient + gradient of proximal term in blockModel.buffer2 
    axpy!(1.0, blockModel.buffer1, blockModel.buffer2)
    L_i = blockModel.proximalCoefficients
    axpy!(L_i, info.solution[blockIndex], blockModel.buffer2)
    axpy!(-L_i, blockModel.extrapolationPoints, blockModel.buffer2)

    # compute new block gradient and add to info.blockDres (new coupling gradient)
    gradientOracle!(blockModel.buffer1, mbp.blocks[blockIndex].f, info.solution[blockIndex])
    axpy!(1.0, blockModel.buffer1, info.blockDres[blockIndex])

    axpy!(-1.0, blockModel.buffer2, info.blockDres[blockIndex])

    info.blockDresL2[blockIndex] = norm(info.blockDres[blockIndex])
    info.blockDresLInf[blockIndex] = norm(info.blockDres[blockIndex], Inf)
end

function updateDualResidual!(solver::BCDProximalLinearSubproblemSolver, 
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
        updateDualResidual!(solver.models[i], i, mbp, info)
    end 
    push!(info.dresL2, norm(info.blockDresL2))
    push!(info.dresLInf, norm(info.blockDresLInf, Inf))
end 