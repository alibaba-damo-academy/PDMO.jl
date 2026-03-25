
"""
    AbstractAdaPDMParam

Abstract base type for AdaPDM algorithm parameters.

All concrete AdaPDM parameter types should inherit from this abstract type.
"""
abstract type AbstractAdaPDMParam end 

"""
    computeNormEstimate(mbp::MultiblockProblem)

Compute the operator norm estimate for the given multiblock problem.

# Arguments
- `mbp::MultiblockProblem`: The multiblock problem to compute the operator norm estimate for.
"""
function computeNormEstimate(mbp::MultiblockProblem)
    operatorNormEstimate = 0.0 
    nPrimalBlocks = length(mbp.blocks) - 1
    for bi in 1:nPrimalBlocks
        block = mbp.blocks[bi]
        mapping = mbp.constraints[1].mappings[block.id]
        operatorNormEstimate += operatorNorm2(mapping)^2
    end 
    return sqrt(operatorNormEstimate)
end 

include("CondatVuParam.jl")
include("AdaPDMParam.jl")
include("AdaPDMPlusParam.jl")
include("MalitskyPockParam.jl")

"""
    getAdaPDMName(param::AbstractAdaPDMParam)

Get the name of the AdaPDM algorithm based on the parameter type.

# Arguments
- `param::AbstractAdaPDMParam`: The parameter object to get the name for.
"""
function getAdaPDMName(param::AbstractAdaPDMParam)
    if param isa AdaPDMParam
        return "AdaPDM"
    elseif param isa AdaPDMPlusParam
        return "AdaPDMPlus"
    elseif param isa MalitskyPockParam
        return "Malitsky-Pock"
    elseif param isa CondatVuParam
        return "Condat-Vu"
    else
        error("getAdaPDMName: Unknown AdaPDMParam type")
    end 
end
