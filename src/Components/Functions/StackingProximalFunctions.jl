"""
    StackingProximalFunctions(proximalFunctions, sizeAlongFirstDimension)

A separable proximal function representing a sum of proximal functions acting on contiguous
sub-arrays of the input along the first dimension.

Given proximal functions f₁, …, f_K and sizes s₁, …, s_K with ∑ sᵢ = size(x,1),
this type represents:

    F(x) = f₁(x[1:s₁, :, ...]) + f₂(x[s₁+1:s₁+s₂, :, ...]) + ... + f_K(x[...])

and its proximal operator is block-separable:

    prox_{γF}(x) = [prox_{γf₁}(x₁); prox_{γf₂}(x₂); ...]
"""
struct StackingProximalFunctions <: AbstractFunction
    proximalFunctions::Vector{AbstractFunction}
    sizeAlongFirstDimension::Vector{Int64}

    function StackingProximalFunctions(proximalFunctions::Vector{<:AbstractFunction},
        sizeAlongFirstDimension::Vector{Int64})
        @assert length(proximalFunctions) == length(sizeAlongFirstDimension) "StackingProximalFunctions: length of proximalFunctions and sizeAlongFirstDimension must be the same"
        @assert all(sizeAlongFirstDimension .> 0) "StackingProximalFunctions: all block sizes must be positive"
        @assert all(g -> isProximal(g), proximalFunctions) "StackingProximalFunctions: all functions must be proximal"
        return new(Vector{AbstractFunction}(proximalFunctions), sizeAlongFirstDimension)
    end
end 

isProximal(::Type{StackingProximalFunctions}) = true
isSmooth(::Type{StackingProximalFunctions}) = false
isSupportedByJuMP(::Type{StackingProximalFunctions}) = false
isSet(::Type{StackingProximalFunctions}) = false
isConvex(::Type{StackingProximalFunctions}) = false
isConvex(f::StackingProximalFunctions) = all(isConvex, f.proximalFunctions)


function (f::StackingProximalFunctions)(x::NumericVariable, enableParallel::Bool=false)
    if isa(x, Number)
        error("StackingProximalFunctions: input must be an array; got scalar")
    end
    @assert(ndims(x) >= 1, "StackingProximalFunctions: input must have at least one dimension")

    totalSize = sum(f.sizeAlongFirstDimension)
    @assert(size(x, 1) == totalSize, "StackingProximalFunctions: size(x,1) must equal sum(sizeAlongFirstDimension)")

    K = length(f.proximalFunctions)
    tail = ndims(x) == 1 ? () : ntuple(_ -> Colon(), ndims(x) - 1)

    if enableParallel && K > 1
        partial = zeros(Float64, Threads.nthreads())
        offsets = cumsum(f.sizeAlongFirstDimension)
        Threads.@threads for k in 1:K
            startIdx = (k == 1) ? 1 : (offsets[k - 1] + 1)
            endIdx = offsets[k]
            xv = ndims(x) == 1 ? (@views x[startIdx:endIdx]) : (@views x[startIdx:endIdx, tail...])
            partial[Threads.threadid()] += f.proximalFunctions[k](xv, false)
        end
        return sum(partial)
    else
        val = 0.0
        startIdx = 1
        for k in 1:K
            endIdx = startIdx + f.sizeAlongFirstDimension[k] - 1
            xv = ndims(x) == 1 ? (@views x[startIdx:endIdx]) : (@views x[startIdx:endIdx, tail...])
            val += f.proximalFunctions[k](xv, enableParallel)
            startIdx = endIdx + 1
        end
        return val
    end
end


function proximalOracle!(y::NumericVariable, f::StackingProximalFunctions, x::NumericVariable,
    gamma::Float64=1.0, enableParallel::Bool=false)

    if isa(x, Number) || isa(y, Number)
        error("StackingProximalFunctions: proximalOracle! does not support scalar inputs/outputs")
    end
    @assert(gamma > 0.0, "StackingProximalFunctions: gamma must be positive")
    @assert(size(x) == size(y), "StackingProximalFunctions: input and output must have the same size")
    @assert(ndims(x) >= 1, "StackingProximalFunctions: input must have at least one dimension")

    totalSize = sum(f.sizeAlongFirstDimension)
    @assert(size(x, 1) == totalSize, "StackingProximalFunctions: size(x,1) must equal sum(sizeAlongFirstDimension)")

    K = length(f.proximalFunctions)
    tail = ndims(x) == 1 ? () : ntuple(_ -> Colon(), ndims(x) - 1)

    if enableParallel && K > 1
        # Parallelize over blocks; avoid nested parallelism inside each prox call.
        offsets = cumsum(f.sizeAlongFirstDimension)
        Threads.@threads for k in 1:K
            startIdx = (k == 1) ? 1 : (offsets[k - 1] + 1)
            endIdx = offsets[k]
            xv = ndims(x) == 1 ? (@views x[startIdx:endIdx]) : (@views x[startIdx:endIdx, tail...])
            yv = ndims(y) == 1 ? (@views y[startIdx:endIdx]) : (@views y[startIdx:endIdx, tail...])
            proximalOracle!(yv, f.proximalFunctions[k], xv, gamma, false)
        end
    else
        startIdx = 1
        for k in 1:K
            endIdx = startIdx + f.sizeAlongFirstDimension[k] - 1
            xv = ndims(x) == 1 ? (@views x[startIdx:endIdx]) : (@views x[startIdx:endIdx, tail...])
            yv = ndims(y) == 1 ? (@views y[startIdx:endIdx]) : (@views y[startIdx:endIdx, tail...])
            proximalOracle!(yv, f.proximalFunctions[k], xv, gamma, enableParallel)
            startIdx = endIdx + 1
        end
    end

    return y
end

function proximalOracle(f::StackingProximalFunctions, x::NumericVariable, gamma::Float64=1.0, enableParallel::Bool=false)
    if isa(x, Number)
        error("StackingProximalFunctions: proximalOracle does not support scalar inputs")
    end
    y = similar(x)
    proximalOracle!(y, f, x, gamma, enableParallel)
    return y
end
