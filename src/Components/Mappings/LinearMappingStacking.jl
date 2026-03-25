"""
    LinearMappingStacking(mappings, sizeAlongFirstDimension)

Stack a vector of mappings vertically (along the first dimension) and return the stacked output.

Given mappings L₁, …, L_K that all act on the same input `x`, this mapping represents:

    L(x) = [L₁(x);
            L₂(x);
            ...
            L_K(x)]

where the stacking is along the first dimension. The vector `sizeAlongFirstDimension`
specifies the first-dimension size of each block output L_k(x).

# Shape requirements
- For each k, `Lk(x)` must be an array with size `(sizeAlongFirstDimension[k], t2, ..., td)` for
  the *same* trailing dimensions `(t2, ..., td)` across all k.
- `ret` must have size `(sum(sizeAlongFirstDimension), t2, ..., td)`.
- For each k, `mappings[k](x, view(ret, block_k, :, ...))` must be dimensionally valid.

# Notes
- This type intentionally does not try to infer output sizes from the mappings (most mappings
  in this codebase do not expose output dimensions generically), so `sizeAlongFirstDimension`
  is required.
"""
struct LinearMappingStacking <: AbstractMapping
    mappings::Vector{AbstractMapping}
    sizeAlongFirstDimension::Vector{Int64}
    totalFirstDimension::Int64

    """
        LinearMappingStacking(mappings::Vector{<:AbstractMapping}, sizeAlongFirstDimension::Vector{Int64})

    Construct a stacking mapping from a list of mappings and the corresponding first-dimension
    sizes of their outputs.

    # Arguments
    - `mappings`: Vector of mappings `L₁, …, L_K`.
    - `sizeAlongFirstDimension`: Vector of positive integers specifying `size(L_k(x), 1)` for each k.

    # Notes
    - All `L_k(x)` must have the same trailing dimensions so they can be stacked.
    """
    function LinearMappingStacking(mappings::Vector{<:AbstractMapping},
        sizeAlongFirstDimension::Vector{Int64})
        @assert length(mappings) == length(sizeAlongFirstDimension) "LinearMappingStacking: length(mappings) and length(sizeAlongFirstDimension) must be the same"
        @assert length(mappings) >= 1 "LinearMappingStacking: must provide at least one mapping"
        @assert all(sizeAlongFirstDimension .> 0) "LinearMappingStacking: all block sizes must be positive"
        return new(Vector{AbstractMapping}(mappings), 
            sizeAlongFirstDimension, 
            sum(sizeAlongFirstDimension))
    end
end


"""
    (L::LinearMappingStacking)(x::NumericVariable, ret::NumericVariable, add::Bool = false)

In-place forward operator for the stacked mapping.

Computes the vertically stacked output along the first dimension:

    ret = [L₁(x); L₂(x); ...; L_K(x)]

# Shape requirements
- `size(ret, 1) == sum(sizeAlongFirstDimension)`.
- The trailing dimensions of `ret` must match the common trailing dimensions of all `L_k(x)`.

# Arguments
- `x`: Input array (scalar inputs are not supported).
- `ret`: Output array buffer to fill/accumulate.
- `add`: If `true`, accumulate into `ret` instead of overwriting.
"""
function (L::LinearMappingStacking)(x::NumericVariable, ret::NumericVariable, add::Bool = false)
    if isa(x, Number) || isa(ret, Number)
        error("LinearMappingStacking: forward mapping does not support scalar inputs/outputs")
    end

    sret = size(ret)
    @assert sret[1] == L.totalFirstDimension "LinearMappingStacking: output array has incorrect first dimension."

    tail = ndims(ret) == 1 ? () : ntuple(_ -> Colon(), ndims(ret) - 1)

    startIdx = 1
    for k in 1:length(L.mappings)
        endIdx = startIdx + L.sizeAlongFirstDimension[k] - 1
        if ndims(ret) == 1
            @views L.mappings[k](x, ret[startIdx:endIdx], add)
        else
            @views L.mappings[k](x, ret[startIdx:endIdx, tail...], add)
        end
        startIdx = endIdx + 1
    end
end


"""
    (L::LinearMappingStacking)(x::NumericVariable)

Allocating forward operator. Returns the stacked output as a new array.

The trailing dimensions are inferred from `L₁(x)` and enforced to be consistent for all blocks.

# Returns
- An array of size `(sum(sizeAlongFirstDimension), t2, ..., td)` where `(t2, ..., td)` are the
  trailing dimensions of each `L_k(x)`.
"""
function (L::LinearMappingStacking)(x::NumericVariable)
    if isa(x, Number)
        error("LinearMappingStacking: out-of-place forward mapping does not support scalar inputs")
    end

    # Infer trailing dimensions from the first mapping output, then enforce consistency.
    y1 = L.mappings[1](x)
    if isa(y1, Number)
        error("LinearMappingStacking: each mapping must return an array output to be stackable")
    end
    @assert size(y1, 1) == L.sizeAlongFirstDimension[1] "LinearMappingStacking: sizeAlongFirstDimension[1] does not match size(L1(x), 1)."

    tailShape = Base.tail(size(y1))
    outShape = (sum(L.sizeAlongFirstDimension), tailShape...)
    ret = length(outShape) <= 2 ? spzeros(outShape) : zeros(outShape)

    # Fill first block from y1 (avoid recomputation).
    if ndims(ret) == 1
        @views ret[1:L.sizeAlongFirstDimension[1]] .= y1
    else
        tail = ntuple(_ -> Colon(), ndims(ret) - 1)
        @views ret[1:L.sizeAlongFirstDimension[1], tail...] .= y1
    end

    # Remaining blocks.
    startIdx = L.sizeAlongFirstDimension[1] + 1
    for k in 2:length(L.mappings)
        yk = L.mappings[k](x)
        if isa(yk, Number)
            error("LinearMappingStacking: each mapping must return an array output to be stackable")
        end
        @assert size(yk, 1) == L.sizeAlongFirstDimension[k] "LinearMappingStacking: sizeAlongFirstDimension[$k] does not match size(Lk(x), 1)."
        @assert Base.tail(size(yk)) == tailShape "LinearMappingStacking: trailing dimensions of Lk(x) must match across all k."

        endIdx = startIdx + L.sizeAlongFirstDimension[k] - 1
        if ndims(ret) == 1
            @views ret[startIdx:endIdx] .= yk
        else
            tail = ntuple(_ -> Colon(), ndims(ret) - 1)
            @views ret[startIdx:endIdx, tail...] .= yk
        end
        startIdx = endIdx + 1
    end

    return ret
end


"""
    adjoint!(L::LinearMappingStacking, y::NumericVariable, ret::NumericVariable, add::Bool = false)

In-place adjoint operator for the stacked mapping.

If `A(x) = [A₁(x); ...; A_K(x)]`, then:

    A*(y) = Σ_k A_k*(y_k)

where `y_k` is the slice of `y` corresponding to block k along the first dimension.

# Shape requirements
- `size(y, 1) == sum(sizeAlongFirstDimension)`.
- `ret` must have the shape of the input space of each block mapping adjoint (the adjoint accumulates into `ret`).
"""
function adjoint!(L::LinearMappingStacking, y::NumericVariable, ret::NumericVariable, add::Bool = false)
    if isa(y, Number) || isa(ret, Number)
        error("LinearMappingStacking (adjoint!): in-place adjoint does not support scalar inputs/outputs")
    end

    sy = size(y)
    @assert sy[1] == sum(L.sizeAlongFirstDimension) "LinearMappingStacking (adjoint!): size(y,1) must equal sum(sizeAlongFirstDimension)."

    # A = [A1; A2; ...], so A' y = sum_i A_i' y_i
    tail = ndims(y) == 1 ? () : ntuple(_ -> Colon(), ndims(y) - 1)
    startIdx = 1
    # First block honors the caller's `add` semantics.
    endIdx = L.sizeAlongFirstDimension[1]
    if ndims(y) == 1
        @views adjoint!(L.mappings[1], y[startIdx:endIdx], ret, add)
    else
        @views adjoint!(L.mappings[1], y[startIdx:endIdx, tail...], ret, add)
    end

    startIdx = endIdx + 1
    for k in 2:length(L.mappings)
        endIdx = startIdx + L.sizeAlongFirstDimension[k] - 1
        if ndims(y) == 1
            @views adjoint!(L.mappings[k], y[startIdx:endIdx], ret, true)
        else
            @views adjoint!(L.mappings[k], y[startIdx:endIdx, tail...], ret, true)
        end
        startIdx = endIdx + 1
    end
end


"""
    adjoint(L::LinearMappingStacking, y::NumericVariable)

Allocating adjoint operator. Returns `A*(y)` as a new array.

Allocation is delegated to the first mapping's allocating `adjoint`, then remaining blocks
are accumulated in-place.
"""
function adjoint(L::LinearMappingStacking, y::NumericVariable)
    if isa(y, Number)
        error("LinearMappingStacking (adjoint): input must be an array (stacked output); got scalar")
    end

    sy = size(y)
    @assert sy[1] == L.totalFirstDimension "LinearMappingStacking (adjoint): size(y,1) must equal sum(sizeAlongFirstDimension)."

    # Allocate using the first block's allocating adjoint (it knows the input shape).
    startIdx = 1
    endIdx = L.sizeAlongFirstDimension[1]
    tail = ndims(y) == 1 ? () : ntuple(_ -> Colon(), ndims(y) - 1)
    y1 = ndims(y) == 1 ? (@views y[startIdx:endIdx]) : (@views y[startIdx:endIdx, tail...])
    ret = copy(adjoint(L.mappings[1], y1))

    # Accumulate remaining blocks in-place.
    startIdx = endIdx + 1
    for k in 2:length(L.mappings)
        endIdx = startIdx + L.sizeAlongFirstDimension[k] - 1
        yk = ndims(y) == 1 ? (@views y[startIdx:endIdx]) : (@views y[startIdx:endIdx, tail...])
        adjoint!(L.mappings[k], yk, ret, true)
        startIdx = endIdx + 1
    end

    return ret
end


# """
# Adjoint mapping wrapper for `LinearMappingStacking`.

# This exists so `createAdjointMapping(::LinearMappingStacking)` can return a valid mapping object.
# """
# struct LinearMappingStackingAdjoint <: AbstractMapping
#     original::LinearMappingStacking
# end

# function (L::LinearMappingStackingAdjoint)(y::NumericVariable, ret::NumericVariable, add::Bool = false)
#     adjoint!(L.original, y, ret, add)
# end

# function (L::LinearMappingStackingAdjoint)(y::NumericVariable)
#     return adjoint(L.original, y)
# end

# function adjoint!(L::LinearMappingStackingAdjoint, x::NumericVariable, ret::NumericVariable, add::Bool = false)
#     L.original(x, ret, add)
# end

# function adjoint(L::LinearMappingStackingAdjoint, x::NumericVariable)
#     return L.original(x)
# end


# function createAdjointMapping(L::LinearMappingStacking)
#     return LinearMappingStackingAdjoint(L)
# end

# function createAdjointMapping(L::LinearMappingStackingAdjoint)
#     return L.original
# end


"""
    operatorNorm2(L::LinearMappingStacking) -> Float64

Compute an upper bound on the operator 2-norm of the stacked mapping.

Uses the inequality:

    ||[A₁; ...; A_K]||₂ ≤ sqrt( Σ_k ||A_k||₂² )
"""
function operatorNorm2(L::LinearMappingStacking)
    # Safe upper bound: ||[A1;...;Ak]||_2 <= sqrt(sum_i ||Ai||_2^2)
    s = 0.0
    for m in L.mappings
        n = operatorNorm2(m)
        s += n * n
    end
    return sqrt(s)
end

# function operatorNorm2(L::LinearMappingStackingAdjoint)
#     return operatorNorm2(L.original)
# end