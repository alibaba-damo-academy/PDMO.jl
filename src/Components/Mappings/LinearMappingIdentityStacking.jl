"""
    LinearMappingIdentityStacking(coe::Float64, sizeStack::Int64)

Linear mapping that vertically stacks `sizeStack` copies of a (scaled) identity mapping
along the first dimension.

For an array input `x`, the forward operator produces:

    L(x) = [coe * x;
            coe * x;
            ...
            coe * x]   # repeated sizeStack times

# Fields
- `coe::Float64`: Scaling coefficient (must be nonzero).
- `sizeStack::Int64`: Number of stacked copies (must be positive).
"""
struct LinearMappingIdentityStacking <: AbstractMapping
    coe::Float64
    sizeStack::Int64 

    function LinearMappingIdentityStacking(coe::Float64, sizeStack::Int64)
        @assert(coe != 0.0, "LinearMappingIdentityStacking: coefficient must be non-zero. ")
        @assert(sizeStack > 0, "LinearMappingIdentityStacking: sizeStack must be positive. ")
        new(coe, sizeStack)
    end
end


"""
    (L::LinearMappingIdentityStacking)(x::NumericVariable, ret::NumericVariable, add::Bool = false)

Apply the "identity-identity stacking" mapping to input `x` and store the result in `ret`.

This mapping returns the vertical stack of `L.sizeStack` (scaled) identity mappings applied to `x`
along the first dimension, i.e.:

    ret = [coe * x;
           coe * x;
           ...
           coe * x]   # repeated sizeStack times

If `add` is `true`, the result is accumulated into `ret` instead of overwriting it.

# Shape requirements
- If `size(x) == (n1, n2, ..., nk)`, then `size(ret)` must be `(n1*sizeStack, n2, ..., nk)`.

# Errors
- Throws an error if `x` is a scalar number (in-place stacking is not supported for scalars).
"""
function (L::LinearMappingIdentityStacking)(x::NumericVariable, ret::NumericVariable, add::Bool = false)
    if isa(x, Number)
        error("LinearMappingIdentityStacking: forward mapping does not support in-place operations for scalar inputs")
    end

    sx = size(x)
    @assert(length(sx) >= 1, "LinearMappingIdentityStacking: input must have at least one dimension. ")
    expectedShape = (sx[1] * L.sizeStack, Base.tail(sx)...)
    @assert(size(ret) == expectedShape, "LinearMappingIdentityStacking: output array has incorrect dimensions. ")

    n1 = sx[1]

    if ndims(x) == 1
        # Vector case: avoid tuple indexing issues (and keep it fast).
        if add == false
            if L.coe == 1.0
                for k in 0:(L.sizeStack - 1)
                    @views ret[(k * n1 + 1):((k + 1) * n1)] .= x
                end
            else
                for k in 0:(L.sizeStack - 1)
                    @views ret[(k * n1 + 1):((k + 1) * n1)] .= L.coe .* x
                end
            end
        else
            if L.coe == 1.0
                for k in 0:(L.sizeStack - 1)
                    @views ret[(k * n1 + 1):((k + 1) * n1)] .+= x
                end
            else
                for k in 0:(L.sizeStack - 1)
                    @views ret[(k * n1 + 1):((k + 1) * n1)] .+= L.coe .* x
                end
            end
        end
    else
        # Multi-dimensional arrays: stack along first dimension, keep all other dimensions.
        tail = ntuple(_ -> Colon(), ndims(x) - 1)
        if add == false
            if L.coe == 1.0
                for k in 0:(L.sizeStack - 1)
                    r1 = (k * n1 + 1):((k + 1) * n1)
                    @views ret[r1, tail...] .= x
                end
            else
                for k in 0:(L.sizeStack - 1)
                    r1 = (k * n1 + 1):((k + 1) * n1)
                    @views ret[r1, tail...] .= L.coe .* x
                end
            end
        else
            if L.coe == 1.0
                for k in 0:(L.sizeStack - 1)
                    r1 = (k * n1 + 1):((k + 1) * n1)
                    @views ret[r1, tail...] .+= x
                end
            else
                for k in 0:(L.sizeStack - 1)
                    r1 = (k * n1 + 1):((k + 1) * n1)
                    @views ret[r1, tail...] .+= L.coe .* x
                end
            end
        end
    end
end 




"""
    (L::LinearMappingIdentityStacking)(x::NumericVariable)

Allocating forward operator. Returns the vertically stacked result `L(x)` as a new array.

# Returns
- If `x` is an array with size `(n1, n2, ..., nk)`, returns an array of size
  `(n1*sizeStack, n2, ..., nk)`.
- If `x` is a scalar number, returns a length-`sizeStack` vector with each entry equal to `coe*x`.
"""
function (L::LinearMappingIdentityStacking)(x::NumericVariable)
    if isa(x, Number)
        # Stacking of a scalar produces a length-sizeStack vector.
        return fill(L.coe == 1.0 ? x : (L.coe * x), L.sizeStack)
    end

    sx = size(x)
    expectedShape = (sx[1] * L.sizeStack, Base.tail(sx)...)

    if ndims(x) == 1
        ret = Vector{Float64}(undef, expectedShape[1])
        L(x, ret)
        return ret
    elseif ndims(x) == 2
        ret = Matrix{Float64}(undef, expectedShape...)
        L(x, ret)
        return ret
    else
        ret = Array{Float64}(undef, expectedShape)
        L(x, ret)
        return ret
    end
end


"""
    adjoint!(L::LinearMappingIdentityStacking, y::NumericVariable, ret::NumericVariable, add::Bool = false)

In-place adjoint operator for the stacking identity mapping.

If `L(x) = coe * [x; x; ...; x]` (stacked `sizeStack` times), then the adjoint is:

    L*(y) = coe * (y₁ + y₂ + ... + y_{sizeStack})

where each `yᵢ` is the corresponding block of `y` along the first dimension.

# Shape requirements
- If `size(y) == (n1*sizeStack, n2, ..., nk)`, then `size(ret)` must be `(n1, n2, ..., nk)`.
"""
function adjoint!(L::LinearMappingIdentityStacking, y::NumericVariable, ret::NumericVariable, add::Bool = false)  
    if isa(y, Number) || isa(ret, Number)
        error("LinearMappingIdentityStacking (adjoint!): in-place adjoint does not support scalar inputs/outputs")
    end

    sy = size(y)
    @assert(length(sy) >= 1, "LinearMappingIdentityStacking (adjoint!): input must have at least one dimension. ")
    @assert(sy[1] % L.sizeStack == 0, "LinearMappingIdentityStacking (adjoint!): input first dimension must be a multiple of sizeStack. ")

    n1 = sy[1] ÷ L.sizeStack
    expectedShape = (n1, Base.tail(sy)...)
    @assert(size(ret) == expectedShape, "LinearMappingIdentityStacking (adjoint!): output array has incorrect dimensions. ")

    if add == false
        ret .= 0.0
    end

    if ndims(y) == 1
        # Vector case
        if L.coe == 1.0
            for k in 0:(L.sizeStack - 1)
                @views ret .+= y[(k * n1 + 1):((k + 1) * n1)]
            end
        else
            for k in 0:(L.sizeStack - 1)
                @views ret .+= L.coe .* y[(k * n1 + 1):((k + 1) * n1)]
            end
        end
    else
        # Multi-dimensional arrays: sum along stacked blocks of first dimension
        tail = ntuple(_ -> Colon(), ndims(y) - 1)
        if L.coe == 1.0
            for k in 0:(L.sizeStack - 1)
                r1 = (k * n1 + 1):((k + 1) * n1)
                @views ret .+= y[r1, tail...]
            end
        else
            for k in 0:(L.sizeStack - 1)
                r1 = (k * n1 + 1):((k + 1) * n1)
                @views ret .+= L.coe .* y[r1, tail...]
            end
        end
    end
end


"""
    adjoint(L::LinearMappingIdentityStacking, y::NumericVariable)

Allocating adjoint operator. Returns `L*(y)` as a new array.

# Returns
- If `y` has size `(n1*sizeStack, n2, ..., nk)`, returns an array of size `(n1, n2, ..., nk)`.
"""
function adjoint(L::LinearMappingIdentityStacking, y::NumericVariable)
    if isa(y, Number)
        error("LinearMappingIdentityStacking (adjoint): input must be an array (stacked output); got scalar")
    end

    sy = size(y)
    @assert(length(sy) >= 1, "LinearMappingIdentityStacking (adjoint): input must have at least one dimension. ")
    @assert(sy[1] % L.sizeStack == 0, "LinearMappingIdentityStacking (adjoint): input first dimension must be a multiple of sizeStack. ")

    n1 = sy[1] ÷ L.sizeStack
    outShape = (n1, Base.tail(sy)...)
    ret = length(outShape) <= 2 ? spzeros(outShape) : zeros(outShape)
    adjoint!(L, y, ret)
    return ret
end


"""
    operatorNorm2(L::LinearMappingIdentityStacking) -> Float64

Compute the operator 2-norm (largest singular value) of the stacking identity mapping.

For `L(x) = coe * [x; ...; x]` stacked `sizeStack` times, the operator norm is:

    |coe| * sqrt(sizeStack)
"""
function operatorNorm2(L::LinearMappingIdentityStacking)
    # L(x) = coe * [x; x; ...; x] (sizeStack times) = (coe * (1 ⊗ I)) x
    # ||1 ⊗ I||_2 = ||1||_2 = sqrt(sizeStack)
    return (L.coe < 0.0 ? -L.coe : L.coe) * sqrt(L.sizeStack)
end 