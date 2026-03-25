"""
    LinearMappingIdentity(coe::Float64=1.0)

Constructs a linear mapping that applies a scaled identity transformation.

The mapping is defined as:

    y = coe * x

where both the input and output have the same dimension.

# Arguments
- `coe::Float64`: The scaling coefficient (must be nonzero). Defaults to 1.0.
"""
struct LinearMappingIdentity <: AbstractMapping
    coe::Float64

    function LinearMappingIdentity(coe::Float64=1.0)
        @assert(coe != 0.0, "LinearMappingIdentity: coefficient must be non-zero. ")
        new(coe)
    end
end 

"""
    (L::LinearMappingIdentity)(x::NumericVariable, ret::NumericVariable, add::Bool = false)

Apply the scaled identity mapping to input `x` and store the result in `ret`.

# Arguments
- `x::NumericVariable`: Input array or number to which the mapping is applied.
- `ret::NumericVariable`: Output array to store the result, must have the same size as `x`.
- `add::Bool`: If `true`, adds the result to `ret` instead of overwriting it. Default is `false`.

# Implementation Details
The function scales `x` by the coefficient `coe` and either assigns to or adds to `ret` 
depending on the `add` parameter. For improved efficiency, when `coe` is 1.0, the function 
avoids the multiplication operation.

# Errors
- Throws an error if `x` is a scalar number as in-place operations are not supported for scalars.
"""
function (L::LinearMappingIdentity)(x::NumericVariable, ret::NumericVariable, add::Bool = false)
    @assert(size(x) == size(ret), "LinearMappingIdentity: input and output must have the same size. ")
    if (isa(x, Number))
        error("LinearMappingIdentity: forward mapping does not support in-place operations for scalar inputs")
    end 
    if add == false 
        ret .= (L.coe == 1.0 ? x : L.coe * x)
    else 
        ret .+= (L.coe == 1.0 ? x : L.coe * x)
    end 
end

"""
    (L::LinearMappingIdentity)(x::NumericVariable)

Apply the scaled identity mapping to input `x` and return the result as a new value.

# Arguments
- `x::NumericVariable`: Input array or number to which the mapping is applied.

# Returns
- The input `x` scaled by the coefficient `coe`.
- For array inputs, returns a fresh array (non-aliasing).
- For scalar inputs, returns the scalar value directly.

# Implementation Details
For `coe == 1.0`, array inputs are copied to preserve allocating-API semantics
(callers can safely mutate the returned value without mutating the input).
"""
function (L::LinearMappingIdentity)(x::NumericVariable)
    if isa(x, Number)
        return L.coe == 1.0 ? x : L.coe * x
    end
    return L.coe == 1.0 ? copy(x) : L.coe * x
end

"""
    adjoint!(L::LinearMappingIdentity, y::NumericVariable, ret::NumericVariable, add::Bool = false)

Apply the adjoint of the scaled identity mapping to input `y` and store the result in `ret`.

# Arguments
- `y::NumericVariable`: Input array or number to which the adjoint mapping is applied.
- `ret::NumericVariable`: Output array to store the result, must have the same size as `y`.
- `add::Bool`: If `true`, adds the result to `ret` instead of overwriting it. Default is `false`.

# Implementation Details
For a scaled identity mapping, the adjoint operation is identical to the forward operation,
so this function simply delegates to the forward operator.
"""
function adjoint!(L::LinearMappingIdentity, y::NumericVariable, ret::NumericVariable, add::Bool = false)  
    L(y, ret, add)
end

"""
    adjoint(L::LinearMappingIdentity, y::NumericVariable)

Apply the adjoint of the scaled identity mapping to input `y` and return the result as a new value.

# Arguments
- `y::NumericVariable`: Input array or number to which the adjoint mapping is applied.

# Returns
- The input `y` scaled by the coefficient `coe`.
- For array inputs, returns a fresh array (non-aliasing).
- For scalar inputs, returns the scalar value directly.

# Implementation Details
For a scaled identity mapping, the adjoint operation is identical to the forward operation,
and follows the same non-aliasing behavior for array outputs.
"""
function adjoint(L::LinearMappingIdentity, y::NumericVariable)
    return L(y)
end 

"""
    createAdjointMapping(L::LinearMappingIdentity)

Create a new mapping that represents the adjoint of the given scaled identity mapping.

# Arguments
- `L::LinearMappingIdentity`: The mapping for which to create an adjoint.

# Returns
- A new `LinearMappingIdentity` with the same coefficient as the input mapping.

# Implementation Details
For a scaled identity mapping, the adjoint mapping is identical to the original mapping,
so this function returns a new instance with the same coefficient.
"""
function createAdjointMapping(L::LinearMappingIdentity)
    return LinearMappingIdentity(L.coe)
end 

"""
    operatorNorm2(L::LinearMappingIdentity)

Compute the operator norm (largest singular value) of the scaled identity mapping.

# Arguments
- `L::LinearMappingIdentity`: The mapping for which to compute the operator norm.

# Returns
- The absolute value of the scaling coefficient `coe`.

# Implementation Details
For a scaled identity mapping, the operator norm is simply the absolute value of the
scaling coefficient, since all singular values are equal to this value.
"""
function operatorNorm2(L::LinearMappingIdentity)
    return L.coe < 0.0 ? -L.coe : L.coe 
end 