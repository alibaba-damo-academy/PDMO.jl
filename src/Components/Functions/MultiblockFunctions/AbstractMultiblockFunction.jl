"""
    AbstractMultiblockFunction <: AbstractFunction

Abstract base type for functions that operate on multiple blocks simultaneously.
These functions take the form f(x₁, x₂, ..., xₙ) where each xᵢ is a NumericVariable.

All multiblock functions are assumed to be smooth and support:
- Function value evaluation
- Partial gradient computation (with respect to specific blocks)
- Full gradient computation (with respect to all blocks)

# Inheritance from AbstractFunction
This type inherits from `AbstractFunction` to benefit from the unified type system
while providing specialized multiblock functionality through its own API.
"""
abstract type AbstractMultiblockFunction <: AbstractFunction end

# =============================================================================
# Core Function Evaluation API
# =============================================================================

"""
    (f::AbstractMultiblockFunction)(x::Vector{NumericVariable})

Evaluate the multiblock function f at the point x = [x₁, x₂, ..., xₙ].

# Arguments
- `f::AbstractMultiblockFunction`: The multiblock function to evaluate
- `x::Vector{NumericVariable}`: Vector of block variables

# Returns
- `Float64`: Function value f(x₁, x₂, ..., xₙ)

# Implementation Note
This method must be implemented by all concrete subtypes.
"""
function (f::AbstractMultiblockFunction)(x::Vector{NumericVariable})
    error("Function evaluation not implemented for $(typeof(f))")
end

function (f::AbstractMultiblockFunction)(x::NumericVariable)
    error("Function evaluation not implemented for $(typeof(f))")
end

# =============================================================================
# Partial Gradient API
# =============================================================================

"""
    partialGradientOracle!(y::NumericVariable, f::AbstractMultiblockFunction, 
                          x::Vector{NumericVariable}, blockIndex::Int)

Compute the partial gradient ∇ₓᵢ f(x₁, x₂, ..., xₙ) in-place.

# Arguments
- `y::NumericVariable`: Pre-allocated output vector for the partial gradient
- `f::AbstractMultiblockFunction`: The multiblock function
- `x::Vector{NumericVariable}`: Current point [x₁, x₂, ..., xₙ]
- `blockIndex::Int`: Index of the block to compute partial gradient for (1-based)

# Implementation Note
This method must be implemented by all concrete subtypes.
The output `y` should have the same dimensions as `x[blockIndex]`.
"""
function partialGradientOracle!(y::NumericVariable, f::AbstractMultiblockFunction, 
                               x::Vector{NumericVariable}, blockIndex::Int)
    error("partialGradientOracle! not implemented for $(typeof(f))")
end

"""
    partialGradientOracle(f::AbstractMultiblockFunction, x::Vector{NumericVariable}, 
                         blockIndex::Int)

Compute the partial gradient ∇ₓᵢ f(x₁, x₂, ..., xₙ) with memory allocation.

# Arguments
- `f::AbstractMultiblockFunction`: The multiblock function
- `x::Vector{NumericVariable}`: Current point [x₁, x₂, ..., xₙ]
- `blockIndex::Int`: Index of the block to compute partial gradient for (1-based)

# Returns
- `Vector{Float64}`: Partial gradient ∇ₓᵢ f(x)

# Implementation Note
Default implementation allocates memory and calls `partialGradientOracle!`.
Can be overridden for efficiency if needed.
"""
function partialGradientOracle(f::AbstractMultiblockFunction, x::Vector{NumericVariable}, 
                              blockIndex::Int)
    @assert 1 <= blockIndex <= length(x) "blockIndex must be between 1 and $(length(x))"
    y = similar(x[blockIndex], Float64)
    partialGradientOracle!(y, f, x, blockIndex)
    return y
end






# =============================================================================
# Traits and Properties
# =============================================================================
"""
    isSmooth(::Type{<:AbstractMultiblockFunction}) -> Bool

All multiblock functions are smooth by default.

Multiblock functions are designed to support gradient-based optimization
algorithms and are assumed to have well-defined gradients everywhere.

# Returns
- `true`: All multiblock functions support gradient computation

# Override
Concrete subtypes can override this if they are not smooth, though this
would be unusual for multiblock functions.
"""
isSmooth(::Type{<:AbstractMultiblockFunction}) = true
isSmooth(::T) where T <: AbstractMultiblockFunction = isSmooth(T)

"""
    isConvex(f::AbstractMultiblockFunction) -> Bool
    isConvex(::Type{<:AbstractMultiblockFunction}) -> Bool

Trait checker for the convex property.

Returns `true` if the multiblock function is convex, `false` otherwise. 
Convex multiblock functions guarantee that any local minimum is a global minimum,
enabling the use of convex optimization algorithms with global optimality guarantees.

# Mathematical Background
A multiblock function f(x₁, x₂, ..., xₙ) is convex if for any λ ∈ [0,1] and 
any two points x, y in the domain:
```
f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)
```

# Implementation Note
The default type-level behavior is `false`. Concrete subtypes should override this
when convexity is known for that function type.
"""
isConvex(::Type{<:AbstractMultiblockFunction}) = false
isConvex(::T) where T <: AbstractMultiblockFunction = isConvex(T)

"""
    isSupportedByJuMP(f::AbstractMultiblockFunction) -> Bool
    isSupportedByJuMP(::Type{<:AbstractMultiblockFunction}) -> Bool

Trait checker for JuMP modeling support.

Returns `true` if the multiblock function can be modeled and solved using JuMP,
`false` otherwise. JuMP-supported functions can be converted to mathematical
programming formulations for use with commercial and open-source solvers.

# Implementation Requirements
If `isSupportedByJuMP(f) = true`, then the function should implement:
- `JuMPAddSmoothFunction(f, model, vars)`: Add function as objective/constraint to JuMP model
- `JuMPAddPartialBlockFunction(f, model, blockIdx, var, vals)`: Add partial block formulation

# Implementation Note
This trait must be implemented by all concrete subtypes. There is no default implementation
since JuMP support depends on whether the function can be expressed using JuMP's
supported mathematical operations (linear, quadratic, conic, nonlinear).
"""
isSupportedByJuMP(::Type{<:AbstractMultiblockFunction}) = error("isSupportedByJuMP not implemented for multiblock function type")
isSupportedByJuMP(f::AbstractMultiblockFunction) = isSupportedByJuMP(typeof(f))



"""
    getNumberOfBlocks(f::AbstractMultiblockFunction)

Get the number of blocks this function operates on.

# Arguments
- `f::AbstractMultiblockFunction`: The multiblock function

# Returns
- `Int`: Number of blocks

# Implementation Note
Must be implemented by concrete subtypes to return the fixed number of blocks.
"""
function getNumberOfBlocks(f::AbstractMultiblockFunction)
    error("getNumberOfBlocks not implemented for $(typeof(f))")
end

"""
    validateBlockDimensions(f::AbstractMultiblockFunction, x::Vector{NumericVariable})

Validate that the input block dimensions are compatible with the function.

# Arguments
- `f::AbstractMultiblockFunction`: The multiblock function
- `x::Vector{NumericVariable}`: Input variables to validate

# Implementation Note
Default implementation does no validation. Override in concrete types to add
dimension checks specific to the function.
"""
function validateBlockDimensions(f::AbstractMultiblockFunction, x::Vector{NumericVariable})
    # Default: no validation
    return nothing
end

# =============================================================================
# Gradient Oracle APIs - Multiple Signatures
# =============================================================================

"""
    gradientOracle!(grad::Vector{NumericVariable}, f::AbstractMultiblockFunction, x::Vector{NumericVariable})

Compute the full gradient of the multiblock function using multiblock format.

This is the primary gradient computation method for multiblock functions.

# Arguments
- `grad::Vector{NumericVariable}`: Pre-allocated output vector for gradients (one per block)
- `f::AbstractMultiblockFunction`: The multiblock function
- `x::Vector{NumericVariable}`: Input variables [x₁, x₂, ..., xₙ]

# Implementation Note
This method must be implemented by all concrete subtypes.
The output `grad` should have the same length as `x`, with `grad[i]` having the same dimensions as `x[i]`.
"""
function gradientOracle!(grad::Vector{NumericVariable}, f::AbstractMultiblockFunction, x::Vector{NumericVariable})
    error("gradientOracle! not implemented for $(typeof(f))")
end

"""
    gradientOracle!(grad::NumericVariable, f::AbstractMultiblockFunction, x::NumericVariable)

Compute the full gradient of the multiblock function using concatenated format.

This provides `AbstractFunction` interface compatibility by working with concatenated vectors.

# Arguments
- `grad::NumericVariable`: Pre-allocated output for concatenated gradient
- `f::AbstractMultiblockFunction`: The multiblock function
- `x::NumericVariable`: Concatenated input vector [x₁; x₂; ...; xₙ]

# Implementation Note
This method must be implemented by concrete subtypes to specify how to split
the concatenated vector `x` into blocks and concatenate the gradient blocks.
"""
function gradientOracle!(grad::NumericVariable, f::AbstractMultiblockFunction, x::NumericVariable)
    error("Concatenated gradientOracle! not implemented for $(typeof(f)). Need to implement concatenated gradient computation.")
end

"""
    gradientOracle(f::AbstractMultiblockFunction, x::Vector{NumericVariable}) -> Vector{NumericVariable}

Compute and return the full gradient using multiblock format.

# Arguments
- `f::AbstractMultiblockFunction`: The multiblock function
- `x::Vector{NumericVariable}`: Input variables [x₁, x₂, ..., xₙ]

# Returns
- `Vector{NumericVariable}`: Gradient blocks [∇₁f, ∇₂f, ..., ∇ₙf]
"""
function gradientOracle(f::AbstractMultiblockFunction, x::Vector{NumericVariable})
    grad = NumericVariable[similar(xi, Float64) for xi in x]
    gradientOracle!(grad, f, x)
    return grad
end

"""
    gradientOracle(f::AbstractMultiblockFunction, x::NumericVariable) -> NumericVariable

Compute and return the full gradient using concatenated format.

# Arguments
- `f::AbstractMultiblockFunction`: The multiblock function
- `x::NumericVariable`: Concatenated input vector [x₁; x₂; ...; xₙ]

# Returns
- `NumericVariable`: Concatenated gradient vector [∇₁f; ∇₂f; ...; ∇ₙf]
"""
function gradientOracle(f::AbstractMultiblockFunction, x::NumericVariable)
    grad = similar(x, Float64)
    gradientOracle!(grad, f, x)
    return grad
end
