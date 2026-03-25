"""
    UserDefinedSmoothFunction(func::Function, gradientFunc::Function, convex::Bool=true)

Wrapper for user-defined smooth functions that have a gradient.

This type lets users inject custom smooth functions by providing both value and
gradient callbacks.

# Arguments
- `func::Function`: Function evaluation callback `f(x) -> Real`.
- `gradientFunc::Function`: Gradient callback `x -> grad` with output shape
  compatible with `x`.
- `convex::Bool=true`: Whether the function is convex

# Properties
- **Smooth**: Yes, by definition
- **Convex**: Instance-specific (`isConvex(f)` returns the stored `convex` flag)
- **Proximal**: No, user-defined functions typically don't have proximal oracles

# Requirements
- `func(x)` must return a Float64 value
- `gradientFunc(x)` must return a gradient with the same shape as `x`
- Both functions must be consistent with the mathematical definition
- For correctness, consider using automatic differentiation tools to compute gradients

# Example
```julia
func = x -> x[1]^2 + 2 * x[2]^2 + x[1] * x[2]
gradientFunc = x -> [2 * x[1] + x[2], 4 * x[2] + x[1]]
f = UserDefinedSmoothFunction(func, gradientFunc, true)
```
"""
struct UserDefinedSmoothFunction <: AbstractFunction
    func::Function
    gradientFunc::Function
    convex::Bool

    function UserDefinedSmoothFunction(func::Function, gradientFunc::Function, convex::Bool=true)
        new(func, gradientFunc, convex)
    end
end

# Override traits for UserDefinedSmoothFunction
isSmooth(::Type{UserDefinedSmoothFunction}) = true
isConvex(f::UserDefinedSmoothFunction) = f.convex
isConvex(::Type{UserDefinedSmoothFunction}) = false  # Default to false, check instance
isProximal(::Type{UserDefinedSmoothFunction}) = false  # User-defined functions typically don't have proximal oracles

# Function evaluation
function (f::UserDefinedSmoothFunction)(x::NumericVariable, enableParallel::Bool=false)
    return f.func(x)
end

# Gradient oracle - in-place version
function gradientOracle!(grad::NumericVariable, f::UserDefinedSmoothFunction, x::NumericVariable, enableParallel::Bool=false)
    if isa(x, Number)
        error("UserDefinedSmoothFunction: gradient oracle does not support in-place operations for scalar inputs.")
    end
    grad .= f.gradientFunc(x)
end

# Gradient oracle - allocating version
function gradientOracle(f::UserDefinedSmoothFunction, x::NumericVariable, enableParallel::Bool=false)
    if isa(x, Number)
        return f.gradientFunc(x)
    else
        grad = similar(x)
        gradientOracle!(grad, f, x, enableParallel)
        return grad
    end
end

# Example usage:
#
# # Simple Quadratic Function
# # f(x) = x₁² + 2x₂² + x₁x₂
# func = x -> x[1]^2 + 2*x[2]^2 + x[1]*x[2]
# gradientFunc = x -> [2*x[1] + x[2], 4*x[2] + x[1]]
# # Non-convex Function
# # f(x) = sin(x₁) + cos(x₂)
# func = x -> sin(x[1]) + cos(x[2])
# gradientFunc = x -> [cos(x[1]), -sin(x[2])]
# # Integration with Bipartization
# # In your optimization problems
# block_x = BlockVariable(xID)
# addBlockVariable!(nlp, block_x) 