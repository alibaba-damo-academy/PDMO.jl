"""
    AbstractFunction

Abstract base type for all functions in the Bipartization optimization framework.

This type serves as the foundation for implementing mathematical functions that can be used
in optimization algorithms. It defines a common interface for function evaluation, 
gradients, and proximal operators.

# Interface Requirements

To implement a new function type, you must:

1. **Define a concrete struct** that inherits from `AbstractFunction`
2. **Implement function evaluation** by overriding the call operator
3. **Implement trait methods** to specify function properties
4. **Implement oracles** for smooth/proximal functions as needed

# Core Interface Methods

## Function Evaluation
```julia
(f::YourFunction)(x::NumericVariable, enableParallel::Bool=false) -> Float64
```

## Trait Methods (override as needed)
```julia
isSmooth(::Type{YourFunction}) = true/false        # Has gradient?
isProximal(::Type{YourFunction}) = true/false      # Has proximal operator?
isConvex(::Type{YourFunction}) = true/false        # Is convex?
isSet(::Type{YourFunction}) = true/false           # Is indicator function?
isSupportedByJuMP(::Type{YourFunction}) = true/false  # Can be modeled in JuMP?
```

## Gradient Oracle (if isSmooth = true)
```julia
gradientOracle!(grad::NumericVariable, f::YourFunction, x::NumericVariable, enableParallel::Bool=false)
gradientOracle(f::YourFunction, x::NumericVariable, enableParallel::Bool=false)
```

## Proximal Oracle (if isProximal = true)
```julia
proximalOracle!(y::NumericVariable, f::YourFunction, x::NumericVariable, gamma::Float64=1.0, enableParallel::Bool=false)
proximalOracle(f::YourFunction, x::NumericVariable, gamma::Float64=1.0, enableParallel::Bool=false)
```

## JuMP Oracles (if isSupportedByJuMP = true)
```julia
# For proximable functions - create variables, constraints, and their objective terms
JuMPAddProximableFunction(g::YourFunction, model::JuMP.Model, dim::Int) -> (var, aux, objTerm)

# For smooth functions - add objective terms using existing variables  
JuMPAddSmoothFunction(f::YourFunction, model::JuMP.Model, var::Vector{<:JuMP.VariableRef}, aux::Union{Vector{<:JuMP.VariableRef}, Nothing}) -> objTerm
```

# Implementation Example

```julia
# Define a simple quadratic function: f(x) = ||x||²
struct SimpleQuadratic <: AbstractFunction
    # No fields needed for this simple case
end

# Specify traits
isSmooth(::Type{SimpleQuadratic}) = true
isConvex(::Type{SimpleQuadratic}) = true
isProximal(::Type{SimpleQuadratic}) = true
isSupportedByJuMP(::Type{SimpleQuadratic}) = true

# Function evaluation
function (f::SimpleQuadratic)(x::NumericVariable, enableParallel::Bool=false)
    return sum(x.^2)
end

# Gradient oracle
function gradientOracle!(grad::NumericVariable, f::SimpleQuadratic, x::NumericVariable, enableParallel::Bool=false)
    grad .= 2.0 .* x
end

# Proximal oracle
function proximalOracle!(y::NumericVariable, f::SimpleQuadratic, x::NumericVariable, gamma::Float64=1.0, enableParallel::Bool=false)
    y .= x ./ (1.0 + 2.0 * gamma)
end

# JuMP proximable function (for when used as a domain/constraint)
function JuMPAddProximableFunction(g::SimpleQuadratic, model::JuMP.Model, dim::Int)
    var = @variable(model, [k = 1:dim])  # No constraints needed for quadratic domain
    aux = nothing  # No auxiliary variables needed
    objTerm = nothing  # Quadratic domain doesn't contribute to objective
    return var, aux, objTerm
end

# JuMP smooth function (for when used as objective)
function JuMPAddSmoothFunction(f::SimpleQuadratic, model::JuMP.Model, var::Vector{<:JuMP.VariableRef}, aux::Union{Vector{<:JuMP.VariableRef}, Nothing})
    obj_expr = JuMP.QuadExpr()
    JuMP.add_to_expression!(obj_expr, sum(v^2 for v in var))
    return obj_expr
end
```

# Built-in Function Types

The framework provides many built-in function types:

- **Basic Functions**: `Zero`, `AffineFunction`, `QuadraticFunction`
- **Norms**: `ElementwiseL1Norm`, `FrobeniusNormSquare`, `MatrixNuclearNorm`
- **Indicators**: `IndicatorBox`, `IndicatorBallL2`, `IndicatorSOC`, `IndicatorPSD`
- **Wrappers**: `WrapperScalingTranslationFunction`, `WrapperScalarInputFunction`
- **User-defined**: `UserDefinedSmoothFunction`, `UserDefinedProximalFunction`

# Design Principles

1. **Modularity**: Each function type is self-contained
2. **Efficiency**: Support for in-place operations and parallel computation
3. **Flexibility**: Trait-based design allows algorithm selection
4. **Extensibility**: Easy to add new function types
5. **Type Safety**: Strong typing helps catch errors early
"""
abstract type AbstractFunction end 

""" 
    NumericVariable

Type alias for function arguments in the Bipartization framework.

This type represents the domain of functions, encompassing both scalar and array inputs.
It provides a unified interface for working with optimization variables of different
dimensions and structures.

# Definition
```julia
NumericVariable = Union{Float64, AbstractArray{Float64, N} where N}
```

# Design Rationale
- **Flexibility**: Supports various optimization variable structures
- **Type Safety**: Ensures only numeric types are used
- **Performance**: Avoids unnecessary type conversions
- **Compatibility**: Works with Julia's array ecosystem
"""
const NumericVariable = Union{Float64, AbstractArray{Float64, N} where N}
""" 
    isProximal(f::AbstractFunction) -> Bool
    isProximal(::Type{<:AbstractFunction}) -> Bool

Trait checker for the proximal operator capability.

Returns `true` if the function has an implemented proximal operator, `false` otherwise.
Functions with proximal operators can be used in proximal algorithms like ADMM,
forward-backward splitting, and Douglas-Rachford splitting.

# Mathematical Background
The proximal operator of a function f is defined as:
```
prox_{γf}(x) = argmin_z { f(z) + (1/(2γ))||z - x||² }
```

# Implementation Requirements
If `isProximal(f) = true`, then the function must implement:
- `proximalOracle!(y, f, x, γ)`: In-place proximal operator
- `proximalOracle(f, x, γ)`: Allocating proximal operator (has default implementation)

"""
isProximal(::Type{<:AbstractFunction}) = false
isProximal(::T) where T <: AbstractFunction = isProximal(T)
""" 
    isSmooth(f::AbstractFunction) -> Bool
    isSmooth(::Type{<:AbstractFunction}) -> Bool

Trait checker for the smooth (differentiable) property.

Returns `true` if the function is smooth (differentiable) everywhere in its domain,
`false` otherwise. Smooth functions can be used in gradient-based optimization
algorithms like gradient descent, Newton's method, and quasi-Newton methods.

# Mathematical Background
A function f is smooth if it is continuously differentiable, meaning:
- The gradient ∇f(x) exists at every point x in the domain
- The gradient is continuous

# Implementation Requirements
If `isSmooth(f) = true`, then the function must implement:
- `gradientOracle!(grad, f, x)`: In-place gradient computation
- `gradientOracle(f, x)`: Allocating gradient computation (has default implementation)
"""
isSmooth(::Type{<:AbstractFunction}) = false
isSmooth(::T) where T <: AbstractFunction = isSmooth(T)

""" 
    isConvex(f::AbstractFunction) -> Bool
    isConvex(::Type{<:AbstractFunction}) -> Bool

Trait checker for the convex property.

Returns `true` if the function is convex, `false` otherwise. Convex functions
have many desirable properties for optimization, including the guarantee that
any local minimum is also a global minimum.
"""
isConvex(::Type{<:AbstractFunction}) = false 
isConvex(::T) where T <: AbstractFunction = isConvex(T)

""" 
    isSet(f::AbstractFunction) -> Bool
    isSet(::Type{<:AbstractFunction}) -> Bool

Trait checker for indicator functions of sets.

Returns `true` if the function is the indicator function of a set, `false` otherwise.
Indicator functions are fundamental in constrained optimization and represent
constraints as functions that are 0 inside the constraint set and ∞ outside.
"""
isSet(T::Type{<:AbstractFunction}) = false
isSet(::T) where T <: AbstractFunction = isSet(T)

""" 
    isSupportedByJuMP(f::AbstractFunction) -> Bool
    isSupportedByJuMP(::Type{<:AbstractFunction}) -> Bool

Trait checker for JuMP modeling support.

Returns `true` if the function can be modeled in JuMP, `false` otherwise.
Functions that support JuMP modeling can be converted to JuMP constraints
(for proximable functions) or objective terms (for smooth functions).

# Implementation Requirements
Functions with `isSupportedByJuMP(f) = true` must implement:
- `JuMPAddProximableFunction` (for proximable functions): Create variables, constraints, and their objective terms
- `JuMPAddSmoothFunction` (for smooth functions): Add objective terms using existing variables

"""
isSupportedByJuMP(::Type{<:AbstractFunction}) = false
isSupportedByJuMP(::T) where T <: AbstractFunction = isSupportedByJuMP(T)

"""
    (f::AbstractFunction)(x::NumericVariable, enableParallel::Bool=false) -> Float64

Function call operator for evaluating the function at a given point.

This is the primary interface for computing function values. Every concrete function 
type must implement this method to define how the function is evaluated.

# Arguments
- `f::AbstractFunction`: The function object to evaluate
- `x::NumericVariable`: The point at which to evaluate the function
- `enableParallel::Bool=false`: Whether to enable parallel computation (when supported)

# Implementation Requirements
Every concrete function type must override this method. The implementation should:
1. Validate input dimensions and types
2. Compute the function value efficiently
3. Handle edge cases appropriately
4. Optionally support parallel computation

# Common Patterns
- **Indicator functions**: Return 0.0 if x is in the constraint set, +∞ otherwise
- **Norm functions**: Return ||x||_p for various norms
- **Regularization**: Return penalty terms for optimization
- **Composite functions**: Combine multiple function evaluations

# Performance Notes
- Implementations should minimize memory allocations
- Use in-place operations when possible
- Consider vectorization for array operations
- Parallel computation should be used for large-scale problems

# Error Handling
- Throw descriptive errors for dimension mismatches
- Use appropriate tolerances for numerical comparisons
- Handle special cases (NaN, Inf, empty inputs) gracefully
"""
function (f::AbstractFunction)(x::NumericVariable, enableParallel::Bool=false)
    error("Function evaluation not implemented for $(typeof(f))")
end

"""
    proximalOracle!(y::NumericVariable, f::AbstractFunction, x::NumericVariable, gamma::Float64=1.0, enableParallel::Bool=false)

In-place computation of the proximal operator for function f.

The proximal operator is a fundamental concept in convex optimization and is defined as:
```
prox_{γf}(x) = argmin_z { f(z) + (1/(2γ))||z - x||² }
```

This function computes the result in-place, storing it in the pre-allocated output `y`.

# Arguments
- `y::NumericVariable`: Pre-allocated output buffer (modified in-place)
- `f::AbstractFunction`: The function for which to compute the proximal operator
- `x::NumericVariable`: The input point
- `gamma::Float64=1.0`: The proximal parameter γ > 0
- `enableParallel::Bool=false`: Whether to enable parallel computation
"""
function proximalOracle!(y::NumericVariable, f::AbstractFunction, x::NumericVariable, gamma::Float64=1.0, enableParallel::Bool=false)
    error("proximalOracle! not implemented for $(typeof(f))")
end

"""
    proximalOracle(f::AbstractFunction, x::NumericVariable, gamma::Float64=1.0, enableParallel::Bool=false) -> NumericVariable

Computes the proximal operator for function f, returning a new array.

This is the allocating version of the proximal operator that creates and returns a new
array containing the result. For performance-critical applications, consider using
the in-place version `proximalOracle!` instead.

# Arguments
- `f::AbstractFunction`: The function for which to compute the proximal operator
- `x::NumericVariable`: The input point
- `gamma::Float64=1.0`: The proximal parameter γ > 0
- `enableParallel::Bool=false`: Whether to enable parallel computation
"""
function proximalOracle(f::AbstractFunction, x::NumericVariable, gamma::Float64=1.0, enableParallel::Bool=false)
    y = similar(x)
    proximalOracle!(y, f, x, gamma, enableParallel)
    return y
end

"""
    gradientOracle!(grad::NumericVariable, f::AbstractFunction, x::NumericVariable, enableParallel::Bool=false)

In-place computation of the gradient of function f at point x.

This function computes the gradient (or subgradient for non-smooth functions) and stores
the result in the pre-allocated output buffer `grad`.

# Arguments
- `grad::NumericVariable`: Pre-allocated output buffer for gradient (modified in-place)
- `f::AbstractFunction`: The function for which to compute the gradient
- `x::NumericVariable`: The point at which to evaluate the gradient
- `enableParallel::Bool=false`: Whether to enable parallel computation
"""
function gradientOracle!(grad::NumericVariable, f::AbstractFunction, x::NumericVariable, enableParallel::Bool=false)
    error("gradientOracle! not implemented for $(typeof(f))")
end
"""
    gradientOracle(f::AbstractFunction, x::NumericVariable, enableParallel::Bool=false) -> NumericVariable

Computes the gradient of function f at point x, returning a new array.

This is the allocating version of the gradient computation that creates and returns a new
array containing the gradient. For performance-critical applications, consider using
the in-place version `gradientOracle!` instead.

# Arguments
- `f::AbstractFunction`: The function for which to compute the gradient
- `x::NumericVariable`: The point at which to evaluate the gradient
- `enableParallel::Bool=false`: Whether to enable parallel computation
"""
function gradientOracle(f::AbstractFunction, x::NumericVariable, enableParallel::Bool=false)
    grad = similar(x)
    gradientOracle!(grad, f, x, enableParallel)
    return grad
end

"""
    JuMPAddProximableFunction(g::AbstractFunction, model::JuMP.Model, var::Vector{<:JuMP.VariableRef}) -> Union{JuMP.AbstractJuMPScalar, Nothing}

Add a proximable function to a JuMP model using existing variables.

This function handles proximable functions that define domains, constraints, or regularizers.
It adds the necessary constraints to the model and returns any objective contribution
the function makes (e.g., L1 norm penalty terms).

# Arguments
- `g::AbstractFunction`: The proximable function to add to the model
- `model::JuMP.Model`: The JuMP model to modify (constraints added in-place)
- `var::Vector{<:JuMP.VariableRef}`: Primary variables to use for this function

# Implementation Requirements
Functions with `isSupportedByJuMP(g) = true` and `isProximal(g) = true` must implement this method.
The implementation should:
1. Add constraint equations to the model using the provided variables
2. Create auxiliary variables internally if needed (e.g., for L1 norm reformulations)
3. Return objective contribution if the function appears in the objective (e.g., regularizers)

# Indicator box constraint: add bounds, no objective contribution
function JuMPAddProximableFunction(g::IndicatorBox, model::JuMP.Model, var::Vector{<:JuMP.VariableRef})
    for k in 1:length(var)
        JuMP.set_lower_bound(var[k], g.lower)
        JuMP.set_upper_bound(var[k], g.upper)
    end
    return nothing  # Box constraints don't contribute to objective
end

# L1 norm: create auxiliary variables and constraints, return objective
function JuMPAddProximableFunction(g::ElementwiseL1Norm, model::JuMP.Model, var::Vector{<:JuMP.VariableRef})
    dim = length(var)
    aux = JuMP.@variable(model, [k = 1:dim], lower_bound = 0.0)
    
    # Add constraints: |x_k| ≤ aux_k
    for k in 1:dim
        JuMP.@constraint(model, var[k] <= aux[k])
        JuMP.@constraint(model, -var[k] <= aux[k])
    end
    
    # Create objective contribution: coefficient * sum(aux)
    objTerm = JuMP.AffExpr(0.0)
    for aux_var in aux
        JuMP.add_to_expression!(objTerm, g.coefficient * aux_var)
    end
    
    return objTerm
end
```

# Common Patterns
- **Indicator functions**: Set variable bounds/constraints, return nothing
- **Conic constraints**: Add second-order cone constraints, no objective contribution
- **Norm functions**: Create auxiliary variables for epigraph reformulations, return penalty objTerm
- **Regularizers**: Both constrain variables and contribute to objective
"""
function JuMPAddProximableFunction(g::AbstractFunction, model::JuMP.Model, var::Vector{<:JuMP.VariableRef})
    error("JuMPAddProximableFunction not implemented for $(typeof(g))")
end
"""
    JuMPAddSmoothFunction(f::AbstractFunction, model::JuMP.Model, var::Vector{<:JuMP.VariableRef}) 
        -> Union{JuMP.AffExpr, JuMP.QuadExpr, NonlinearExpr, Nothing}

Add a smooth function's objective contribution to a JuMP model using existing variables.

This function takes a smooth function and creates its objective expression using the provided
variables. It handles pure objective terms and may create auxiliary variables internally
if needed for the expression.

# Arguments
- `f::AbstractFunction`: The smooth function to add to the objective
- `model::JuMP.Model`: The JuMP model (may be modified to add auxiliary variables)
- `var::Vector{<:JuMP.VariableRef}`: Primary variables to use in the expression

# Implementation Requirements
Functions with `isSupportedByJuMP(f) = true` and `isSmooth(f) = true` must implement this method.
The implementation should:
1. Create appropriate expression type based on function complexity
2. Use provided variables as primary variables
3. Create auxiliary variables internally if needed for complex expressions
4. Return the expression ready for use in objective

# Example: Quadratic function: return QuadExpr
function JuMPAddSmoothFunction(f::QuadraticFunction, model::JuMP.Model, var::Vector{<:JuMP.VariableRef})
    obj_expr = JuMP.QuadExpr()
    # Add quadratic terms: x'Qx (note: no 1/2 factor)
    JuMP.add_to_expression!(obj_expr, var' * f.Q * var)
    JuMP.add_to_expression!(obj_expr, f.q' * var + f.r)
    return obj_expr
end
"""
function JuMPAddSmoothFunction(f::AbstractFunction, model::JuMP.Model, var::Vector{<:JuMP.VariableRef})
    error("JuMPAddSmoothFunction not implemented for $(typeof(f))")
end


# Include and export all function implementations
include("AffineFunction.jl")
include("ComponentwiseExponentialFunction.jl")
include("ElementwiseL1Norm.jl")
include("FrobeniusNormSquare.jl")
include("IndicatorBallL2.jl")
include("IndicatorBinary.jl")
include("IndicatorBox.jl")
include("IndicatorHyperplane.jl")
include("IndicatorLinearSubspace.jl")
include("IndicatorMixedInteger.jl")
include("IndicatorNonnegativeOrthant.jl")
include("IndicatorPSD.jl")
include("IndicatorRotatedSOC.jl")
include("IndicatorSOC.jl")
include("IndicatorSumOfNVariables.jl")
include("MatrixNuclearNorm.jl")
include("QuadraticFunction.jl")
include("StackingProximalFunctions.jl")
include("UserDefinedProximalFunction.jl")
include("UserDefinedSmoothFunction.jl")
include("WeightedMatrixL1Norm.jl")
include("WrapperScalarInputFunction.jl")
include("WrapperScalingTranslationFunction.jl")
include("Zero.jl")

# MultiblockFunctions
include("MultiblockFunctions/AbstractMultiblockFunction.jl")
include("MultiblockFunctions/QuadraticMultiblockFunction.jl")

# Utility functions 
include("AbstractFunctionUtil.jl")

# flow of function transformation  
# f --vectorize--> WrapperScalarInputFunction(f) --scaling--> WrapperScalingTranslationFunction(WrapperScalarInputFunction(f)) 
# g --vectorize--> WrapperScalarInputFunction(g) --scaling--> WrapperScalingTranslationFunction(WrapperScalarInputFunction(g)) 


# flow of function modeling in JuMP 
# JuMPAddSmoothFunction of WrapperScalingTranslationFunction 
# -> JuMPAddSmoothFunction of WrapperScalarInputFunction 
# -> JuMPAddSmoothFunction of f

# JuMPAddProximableFunction of WrapperScalingTranslationFunction -> 
# JuMPAddProximableFunction of WrapperScalarInputFunction -> 
# JuMPAddProximableFunction of g 