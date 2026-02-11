"""
    AbstractMapping

This module defines the abstract interface for mappings in the optimization framework.

Mappings represent linear operators that transform variables from one space to another.
They are a fundamental component used in formulating and solving optimization problems.

# Interface Functions
All concrete mapping types should implement:
- Forward operator: `(L::ConcreteMapping)(x, ret, add=false)` - In-place application
- Forward operator: `(L::ConcreteMapping)(x)` - Out-of-place application
- Adjoint operator: `adjoint!(L, y, ret, add=false)` - In-place adjoint application
- Adjoint operator: `adjoint(L, y)` - Out-of-place adjoint application
- `createAdjointMapping(L)` - Creates a new mapping that represents the adjoint
- `operatorNorm2(L)` - Computes the operator norm (largest singular value)

# Available Mappings
- `NullMapping`: A placeholder mapping that does nothing
- `LinearMappingIdentity`: A scaled identity mapping
- `LinearMappingExtraction`: Extracts a subset of elements from input array
- `LinearMappingMatrix`: Matrix-based linear mapping

# Examples
```julia
# Create an identity mapping with coefficient 2.0
L = LinearMappingIdentity(2.0)

# Apply the mapping to a vector
x = [1.0, 2.0, 3.0]
y = L(x)  # Returns [2.0, 4.0, 6.0]

# Create the adjoint mapping
L_adj = createAdjointMapping(L)
```
"""
abstract type AbstractMapping end 

"""
    (L::AbstractMapping)(x::NumericVariable) -> NumericVariable

Apply the mapping to input `x` and return the result as a new variable.

This is the out-of-place forward operator that all concrete mapping types must implement.
The function computes y = L(x) where L is the linear mapping.

# Arguments
- `x::NumericVariable`: Input variable (scalar, vector, matrix, or higher-dimensional array)

# Returns
- `NumericVariable`: The result of applying the mapping to `x`

# Implementation Notes
This is an abstract interface function that must be implemented by all concrete mapping types.
The default implementation throws an error indicating the mapping type is not implemented.

# Examples
```julia
# For a concrete mapping implementation
L = LinearMappingIdentity(2.0)
x = [1.0, 2.0, 3.0]
y = L(x)  # Returns [2.0, 4.0, 6.0]
```
"""
function (L::AbstractMapping)(x::NumericVariable)
    error("$(typeof(L)) is not implemented")
end

"""
    (L::AbstractMapping)(x::NumericVariable, ret::NumericVariable, add::Bool = false)

Apply the mapping to input `x` and store the result in `ret`.

This is the in-place forward operator that all concrete mapping types must implement.
The function computes ret = L(x) or ret += L(x) depending on the `add` parameter.

# Arguments
- `x::NumericVariable`: Input variable (scalar, vector, matrix, or higher-dimensional array)
- `ret::NumericVariable`: Pre-allocated output variable to store the result
- `add::Bool`: If `true`, adds the result to `ret`; if `false`, overwrites `ret`. Default is `false`.

# Implementation Notes
This is an abstract interface function that must be implemented by all concrete mapping types.
The default implementation throws an error indicating the mapping type is not implemented.

Concrete implementations should:
- Verify that input and output dimensions are compatible
- Handle the `add` parameter correctly (overwrite vs accumulate)
- Optimize for performance when possible (e.g., avoid allocation)

# Examples
```julia
# For a concrete mapping implementation
L = LinearMappingIdentity(2.0)
x = [1.0, 2.0, 3.0]
ret = zeros(3)
L(x, ret)  # ret now contains [2.0, 4.0, 6.0]
```
"""
function (L::AbstractMapping)(x::NumericVariable, ret::NumericVariable, add::Bool = false)
    error("$(typeof(L)) is not implemented")
end 

"""
    adjoint!(L::AbstractMapping, y::NumericVariable, ret::NumericVariable, add::Bool = false)

Apply the adjoint of the mapping to input `y` and store the result in `ret`.

This is the in-place adjoint operator that all concrete mapping types must implement.
The function computes ret = L*(y) or ret += L*(y) depending on the `add` parameter,
where L* denotes the adjoint (transpose) of the linear mapping L.

# Arguments
- `y::NumericVariable`: Input variable in the range space of the mapping
- `ret::NumericVariable`: Pre-allocated output variable in the domain space
- `add::Bool`: If `true`, adds the result to `ret`; if `false`, overwrites `ret`. Default is `false`.

# Mathematical Background
For a linear mapping L: X → Y, the adjoint L*: Y → X satisfies:
⟨L(x), y⟩_Y = ⟨x, L*(y)⟩_X for all x ∈ X, y ∈ Y

# Implementation Notes
This is an abstract interface function that must be implemented by all concrete mapping types.
The default implementation throws an error indicating the mapping type is not implemented.

Concrete implementations should:
- Verify that input and output dimensions are compatible
- Handle the `add` parameter correctly (overwrite vs accumulate)
- Implement the mathematically correct adjoint operation

# Examples
```julia
# For a concrete mapping implementation
L = LinearMappingMatrix(A)  # where A is a matrix
y = [1.0, 2.0]  # in range space
ret = zeros(size(A, 2))  # in domain space
adjoint!(L, y, ret)  # ret = A' * y
```
"""
function adjoint!(L::AbstractMapping, y::NumericVariable, ret::NumericVariable, add::Bool = false)
    error("$(typeof(L)) is not implemented")
end

"""
    adjoint(L::AbstractMapping, y::NumericVariable) -> NumericVariable

Apply the adjoint of the mapping to input `y` and return the result as a new variable.

This is the out-of-place adjoint operator that all concrete mapping types must implement.
The function computes and returns L*(y) where L* denotes the adjoint (transpose) of the linear mapping L.

# Arguments
- `y::NumericVariable`: Input variable in the range space of the mapping

# Returns
- `NumericVariable`: The result of applying the adjoint mapping to `y`

# Mathematical Background
For a linear mapping L: X → Y, the adjoint L*: Y → X satisfies:
⟨L(x), y⟩_Y = ⟨x, L*(y)⟩_X for all x ∈ X, y ∈ Y

# Implementation Notes
This is an abstract interface function that must be implemented by all concrete mapping types.
The default implementation throws an error indicating the mapping type is not implemented.

Most concrete implementations will:
1. Allocate an appropriate output variable
2. Delegate to the in-place version `adjoint!(L, y, ret)`
3. Return the result

# Examples
```julia
# For a concrete mapping implementation
L = LinearMappingMatrix(A)  # where A is a matrix
y = [1.0, 2.0]  # in range space
x = adjoint(L, y)  # Returns A' * y
```
"""
function adjoint(L::AbstractMapping, y::NumericVariable)
    error("$(typeof(L)) is not implemented")
end

"""
    operatorNorm2(L::AbstractMapping)

Compute the operator norm  of the mapping.

# Returns
- `Float64`: The operator norm of the mapping
"""
function operatorNorm2(L::AbstractMapping)
    error("$(typeof(L)) is not implemented")
end


""" 
    NullMapping <: AbstractMapping 

A null mapping that does nothing. Placeholder in SolverData. 
"""
struct NullMapping <: AbstractMapping end 

# include specific mappings here 
include("LinearMappingIdentity.jl")
include("LinearMappingExtraction.jl")
include("LinearMappingMatrix.jl")
include("LinearMappingIdentityStacking.jl")
include("LinearMappingStacking.jl")

""" 
    adjoint(mapping1::AbstractMapping, mapping2::AbstractMapping)

Returns the adjoint of the product of two mappings, which is equivalent to the composition
of their individual adjoints applied in reverse order.

This function implements the mathematical identity that for mappings A and B:
(A ∘ B)* = B* ∘ A*

where * denotes the adjoint operator and ∘ denotes composition.

The function handles various combinations of mapping types and optimizes for specific
mapping combinations to avoid creating unnecessary intermediate mappings.

# Arguments
- `mapping1::AbstractMapping`: The first mapping in the composition (applied second).
- `mapping2::AbstractMapping`: The second mapping in the composition (applied first).

# Returns
- An `AbstractMapping` representing the adjoint of the composition.
- Returns `NullMapping()` if the adjoint cannot be computed.

# Special Cases
- For identity mappings with coefficient 1.0, the other mapping's adjoint is returned.
- For matrix mappings, the adjoint matrices are multiplied directly.
- For extraction mappings with identical dimensions and indices, their coefficients are multiplied.
- For combinations of identity and matrix mappings, optimized matrix operations are used.
"""
function adjoint(mapping1::AbstractMapping, mapping2::AbstractMapping)
    if isa(mapping1, LinearMappingMatrix) && isa(mapping2, LinearMappingMatrix)
        return LinearMappingMatrix(mapping1.A' * mapping2.A)
    end

    if isa(mapping1, LinearMappingIdentity) && isa(mapping2, LinearMappingIdentity)
        return LinearMappingIdentity(mapping1.coe * mapping2.coe)
    end 
    
    if isa(mapping1, LinearMappingExtraction) && isa(mapping2, LinearMappingExtraction) && 
       mapping1.dim == mapping2.dim && 
       mapping1.indexStart == mapping2.indexStart && 
       mapping1.indexEnd == mapping2.indexEnd
        return LinearMappingExtraction(mapping1.dim, 
            mapping1.coe * mapping2.coe, 
            mapping1.indexStart, 
            mapping1.indexEnd)
    end 

    if isa(mapping1, LinearMappingIdentity) && isa(mapping2, LinearMappingMatrix)
        return LinearMappingMatrix(mapping1.coe * mapping2.A)
    end 
    
    if isa(mapping1, LinearMappingMatrix) && isa(mapping2, LinearMappingIdentity)
        return LinearMappingMatrix(mapping2.coe * mapping1.A')
    end 
    
    if isa(mapping1, LinearMappingIdentity) && mapping1.coe == 1.0 
        return mapping2 
    end 

    if (isa(mapping2, LinearMappingIdentity) && mapping2.coe == 1.0)
        try 
            return createAdjointMapping(mapping1)
        catch # in case the adjoint is not defined
            return NullMapping()
        end 
    end 

    return NullMapping()
end 
