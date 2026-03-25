"""
    QuadraticMultiblockFunction

A quadratic multiblock function of the form:
f(x₁, x₂, ..., xₙ) = xᵀ Q x + qᵀ x + r

where x = [x₁; x₂; ...; xₙ] is the concatenated vector of all blocks,
Q is a (typically symmetric) quadratic coefficient matrix, q is a linear coefficient
vector, and r is a scalar constant.

# Fields
- `Q::AbstractMatrix{Float64}`: Quadratic coefficient matrix (n×n where n is total dimension)
- `q::Vector{Float64}`: Linear coefficient vector (length n)
- `r::Float64`: Constant term
- `blockDims::Vector{Int}`: Dimensions of each block [dim₁, dim₂, ..., dimₙ]
- `buffer::Vector{Float64}`: Pre-allocated buffer for concatenated vector (for efficiency)
- `blockIndices::Vector{UnitRange{Int}}`: Precomputed mapping from block index to indices in concatenated vector

# Constructor Behavior
- If `Q` is not symmetric (within tolerance), it is symmetrized as `(Q + Q')/2`.
- Positive semidefiniteness is not explicitly enforced by the constructor.

# Mathematical Form
The function evaluates to:
```
f(x₁, x₂, ..., xₙ) = [x₁; x₂; ...; xₙ]ᵀ Q [x₁; x₂; ...; xₙ] + qᵀ [x₁; x₂; ...; xₙ] + r
```

# Gradient Structure
- Partial gradient with respect to block i: ∇ₓᵢ f(x) = 2 * Q[blockIdx, :] * x + q[blockIdx]
- Full gradient: ∇f(x) = 2 * Q * x + q

# Constructor
```julia
QuadraticMultiblockFunction(Q, q, r, blockDims)
```

# Examples
```julia
# Two blocks of dimensions 2 and 3
blockDims = [2, 3]
Q = [1.0 0.5 0.1 0.0 0.0;
     0.5 2.0 0.0 0.1 0.0;
     0.1 0.0 1.5 0.2 0.1;
     0.0 0.1 0.2 1.0 0.0;
     0.0 0.0 0.1 0.0 2.0]
q = [1.0, -1.0, 0.5, -0.5, 1.0]
r = 0.0

f = QuadraticMultiblockFunction(Q, q, r, blockDims)

# Evaluate at point x = [x₁, x₂] where x₁ ∈ ℝ² and x₂ ∈ ℝ³
x₁ = [1.0, 2.0]
x₂ = [0.5, 1.0, -0.5]
value = f([x₁, x₂])
```
"""
mutable struct QuadraticMultiblockFunction <: AbstractMultiblockFunction
    Q::AbstractMatrix{Float64}
    q::Vector{Float64}
    r::Float64
    blockDims::Vector{Int}
    buffer::Vector{Float64}
    blockIndices::Vector{UnitRange{Int}}  # Precomputed mapping: blockIndex -> indices in concatenated vector
    function QuadraticMultiblockFunction(Q::AbstractMatrix{Float64}, 
                                       q::Vector{Float64}, 
                                       r::Float64, 
                                       blockDims::Vector{Int})
        # Validation
        totalDim = sum(blockDims)
        @assert size(Q, 1) == size(Q, 2) "Q must be square"
        @assert size(Q, 1) == totalDim "Q dimensions must match total block dimensions"
        @assert length(q) == totalDim "q length must match total block dimensions"
        @assert all(blockDims .> 0) "All block dimensions must be positive"
        
        # Check if Q is symmetric (within numerical tolerance)
        if isapprox(Q, Q', atol=1e-12) == false 
            @warn "Q is not symmetric; symmetrizing by taking (Q + Q')/2"
            Q = (Q + Q') / 2
        end
        
        # Initialize buffer for concatenated vector
        totalDim = sum(blockDims)
        buffer = zeros(totalDim)
        
        # Precompute block indices mapping
        blockIndices = Vector{UnitRange{Int}}(undef, length(blockDims))
        startIdx = 1
        for i in 1:length(blockDims)
            endIdx = startIdx + blockDims[i] - 1
            blockIndices[i] = startIdx:endIdx
            startIdx = endIdx + 1
        end
        
        new(Q, q, r, blockDims, buffer, blockIndices)
    end
end

# =============================================================================
# Utility Functions
# =============================================================================

"""
    _copyBlocksToBuffer!(buffer::Vector{Float64}, x::Vector{NumericVariable}, blockIndices::Vector{UnitRange{Int}})

Copy block variables into the pre-allocated buffer efficiently using copyto!.

# Arguments
- `buffer::Vector{Float64}`: Pre-allocated buffer to store concatenated vector
- `x::Vector{NumericVariable}`: Vector of block variables
- `blockIndices::Vector{UnitRange{Int}}`: Precomputed block index ranges

# Implementation Note
Uses copyto! for each block which is much more efficient than element-by-element copying.
"""
function _copyBlocksToBuffer!(buffer::Vector{Float64}, x::Vector{NumericVariable}, blockIndices::Vector{UnitRange{Int}})
    for i in 1:length(x)
        @assert isa(x[i], Number) == false "QuadraticMultiblockFunction: scalar blocks are not supported; use length-1 arrays"
        xi_vec = vec(x[i])  # Ensure it's a vector
        blockIdx = blockIndices[i]
        @assert length(xi_vec) == length(blockIdx) "Block $i size mismatch: expected $(length(blockIdx)), got $(length(xi_vec))"
        
        # Copy block into buffer using efficient copyto!
        copyto!(buffer, first(blockIdx), xi_vec, 1, length(xi_vec))
    end
end

"""
    _concatenateBlocks(x::Vector{NumericVariable}) -> Vector{Float64}

Concatenate block variables into a single vector.
This is kept for compatibility but _copyBlocksToBuffer! is more efficient.
"""
function _concatenateBlocks(x::Vector{NumericVariable})
    return vcat([vec(xi) for xi in x]...)
end



# =============================================================================
# Core Function Evaluation API Implementation
# =============================================================================

"""
    (f::QuadraticMultiblockFunction)(x::Vector{NumericVariable})

Evaluate the quadratic multiblock function at point x.

# Arguments
- `x::Vector{NumericVariable}`: Vector of block variables [x₁, x₂, ..., xₙ]

# Returns
- `Float64`: Function value f(x₁, x₂, ..., xₙ) = xᵀ Q x + qᵀ x + r
"""
function (f::QuadraticMultiblockFunction)(x::Vector{NumericVariable}, enableParallel::Bool=false)
    validateBlockDimensions(f, x)
    
    # Copy all blocks into the pre-allocated buffer
    _copyBlocksToBuffer!(f.buffer, x, f.blockIndices)
    
    # Evaluate quadratic function: xᵀ Q x + qᵀ x + r
    return dot(f.buffer, f.Q * f.buffer) + dot(f.q, f.buffer) + f.r
end

function (f::QuadraticMultiblockFunction)(x::Vector{Float64}, enableParallel::Bool=false)
    validateBlockDimensions(f, x)
    return dot(x, f.Q * x) + dot(f.q, x) + f.r
end

# =============================================================================
# Partial Gradient API Implementation  
# =============================================================================

"""
    partialGradientOracle!(y::NumericVariable, f::QuadraticMultiblockFunction, 
                          x::Vector{NumericVariable}, blockIndex::Int)

Compute the partial gradient ∇ₓᵢ f(x₁, x₂, ..., xₙ) in-place.

For a quadratic function f(x) = xᵀ Q x + qᵀ x + r, the partial gradient
with respect to block i is: ∇ₓᵢ f(x) = 2 * Q[blockIdx, :] * x + q[blockIdx]
where blockIdx are the indices corresponding to block i.

# Arguments
- `y::NumericVariable`: Pre-allocated output vector for the partial gradient
- `f::QuadraticMultiblockFunction`: The quadratic multiblock function
- `x::Vector{NumericVariable}`: Current point [x₁, x₂, ..., xₙ]
- `blockIndex::Int`: Index of the block to compute partial gradient for (1-based)
"""
function partialGradientOracle!(y::NumericVariable, f::QuadraticMultiblockFunction, 
                               x::Vector{NumericVariable}, blockIndex::Int)
    validateBlockDimensions(f, x)
    @assert 1 <= blockIndex <= length(x) "blockIndex must be between 1 and $(length(x))"
    @assert length(y) == length(x[blockIndex]) "Output vector y must have same dimensions as x[$blockIndex]"
    
    # Get precomputed indices for this block
    blockIdx = f.blockIndices[blockIndex]
    
    # Copy all blocks into the pre-allocated buffer
    _copyBlocksToBuffer!(f.buffer, x, f.blockIndices)
    
    # Compute partial gradient efficiently: 2 * Q[blockIdx, :] * x + q[blockIdx]
    # Only multiply the rows of Q corresponding to this block
    Q_block = @view f.Q[blockIdx, :]
    q_block = @view f.q[blockIdx]
    y .= 2 * Q_block * f.buffer + q_block
end

"""
    partialGradientOracle(f::QuadraticMultiblockFunction, x::Vector{NumericVariable}, blockIndex::Int)

Compute the partial gradient ∇ₓᵢ f(x₁, x₂, ..., xₙ) with memory allocation.

For a quadratic function f(x) = xᵀ Q x + qᵀ x + r, the partial gradient
with respect to block i is: ∇ₓᵢ f(x) = 2 * Q[blockIdx, :] * x + q[blockIdx]
where blockIdx are the indices corresponding to block i.

# Arguments
- `f::QuadraticMultiblockFunction`: The quadratic multiblock function
- `x::Vector{NumericVariable}`: Current point [x₁, x₂, ..., xₙ]
- `blockIndex::Int`: Index of the block to compute partial gradient for (1-based)

# Returns
- `Vector{Float64}`: Partial gradient ∇ₓᵢ f(x)

# Implementation Note
This method allocates memory and calls the in-place version `partialGradientOracle!`.
"""
function partialGradientOracle(f::QuadraticMultiblockFunction, x::Vector{NumericVariable}, blockIndex::Int)
    @assert 1 <= blockIndex <= length(x) "blockIndex must be between 1 and $(length(x))"
    
    # Allocate output vector with same dimensions as the target block
    y = similar(x[blockIndex], Float64)
    
    # Call the in-place version
    partialGradientOracle!(y, f, x, blockIndex)
    
    return y
end

# =============================================================================
# Full Gradient Oracle API Implementation - All 4 Signatures
# =============================================================================

"""
    gradientOracle!(grad::Vector{NumericVariable}, f::QuadraticMultiblockFunction, x::Vector{NumericVariable})

Compute the full gradient [∇ₓ₁ f(x), ∇ₓ₂ f(x), ..., ∇ₓₙ f(x)] in-place using multiblock format.

For a quadratic function f(x) = xᵀ Q x + qᵀ x + r, the full gradient is:
∇f(x) = 2 * Q * x + q

# Arguments
- `grad::Vector{NumericVariable}`: Pre-allocated output vector for gradients
- `f::QuadraticMultiblockFunction`: The quadratic multiblock function
- `x::Vector{NumericVariable}`: Current point [x₁, x₂, ..., xₙ]
"""
function gradientOracle!(grad::Vector{NumericVariable}, f::QuadraticMultiblockFunction, x::Vector{NumericVariable}, enableParallel::Bool=false )
    validateBlockDimensions(f, x)
    @assert length(grad) == length(x) "Output vector grad must have same length as input x"
    
    # Copy all blocks into the pre-allocated buffer
    _copyBlocksToBuffer!(f.buffer, x, f.blockIndices)
    
    # Compute full gradient: 2 * Q * x + q
    gradient_full = 2 * f.Q * f.buffer + f.q
    
    # Split gradient back into blocks
    for i in 1:length(x)
        blockIdx = f.blockIndices[i]
        @assert length(grad[i]) == length(x[i]) "Output block grad[$i] must have same dimensions as input block x[$i]"
        grad[i] .= gradient_full[blockIdx]
    end
end

"""
    gradientOracle(f::QuadraticMultiblockFunction, x::Vector{NumericVariable}) -> Vector{NumericVariable}

Compute and return the full gradient using multiblock format.

For a quadratic function f(x) = xᵀ Q x + qᵀ x + r, the full gradient is:
∇f(x) = 2 * Q * x + q

# Arguments
- `f::QuadraticMultiblockFunction`: The quadratic multiblock function
- `x::Vector{NumericVariable}`: Current point [x₁, x₂, ..., xₙ]

# Returns
- `Vector{NumericVariable}`: Full gradient vector [∇ₓ₁ f(x), ∇ₓ₂ f(x), ..., ∇ₓₙ f(x)]
"""
function gradientOracle(f::QuadraticMultiblockFunction, x::Vector{NumericVariable}, enableParallel::Bool=false)
    # Allocate output vectors with same dimensions as input blocks
    grad = NumericVariable[similar(xi, Float64) for xi in x]
    
    # Call the in-place version
    gradientOracle!(grad, f, x)
    
    return grad
end

"""
    gradientOracle!(grad::NumericVariable, f::QuadraticMultiblockFunction, x::NumericVariable)

Compute the full gradient using concatenated input/output format.

This provides `AbstractFunction` interface compatibility by working with concatenated vectors.
For f(x) = xᵀ Q x + qᵀ x + r, the gradient is ∇f(x) = 2 Q x + q.

# Arguments
- `grad::NumericVariable`: Pre-allocated output for concatenated gradient
- `f::QuadraticMultiblockFunction`: The quadratic multiblock function
- `x::NumericVariable`: Concatenated input vector [x₁; x₂; ...; xₙ]

# Mathematical Background
The concatenated gradient computation is more efficient than the block-based approach
for this signature since we can directly compute: ∇f(x) = 2 Q x + q.
"""
function gradientOracle!(grad::NumericVariable, f::QuadraticMultiblockFunction, x::NumericVariable, enableParallel::Bool=false)
    @assert isa(x, Number) == false "QuadraticMultiblockFunction: concatenated gradient API requires array input"
    @assert isa(grad, Number) == false "QuadraticMultiblockFunction: concatenated gradient API requires array output"
    @assert length(x) == sum(f.blockDims) "Input dimension mismatch: expected $(sum(f.blockDims)), got $(length(x))"
    @assert length(grad) == length(x) "Output dimension mismatch: expected $(length(x)), got $(length(grad))"
    # Direct computation using concatenated format: ∇f(x) = 2 Q x + q
    grad .= 2 * (f.Q * x) + f.q
end

"""
    gradientOracle(f::QuadraticMultiblockFunction, x::NumericVariable) -> NumericVariable

Compute and return the full gradient using concatenated format.

For f(x) = xᵀ Q x + qᵀ x + r, the gradient is ∇f(x) = 2 Q x + q.

# Arguments
- `f::QuadraticMultiblockFunction`: The quadratic multiblock function
- `x::NumericVariable`: Concatenated input vector [x₁; x₂; ...; xₙ]

# Returns
- `NumericVariable`: Concatenated gradient vector [∇₁f; ∇₂f; ...; ∇ₙf]
"""
function gradientOracle(f::QuadraticMultiblockFunction, x::NumericVariable, enableParallel::Bool=false)
    # Allocate output vector with same dimensions as input
    grad = similar(x, Float64)
    
    # Call the in-place version
    gradientOracle!(grad, f, x)
    
    return grad
end

# =============================================================================
# Traits and Properties Implementation
# =============================================================================

# Specify that QuadraticMultiblockFunction supports JuMP
# Note: isSmooth is inherited as true from AbstractMultiblockFunction
isSupportedByJuMP(::Type{QuadraticMultiblockFunction}) = true

"""
    getNumberOfBlocks(f::QuadraticMultiblockFunction)

Get the number of blocks this function operates on.

# Arguments
- `f::QuadraticMultiblockFunction`: The quadratic multiblock function

# Returns
- `Int`: Number of blocks
"""
function getNumberOfBlocks(f::QuadraticMultiblockFunction)
    return length(f.blockDims)
end

"""
    validateBlockDimensions(f::QuadraticMultiblockFunction, x::Vector{NumericVariable})

Validate that the input block dimensions are compatible with the function.

# Arguments
- `f::QuadraticMultiblockFunction`: The quadratic multiblock function
- `x::Vector{NumericVariable}`: Input variables to validate

# Throws
- `AssertionError`: If dimensions are incompatible
"""
function validateBlockDimensions(f::QuadraticMultiblockFunction, x::Vector{NumericVariable})
    @assert length(x) == length(f.blockDims) "Number of blocks mismatch: expected $(length(f.blockDims)), got $(length(x))"
    
    for i in 1:length(x)
        @assert isa(x[i], Number) == false "Block $i is scalar; scalar blocks are not supported. Use a length-1 array."
        @assert length(x[i]) == f.blockDims[i] "Block $i dimension mismatch: expected $(f.blockDims[i]), got $(length(x[i]))"
    end
end

function validateBlockDimensions(f::QuadraticMultiblockFunction, x::Vector{Float64})
    @assert length(x) == sum(f.blockDims) "Number of blocks mismatch: expected $(sum(f.blockDims)), got $(length(x))"
end


# =============================================================================
# JuMP Integration for Whole Multiblock Function
# =============================================================================

"""
    JuMPAddSmoothFunction(f::QuadraticMultiblockFunction, model::JuMP.Model, 
                         vars::Vector{Vector{JuMP.VariableRef}})

Add the quadratic multiblock function as an objective term to a JuMP model.

This creates a quadratic expression representing f(x₁, x₂, ..., xₙ) = xᵀ Q x + qᵀ x + r
where x is the concatenation of all block variables.

# Arguments
- `f::QuadraticMultiblockFunction`: The quadratic multiblock function
- `model::JuMP.Model`: The JuMP model to add the function to
- `vars::Vector{Vector{JuMP.VariableRef}}`: Vector of JuMP variable vectors for each block

# Returns
- `JuMP.QuadExpr`: Quadratic expression representing the function

# Example
```julia
# Create a 2-block quadratic function
blockDims = [2, 3]
Q = Matrix(I, 5, 5)  # Identity matrix
q = ones(5)
r = 0.0
f = QuadraticMultiblockFunction(Q, q, r, blockDims)

# Create JuMP model and variables
model = JuMP.Model()
vars = Vector{Vector{JuMP.VariableRef}}()
push!(vars, JuMP.@variable(model, [1:2]))  # Block 1
push!(vars, JuMP.@variable(model, [1:3]))  # Block 2

# Add function to model
obj_expr = JuMPAddSmoothFunction(f, model, vars)
JuMP.@objective(model, Min, obj_expr)
```
"""
function JuMPAddSmoothFunction(f::QuadraticMultiblockFunction, model::JuMP.Model, 
                              vars::Vector{Vector{JuMP.VariableRef}})
    
    # Validate that we have the right number of blocks
    @assert length(vars) == length(f.blockDims) "Number of variable blocks ($(length(vars))) doesn't match function blocks ($(length(f.blockDims)))"
    
    # Validate dimensions for each block
    for i in 1:length(vars)
        @assert length(vars[i]) == f.blockDims[i] "Block $i variable dimension ($(length(vars[i]))) doesn't match expected ($(f.blockDims[i]))"
    end
    
    # Concatenate all variables in order
    var_concat = vcat(vars...)
    
    # Validate total dimensions
    totalDim = sum(f.blockDims)
    @assert length(var_concat) == totalDim "Total variable dimension ($(length(var_concat))) doesn't match expected ($(totalDim))"
    
    # Create quadratic expression: xᵀ Q x + qᵀ x + r
    obj_expr = var_concat' * f.Q * var_concat + f.q' * var_concat + f.r
    
    return obj_expr
end

# =============================================================================
# JuMP Integration for Partial Block Modeling
# =============================================================================

"""
    JuMPAddPartialBlockFunction(f::QuadraticMultiblockFunction, model::JuMP.Model,
                               blockIdx::Int, var::Vector{JuMP.VariableRef}, vals::Vector{NumericVariable})

Create a JuMP objective expression for the quadratic multiblock function when all blocks except one are fixed.

This is useful for block coordinate descent algorithms where we optimize over one block
while keeping others fixed. The resulting function is quadratic in the free block variables.

# Arguments
- `f::QuadraticMultiblockFunction`: The original multiblock function
- `model::JuMP.Model`: The JuMP model (for consistency, though not used in this implementation)
- `blockIdx::Int`: Index of the block to optimize over (1-based)
- `var::Vector{JuMP.VariableRef}`: JuMP variables for the free block
- `vals::Vector{NumericVariable}`: Current values of all blocks (including the free block)

# Returns
- `JuMP.QuadExpr`: Quadratic expression in terms of the free block variables

# Mathematical Details
Given f(x₁, ..., xₙ) = xᵀ Q x + qᵀ x + r, when blocks except block k are fixed,
we get a quadratic function in xₖ:

g(xₖ) = xₖᵀ Qₖₖ xₖ + (2 * Qₖ,₋ₖ x₋ₖ + qₖ)ᵀ xₖ + constant

where:
- Qₖₖ is the block of Q corresponding to block k
- Qₖ,₋ₖ is the block of Q coupling block k with other blocks  
- x₋ₖ are the fixed values of other blocks
- qₖ is the portion of q corresponding to block k

# Example
```julia
# 2-block function with blocks of size [2, 3]
blockDims = [2, 3]
Q = rand(5, 5); Q = Q'Q  # Make positive semidefinite
q = rand(5)
r = 1.0
f = QuadraticMultiblockFunction(Q, q, r, blockDims)

# Current values of all blocks
vals = [[1.0, 2.0], [0.5, 1.0, -0.5]]

# Create variables for block 1 and get objective for optimizing over block 1
model = JuMP.Model()
var = JuMP.@variable(model, [1:2])
obj_expr = JuMPAddPartialBlockFunction(f, model, 1, var, vals)

JuMP.@objective(model, Min, obj_expr)
```
"""
function JuMPAddPartialBlockFunction(f::QuadraticMultiblockFunction, 
    model::JuMP.Model,
    blockIdx::Int, 
    var::Vector{JuMP.VariableRef}, 
    vals::Vector{NumericVariable})
    
    # Validate inputs
    @assert 1 <= blockIdx <= length(f.blockDims) "blockIdx must be between 1 and $(length(f.blockDims))"
    @assert length(vals) == length(f.blockDims) "vals must have $(length(f.blockDims)) blocks, got $(length(vals))"
    @assert length(var) == f.blockDims[blockIdx] "var must have $(f.blockDims[blockIdx]) variables for block $blockIdx, got $(length(var))"
    
    # Validate that each vals[i] is a vector (not a scalar)
    for i in 1:length(vals)
        @assert vals[i] isa AbstractArray "Block $i in vals must be a vector, got $(typeof(vals[i]))"
        @assert length(vals[i]) == f.blockDims[i] "Block $i in vals has dimension $(length(vals[i])), expected $(f.blockDims[i])"
    end
    
    # Get precomputed indices for the free block
    freeBlockIdx = f.blockIndices[blockIdx]
    
    # Reuse the existing buffer to store all current values
    # Copy all blocks from vals into the buffer efficiently
    for i in 1:length(vals)
        blockRange = f.blockIndices[i]
        copyto!(f.buffer, first(blockRange), vals[i], 1, length(vals[i]))
    end
    
    # Extract relevant matrices and vectors using views (no allocation)
    # Q_kk: quadratic terms within the free block
    Q_kk = @view f.Q[freeBlockIdx, freeBlockIdx]
    
    # Q_k_others: coupling terms between free block and fixed blocks
    otherIdx = setdiff(1:length(f.buffer), freeBlockIdx)
    Q_k_others = @view f.Q[freeBlockIdx, otherIdx]
    x_others = @view f.buffer[otherIdx]
    
    # q_k: linear terms for the free block
    q_k = @view f.q[freeBlockIdx]
    
    # Construct the quadratic expression for the free block:
    # g(x_k) = x_k' Q_kk x_k + (2 * Q_k_others * x_others + q_k)' x_k + constant
    
    # Compute effective linear term: 2 * Q_k_others * x_others + q_k
    # Note: This still allocates, but it's unavoidable for the JuMP expression
    effective_linear = 2 * Q_k_others * x_others + q_k
    
    # Compute constant term: x_others' Q_others_others x_others + q_others' x_others + r
    Q_others_others = @view f.Q[otherIdx, otherIdx]
    q_others = @view f.q[otherIdx]
    constant_term = dot(x_others, Q_others_others * x_others) + dot(q_others, x_others) + f.r
    
    # Create quadratic expression using matrix form: x_k' Q_kk x_k + effective_linear' x_k + constant
    obj_expr = var' * Q_kk * var + effective_linear' * var + constant_term
    
    return obj_expr
end
