"""
    WeightedMatrixL1Norm(A, inNonnegativeOrthant=false)

Weighted L1 norm of a matrix, defined as:
    ||A ⊙ x||₁ = ∑ᵢⱼ A_{i,j}|x_{i,j}|

When inNonnegativeOrthant is true, the function becomes:
    ||A ⊙ x||₁ + indicator(x ≥ 0)

where A is a sparse matrix of non-negative weights and ⊙ denotes element-wise multiplication.

# Arguments
- `A::SparseMatrixCSC{Float64, Int64}`: Matrix of non-negative weights
- `inNonnegativeOrthant::Bool=false`: If true, restricts the domain to the non-negative orthant

# Properties
- Convex: Yes
- Proximal: Yes
- Proximal Operator: 
  - If inNonnegativeOrthant=false: Element-wise soft thresholding with weights A
  - If inNonnegativeOrthant=true: Project onto non-negative orthant after soft thresholding with weights A

# Example
```julia
A = sparse([1.0 2.0; 3.0 4.0])
f = WeightedMatrixL1Norm(A)            # Standard weighted L1 norm
g = WeightedMatrixL1Norm(A, true)      # Weighted L1 norm restricted to non-negative orthant
x = [1.0 -1.0; 2.0 -2.0]
val = f(x)  # Computes weighted L1 norm
```
"""
struct WeightedMatrixL1Norm <: AbstractFunction 
    A::SparseMatrixCSC{Float64, Int64}
    numberRows::Int64
    numberColumns::Int64
    inNonnegativeOrthant::Bool
    
    function WeightedMatrixL1Norm(A::SparseMatrixCSC{Float64, Int64}; inNonnegativeOrthant::Bool=false)
        if any(A.nzval .< -FeasTolerance)
            error("WeightedMatrixL1Norm: A must be a sparse matrix with non-negative values.")
        end 
        return new(A, size(A, 1), size(A, 2), inNonnegativeOrthant)
    end 
end 

isConvex(::Type{WeightedMatrixL1Norm}) = true 
isProximal(::Type{WeightedMatrixL1Norm}) = true 

function (f::WeightedMatrixL1Norm)(x::NumericVariable, enableParallel::Bool=false)
    @assert(size(x) == (f.numberRows, f.numberColumns), "WeightedMatrixL1Norm: input dimension mismatch.")
    
    if f.inNonnegativeOrthant && any(x .< -FeasTolerance * 100.0)
        return Inf
    end
       
    return sum(abs.(f.A .* x))
end 


function proximalOracle!(y::NumericVariable, f::WeightedMatrixL1Norm, x::NumericVariable, gamma::Float64 = 1.0, enableParallel::Bool=false)
    @assert(size(x) == (f.numberRows, f.numberColumns), "WeightedMatrixL1Norm: input dimension mismatch.")
    @assert(size(y) == (f.numberRows, f.numberColumns), "WeightedMatrixL1Norm: output dimension mismatch.")
    @assert(gamma > 0.0, "WeightedMatrixL1Norm: gamma must be positive.")
    
    if f.inNonnegativeOrthant
        if enableParallel && length(x) > 1000  # some threshold
            Threads.@threads for i in eachindex(x)
                @inbounds y[i] = max(0, x[i] - gamma * f.A[i])
            end
        else
            y .= max.(0, x .- gamma .* f.A)
        end
    else
        # Standard soft thresholding
        if enableParallel && length(x) > 1000  # some threshold
            Threads.@threads for i in eachindex(x)
                @inbounds y[i] = sign(x[i]) * max(0, abs(x[i]) - gamma * f.A[i])
            end
        else
            y .= sign.(x) .* max.(0, abs.(x) .- gamma .* f.A) 
        end
    end
end 

function proximalOracle(f::WeightedMatrixL1Norm, x::NumericVariable, gamma::Float64 = 1.0, enableParallel::Bool=false)
    y = similar(x)
    proximalOracle!(y, f, x, gamma, enableParallel)
    return y
end 

