""" 
    MatrixNuclearNorm(b, rows, cols)

Weighted nuclear norm of a matrix with component-wise weights, defined as:
    ||W||_{b,*} = ∑ᵢ bᵢσᵢ(W)
    
where σᵢ(W) are the singular values of W, b ∈ ℝᵐ₊ is a vector of positive weights 
(m = min(rows,cols)), and the sum goes up to the number of singular values.

# Arguments
- `b::Vector{Float64}`: Vector of positive weights, one per singular value
- `rows::Int64`: Number of rows in the matrix
- `cols::Int64`: Number of columns in the matrix

# Properties
- Proximal: Yes
- Proximal Operator: Component-wise soft thresholding of singular values
- Convex: No. Only when all entries of b are equal

Note: The number of weights in b must equal min(rows, cols), as this is the 
maximum possible number of non-zero singular values.
"""
struct MatrixNuclearNorm <: AbstractFunction 
    b::Vector{Float64}
    numberRows::Int64
    numberColumns::Int64

    function MatrixNuclearNorm(b::Vector{Float64}, rows::Int64, cols::Int64)
        if any(b .<= 0.0)
            error("MatrixNuclearNorm: all weights must be positive")
        end
        if rows <= 0 || cols <= 0
            error("MatrixNuclearNorm: dimensions must be positive")
        end
        if length(b) != min(rows, cols)
            error("MatrixNuclearNorm: number of weights must equal min(rows, cols)")
        end
        return new(b, rows, cols)
    end
end

# isConvex(::Type{MatrixNuclearNorm}) = true # the function is convex iff b has same entries
isProximal(::Type{MatrixNuclearNorm}) = true 

"""
Evaluate the weighted nuclear norm: ∑ᵢ bᵢσᵢ(x)
"""
function (f::MatrixNuclearNorm)(x::NumericVariable, enableParallel::Bool=false)
    @assert(size(x) == (f.numberRows, f.numberColumns), "MatrixNuclearNorm: input dimension mismatch")
    # NOTE: In Julia >= 1.11, `svd(::SparseMatrixCSC)` may error (it routes to an `svd!` method
    # that is not defined for sparse matrices with the required keyword arguments).
    # Convert sparse input to a dense matrix before calling SVD.
    F = issparse(x) ? svd(Matrix(x)) : svd(x)
    return sum(f.b .* F.S)
end

"""
Proximal operator for weighted nuclear norm.
The solution is given by U * diag(max(0, σᵢ - γbᵢ)) * Vᵀ
where U, σᵢ, V come from the SVD of x.
"""
function proximalOracle!(y::NumericVariable, f::MatrixNuclearNorm, x::NumericVariable, gamma::Float64 = 1.0, enableParallel::Bool=false)
    @assert(size(x) == (f.numberRows, f.numberColumns), "MatrixNuclearNorm: input dimension mismatch")
    @assert(size(y) == (f.numberRows, f.numberColumns), "MatrixNuclearNorm: output dimension mismatch")
    @assert(gamma > 0.0, "MatrixNuclearNorm: gamma must be positive")  
    
    # See note in function evaluation: handle sparse inputs robustly.
    F = issparse(x) ? svd(Matrix(x)) : svd(x)
    S_prox = max.(0, F.S .- gamma .* f.b)
    Ydense = F.U * Diagonal(S_prox) * F.Vt

    # Support both dense and sparse output buffers.
    if issparse(y)
        copyto!(y, sparse(Ydense))
    else
        y .= Ydense
    end
end     

"""
Non-mutating version of the proximal operator
"""
function proximalOracle(f::MatrixNuclearNorm, x::NumericVariable, gamma::Float64 = 1.0, enableParallel::Bool=false)
    # If x is sparse, `similar(x)` is sparse, but the proximal result is generally dense.
    # Allocate a dense buffer in that case.
    y = issparse(x) ? Matrix{Float64}(undef, size(x)...) : similar(x)
    proximalOracle!(y, f, x, gamma, enableParallel)
    return y
end









