"""
Given input x of size (2 × numberRows × numberColumns), representing the horizontal and vertical finite differences of some matrix,
the function computes the value 
    
    f(x) = coefficient * sum_{i=1}^{numberRows} sum_{j=1}^{numberColumns} sqrt(x[1,i,j]^2 + x[2,i,j]^2)

In particular, if we reshape x as a (numberRows * numberColumns) × 2 matrix, then this is the L21 norm of the reshaped matrix.

"""
struct FiniteDifferenceMatrixNorm <: AbstractFunction 
    numberRows::Int64
    numberColumns::Int64
    coefficient::Float64
    buffer::Vector{Float64}  # Buffer for row-wise sums

    function FiniteDifferenceMatrixNorm(numberRows::Int64, numberColumns::Int64, coefficient::Float64=1.0)
        numberRows > 0 || throw(ArgumentError("numberRows must be positive"))
        numberColumns > 0 || throw(ArgumentError("numberColumns must be positive"))
        coefficient > 0.0 || throw(ArgumentError("coefficient must be positive"))
        
        new(numberRows, numberColumns, coefficient, zeros(numberRows))
    end 
end 

# override traits
isProximal(f::Type{<:FiniteDifferenceMatrixNorm}) = true 
isConvex(f::Type{<:FiniteDifferenceMatrixNorm}) = true

# Stable norm computation helper function
function stableNorm(a::Float64, b::Float64)
    # Handle zero case
    if a == 0.0 && b == 0.0
        return 0.0
    end
    
    # Get absolute values
    abs_a = abs(a)
    abs_b = abs(b)
    
    # Put larger value first
    if abs_a < abs_b
        abs_a, abs_b = abs_b, abs_a
    end
    
    # Use stable formula
    ratio = abs_b / abs_a
    return abs_a * sqrt(1.0 + ratio * ratio)
end

# function value
function (f::FiniteDifferenceMatrixNorm)(x::AbstractArray{Float64,3}, enableParallel::Bool=false)
    # Input validation
    size(x) == (2, f.numberRows, f.numberColumns) || 
        throw(DimensionMismatch("Input must be size (2, numberRows, numberColumns)"))
    
    # Check for NaN/Inf
    all(isfinite, x) || throw(ArgumentError("Input contains NaN or Inf"))

    # Reset buffer
    fill!(f.buffer, 0.0)

    if enableParallel
        # Parallel implementation
        @threads for i in 1:f.numberRows
            rowSum = 0.0
            @inbounds for j in 1:f.numberColumns
                rowSum += stableNorm(x[1,i,j], x[2,i,j])
            end
            f.buffer[i] = rowSum
        end
        return f.coefficient * sum(f.buffer)
    else 
        @inbounds for i in 1:f.numberRows
            for j in 1:f.numberColumns
                f.buffer[i] += stableNorm(x[1,i,j], x[2,i,j])
            end
        end
        return f.coefficient * sum(f.buffer)
    end 
end 

function proximalOracle!(y::AbstractArray{Float64,3}, 
    f::FiniteDifferenceMatrixNorm, 
    x::AbstractArray{Float64,3}, 
    gamma::Float64=1.0, 
    enableParallel::Bool=false)

    # Input validation
    size(x) == (2, f.numberRows, f.numberColumns) || 
        throw(DimensionMismatch("Input must be size (2, numberRows, numberColumns)"))
    size(y) == size(x) || 
        throw(DimensionMismatch("Output must have same size as input"))
    gamma > 0 || throw(ArgumentError("gamma must be positive"))

    lambda = gamma * f.coefficient
    
    if enableParallel
        @threads for i in 1:f.numberRows
            @inbounds @simd for j in 1:f.numberColumns
                # Compute norm using stable helper function
                norm_ij = stableNorm(x[1,i,j], x[2,i,j])
                
                # Compute scaling factor with protection against division by zero
                scalar = max(0.0, 1.0 - lambda / max(norm_ij, ZeroTolerance))
                
                # Apply scaling
                y[1,i,j] = scalar * x[1,i,j]
                y[2,i,j] = scalar * x[2,i,j]
            end 
        end 
    else 
        @inbounds for i in 1:f.numberRows
            @simd for j in 1:f.numberColumns
                # Same computation as above
                norm_ij = stableNorm(x[1,i,j], x[2,i,j])
                
                scalar = max(0.0, 1.0 - lambda / max(norm_ij, ZeroTolerance))
                
                y[1,i,j] = scalar * x[1,i,j]
                y[2,i,j] = scalar * x[2,i,j]
            end 
        end 
    end 
end 

function proximalOracle(f::FiniteDifferenceMatrixNorm, 
    x::AbstractArray{Float64,3}, 
    gamma::Float64=1.0, 
    enableParallel::Bool=false)
    y = Array{Float64,3}(undef, size(x)...)
    proximalOracle!(y, f, x, gamma, enableParallel)
    return y
end 

function testFiniteDifferenceMatrixNorm()
    # Define dimensions and coefficient
    numberRows = 5
    numberColumns = 4
    coefficient = 2.0

    # Create an instance of the FiniteDifferenceMatrixNorm operator
    f = FiniteDifferenceMatrixNorm(numberRows, numberColumns, coefficient)

    # Generate a random input x of size (2, numberRows, numberColumns)
    x = randn(2, numberRows, numberColumns)

    # --- Test the function value ---
    # Manually compute the L21 norm: sum over (i,j) sqrt(x[1,i,j]^2 + x[2,i,j]^2)
    manual_sum = 0.0
    for i in 1:numberRows, j in 1:numberColumns
        a = x[1, i, j]
        b = x[2, i, j]
        manual_sum += sqrt(a^2 + b^2)
    end
    expected_value = coefficient * manual_sum

    # Compute the value using our operator
    computed_value = f(x, false)
    @test isapprox(computed_value, expected_value, atol=1e-8)

    # --- Test the proximal operator ---
    gamma = 1.5  # Choose a positive gamma value
    y = proximalOracle(f, x, gamma, false)

    # For each pixel, check that the proximal mapping performs:
    # y[i,j] = scalar * x[i,j]  with scalar = max(0, 1 - (gamma * coefficient) / max(norm(x[i,j]), ZeroTolerance))
    for i in 1:numberRows, j in 1:numberColumns
        a = x[1, i, j]
        b = x[2, i, j]
        norm_ij = sqrt(a^2 + b^2)
        expected_scalar = max(0.0, 1.0 - (gamma * coefficient) / max(norm_ij, ZeroTolerance))
        @test isapprox(y[1, i, j], expected_scalar * a, atol=1e-8)
        @test isapprox(y[2, i, j], expected_scalar * b, atol=1e-8)
    end

    println("FiniteDifferenceMatrixNorm: all tests passed.")
end
