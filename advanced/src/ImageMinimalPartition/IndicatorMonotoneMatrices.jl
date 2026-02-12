"""
    IndicatorMonotoneMatrices(numberMatrices::Int64, numberRows::Int64, numberColumns::Int64, 
                             lb::Float64=0.0, ub::Float64=1.0)

Indicator function for monotone matrices constraint:
{Θ ∈ ℝ^{n×N₁×N₂} | θ₁ ≥ θ₂ ≥ ... ≥ θₙ}
where θᵢ = Θ[i,:,:] is the i-th N₁×N₂ matrix and lb ≤ θᵢ ≤ ub elementwise.

# Arguments
- `numberMatrices::Int64`: Number of matrices n
- `numberRows::Int64`: Number of rows N₁
- `numberColumns::Int64`: Number of columns N₂
- `lb::Float64=0.0`: Lower bound for all elements
- `ub::Float64=1.0`: Upper bound for all elements

# Returns
f(Θ) = 0 if Θ satisfies constraints, +∞ otherwise
"""
mutable struct IndicatorMonotoneMatrices <: AbstractFunction 
    numberMatrices::Int64
    numberRows::Int64
    numberColumns::Int64
    lb::Float64
    ub::Float64
    buffer::Vector{Vector{Float64}}  # Buffer for isotonic regression

    function IndicatorMonotoneMatrices(numberMatrices::Int64, 
        numberRows::Int64, 
        numberColumns::Int64, 
        lb::Float64 = 0.0, 
        ub::Float64 = 1.0)
        
        # Input validation
        @assert numberMatrices >= 2 "Need at least 2 matrices"
        @assert numberRows > 0 "Number of rows must be positive"
        @assert numberColumns > 0 "Number of columns must be positive"
        @assert lb <= ub "Lower bound must not exceed upper bound"

        # Initialize buffers
        numberElements = numberRows * numberColumns
        buffer = [zeros(numberMatrices) for _ in 1:numberElements]

        new(numberMatrices, numberRows, numberColumns, lb, ub, buffer)
    end 
end 

# Define function traits
isProximal(::Type{<:IndicatorMonotoneMatrices}) = true 
isConvex(::Type{<:IndicatorMonotoneMatrices}) = true 
isSet(::Type{<:IndicatorMonotoneMatrices}) = true 

"""
    isotonic_regression!(y::Vector{Float64}, weights::Vector{Float64})

Perform isotonic regression in-place using the Pool Adjacent Violators Algorithm (PAVA).
source: https://github.com/ajtulloch/Isotonic.jl/blob/master/src/linear_pava.jl
"""
function isotonic_regression!(y::Vector{Float64}, weights::Vector{Float64})
    n = length(y)
    n <= 1 && return y
    
    length(y) == length(weights) || throw(DimensionMismatch("Lengths of values and weights mismatch"))

    @inbounds begin
        n -= 1
        while true
            i = 1
            pooled = 0
            while i <= n
                k = i
                while k <= n && y[k] >= y[k+1]
                    k += 1
                end

                if y[i] != y[k]
                    numerator = 0.0
                    denominator = 0.0
                    for j in i:k
                        numerator += y[j] * weights[j]
                        denominator += weights[j]
                    end

                    for j in i:k
                        y[j] = numerator / denominator
                    end
                    pooled = 1
                end
                i = k + 1
            end
            if pooled == 0
                break
            end
        end
    end
end

isotonic_regression!(y::Vector{Float64}) = isotonic_regression!(y, ones(length(y)))

"""
    (f::IndicatorMonotoneMatrices)(x::Array{Float64,3})

Evaluate the indicator function. Returns 0.0 if constraints are satisfied, Inf otherwise.
"""
function (f::IndicatorMonotoneMatrices)(x::Array{Float64,3})
    size(x) == (f.numberMatrices, f.numberRows, f.numberColumns) || 
        throw(DimensionMismatch("Expected size ($(f.numberMatrices), $(f.numberRows), $(f.numberColumns)), got $(size(x))"))
    
    # Check bounds
    if any(x .> f.ub + FeasTolerance) || any(x .< f.lb - FeasTolerance)
        return Inf
    end
    
    # Check monotonicity
    for i in 2:f.numberMatrices
        if any(@view(x[i-1,:,:]) .< @view(x[i,:,:]) .- FeasTolerance)
            return Inf
        end
    end
    
    return 0.0
end

"""
    proximalOracle!(y::Array{Float64,3}, 
                   f::IndicatorMonotoneMatrices, 
                   x::Array{Float64,3}, 
                   gamma::Float64 = 1.0, 
                   enableParallel::Bool = false)

Compute the proximal operator for the monotone matrices constraint.
"""
function proximalOracle!(y::Array{Float64,3}, 
                        f::IndicatorMonotoneMatrices, 
                        x::Array{Float64,3}, 
                        gamma::Float64 = 1.0, 
                        enableParallel::Bool = false)
    
    # Input validation
    size(x) == size(y) == (f.numberMatrices, f.numberRows, f.numberColumns) || 
        throw(DimensionMismatch("Input/output size mismatch"))

    function processTask(idx::Int64)
        row = div(idx - 1, f.numberColumns) + 1
        col = mod(idx - 1, f.numberColumns) + 1
        
        # Extract values for current position
        @inbounds for k in 1:f.numberMatrices
            f.buffer[idx][k] = x[k,row,col]
        end
    
        # Apply isotonic regression (with sign change for ≥ constraint)
        @inbounds begin
            @. f.buffer[idx] = -f.buffer[idx] 
            isotonic_regression!(f.buffer[idx])
            @. f.buffer[idx] = -f.buffer[idx]
        end

        # Write back results with bounds enforcement
        @inbounds for k in 1:f.numberMatrices
            y[k,row,col] = clamp(f.buffer[idx][k], f.lb, f.ub)
        end
    end

    totalTasks = f.numberRows * f.numberColumns
    if enableParallel && totalTasks > 1000  # Only parallelize for larger problems
        Threads.@threads for idx in 1:totalTasks
            processTask(idx)
        end
    else
        for idx in 1:totalTasks
            processTask(idx)
        end
    end
end 

"""
    proximalOracle(f::IndicatorMonotoneMatrices, x::Array{Float64,3}, 
                   gamma::Float64 = 1.0, enableParallel::Bool = false)

Non-mutating version of the proximal operator.
"""
function proximalOracle(f::IndicatorMonotoneMatrices, 
                       x::Array{Float64,3}, 
                       gamma::Float64 = 1.0, 
                       enableParallel::Bool = false)
    y = similar(x)
    proximalOracle!(y, f, x, gamma, enableParallel)
    return y 
end

function testIndicatorMonotoneMatrices()
    # Problem dimensions and bounds
    numberMatrices = 3
    numberRows = 4
    numberColumns = 5
    lb = 0.0
    ub = 1.0

    # Create an instance of the indicator function
    f = IndicatorMonotoneMatrices(numberMatrices, numberRows, numberColumns, lb, ub)

    # ---------------------------
    # Test 1: Feasible monotone sequence
    # ---------------------------
    # Create a 3D array X with monotone decreasing slices:
    # X[1,:,:] = 0.9, X[2,:,:] = 0.5, X[3,:,:] = 0.1
    X = zeros(numberMatrices, numberRows, numberColumns)
    X[1, :, :] .= 0.9
    X[2, :, :] .= 0.5
    X[3, :, :] .= 0.1

    @assert f(X) == 0.0 "IndicatorMonotoneMatrices: Test 1 failed"
    
    # ---------------------------
    # Test 2: Non-monotone sequence
    # ---------------------------
    Y = copy(X)
    # Introduce a violation: set an element in the second slice greater than the corresponding element in the first slice.
    Y[2, 1, 1] = 0.95
    @assert f(Y) == Inf "IndicatorMonotoneMatrices: Test 2 failed"

    # ---------------------------
    # Test 3: Out-of-bound sequence
    # ---------------------------
    Z = copy(X)
    # Introduce an out-of-bound violation: set an element to be above ub.
    Z[1, 2, 3] = 1.1
    @assert f(Z) == Inf "IndicatorMonotoneMatrices: Test 3 failed"

    # ---------------------------
    # Test 4: Proximal operator projects non-monotone input to a feasible solution
    # ---------------------------
    # Use the non-monotone Y from Test 2 and project it.
    Yproj = proximalOracle(f, Y, 1.0, false)
    # println(Yproj[1,:,:])
    # println(Yproj[2,:,:])
    # println(Yproj[3,:,:])
    # After projection, the indicator should return 0 (i.e. feasibility is enforced)
    @assert f(Yproj) == 0.0 "IndicatorMonotoneMatrices: Test 4 failed"

    # Optionally, one can check that the projected solution is "close" to the original Y in some metric.
    # For example, the correction should be minimal in the least-squares sense.
    
    println("All tests passed!")
end