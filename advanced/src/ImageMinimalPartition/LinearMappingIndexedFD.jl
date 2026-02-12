"""
    LinearMappingIndexedFD

Linear mapping that computes horizontal and vertical finite differences of matrix differences. 

Let x be a 3D array of size (numberMatrices × numberRows × numberColumns), and 1<=q<=numberMatrices+1.
For a specific index q:
- For q = 1: D(1 - X₁)
- For q = numberMatrices+1: D(X_{numberMatrices} - 0)
- For 1 < q <= numberMatrices: D(X_{q-1} - X_q)
where D computes horizontal and vertical differences with cyclic boundary conditions. 

Output is a (2 × numberRows × numberColumns) array: [horizontal_diff; vertical_diff] for the specific q.

"""


struct LinearMappingIndexedFD <: AbstractMapping
    numberMatrices::Int64 
    numberRows::Int64 
    numberColumns::Int64 
    q::Int64  # Index for which to compute differences
    tempMatrix::Matrix{Float64}  # Workspace for difference computation

    function LinearMappingIndexedFD(numberMatrices::Int64, numberRows::Int64, numberColumns::Int64, q::Int64)
        @assert numberMatrices > 0 "Number of matrices must be positive"
        @assert numberRows > 0 "Number of rows must be positive"
        @assert numberColumns > 0 "Number of columns must be positive"
        @assert 1 <= q <= numberMatrices + 1 "Index q must be between 1 and numberMatrices+1"
        new(numberMatrices, numberRows, numberColumns, q, zeros(numberRows, numberColumns))
    end
end 

function (L::LinearMappingIndexedFD)(x::AbstractArray{Float64,3}, ret::AbstractArray{Float64,3}, add::Bool = false)
    # Input validation
    size(x) == (L.numberMatrices, L.numberRows, L.numberColumns) || 
        throw(DimensionMismatch("Input size mismatch"))
    size(ret) == (2, L.numberRows, L.numberColumns) || 
        throw(DimensionMismatch("Output must be size (2, numberRows, numberColumns)"))

    # If not accumulating, zero out the return matrices
    if !add
        fill!(ret, 0.0)
    end

    # Compute X_{q-1} - X_q for the specific q
    if L.q == 1
        # For q = 1, use all-ones matrix as X_0
        fill!(L.tempMatrix, 1.0)
        L.tempMatrix .-= @view x[1,:,:]
    elseif L.q == L.numberMatrices + 1
        copyto!(L.tempMatrix, @view x[L.numberMatrices,:,:])
    else
        copyto!(L.tempMatrix, @view x[L.q-1,:,:])
        L.tempMatrix .-= @view x[L.q,:,:]
    end

    # Horizontal differences
    @threads for i in 1:L.numberRows
        # Create views to avoid extra allocations and improve cache locality
        tempRow = @view L.tempMatrix[i, :]
        retRow = @view ret[1,i,:]
        # Handle cyclic boundary: last column gets difference of first and last column
        @inbounds retRow[L.numberColumns] += tempRow[1] - tempRow[L.numberColumns]
        # Compute differences for remaining columns
        @inbounds @simd for j in 1:(L.numberColumns - 1)
            retRow[j] += tempRow[j+1] - tempRow[j]
        end
    end

    # Vertical differences
    @threads for j in 1:L.numberColumns
        # Create views for the current column
        tempCol = @view L.tempMatrix[:,j]
        retCol = @view ret[2,:,j]
        # Handle cyclic boundary: last row gets difference of first and last row
        @inbounds retCol[L.numberRows] += tempCol[1] - tempCol[L.numberRows]
        # Compute differences for the remaining rows
        @inbounds @simd for i in 1:(L.numberRows - 1)
            retCol[i] += tempCol[i+1] - tempCol[i]
        end
    end
end

function (L::LinearMappingIndexedFD)(x::AbstractArray{Float64,3})
    ret = zeros(2, L.numberRows, L.numberColumns)
    L(x, ret)
    return ret
end 

function adjoint!(L::LinearMappingIndexedFD, 
        y::AbstractArray{Float64,3},  # Input: dual variables [horizontal_diff; vertical_diff]
        ret::AbstractArray{Float64,3}, # Output: adjoint result
        add::Bool = false)
    
    # Input validation
    size(y) == (2, L.numberRows, L.numberColumns) || 
        throw(DimensionMismatch("Input must be size (2, numberRows, numberColumns)"))
    size(ret) == (L.numberMatrices, L.numberRows, L.numberColumns) || 
        throw(DimensionMismatch("Output size mismatch"))

    # If not accumulating, zero out the return matrices
    if !add
        fill!(ret, 0.0)
    end

    if L.q == 1
        # Case: D(1 - X₁)
        # Horizontal adjoint update
        @threads for i in 1:L.numberRows
            @inbounds @simd for j in 1:L.numberColumns
                next_j = (j == L.numberColumns ? 1 : j + 1)
                # X₁ gets positive contribution (due to minus sign in 1 - X₁)
                ret[1,i,j] += y[1,i,j]
                ret[1,i,next_j] -= y[1,i,j]
            end
        end

        # Vertical adjoint update
        @threads for j in 1:L.numberColumns
            @inbounds @simd for i in 1:L.numberRows
                next_i = (i == L.numberRows ? 1 : i + 1)
                ret[1,i,j] += y[2,i,j]
                ret[1,next_i,j] -= y[2,i,j]
            end
        end

    elseif L.q == L.numberMatrices + 1
        # Case: D(X_{numberMatrices} - 0)
        @threads for i in 1:L.numberRows
            @inbounds @simd for j in 1:L.numberColumns
                next_j = (j == L.numberColumns ? 1 : j + 1)
                ret[L.numberMatrices,i,j] -= y[1,i,j]
                ret[L.numberMatrices,i,next_j] += y[1,i,j]
            end
        end

        @threads for j in 1:L.numberColumns
            @inbounds @simd for i in 1:L.numberRows
                next_i = (i == L.numberRows ? 1 : i + 1)
                ret[L.numberMatrices,i,j] -= y[2,i,j]
                ret[L.numberMatrices,next_i,j] += y[2,i,j]
            end
        end

    else
        # Case: D(X_{q-1} - X_q) for 1 < q <= numberMatrices
        # Horizontal adjoint update
        @threads for i in 1:L.numberRows
            @inbounds @simd for j in 1:L.numberColumns
                next_j = (j == L.numberColumns ? 1 : j + 1)
                # X_{q-1} gets negative contribution
                ret[L.q-1,i,j] -= y[1,i,j]
                ret[L.q-1,i,next_j] += y[1,i,j]
                # X_q gets positive contribution
                ret[L.q,i,j] += y[1,i,j]
                ret[L.q,i,next_j] -= y[1,i,j]
            end
        end

        # Vertical adjoint update
        @threads for j in 1:L.numberColumns
            @inbounds @simd for i in 1:L.numberRows
                next_i = (i == L.numberRows ? 1 : i + 1)
                # X_{q-1} gets negative contribution
                ret[L.q-1,i,j] -= y[2,i,j]
                ret[L.q-1,next_i,j] += y[2,i,j]
                # X_q gets positive contribution
                ret[L.q,i,j] += y[2,i,j]
                ret[L.q,next_i,j] -= y[2,i,j]
            end
        end
    end
end

function adjoint(L::LinearMappingIndexedFD, y::AbstractArray{Float64,3})
    ret = zeros(L.numberMatrices, L.numberRows, L.numberColumns)
    adjoint!(L, y, ret)
    return ret
end 

function operatorNorm2(L::LinearMappingIndexedFD)
    # For finite difference operators with cyclic boundary conditions,
    # the operator norm is bounded by 2√2 for a single difference operation
    return 2.0 * sqrt(2.0)
end 


function testLinearMappingIndexedFD(numberMatrices::Int64 = 3,
    numberRows::Int64 = 5,
    numberColumns::Int64 = 4)

    println("--------------------------------------------------")
    # We'll test for three cases: q = 1, q = 2, and q = numberMatrices+1
    for q in 1:numberMatrices+1
        println("Testing for q = $q")
        # Create our operator instance
        L = LinearMappingIndexedFD(numberMatrices, numberRows, numberColumns, q)
        
        # Generate random input x: a 3D array (numberMatrices, numberRows, numberColumns)
        x = randn(numberMatrices, numberRows, numberColumns)
        
        # Generate random dual variable y: a 3D array (2, numberRows, numberColumns)
        y = randn(2, numberRows, numberColumns)
        
        # Compute forward mapping: L(x) returns a (2, numberRows, numberColumns) array.
        Lx = L(x)
        
        # Compute inner product <L(x), y>
        ip_forward = dot(Lx, y)
        
        # Compute adjoint mapping: L^*(y) returns a (numberMatrices, numberRows, numberColumns) array.
        Lty = adjoint(L, y)
        
        # Compute inner product <x, L^*(y)>
        ip_adjoint = dot(x, Lty)    

        # println("  <L(x), y> = ", ip_forward)
        # println("  <x, L*(y)> = ", ip_adjoint)
        diff = abs(ip_forward - ip_adjoint)
        @assert diff < ZeroTolerance "LinearMappingIndexedFD: <L(x), y> = $(ip_forward), <x, L*(y)> = $(ip_adjoint), Difference = $diff"
        println("OK: Difference = $diff")
        println("--------------------------------------------------")
    end
end 