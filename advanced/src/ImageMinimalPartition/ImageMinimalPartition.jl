using Images, FileIO

include("FiniteDifferenceMatrixNorm.jl")
include("IndicatorMonotoneMatrices.jl")
include("LinearMappingIndexedFD.jl")

function processImage(imageDir::String, Q::Int64, color::Bool = true)
    img = load(imageDir)
    (numberRows, numberCols) = size(img)
    println("processImage: read image from $imageDir, size: $numberRows x $numberCols")
    # Pre-allocate with known size
    n_channels = (eltype(img) <: Gray || !color) ? 1 : 3
    imageVec = Vector{Matrix{Float64}}(undef, n_channels)
    left = Vector{Matrix{Float64}}(undef, n_channels)
    right = Vector{Matrix{Float64}}(undef, n_channels)

    # Fill imageVec
    if eltype(img) <: Gray || !color
        imageVec[1] = 255.0 * Float64.(Gray.(img)) 
    else 
        for c in 1:3
            imageVec[c] = 255.0 * Float64.(channelview(img)[c, :, :])
        end
    end 

    std_dev = 10.0
    # Normalize std_dev to match the [0,1] sc
    
    for (i, m) in enumerate(imageVec)
        # Add noise to normalized images
        left[i] = clamp.(round.(m .+ std_dev * randn(size(m))), 0, 255)
        right[i] = clamp.(round.(m .+ std_dev * randn(size(m))), 0, 255)
    end 
    
    # Pre-allocate vecEta
    vecEta = [zeros(numberRows, numberCols) for _ in 1:Q]
    C = length(left)
    
    # Compute eta values more efficiently
    for q in 1:Q 
        for i in 1:numberRows 
            for j in 1:numberCols 
                right_col = clamp(j - q + 1, 1, numberCols)
                sum_buffer = 0.0
                for c in 1:C
                    @inbounds sum_buffer += abs(left[c][i,j] - right[c][i, right_col])
                end
                vecEta[q][i,j] = sum_buffer
            end 
        end 
    end 
    return vecEta
end 

function ImageMinimalPartition(imageDir::String, Q::Int64, lambda::Float64 = 1.0, color::Bool = false)
    isfile(imageDir) || throw(ArgumentError("Image file not found: $imageDir"))
    Q > 2 || throw(ArgumentError("Q must be greater than 2"))

    vecEta = processImage(imageDir, Q, color)
    numberMatrices = Q-1 
    numberRows, numberCols = size(vecEta[1])
    mbp = MultiblockProblem() 
    
    # Theta block 
    thetaBlock = BlockVariable(0) 
    A = zeros(numberMatrices, numberRows, numberCols)
    for q in 1:numberMatrices 
        A[q,:,:] .= vecEta[q+1] .- vecEta[q]
    end 
    thetaBlock.f = AffineFunction(A, 0.0)
    vecEta = nothing
    
    thetaBlock.g = IndicatorMonotoneMatrices(numberMatrices, numberRows, numberCols)
    thetaBlock.val = rand(numberMatrices, numberRows, numberCols)
    addBlockVariable!(mbp, thetaBlock)

    # Phi blocks (Q of them)
    TVCoeff = sqrt(1/8)
    for q in 1:Q
        block = BlockVariable(q) 
        block.g = FiniteDifferenceMatrixNorm(numberRows, numberCols, TVCoeff * lambda)
        block.val = rand(2, numberRows, numberCols)
        # println("block.val norm: ", norm(block.val))
        addBlockVariable!(mbp, block)
    end
    
    # constraints
    for q in 1:Q 
        constr = BlockConstraint(q)
        addBlockMappingToConstraint!(constr, 0, LinearMappingIndexedFD(numberMatrices, numberRows, numberCols, q))
        addBlockMappingToConstraint!(constr, q, LinearMappingIdentity(-1.0))
        constr.rhs = zeros(2, numberRows, numberCols)
        addBlockConstraint!(mbp, constr) 
    end
    return mbp  
end


function ImageMinimalPartition_PDM(imageDir::String, Q::Int64, lambda::Float64 = 1.0, color::Bool = false)
    isfile(imageDir) || throw(ArgumentError("Image file not found: $imageDir"))

    vecEta = processImage(imageDir, Q, color)
    numberMatrices = Q-1 
    numberRows, numberCols = size(vecEta[1])
    mbp = MultiblockProblem() 

    # Theta block 
    thetaBlock = BlockVariable(0) 
    A = zeros(numberMatrices, numberRows, numberCols)
    for q in 1:numberMatrices 
        A[q,:,:] .= vecEta[q+1] .- vecEta[q]
    end 
    thetaBlock.f = AffineFunction(A, 0.0)
    vecEta = nothing
    
    thetaBlock.g = IndicatorMonotoneMatrices(numberMatrices, numberRows, numberCols)
    thetaBlock.val = rand(numberMatrices, numberRows, numberCols)
    addBlockVariable!(mbp, thetaBlock)

    # Phi block 
    TVCoeff = sqrt(1/8)
    PhiBlock = BlockVariable(1)
    proximalFunctions = Vector{AbstractFunction}(undef, Q)
    for q in 1:Q
        proximalFunctions[q] = FiniteDifferenceMatrixNorm(numberRows, numberCols, TVCoeff * lambda)
    end
    PhiBlock.g = StackingProximalFunctions(proximalFunctions, Int64[2 for _ in 1:Q])
    PhiBlock.val = rand(Q*2, numberRows, numberCols)
    addBlockVariable!(mbp, PhiBlock)

    # constraints
    constr = BlockConstraint(1)
    mappings = Vector{AbstractMapping}(undef, Q)
    for q in 1:Q 
        mappings[q] = LinearMappingIndexedFD(numberMatrices, numberRows, numberCols, q)
    end 
    addBlockMappingToConstraint!(constr, 0, LinearMappingStacking(mappings, Int64[2 for _ in 1:Q]))
    addBlockMappingToConstraint!(constr, 1, LinearMappingIdentity(-1.0))
    constr.rhs = zeros(2 * Q, numberRows, numberCols)
    addBlockConstraint!(mbp, constr)
    return mbp 
end