""" 
    The hyperspectral image unmixing problem is a problem of decomposing a hyperspectral image into a set of endmembers and their corresponding abundance maps. 
    The problem can be formulated as:
    
    min_W f_1(W) + f_2(w) + f_3(w) + f_4(w)

    where f_1(x) = 0.5||Y-Phi W||_F^2, 
          f_2(x) = coefL1Norm * ||A circ W||_1 
          f_3(x) = coefNuclearNorm * ||W||_{b,*}
          f_4(x) = IndicatorNonnegativeOrthant(w)

    One formulation is to introduce a 4 auxiliary blocks and solve: 

    min_{W, Omega_1, Omega_2, Omega_3, Omega_4} 0.5||Omega_1 - Y||_F^2 + f_2(Omega_2) + f_3(Omega_3) + f_4(Omega_4)
    s.t. Phi W - Omega_1 = 0
         W - Omega_2 = 0
         W - Omega_3 = 0
         W - Omega_4 = 0
        
    The resulting formulation has 5 blocks. 

    References: https://arxiv.org/pdf/1504.01515
"""
function HyperspectralImageUnmixing_5Blocks(Phi::Matrix{Float64},
    Y::Matrix{Float64},                
    A::SparseMatrixCSC{Float64, Int64}, # weights for the L1 norm
    b::Vector{Float64},                     # weights for the nuclear norm   
    coefL1Norm::Float64,                 # coefficient for the L1 norm
    coefNuclearNorm::Float64)            # coefficient for the nuclear norm

    @assert(size(Phi, 1) == size(Y, 1), "HypersoectralImageUnmixing_5Blocks: Phi and Y must have the same number of rows.")
    numberRows = size(Phi, 2)
    numberColumns = size(Y, 2)

    mbp = MultiblockProblem() 
    
    # concensus block: W
    block0 = BlockVariable(0)
    block0.val = spzeros(numberRows, numberColumns)
    addBlockVariable!(mbp, block0)

    # first block: 0.5||Omega_1 - Y||_F^2
    block1 = BlockVariable(1)
    block1.g = FrobeniusNormSquare(Matrix{Float64}(I, size(Y, 1), size(Y, 1)), Y, size(Y, 1), size(Y, 2), 0.5)
    block1.val = proximalOracle(block1.g, spzeros(size(Y, 1), size(Y, 2)))
    addBlockVariable!(mbp, block1)

    # second block: coefL1Norm * ||A circ Omega_2||_1
    block2 = BlockVariable(2)
    block2.g = WeightedMatrixL1Norm(coefL1Norm * A) # weights and coefficients are combined 
    block2.val = proximalOracle(block2.g, spzeros(numberRows, numberColumns))
    addBlockVariable!(mbp, block2)

    # third block: coefNuclearNorm * ||Omega_3||_{b,*}
    block3 = BlockVariable(3)
    block3.g = MatrixNuclearNorm(coefNuclearNorm * b, numberRows, numberColumns)
    block3.val = proximalOracle(block3.g, spzeros(numberRows, numberColumns))
    addBlockVariable!(mbp, block3)

    # fourth block: IndicatorNonnegativeOrthant(O)
    block4 = BlockVariable(4)
    block4.g = IndicatorNonnegativeOrthant()
    block4.val = proximalOracle(block4.g, spzeros(numberRows, numberColumns))
    addBlockVariable!(mbp, block4)


    # constraints: 
    # 1: Phi W - Omega_1 = 0
    constr1 = BlockConstraint(1)
    addBlockMappingToConstraint!(constr1, 0, LinearMappingMatrix(sparse(Phi))) 
    addBlockMappingToConstraint!(constr1, 1, LinearMappingIdentity(-1.0))
    constr1.rhs = spzeros(size(Y,1), size(Y,2))
    addBlockConstraint!(mbp, constr1)

    # 2: W - Omega_2 = 0
    constr2 = BlockConstraint(2)
    addBlockMappingToConstraint!(constr2, 0, LinearMappingIdentity(1.0))
    addBlockMappingToConstraint!(constr2, 2, LinearMappingIdentity(-1.0))
    constr2.rhs = spzeros(numberRows, numberColumns)
    addBlockConstraint!(mbp, constr2)

    # 3: W - Omega_3 = 0
    constr3 = BlockConstraint(3)
    addBlockMappingToConstraint!(constr3, 0, LinearMappingIdentity(1.0))
    addBlockMappingToConstraint!(constr3, 3, LinearMappingIdentity(-1.0))
    constr3.rhs = spzeros(numberRows, numberColumns)
    addBlockConstraint!(mbp, constr3) 

    # 4: W - Omega_4 = 0
    constr4 = BlockConstraint(4)
    addBlockMappingToConstraint!(constr4, 0, LinearMappingIdentity(1.0))
    addBlockMappingToConstraint!(constr4, 4, LinearMappingIdentity(-1.0))
    constr4.rhs = spzeros(numberRows, numberColumns)
    addBlockConstraint!(mbp, constr4)

    return mbp 
end 

function HyperspectralImageUnmixing_2Blocks(Phi::Matrix{Float64},
    Y::Matrix{Float64},                
    A::SparseMatrixCSC{Float64, Int64}, # weights for the L1 norm
    b::Vector{Float64},                     # weights for the nuclear norm   
    coefL1Norm::Float64,                 # coefficient for the L1 norm
    coefNuclearNorm::Float64)            # coefficient for the nuclear norm

    @assert(size(Phi, 1) == size(Y, 1), "HypersoectralImageUnmixing_2Blocks: Phi and Y must have the same number of rows.")
    numberRows = size(Phi, 2)
    numberColumns = size(Y, 2)

    mbp = MultiblockProblem() 
    
    # first block: f + g where 
    # f = 0.5||Phi Omega1 - Y||_F^2 
    # g = coefL1Norm * ||A circ Omega_1||_1 + indicatorNonnegativeOrthant(Omega_1)
    block1 = BlockVariable(1)
    block1.f = FrobeniusNormSquare(Phi, Y, numberRows, numberColumns, 0.5)
    block1.g = WeightedMatrixL1Norm(coefL1Norm * A; inNonnegativeOrthant=true)
    block1.val = proximalOracle(block1.g, spzeros(numberRows, numberColumns))
    addBlockVariable!(mbp, block1)

    # second block: coefNuclearNorm * ||Omega_2||_{b,*} 
    block2 = BlockVariable(2)
    block2.g = MatrixNuclearNorm(coefNuclearNorm * b, numberRows, numberColumns)
    block2.val = proximalOracle(block2.g, spzeros(numberRows, numberColumns)) # SVD does not apply to sparse matrix
    addBlockVariable!(mbp, block2)

    # constraints: 
    # omega_1 - omega_2 = 0
    constr1 = BlockConstraint(1)
    addBlockMappingToConstraint!(constr1, 1, LinearMappingIdentity(1.0))
    addBlockMappingToConstraint!(constr1, 2, LinearMappingIdentity(-1.0))
    constr1.rhs = spzeros(numberRows, numberColumns)
    addBlockConstraint!(mbp, constr1)

    return mbp 
end 


function HyperspectralImageUnmixing_3Blocks(Phi::Matrix{Float64},
    Y::Matrix{Float64},                
    A::SparseMatrixCSC{Float64, Int64}, # weights for the L1 norm
    b::Vector{Float64},                     # weights for the nuclear norm   
    coefL1Norm::Float64,                 # coefficient for the L1 norm
    coefNuclearNorm::Float64)            # coefficient for the nuclear norm

    @assert(size(Phi, 1) == size(Y, 1), "HypersoectralImageUnmixing_3Blocks: Phi and Y must have the same number of rows.")
    numberRows = size(Phi, 2)
    numberColumns = size(Y, 2)

    mbp = MultiblockProblem() 
    
    # first block: 0.5||Phi Omega_1 - Y||_F^2
    block1 = BlockVariable(1)
    block1.g = FrobeniusNormSquare(Phi, Y, numberRows, numberColumns, 0.5)
    block1.val = proximalOracle(block1.g, spzeros(numberRows, numberColumns))
    addBlockVariable!(mbp, block1)

    # second block: coefL1Norm * ||A circ Omega_2||_1 + indicatorNonnegativeOrthant(Omega_2)
    block2 = BlockVariable(2)
    block2.g = WeightedMatrixL1Norm(coefL1Norm * A; inNonnegativeOrthant=true) # weights and coefficients are combined 
    block2.val = proximalOracle(block2.g, spzeros(numberRows, numberColumns))
    addBlockVariable!(mbp, block2)

    # third block: coefNuclearNorm * ||Omega_3||_{b,*}
    block3 = BlockVariable(3)
    block3.g = MatrixNuclearNorm(coefNuclearNorm * b, numberRows, numberColumns)
    block3.val = proximalOracle(block3.g, spzeros(numberRows, numberColumns)) # SVD does not apply to sparse matrix
    addBlockVariable!(mbp, block3)

    # constraints: 
    # omega_1 - omega_3 = 0
    constr1 = BlockConstraint(1)
    addBlockMappingToConstraint!(constr1, 1, LinearMappingIdentity(1.0))
    addBlockMappingToConstraint!(constr1, 2, LinearMappingIdentity(-1.0))
    constr1.rhs = spzeros(numberRows, numberColumns)
    addBlockConstraint!(mbp, constr1)

    # omega_2 - omega_3 = 0
    constr2 = BlockConstraint(2)
    addBlockMappingToConstraint!(constr2, 2, LinearMappingIdentity(1.0))
    addBlockMappingToConstraint!(constr2, 3, LinearMappingIdentity(-1.0))
    constr2.rhs = spzeros(numberRows, numberColumns)
    addBlockConstraint!(mbp, constr2)


    return mbp 
end 

function HyperspectralImageUnmixing_3Blocks_PDM(Phi::Matrix{Float64},
    Y::Matrix{Float64},                
    A::SparseMatrixCSC{Float64, Int64}, # weights for the L1 norm
    b::Vector{Float64},                     # weights for the nuclear norm   
    coefL1Norm::Float64,                 # coefficient for the L1 norm
    coefNuclearNorm::Float64)            # coefficient for the nuclear norm

    @assert(size(Phi, 1) == size(Y, 1), "HypersoectralImageUnmixing_3Blocks_PDM: Phi and Y must have the same number of rows.")
    numberRows = size(Phi, 2)
    numberColumns = size(Y, 2)

    mbp = MultiblockProblem() 

    block1 = BlockVariable(1)
    block1.g = MatrixNuclearNorm(coefNuclearNorm * b, numberRows, numberColumns)
    block1.val = proximalOracle(block1.g, spzeros(numberRows, numberColumns)) # SVD does not apply to sparse matrix
    addBlockVariable!(mbp, block1)
    
    block2 = BlockVariable(2)
    block2.g = StackingProximalFunctions(
        AbstractFunction[FrobeniusNormSquare(Phi, Y, numberRows, numberColumns, 0.5), WeightedMatrixL1Norm(coefL1Norm * A; inNonnegativeOrthant=true)], 
        Int64[numberRows, numberRows]
    )
    block2.val = proximalOracle(block2.g, spzeros(2*numberRows, numberColumns))
    addBlockVariable!(mbp, block2)

    constr  = BlockConstraint(1)
    addBlockMappingToConstraint!(constr, 1, LinearMappingIdentityStacking(1.0, 2))
    addBlockMappingToConstraint!(constr, 2, LinearMappingIdentity(-1.0))
    constr.rhs = spzeros(2 * numberRows, numberColumns)
    addBlockConstraint!(mbp, constr)

    return mbp 
end 


function HyperspectralImageUnmixing_4Blocks(Phi::Matrix{Float64},
    Y::Matrix{Float64},                
    A::SparseMatrixCSC{Float64, Int64}, # weights for the L1 norm
    b::Vector{Float64},                     # weights for the nuclear norm   
    coefL1Norm::Float64,                 # coefficient for the L1 norm
    coefNuclearNorm::Float64)            # coefficient for the nuclear norm

    @assert(size(Phi, 1) == size(Y, 1), "HypersoectralImageUnmixing_5Blocks: Phi and Y must have the same number of rows.")
    numberRows = size(Phi, 2)
    numberColumns = size(Y, 2)

    mbp = MultiblockProblem() 
    
    # concensus block: W
    block0 = BlockVariable(0)
    block0.val = spzeros(numberRows, numberColumns)
    addBlockVariable!(mbp, block0)

    # first block: 0.5||Phi Omega_1 - Y||_F^2
    block1 = BlockVariable(1)
    block1.g = FrobeniusNormSquare(Phi, Y, numberRows, numberColumns, 0.5)
    block1.val = proximalOracle(block1.g, spzeros(numberRows, numberColumns))
    addBlockVariable!(mbp, block1)

    # second block: coefL1Norm * ||A circ Omega_2||_1 + indicatorNonnegativeOrthant(Omega_2)
    block2 = BlockVariable(2)
    block2.g = WeightedMatrixL1Norm(coefL1Norm * A; inNonnegativeOrthant=true) # weights and coefficients are combined 
    block2.val = proximalOracle(block2.g, spzeros(numberRows, numberColumns))
    addBlockVariable!(mbp, block2)

    # third block: coefNuclearNorm * ||Omega_3||_{b,*}
    block3 = BlockVariable(3)
    block3.g = MatrixNuclearNorm(coefNuclearNorm * b, numberRows, numberColumns)
    block3.val = proximalOracle(block3.g, spzeros(numberRows, numberColumns)) # SVD does not apply to sparse matrix
    addBlockVariable!(mbp, block3)


    # constraints: 
    # 1: W - Omega_1 = 0
    constr1 = BlockConstraint(1)
    addBlockMappingToConstraint!(constr1, 0, LinearMappingIdentity(1.0))
    addBlockMappingToConstraint!(constr1, 1, LinearMappingIdentity(-1.0))
    constr1.rhs = spzeros(numberRows, numberColumns)
    addBlockConstraint!(mbp, constr1)

    # 2: W - Omega_2 = 0
    constr2 = BlockConstraint(2)
    addBlockMappingToConstraint!(constr2, 0, LinearMappingIdentity(1.0))
    addBlockMappingToConstraint!(constr2, 2, LinearMappingIdentity(-1.0))
    constr2.rhs = spzeros(numberRows, numberColumns)
    addBlockConstraint!(mbp, constr2)

    # 3: W - Omega_3 = 0
    constr3 = BlockConstraint(3)
    addBlockMappingToConstraint!(constr3, 0, LinearMappingIdentity(1.0))
    addBlockMappingToConstraint!(constr3, 3, LinearMappingIdentity(-1.0))
    constr3.rhs = spzeros(numberRows, numberColumns)
    addBlockConstraint!(mbp, constr3) 


    return mbp 
end 


function HyperspectralImageUnmixing_4Blocks_PDM(Phi::Matrix{Float64},
    Y::Matrix{Float64},                
    A::SparseMatrixCSC{Float64, Int64}, # weights for the L1 norm
    b::Vector{Float64},                     # weights for the nuclear norm   
    coefL1Norm::Float64,                 # coefficient for the L1 norm
    coefNuclearNorm::Float64)            # coefficient for the nuclear norm

    @assert(size(Phi, 1) == size(Y, 1), "HypersoectralImageUnmixing_5Blocks: Phi and Y must have the same number of rows.")
    numberRows = size(Phi, 2)
    numberColumns = size(Y, 2)

    mbp = MultiblockProblem() 
    
    # concensus block: W
    block0 = BlockVariable(0)
    block0.val = spzeros(numberRows, numberColumns)
    addBlockVariable!(mbp, block0)


    block1 = BlockVariable(1)
    block1.g = StackingProximalFunctions(
        AbstractFunction[FrobeniusNormSquare(Phi, Y, numberRows, numberColumns, 0.5), 
        WeightedMatrixL1Norm(coefL1Norm * A; inNonnegativeOrthant=true), 
        MatrixNuclearNorm(coefNuclearNorm * b, numberRows, numberColumns)], 
        Int64[numberRows, numberRows, numberRows]
    )
    block1.val = proximalOracle(block1.g, spzeros(3*numberRows, numberColumns))
    addBlockVariable!(mbp, block1)

    constr = BlockConstraint(1)
    addBlockMappingToConstraint!(constr, 0, LinearMappingIdentityStacking(1.0, 3))
    addBlockMappingToConstraint!(constr, 1, LinearMappingIdentity(-1.0))
    constr.rhs = spzeros(3 * numberRows, numberColumns)
    addBlockConstraint!(mbp, constr)


    return mbp 
end 

function hyperSpectralImageData(Q::AbstractMatrix{<:Real}, 
    N::Int64, 
    K::Int64, 
    isConvex = false, 
    SNR::Float64 = 30.0, 
    rng::AbstractRNG = Random.GLOBAL_RNG)

    L, M = size(Q)
    @assert N ≤ M "hyperSpectralImageData: Cannot pick more endmembers than available columns"

    # 1) sample N endmembers without replacement
    idx = randperm(rng, M)[1:N]
    Phi = Float64.(Q[:, idx])   # L×N

    # 2) generate K abundance vectors on the simplex
    W = zeros(N, K)
    for j in 1:K
        w = rand(rng, N)             # N independent positives
        W[:, j] = w ./ sum(w)        # normalize so ∑ wᵢ = 1
    end

    # 3) form clean data and compute noise level for desired SNR
    Y0 = Phi * W                     # L×K
    sig_power = norm(Y0, 2)^2
    noise_power = sig_power / (10^(SNR/10))
    sigma2 = noise_power / (L*K)     # variance per entry

    # 4) sample Gaussian noise and add
    E = sqrt(sigma2) .* randn(rng, L, K)
    Y = Y0 + E

    # 5) static ℓ₁‐weights from unconstrained LS: W_ls = Φ \ Y
    W_ls = Phi \ Y                   # solves min‖ΦX - Y‖_F
    A = spzeros(N,K)
    A .= 1.0 ./( abs.(W_ls) .+ ZeroTolerance) # N×K

    # 6) compute vector b where b_i = 1/(σᵢ(W_ls) + eps)^2
    F = svd(W_ls)
    b = 1.0 ./ (F.S .+ ZeroTolerance)   

    if isConvex # if we want to make the function convex, make all entries of b equal
        mean = sum(b) / length(b)
        b .= mean
    end 

    return Phi, Y, A, b
end 

