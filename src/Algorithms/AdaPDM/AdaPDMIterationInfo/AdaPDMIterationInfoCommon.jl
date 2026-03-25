"""
    computePartialObjective!(info::AdaPDMIterationInfo, mbp::MultiblockProblem)

Compute the partial objective value f(x) + g(x) and update bufferAx with Ax.

# Arguments
- `info::AdaPDMIterationInfo`: The iteration info object containing current solutions
- `mbp::MultiblockProblem`: The multiblock problem being solved

# Returns
- `Float64`: The partial objective value

"""
function computePartialObjective!(info::AdaPDMIterationInfo, mbp::MultiblockProblem)
    obj = 0.0
    info.bufferAx .= 0.0
    mappings = mbp.constraints[1].mappings
    nPrimalBlocks = length(mbp.blocks) - 1
    for bi in 1:nPrimalBlocks
        block = mbp.blocks[bi]
        obj += (block.f)(info.primalSol[block.id]) + (block.g)(info.primalSol[block.id])
        mappings[block.id](info.primalSol[block.id], info.bufferAx, true)
    end 
    return obj 
end 


"""
    computeLipschitzAndCocoercivityEstimate(mbp::MultiblockProblem, info::AdaPDMIterationInfo) 

Compute the Lipschitz and cocoercivity constants of the problem based on current iterations.

# Arguments
- `mbp::MultiblockProblem`: The multiblock problem being solved
- `info::AdaPDMIterationInfo`: The current iteration information

# Returns
- `Float64, Float64`: Estimated Lipschitz constant and cocoercivity constant

# Details
This function estimates the Lipschitz constant of the gradient and the cocoercivity
constant by examining the difference between current and previous iterations.
"""
function computeLipschitzAndCocoercivityEstimate(mbp::MultiblockProblem, info::AdaPDMIterationInfo) 
    l = 0.0 
    c = 0.0
    nPrimalBlocks = length(mbp.blocks) - 1
    for bi in 1:nPrimalBlocks
        block = mbp.blocks[bi]
        # primalBuffer2 = nabla f(x^{k-1}) - nabla f(x^k)
        gradientOracle!(info.primalBuffer1[block.id], block.f, info.primalSol[block.id],)
        gradientOracle!(info.primalBuffer2[block.id], block.f, info.primalSolPrev[block.id])
        axpy!(-1.0, info.primalBuffer1[block.id], info.primalBuffer2[block.id])

        # primalBuffer1 = x^{k-1} - x^k
        copyto!(info.primalBuffer1[block.id], info.primalSolPrev[block.id])
        axpy!(-1.0, info.primalSol[block.id], info.primalBuffer1[block.id])

        innerProduct = dot(info.primalBuffer1[block.id], info.primalBuffer2[block.id])
        primalDiffSquare = dot(info.primalBuffer1[block.id], info.primalBuffer1[block.id])
        gradientDiffSquare = dot(info.primalBuffer2[block.id], info.primalBuffer2[block.id])

        blockLipschitz = primalDiffSquare < ZeroTolerance ? 0.0 : innerProduct / primalDiffSquare
        blockCocoercivity = abs(innerProduct) < ZeroTolerance ? 0.0 : gradientDiffSquare / innerProduct

        l = max(l, blockLipschitz)
        c = max(c, blockCocoercivity)
    end 
    return l, c
end 

"""
    prepareProximalCenterForConjugateProximalOracle!(info::AdaPDMIterationInfo, dualStepSize::Float64, ratio1::Float64, ratio2::Float64)

Prepare the proximal center for the conjugate proximal oracle.

# Arguments
- `info::AdaPDMIterationInfo`: The iteration information object
- `dualStepSize::Float64`: The dual step size (sigma)
- `ratio1::Float64`: Coefficient for bufferAx (typically 1.0 + newGamma/gamma)
- `ratio2::Float64`: Coefficient for bufferAxPrev (typically newGamma/gamma)

# Details
This function prepares the proximal center for the conjugate proximal oracle by calculating:
    dualBuffer <- y^{k} + dualStepSize * (ratio1 * Ax^{k+1} - ratio2 * Ax^k)

The result is stored in `info.dualBuffer` for subsequent use in the proximal oracle computation.
"""
function prepareProximalCenterForConjugateProximalOracle!(info::AdaPDMIterationInfo, 
    dualStepSize::Float64, 
    ratio1::Float64, # coefficient for bufferAx 
    ratio2::Float64) # coefficient for bufferAxPrev
    copyto!(info.dualBuffer, info.dualSol)
    axpy!(dualStepSize * ratio1, info.bufferAx, info.dualBuffer)
    if ratio2 != 0.0
        axpy!(-dualStepSize * ratio2 , info.bufferAxPrev, info.dualBuffer)
    end 
end 

"""
    prepareProximalCenterForPrimalUpdate!(info::AdaPDMIterationInfo, blockID::BlockID, mbp::MultiblockProblem, primalStepSize::Float64, dualBuffer::NumericVariable, gradientBuffer::NumericVariable)

Prepare the proximal center in primalBuffer1 for primal variable update.

# Arguments
- `info::AdaPDMIterationInfo`: The iteration information object
- `blockID::BlockID`: The ID of the block to update
- `mbp::MultiblockProblem`: The multiblock problem being solved
- `primalStepSize::Float64`: The primal step size (gamma)
- `dualBuffer::NumericVariable`: The dual variable to use (typically info.dualSol)
- `gradientBuffer::NumericVariable`: Buffer containing the gradient (typically info.primalBuffer2)

# Details
This function prepares the proximal center for the primal update by calculating:
    primalBuffer1 <- x^k - primalStepSize * (∇f(x^k) + A'y^k)

The result is stored in `info.primalBuffer1[blockID]` for subsequent use in the proximal oracle computation.
"""
function prepareProximalCenterForPrimalUpdate!(info::AdaPDMIterationInfo, 
    blockID::BlockID, 
    mbp::MultiblockProblem, 
    primalStepSize::Float64, 
    dualBuffer::NumericVariable, 
    gradientBuffer::NumericVariable)  # usually we use primalBuffer2 here for gradient buffer  

    mappings = mbp.constraints[1].mappings 
    copyto!(info.primalBuffer1[blockID], gradientBuffer)
    adjoint!(mappings[blockID], dualBuffer, info.primalBuffer1[blockID], true)
    axpby!(1.0, info.primalSol[blockID], -primalStepSize, info.primalBuffer1[blockID])
end 

"""
    updatePrimalSolution!(block::BlockVariable, mbp::MultiblockProblem, info::AdaPDMIterationInfo)

Update the primal solution for a specific block using the proximal oracle.

# Arguments
- `block::BlockVariable`: The block variable to update
- `mbp::MultiblockProblem`: The multiblock problem being solved
- `info::AdaPDMIterationInfo`: The current iteration information

# Details
This function:
1. Computes the gradient of the objective function at the current solution
2. Prepares the proximal center by applying the gradient and adjoint operators
3. Saves the current solution as the previous solution
4. Updates the primal solution using the proximal oracle

The function is used by both AdaPDM and AdaPDMPlus algorithms to update individual block variables.
"""
function updatePrimalSolution!(block::BlockVariable, 
    mbp::MultiblockProblem, 
    info::AdaPDMIterationInfo)

    blockID = block.id 
    gradientOracle!(info.primalBuffer2[blockID], block.f, info.primalSol[blockID])
    prepareProximalCenterForPrimalUpdate!(info, 
        blockID, 
        mbp, 
        info.primalStepSize, 
        info.dualSol, 
        info.primalBuffer2[blockID])
    copyto!(info.primalSolPrev[blockID], info.primalSol[blockID])
    proximalOracle!(info.primalSol[blockID], block.g, info.primalBuffer1[blockID], info.primalStepSize)
end 
