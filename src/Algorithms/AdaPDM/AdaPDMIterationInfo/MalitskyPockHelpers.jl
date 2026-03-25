"""
    updateDualSolution!(mbp::MultiblockProblem, info::AdaPDMIterationInfo, param::MalitskyPockParam)

Updates the dual solution in the Malitsky-Pock algorithm. This function:
1. Prepares the proximal center for the conjugate proximal oracle
2. Stores the current dual solution in the previous dual solution buffer
3. Applies the proximal oracle of the conjugate function to compute the new dual solution
"""
function updateDualSolution!(mbp::MultiblockProblem, info::AdaPDMIterationInfo, param::MalitskyPockParam) 
    # # dualBuffer <- y^k + sigma Ax^k
    # copyto!(info.dualBuffer, info.dualSol)
    # axpy!(info.dualStepSize, info.bufferAx, info.dualBuffer)
    prepareProximalCenterForConjugateProximalOracle!(info, info.dualStepSize, 1.0, 0.0)
    copyto!(info.dualSolPrev, info.dualSol)
    proximalOracleOfConjugate!(info.dualSol, mbp.blocks[end].g, info.dualBuffer, info.dualStepSize)
end


"""
    updatePrimalSolution!(mbp::MultiblockProblem, info::AdaPDMIterationInfo, param::MalitskyPockParam)

Updates the primal solution in the Malitsky-Pock algorithm. This function:
1. Computes the new step sizes based on the adaptive scheme
2. Computes gradients for each primal block
3. Performs line search to find suitable step sizes satisfying the descent condition
4. Updates the primal and dual step sizes and solutions

The line search is based on a sufficient decrease condition that balances
the change in objective value with the squared norm of the primal variable difference.
"""
function updatePrimalSolution!(mbp::MultiblockProblem, info::AdaPDMIterationInfo, param::MalitskyPockParam)
    theta = info.dualStepSize / info.dualStepSizePrev
    newSigma = info.dualStepSize * sqrt(1.0 + theta)
    newGamma = 0.0 
     
    # prepare gradient for blocks
    nPrimalBlocks = length(mbp.blocks) - 1
    for bi in 1:nPrimalBlocks
        block = mbp.blocks[bi]
        gradientOracle!(info.primalBuffer2[block.id], block.f, info.primalSol[block.id])
    end 

    mappings = mbp.constraints[1].mappings

    countBacktracks = 0 
    for iter in 1:param.lineSearchMaxIter
        countBacktracks += 1
        newTheta = newSigma / info.dualStepSize 
        newGamma = param.t^2 * newSigma 

        copyto!(info.lineSearchDualBuffer, info.dualSol)
        axpby!(-newTheta, info.dualSolPrev, (1.0 + newTheta), info.lineSearchDualBuffer)

        @threads for bi in 1:nPrimalBlocks
            block = mbp.blocks[bi]
            prepareProximalCenterForPrimalUpdate!(info, 
                block.id, 
                mbp, 
                newGamma, 
                info.lineSearchDualBuffer, 
                info.primalBuffer2[block.id])
            proximalOracle!(info.lineSearchPrimalBuffer[block.id], 
                block.g, 
                info.primalBuffer1[block.id], 
                newGamma)  
                
            # store primal difference in primalBuffer1
            copyto!(info.primalBuffer1[block.id], info.lineSearchPrimalBuffer[block.id])
            axpy!(-1.0, info.primalSol[block.id], info.primalBuffer1[block.id])
        end


        info.dualBuffer .= 0.0 
        objDiffEstimate = 0.0 
        primalDiffSquare = 0.0 
        for bi in 1:nPrimalBlocks
            block = mbp.blocks[bi]
            # store Ax^{k+1} in dualBuffer
            mappings[block.id](info.lineSearchPrimalBuffer[block.id], info.dualBuffer, true) 
            # compute f(x^{k+1}) - f(x^k) - <nabla f(x^k), x^{k+1} - x^k>
            objDiffEstimate += block.f(info.lineSearchPrimalBuffer[block.id]) - block.f(info.primalSol[block.id]) 
            objDiffEstimate -= dot(info.primalBuffer2[block.id], info.primalBuffer1[block.id])
            # compute ||x^{k+1} - x^k||^2
            primalDiffSquare += dot(info.primalBuffer1[block.id], info.primalBuffer1[block.id])
        end 
        
        # store Ax^{k+1} - Ax^k in dualBuffer
        axpy!(-1.0, info.bufferAx, info.dualBuffer)

        # compute the left hand side of the inequality
        checkLHS = newGamma * newSigma * dot(info.dualBuffer, info.dualBuffer) + 2.0 * newGamma * objDiffEstimate  
        # compute the right hand side of the inequality
        checkRHS = param.backtrackDescentRatio * primalDiffSquare 

        if checkLHS <= checkRHS 
            break 
        else 
            newSigma = newSigma * param.mu 
        end 

        if iter == param.lineSearchMaxIter 
            @PDMOWarn param.logLevel "MaliskyPock: linesearch failed to terminate in $(param.lineSearchMaxIter) iterations."
        end         
    end 

    for bi in 1:nPrimalBlocks
        block = mbp.blocks[bi]
        copyto!(info.primalSolPrev[block.id], info.primalSol[block.id])
        copyto!(info.primalSol[block.id], info.lineSearchPrimalBuffer[block.id])
    end 

    push!(info.numberBacktracks, countBacktracks)
    info.dualStepSizePrev = info.dualStepSize 
    info.dualStepSize = newSigma 
    info.primalStepSize = newGamma 
end 


"""
    setupInitialPrimalDualStepSize!(info::AdaPDMIterationInfo, param::MalitskyPockParam)

Initializes the primal and dual step sizes for the Malitsky-Pock algorithm.
The primal step size is initially set to 1, while the dual step size is set based on 
the parameters provided in `param`. The previous dual step size is set as a fraction 
of the initial dual step size, controlled by the `initialTheta` parameter.
"""
function setupInitialPrimalDualStepSize!(info::AdaPDMIterationInfo, param::MalitskyPockParam)
    info.primalStepSize = 1.0
    info.dualStepSize = param.initialSigma
    info.dualStepSizePrev = param.initialSigma * param.initialTheta 
end