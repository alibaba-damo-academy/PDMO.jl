"""
    updateDualSolution!(mbp::MultiblockProblem, info::AdaPDMIterationInfo, param::AdaPDMPlusParam)

Update the dual solution and step sizes for the Adaptive Primal-Dual Method Plus, which includes
adaptive operator norm estimation and backtracking line search.

# Arguments
- `mbp::MultiblockProblem`: The multiblock problem being solved
- `info::AdaPDMIterationInfo`: The current iteration information
- `param::AdaPDMPlusParam`: The parameters for the AdaPDMPlus algorithm

# Details
This function:
1. Computes Lipschitz and cocoercivity estimates based on current iterates
2. Performs an adaptive line search to find an appropriate operator norm estimate:
   - Starts with a reduced operator norm estimate from the previous iteration
   - Calculates a new primal step size based on this estimate
   - Checks if the estimate satisfies a specific criterion involving the dual iterates
   - If not, increases the estimate and repeats
3. Updates the dual solution using the proximal oracle with the computed step size
4. Updates the step size and operator norm estimate information in the iteration info object

The line search helps ensure robustness by adaptively finding appropriate step sizes
that maintain algorithm stability while potentially allowing larger steps than the
standard AdaPDM algorithm.
"""
function updateDualSolution!(mbp::MultiblockProblem, info::AdaPDMIterationInfo, param::AdaPDMPlusParam) 
    # compute Lipschitz and cocoercivity estimates and step size parameters 
    lipschitzEstimate, cocoercivityEstimate = computeLipschitzAndCocoercivityEstimate(mbp, info)
    delta = info.primalStepSize * lipschitzEstimate * (info.primalStepSize * cocoercivityEstimate - 1.0)
    xi = (param.t * info.primalStepSize * info.opNormEstimate * (1.0 + param.stepSizeEpsilon))^2 
    
    newEta = info.opNormEstimate * param.normEstimateShrinkingFactor 
    newGamma = 0.0 
    sigma = 0.0
    mappings = mbp.constraints[1].mappings
    nPrimalBlocks = length(mbp.blocks) - 1
    
    countBacktracks = 0 
    for iter in 1:param.lineSearchMaxIter
        countBacktracks += 1
        # compute the new gamma 
        v1 = info.primalStepSize * sqrt(1.0 + info.primalStepSize / info.primalStepSizePrev)
        v2 = 1.0 / (2.0 * param.stepSizeNu * param.t * newEta)
        numerator = 1.0 - 4.0 * xi 
        denominator = 2.0 * (1.0 + param.stepSizeEpsilon) * (sqrt(delta^2 + (param.t * newEta * info.primalStepSize)^2 * numerator) + delta)
        v3 = info.primalStepSize * sqrt(numerator / denominator)
        newGamma = min(v1, v2, v3)

        sigma = param.t^2 * newGamma

        ratio2 = newGamma / info.primalStepSize 
        ratio1 = ratio2 + 1.0
        prepareProximalCenterForConjugateProximalOracle!(info, sigma, ratio1, ratio2) 
        proximalOracleOfConjugate!(info.lineSearchDualBuffer, mbp.blocks[end].g, info.dualBuffer, sigma)

        # dualBuffer <- y^{k+1} - y^k
        copyto!(info.dualBuffer, info.lineSearchDualBuffer)
        axpy!(-1.0, info.dualSol, info.dualBuffer)

        denominator = norm(info.dualBuffer, 2) # norm of dual difference 
        numerator = 0.0 
        for bi in 1:nPrimalBlocks
            block = mbp.blocks[bi]
            adjoint!(mappings[block.id], info.dualBuffer, info.primalBuffer1[block.id])
            numerator += dot(info.primalBuffer1[block.id], info.primalBuffer1[block.id])
        end 
        numerator = sqrt(numerator)

        ratio = denominator < ZeroTolerance ? 0.0 : numerator / denominator 
        if newEta >= ratio 
            break 
        else 
            newEta = newEta * param.backtrackingFactor 
        end  

        if iter == param.lineSearchMaxIter
            @PDMOWarn param.logLevel "AdaPDM: linesearch failed to terminate in $(param.lineSearchMaxIter) iterations."
        end 
    end 

    copyto!(info.dualSolPrev, info.dualSol)
    copyto!(info.dualSol, info.lineSearchDualBuffer)
    
    push!(info.numberBacktracks, countBacktracks)
    info.opNormEstimate = newEta 
    info.primalStepSizePrev = info.primalStepSize
    info.primalStepSize = newGamma 
    info.dualStepSize = sigma
end 

"""
    updatePrimalSolution!(mbp::MultiblockProblem, info::AdaPDMIterationInfo, param::AdaPDMPlusParam)

Update the primal solutions for all blocks in the multiblock problem using the proximal oracle.

# Arguments
- `mbp::MultiblockProblem`: The multiblock problem being solved
- `info::AdaPDMIterationInfo`: The current iteration information
- `param::AdaPDMPlusParam`: The parameters for the AdaPDMPlus algorithm

# Details
This function updates the primal solution for each block in the multiblock problem
by calling the `updatePrimalSolution!` function for each block. The function iterates
over all blocks except the last one, which corresponds to the dual variable.
"""
function updatePrimalSolution!(mbp::MultiblockProblem, 
    info::AdaPDMIterationInfo, 
    param::AdaPDMPlusParam)
    nPrimalBlocks = length(mbp.blocks) - 1
    @threads for bi in 1:nPrimalBlocks
        updatePrimalSolution!(mbp.blocks[bi], mbp, info)
    end 
end 

""" 
    setupInitialPrimalDualStepSize!(info::AdaPDMIterationInfo, param::AdaPDMPlusParam)

Setup the initial primal and dual step sizes for the AdaPDMPlus algorithm.

# Arguments
- `info::AdaPDMIterationInfo`: The iteration information object to initialize
- `param::AdaPDMPlusParam`: The parameters for the AdaPDMPlus algorithm

# Details
This function initializes the primal step size to the initial gamma value from the parameters,
the previous primal step size to the initial gamma previous value, and sets the dual step size to zero.
These values are used in the first iteration of the algorithm before adaptive step sizes are computed.
"""
function setupInitialPrimalDualStepSize!(info::AdaPDMIterationInfo, param::AdaPDMPlusParam)
    info.primalStepSize = param.initialGamma
    info.primalStepSizePrev = param.initialGammaPrev
    info.dualStepSize = 0.0
end 