"""
    updateDualSolution!(mbp::MultiblockProblem, info::AdaPDMIterationInfo, param::AdaPDMParam)

Update the dual solution and step sizes for the standard Adaptive Primal-Dual Method.

# Arguments
- `mbp::MultiblockProblem`: The multiblock problem being solved
- `info::AdaPDMIterationInfo`: The current iteration information
- `param::AdaPDMParam`: The parameters for the AdaPDM algorithm

# Details
This function:
1. Computes Lipschitz and cocoercivity estimates based on current iterates
2. Calculates the new primal step size (gamma) using three different criteria and takes the minimum
3. Computes the dual step size (sigma) based on the primal step size and the parameter t
4. Updates the proximal center for the conjugate proximal oracle
5. Performs the proximal oracle step to compute the new dual solution
6. Updates the step size information in the iteration info object

The new gamma is computed as the minimum of:
- v1: Controlled increase based on previous step size
- v2: Based on operator norm estimate
- v3: Based on parameter delta, xi, and epsilon calculations
"""
function updateDualSolution!(mbp::MultiblockProblem, info::AdaPDMIterationInfo, param::AdaPDMParam)
    # compute Lipschitz and cocoercivity estimates and step size parameters 
    lipschitzEstimate, cocoercivityEstimate = computeLipschitzAndCocoercivityEstimate(mbp, info)
    delta = info.primalStepSize * lipschitzEstimate * (info.primalStepSize * cocoercivityEstimate - 1.0)
    xi = (param.t * info.primalStepSize * info.opNormEstimate)^2 

    # compute the new gamma 
    v1 = info.primalStepSize * sqrt(1.0 + info.primalStepSize / info.primalStepSizePrev)
    v2 = 1.0 / (2.0 * param.stepSizeNu * param.t * info.opNormEstimate)
    numerator = 1.0 - 4.0 * xi * (1.0 + param.stepSizeEpsilon)^2
    denominator = 2.0 * (1.0 + param.stepSizeEpsilon) * (sqrt(delta^2 + xi * numerator) + delta)
    v3 = info.primalStepSize * sqrt(numerator / denominator)
    newGamma = min(v1, v2, v3)
    sigma = param.t^2 * newGamma 

    # update the primal dual solution 
    ratio2 = newGamma / info.primalStepSize     
    ratio1 = ratio2 + 1.0
    prepareProximalCenterForConjugateProximalOracle!(info, sigma, ratio1, ratio2) 
    copyto!(info.dualSolPrev, info.dualSol)
    proximalOracleOfConjugate!(info.dualSol, mbp.blocks[end].g, info.dualBuffer, sigma)

    info.primalStepSizePrev = info.primalStepSize
    info.primalStepSize = newGamma 
    info.dualStepSize = sigma
end 

"""
    updatePrimalSolution!(mbp::MultiblockProblem, info::AdaPDMIterationInfo, param::AdaPDMParam)

Update the primal solutions for all blocks in the multiblock problem using the proximal oracle.

# Arguments
- `mbp::MultiblockProblem`: The multiblock problem being solved
- `info::AdaPDMIterationInfo`: The current iteration information
- `param::AdaPDMParam`: The parameters for the AdaPDM algorithm

# Details
This function updates the primal solution for each block in the multiblock problem
by calling the `updatePrimalSolution!` function for each block. The function iterates
over all blocks except the last one, which corresponds to the dual variable.
"""
function updatePrimalSolution!(mbp::MultiblockProblem, 
    info::AdaPDMIterationInfo, 
    param::AdaPDMParam)
    nPrimalBlocks = length(mbp.blocks) - 1
    @threads for bi in 1:nPrimalBlocks
        updatePrimalSolution!(mbp.blocks[bi], mbp, info)
    end 
end 

""" 
    setupInitialPrimalDualStepSize!(info::AdaPDMIterationInfo, param::AdaPDMParam)

Setup the initial primal and dual step sizes for the AdaPDM algorithm.

# Arguments
- `info::AdaPDMIterationInfo`: The iteration information object to initialize
- `param::AdaPDMParam`: The parameters for the AdaPDM algorithm

# Details
This function initializes the primal step size to the initial gamma value from the parameters,
the previous primal step size to the initial gamma previous value, and sets the dual step size to zero.
These values are used in the first iteration of the algorithm before adaptive step sizes are computed.
"""
function setupInitialPrimalDualStepSize!(info::AdaPDMIterationInfo, param::AdaPDMParam)
    info.primalStepSize = param.initialGamma
    info.primalStepSizePrev = param.initialGammaPrev
    info.dualStepSize = 0.0
end 
