"""
    updateDualSolution!(mbp::MultiblockProblem, info::AdaPDMIterationInfo, param::CondatVuParam)

Updates the dual solution in the Condat-Vu algorithm. This function:
1. Prepares the proximal center for the conjugate proximal oracle
2. Stores the current dual solution in the previous dual solution buffer
3. Applies the proximal oracle of the conjugate function to compute the new dual solution
"""
function updateDualSolution!(mbp::MultiblockProblem, info::AdaPDMIterationInfo, param::CondatVuParam) 
    prepareProximalCenterForConjugateProximalOracle!(info, info.dualStepSize, 2.0, 1.0)
    copyto!(info.dualSolPrev, info.dualSol)
    proximalOracleOfConjugate!(info.dualSol, mbp.blocks[end].g, info.dualBuffer, info.dualStepSize)
end


"""
    updatePrimalSolution!(mbp::MultiblockProblem, info::AdaPDMIterationInfo, param::CondatVuParam)

Updates the primal solution in the Condat-Vu algorithm. This function:
1. Computes the new step sizes based on the adaptive scheme
2. Computes gradients for each primal block
3. Performs line search to find suitable step sizes satisfying the descent condition
4. Updates the primal and dual step sizes and solutions

The line search is based on a sufficient decrease condition that balances
the change in objective value with the squared norm of the primal variable difference.
"""
function updatePrimalSolution!(mbp::MultiblockProblem, 
    info::AdaPDMIterationInfo, 
    param::CondatVuParam)
    nPrimalBlocks = length(mbp.blocks) - 1
    @threads for bi in 1:nPrimalBlocks
        updatePrimalSolution!(mbp.blocks[bi], mbp, info)
    end 
end 


"""
    setupInitialPrimalDualStepSize!(info::AdaPDMIterationInfo, param::CondatVuParam)

Initializes the primal and dual step sizes for the Condat-Vu algorithm.
The primal step size is initially set to the primal step size provided in `param`,
while the dual step size is set to the dual step size provided in `param`.
The previous primal and dual step sizes are set to the initial values.
"""
function setupInitialPrimalDualStepSize!(info::AdaPDMIterationInfo, param::CondatVuParam)
    info.primalStepSize = param.primalStepSize
    info.primalStepSizePrev = info.primalStepSize 
    info.dualStepSize = param.dualStepSize
    info.dualStepSizePrev = info.dualStepSize 
end