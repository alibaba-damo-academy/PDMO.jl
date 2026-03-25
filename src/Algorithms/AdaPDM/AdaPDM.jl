include("AbstractAdaPDMParam/AbstractAdaPDMParam.jl")
include("AdaPDMIterationInfo/AdaPDMIterationInfo.jl")
include("AdaPDMTerminationCriteria.jl")
include("AdaPDMUtil.jl")

"""
    AdaptivePrimalDualMethod(mbp::MultiblockProblem, param::AbstractAdaPDMParam)

Internal AdaPDM loop used by `runAdaPDM`.

# Arguments
- `mbp::MultiblockProblem`: Composite multiblock problem.
- `param::AbstractAdaPDMParam`: Parameter object selecting the algorithm variant
  (`AdaPDMParam`, `AdaPDMPlusParam`, `MalitskyPockParam`, or `CondatVuParam`).

# Returns
- `AdaPDMIterationInfo`: Iteration history, residuals, objective values, timing, and
  termination status.

See also: `runAdaPDM`, `AdaPDMIterationInfo`, `AdaPDMTerminationCriteria`
"""
function AdaptivePrimalDualMethod(mbp::MultiblockProblem, param::AbstractAdaPDMParam) 
    startTime = time() 

    @PDMOInfo param.logLevel "#"^40 * " Adaptive Primal-dual Method " * "#"^40
    @PDMOInfo param.logLevel "Method = $(getAdaPDMName(param))"
    if checkCompositeProblemValidity!(mbp) == false 
        @PDMOError param.logLevel "AdaptiveProximalGradientMethod: the input problem is not a valid composite problem."
        return
    end 

    # Initialize iteration info and termination criteria 
    info = AdaPDMIterationInfo(mbp, param)
    terminationCriteria = AdaPDMTerminationCriteria(param)

    msg = Printf.@sprintf("AdaPDM: initialization took %.2f seconds \n", time() - startTime)
    @PDMOInfo param.logLevel msg 

    startTime = time()
    AdaPDMLog(0, info, param)

    # start the iteration 
    for iter in 1:param.maxIter 
        updateDualSolution!(mbp, info, param)
        updatePrimalSolution!(mbp, info, param)
        computePDMResidualsAndObjective!(info, mbp, param)

        # log iteration info 
        info.totalTime = time() - startTime 
        iterLogged = AdaPDMLog(iter, info, param)

        # check termination criteria 
        checkTerminationCriteria(info, terminationCriteria)
        if terminationCriteria.terminated 
            if iterLogged == false 
                AdaPDMLog(iter, info, param; final = true)
            end 
            break 
        end 
    end 

    return info 
end
