"""
    AdaPDMPlusParam <: AbstractAdaPDMParam

Parameters for the Adaptive Primal-Dual Method Plus, which extends the basic AdaPDM with additional 
parameters for line search and operator norm estimation.

# Fields
- `t::Float64`: Primal/dual step size ratio
- `stepSizeEpsilon::Float64`: Step size parameter epsilon, i.e., 1e-6 in the paper
- `stepSizeNu::Float64`: Step size parameter nu, i.e., 1.2 in the paper
- `initialGamma::Float64`: Initial gamma, i.e., gamma_0
- `initialGammaPrev::Float64`: Initial gamma prev, i.e., gamma_{-1}
- `initialNormEstimate::Float64`: ||A|| or its estimate eta_0
- `backtrackingFactor::Float64`: Backtracking factor for linesearch, i.e., r = 2.0 in the paper
- `normEstimateShrinkingFactor::Float64`: Shrinking factor for sequence of eta, i.e., 0.95 in the paper
- `presTolL2::Float64`: Primal residual tolerance in L2 norm
- `dresTolL2::Float64`: Dual residual tolerance in L2 norm
- `presTolLInf::Float64`: Primal residual tolerance in LInf norm
- `dresTolLInf::Float64`: Dual residual tolerance in LInf norm
- `maxIter::Int64`: Maximum number of iterations
- `logInterval::Int64`: Interval for logging information
- `timeLimit::Float64`: Time limit in seconds
- `logLevel::Int64`: Level for logging information
"""
mutable struct AdaPDMPlusParam <: AbstractAdaPDMParam
    # parameters for the algorithm 
    t::Float64                  # primal/dual step size ratio 
    stepSizeEpsilon::Float64    # step size parameter epsilon, i.e., 1e-6 in the paper 
    stepSizeNu::Float64         # step size parameter nu, i.e., 1.2 in the paper 
    initialGamma::Float64       # initial gamma, i.e., gamma_0
    initialGammaPrev::Float64   # initial gamma prev, i.e., gamma_{-1}
    
    # estimate of ||A||
    initialNormEstimate::Float64           # ||A|| or its estimate eta_0 
    backtrackingFactor::Float64            # backtracking factor for linesearch, i.e., r = 2.0 in the paper 
    normEstimateShrinkingFactor::Float64   # shrinking factor for sequence of eta, i.e., 0.95 in the paper 

    # tolerances for the stopping criteria 
    presTolL2::Float64          # primal residual tolerance in L2 norm 
    dresTolL2::Float64          # dual residual tolerance in L2 norm 
    presTolLInf::Float64        # primal residual tolerance in LInf norm 
    dresTolLInf::Float64        # dual residual tolerance in LInf norm 

    # limit and logging parameters 
    lineSearchMaxIter::Int64 # Maximum number of iterations for line search
    maxIter::Int64           # Maximum number of iterations
    logInterval::Int64       # Interval for logging information
    timeLimit::Float64       # Time limit in seconds
    logLevel::Int64          # Level for logging information
end

"""
    AdaPDMPlusParam(mbp::MultiblockProblem; t::Float64=1.0, stepSizeEpsilon::Float64=1e-6, stepSizeNu::Float64=1.2, opNormEstimateProvided::Float64=Inf, backtrackingFactor::Float64=2.0, normEstimateShrinkingFactor::Float64=0.95, presTolL2::Float64=1e-4, dresTolL2::Float64=1e-4, presTolLInf::Float64=1e-6, dresTolLInf::Float64=1e-6, maxIter::Int64=10000, logInterval::Int64=1000, timeLimit::Float64=3600.0)

Construct a parameter object for the Adaptive Primal-Dual Method Plus, which includes additional parameters
for line search and adaptive operator norm estimation.

# Arguments
- `mbp::MultiblockProblem`: The multiblock problem to analyze.      
- `t::Float64=1.0`: Primal/dual step size ratio.
- `stepSizeEpsilon::Float64=1e-6`: Step size parameter epsilon.
- `stepSizeNu::Float64=1.2`: Step size parameter nu.
- `opNormEstimateProvided::Float64=Inf`: Provided estimate of the operator norm. If Inf, it will be computed.
- `backtrackingFactor::Float64=2.0`: Backtracking factor for linesearch.
- `normEstimateShrinkingFactor::Float64=0.95`: Shrinking factor for sequence of eta.
- `presTolL2::Float64=1e-4`: Primal residual tolerance in L2 norm.
- `dresTolL2::Float64=1e-4`: Dual residual tolerance in L2 norm.
- `presTolLInf::Float64=1e-6`: Primal residual tolerance in LInf norm.
- `dresTolLInf::Float64=1e-6`: Dual residual tolerance in LInf norm.
- `maxIter::Int64=10000`: Maximum number of iterations.
- `logInterval::Int64=1000`: Interval for logging information.
- `timeLimit::Float64=3600.0`: Time limit in seconds.
- `logLevel::Int64=1`: Level for logging information.

# Returns
- `AdaPDMPlusParam`: A parameter object for the Adaptive Primal-Dual Method Plus.

# Throws
- `ErrorException`: If the input problem is not a valid composite problem.
"""
function AdaPDMPlusParam(mbp::MultiblockProblem; 
    t::Float64=1.0,
    stepSizeEpsilon::Float64=1e-6,
    stepSizeNu::Float64=1.2,
    opNormEstimateProvided::Float64=Inf, 
    backtrackingFactor::Float64=2.0, 
    normEstimateShrinkingFactor::Float64=1.00,
    presTolL2::Float64=1e-4,
    dresTolL2::Float64=1e-4,
    presTolLInf::Float64=1e-6,
    dresTolLInf::Float64=1e-6,
    lineSearchMaxIter::Int64=1000,
    maxIter::Int64=10000,
    logInterval::Int64=1000,
    timeLimit::Float64=3600.0,
    logLevel::Int64=1)
    
    if checkCompositeProblemValidity!(mbp) == false 
        error("AdaPDMParam: the input problem is not a valid composite problem.")
    end 
    
    # estimate ||A|| 
    opNormEstimate = opNormEstimateProvided < Inf ? opNormEstimateProvided : computeNormEstimate(mbp)

    # estimate gamma_0 and gamma_{-1}
    initialGamma = 1/(2.0 * stepSizeNu * t * opNormEstimate)
    initialGammaPrev = initialGamma 

    # initialize the parameter object 
    return AdaPDMPlusParam(
        t, 
        stepSizeEpsilon, 
        stepSizeNu, 
        initialGamma, 
        initialGammaPrev, 
        opNormEstimate, 
        backtrackingFactor,     
        normEstimateShrinkingFactor, 
        presTolL2, 
        dresTolL2, 
        presTolLInf, 
        dresTolLInf, 
        lineSearchMaxIter,
        maxIter, 
        logInterval, 
        timeLimit, 
        logLevel)
end