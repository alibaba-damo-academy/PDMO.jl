"""
    CondatVuParam <: AbstractAdaPDMParam

Parameters for the Condat-Vũ primal-dual algorithm.

The Condat-Vũ algorithm is a primal-dual method for solving composite optimization problems
of the form: minimize f(x) + g(Ax), where f is smooth and g is proximable. It is particularly
well-suited for problems where the linear operator A has a known operator norm.

# Fields

## Algorithm Parameters
- `primalStepSize::Float64`: Step size for primal variable updates (α)
- `dualStepSize::Float64`: Step size for dual variable updates (β)  
- `opNormEstimate::Float64`: Estimate of the operator norm ||A||
- `LipschitzConstantEstimate::Float64`: Estimate of the Lipschitz constant of ∇f

## Convergence Tolerances
- `presTolL2::Float64`: Primal residual tolerance in L2 norm
- `dresTolL2::Float64`: Dual residual tolerance in L2 norm
- `presTolLInf::Float64`: Primal residual tolerance in L∞ norm
- `dresTolLInf::Float64`: Dual residual tolerance in L∞ norm

## Iteration Control
- `maxIter::Int64`: Maximum number of iterations
- `logInterval::Int64`: Interval for logging algorithm progress
- `timeLimit::Float64`: Time limit in seconds
- `logLevel::Int64`: Level for logging information

# Algorithm Theory
The Condat-Vũ algorithm uses the following update scheme:
1. x^{k+1} = prox_{αf}(x^k - α A^T y^k)
2. y^{k+1} = prox_{βg*}(y^k + β A(2x^{k+1} - x^k))

The step sizes must satisfy: αβ||A||² < 1 for convergence.

# References
- Condat, L. (2013). A primal-dual splitting method for convex optimization involving Lipschitzian, proximable and linear composite terms
- Vũ, B. C. (2013). A splitting algorithm for dual monotone inclusions involving cocoercive operators

See also: `AdaPDMParam`, `MalitskyPockParam`, `runAdaPDM`
"""
mutable struct CondatVuParam <: AbstractAdaPDMParam
    # parameters for the algorithm
    primalStepSize::Float64 
    dualStepSize::Float64 
    opNormEstimate::Float64
    LipschitzConstantEstimate::Float64 

    # tolerances for the stopping criteria 
    presTolL2::Float64        # primal residual tolerance in L2 norm 
    dresTolL2::Float64        # dual residual tolerance in L2 norm 
    presTolLInf::Float64      # primal residual tolerance in LInf norm 
    dresTolLInf::Float64      # dual residual tolerance in LInf norm 

    # limit and logging parameters 
    maxIter::Int64           # Maximum number of iterations
    logInterval::Int64       # Interval for logging information
    timeLimit::Float64       # Time limit in seconds
    logLevel::Int64          # Level for logging information
end


"""
    CondatVuParam(mbp::MultiblockProblem; kwargs...)

Create parameters for the Condat-Vũ primal-dual algorithm.

This constructor automatically estimates algorithm parameters based on the problem structure
and applies the Condat-Vũ convergence conditions. The step sizes are chosen to satisfy
the convergence criterion: αβ||A||² < 1.

# Arguments
- `mbp::MultiblockProblem`: The composite multiblock problem to solve

# Keyword Arguments

## Step Size Parameters
- `alphaProvided::Float64=Inf`: Primal step size (α). If Inf, automatically computed as 1/(2L) where L is the Lipschitz constant
- `betaProvided::Float64=Inf`: Dual step size (β). If Inf, automatically computed based on operator norm
- `opNormEstimateProvided::Float64=Inf`: Operator norm estimate ||A||. If Inf, computed automatically
- `LipschitzConstantEstimateProvided::Float64=Inf`: Lipschitz constant estimate. If Inf, estimated from smooth functions

## Convergence Tolerances
- `presTolL2::Float64=1e-4`: Primal residual tolerance in L2 norm
- `dresTolL2::Float64=1e-4`: Dual residual tolerance in L2 norm  
- `presTolLInf::Float64=1e-6`: Primal residual tolerance in L∞ norm
- `dresTolLInf::Float64=1e-6`: Dual residual tolerance in L∞ norm

## Algorithm Control
- `lineSearchMaxIter::Int64=1000`: Maximum iterations for line search (unused in this implementation)
- `maxIter::Int64=10000`: Maximum number of algorithm iterations
- `logInterval::Int64=1000`: Logging interval for progress reporting
- `timeLimit::Float64=3600.0`: Time limit in seconds
- `logLevel::Int64=1`: Level for logging information
# Returns
- `CondatVuParam`: Configured parameters for the Condat-Vũ algorithm

# Examples
```julia
# Basic usage with automatic parameter selection
mbp = MultiblockProblem()
# ... set up composite problem ...
param = CondatVuParam(mbp)

# Custom tolerances
param = CondatVuParam(mbp; presTolL2=1e-6, maxIter=5000)

# Manual step size specification
param = CondatVuParam(mbp; alphaProvided=0.01, betaProvided=0.1)

# Solve with Condat-Vũ algorithm
result = runAdaPDM(mbp, param)
```

# Algorithm Details
The constructor performs the following automatic parameter estimation:
1. **Operator Norm**: Estimates ||A|| from the linear mappings in constraints
2. **Lipschitz Constant**: Estimates L from the smooth functions in each block
3. **Step Sizes**: Computes α = 1/(2L) and β = L/||A||² to ensure convergence

# Problem Requirements
The problem must satisfy the composite structure required by Condat-Vũ:
- Each block (except the last) contains smooth functions with gradient oracles
- The last block represents the coupling constraint
- The problem structure is validated using `checkCompositeProblemValidity!`

# Error Conditions
Throws an error if the input problem is not a valid composite problem structure.

See also: `runAdaPDM`, `AdaPDMParam`, `MalitskyPockParam`
"""
function CondatVuParam(mbp::MultiblockProblem; 
    alphaProvided::Float64=Inf,
    betaProvided::Float64=Inf, 
    opNormEstimateProvided::Float64=Inf, 
    LipschitzConstantEstimateProvided::Float64=Inf, 
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

    opNormEstimate = opNormEstimateProvided < Inf ? opNormEstimateProvided : computeNormEstimate(mbp)

    LipschitzConstantEstimate = 0.0 
    if LipschitzConstantEstimateProvided < Inf 
        LipschitzConstantEstimate = LipschitzConstantEstimateProvided 
    else 
        nPrimalBlocks = length(mbp.blocks) - 1
        for bi in 1:nPrimalBlocks
            block = mbp.blocks[bi]
            LipschitzConstantEstimate = max(LipschitzConstantEstimate, estimateLipschitzConstant(block.f, block.val))
        end 
    end 

    alpha = alphaProvided < Inf ? alphaProvided : 1.0 / (2.0 * max(LipschitzConstantEstimate, 1.0))
    beta = betaProvided < Inf ? betaProvided : max(1.0, LipschitzConstantEstimate) / max(1.0, opNormEstimate^2)

    return CondatVuParam(
        alpha, 
        beta, 
        opNormEstimate, 
        LipschitzConstantEstimate, 
        presTolL2, 
        dresTolL2, 
        presTolLInf, 
        dresTolLInf, 
        maxIter, 
        logInterval, 
        timeLimit, 
        logLevel)
end 