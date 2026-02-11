"""
    AdaPDMTerminationStatus

Enum representing the termination status of the Adaptive Primal-Dual Method.

# Values
- `ADA_PDM_TERMINATION_UNSPECIFIED`: Termination status not yet determined
- `ADA_PDM_TERMINATION_OPTIMAL`: Converged to optimal solution
- `ADA_PDM_TERMINATION_ITERATION_LIMIT`: Reached maximum number of iterations
- `ADA_PDM_TERMINATION_TIME_LIMIT`: Reached time limit
- `ADA_PDM_TERMINATION_UNBOUNDED`: Problem is unbounded
- `ADA_PDM_TERMINATION_UNKNOWN`: Terminated for unknown reason
"""
@enum AdaPDMTerminationStatus begin
    ADA_PDM_TERMINATION_UNSPECIFIED
    ADA_PDM_TERMINATION_OPTIMAL
    ADA_PDM_TERMINATION_ITERATION_LIMIT
    ADA_PDM_TERMINATION_TIME_LIMIT
    ADA_PDM_TERMINATION_UNBOUNDED
    ADA_PDM_TERMINATION_UNKNOWN
end
 
"""
    AdaPDMIterationInfo

Structure to store and track the information about the iterations of the Adaptive Primal-Dual Method.

# Fields
- `presL2::Vector{Float64}`: Primal residuals in L2 norm at each iteration
- `dresL2::Vector{Float64}`: Dual residuals in L2 norm at each iteration
- `presLInf::Vector{Float64}`: Primal residuals in L-infinity norm at each iteration
- `dresLInf::Vector{Float64}`: Dual residuals in L-infinity norm at each iteration
- `lagrangianObj::Vector{Float64}`: Lagrangian objective values at each iteration
- `numberBacktracks::Vector{Int64}`: Number of backtracks in the linesearch for each iteration 
- `primalSol::Dict{BlockID, NumericVariable}`: Current primal solution (x^{k+1})
- `primalSolPrev::Dict{BlockID, NumericVariable}`: Previous primal solution (x^k)
- `primalBuffer1::Dict{BlockID, NumericVariable}`: Buffer for primal variable computations
- `primalBuffer2::Dict{BlockID, NumericVariable}`: Additional buffer for primal variable computations
- `lineSearchPrimalBuffer::Dict{BlockID, NumericVariable}`: Buffer for line search operations
- `dualSol::NumericVariable`: Current dual solution (y^{k+1})
- `dualSolPrev::NumericVariable`: Previous dual solution (y^k)
- `bufferAx::NumericVariable`: Buffer for Ax^{k+1}
- `bufferAxPrev::NumericVariable`: Buffer for Ax^k
- `lineSearchDualBuffer::NumericVariable`: Buffer for line search operations
- `dualBuffer::NumericVariable`: Buffer for dual variable computations
- `primalStepSize::Float64`: Current primal step size (gamma_{k+1})
- `primalStepSizePrev::Float64`: Previous primal step size (gamma_k)
- `dualStepSize::Float64`: Current dual step size (sigma_{k+1})
- `opNormEstimate::Float64`: Estimate of the operator norm
- `stopIter::Int64`: Iteration at which the algorithm stopped
- `totalTime::Float64`: Total execution time
- `terminationStatus::AdaPDMTerminationStatus`: Status indicating how the algorithm terminated
"""
mutable struct AdaPDMIterationInfo
    # algorithmic history
    presL2::Vector{Float64}
    dresL2::Vector{Float64}
    presLInf::Vector{Float64}
    dresLInf::Vector{Float64}
    lagrangianObj::Vector{Float64}
    objectiveValue::Vector{Float64}   # primal objective: Σ_i (f_i(x_i)+g_i(x_i)) + h(Ax)
    numberBacktracks::Vector{Int64} # number of backtracks in the linesearch for each iteration 
    
    # primal variables 
    primalSol::Dict{BlockID, NumericVariable}              # x^{k+1}         
    primalSolPrev::Dict{BlockID, NumericVariable}          # x^{k}
    primalBuffer1::Dict{BlockID, NumericVariable}          # buffer for intermediate computations 
    primalBuffer2::Dict{BlockID, NumericVariable}          # buffer for intermediate computations 
    lineSearchPrimalBuffer::Dict{BlockID, NumericVariable} # buffer for linesearch 

    # dual variables 
    dualSol::NumericVariable                     # y^{k+1}
    dualSolPrev::NumericVariable                 # y^k
    bufferAx::NumericVariable                    # Ax^{k+1}
    bufferAxPrev::NumericVariable                # Ax^k
    lineSearchDualBuffer::NumericVariable        # buffer for linesearch 
    dualBuffer::NumericVariable                  # buffer for intermediate computations 
    
    # step sizes 
    primalStepSize::Float64 
    primalStepSizePrev::Float64
    dualStepSize::Float64
    dualStepSizePrev::Float64 

    # dynamic estimate of operator norm ||A||
    opNormEstimate::Float64

    # stopping criteria 
    stopIter::Int64
    totalTime::Float64
    terminationStatus::AdaPDMTerminationStatus

    """
        AdaPDMIterationInfo()

    Default constructor for AdaPDMIterationInfo, initializing all fields with default values.
    """
    AdaPDMIterationInfo() = new(
        Vector{Float64}(),
        Vector{Float64}(),
        Vector{Float64}(),
        Vector{Float64}(),
        Vector{Float64}(),
        Vector{Float64}(),
        Vector{Int64}(),
        Dict{BlockID, NumericVariable}(),
        Dict{BlockID, NumericVariable}(),
        Dict{BlockID, NumericVariable}(),
        Dict{BlockID, NumericVariable}(),
        Dict{BlockID, NumericVariable}(),
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # dualSol, dualSolPrev, bufferAx, bufferAxPrev, dualBuffer
        Inf, Inf, Inf, Inf, # primalStepSize, primalStepSizePrev, dualStepSize, dualStepSizePrev
        0.0, # normEstimate
        -1, 0.0,
        ADA_PDM_TERMINATION_UNSPECIFIED)
end

include("AdaPDMIterationInfoCommon.jl")
include("AdaPDMHelpers.jl")
include("AdaPDMPlusHelpers.jl")
include("MalitskyPockHelpers.jl")
include("CondatVuHelpers.jl")


"""
    computeAdaPDMDualResiduals!(info::AdaPDMIterationInfo, mbp::MultiblockProblem, param::AbstractAdaPDMParam)

Compute the dual residuals for the Adaptive Primal-Dual Method.

# Arguments
- `info::AdaPDMIterationInfo`: The iteration information object to update
- `mbp::MultiblockProblem`: The multiblock problem being solved
- `param::AbstractAdaPDMParam`: Parameters for the algorithm

# Details
This function computes the dual residuals by:
1. For Malitsky-Pock algorithm, calculating y^{k+1} - bar{y}^{k+1}
2. For each block, computing gradient differences and step-size adjusted variable differences
3. For Malitsky-Pock, applying the adjoint mapping
4. Computing both L2 and L-infinity norms of the residuals
5. Storing the computed residuals in the iteration info object
"""
function computeAdaPDMDualResiduals!(info::AdaPDMIterationInfo, mbp::MultiblockProblem, param::AbstractAdaPDMParam)
    # compute dual residuals: 
    dresL2 = 0.0
    dresLInf = 0.0

    if isa(param, MalitskyPockParam)
        axpby!(1.0, info.dualSol, -1.0, info.lineSearchDualBuffer) # y^{k+1} - \bar{y}^{k+1}
    end 

    mappings = mbp.constraints[1].mappings 
    for block in mbp.blocks[1:end-1]
        gradientOracle!(info.primalBuffer1[block.id], block.f, info.primalSol[block.id])
        gradientOracle!(info.primalBuffer2[block.id], block.f, info.primalSolPrev[block.id])
        axpy!(-1.0, info.primalBuffer2[block.id], info.primalBuffer1[block.id])
        axpy!(1/info.primalStepSize, info.primalSolPrev[block.id], info.primalBuffer1[block.id])
        axpy!(-1/info.primalStepSize, info.primalSol[block.id], info.primalBuffer1[block.id])

        if isa(param, MalitskyPockParam)
            adjoint!(mappings[block.id], info.lineSearchDualBuffer, info.primalBuffer1[block.id], true)
        end 

        blockDresL2 = norm(info.primalBuffer1[block.id], 2)
        blockDresLInf = norm(info.primalBuffer1[block.id], Inf)
        dresL2 += blockDresL2^2
        dresLInf = max(dresLInf, blockDresLInf)
    end 
    dresL2 = sqrt(dresL2)
    push!(info.dresL2, dresL2)
    push!(info.dresLInf, dresLInf)
end

"""
    computeAdaPDMPrimalResiduals!(info::AdaPDMIterationInfo, mbp::MultiblockProblem, param::AbstractAdaPDMParam)

Compute the primal residuals for the Adaptive Primal-Dual Method.

# Arguments
- `info::AdaPDMIterationInfo`: The iteration information object to update
- `mbp::MultiblockProblem`: The multiblock problem being solved
- `param::AbstractAdaPDMParam`: Parameters for the algorithm

# Details
This function:
1. Computes Ax^{k+1} and stores it in dualBuffer
2. Calculates Ax^k - Ax^{k+1}
3. For non-Malitsky-Pock algorithms, applies step size ratio adjustments
4. Computes (1/sigma)*(y^k - y^{k+1}) + step-size adjusted differences
5. Calculates both L2 and L-infinity norms of the residuals
6. Stores the computed residuals in the iteration info object
"""
function computeAdaPDMPrimalResiduals!(info::AdaPDMIterationInfo, mbp::MultiblockProblem, param::AbstractAdaPDMParam)
    # compute primal residuals 
    # store Ax^{k+1} in duaBuffer
    info.dualBuffer .= 0.0 
    addToBuffer = true 
    mappings = mbp.constraints[1].mappings 
    for block in mbp.blocks[1:end-1]
        mappings[block.id](info.primalSol[block.id], info.dualBuffer, addToBuffer)
    end 

    # dualBuffer <- Ax^k - Ax^{k+1}
    axpby!(1.0, info.bufferAx, -1.0, info.dualBuffer)
    
    # dualBuffer <- (gamma_{k+1}/gamma_k) * (Ax^k - Ax^{k+1})
    if isa(param, MalitskyPockParam) == false 
        gammaRatio = info.primalStepSize / info.primalStepSizePrev
        axpy!(gammaRatio, info.bufferAx, info.dualBuffer)
        axpy!(-gammaRatio, info.bufferAxPrev, info.dualBuffer)
    end 

    # dualBuffer <- (1/sigma) *(y^k - y^{k+1}) + (gamma_{k+1}/gamma_k) * (Ax^k - Ax^{k+1})
    axpy!(1.0 / info.dualStepSize, info.dualSolPrev, info.dualBuffer)
    axpy!(-1.0 / info.dualStepSize, info.dualSol, info.dualBuffer)

    presL2 = norm(info.dualBuffer, 2)
    presLInf = norm(info.dualBuffer, Inf)
    push!(info.presL2, presL2)
    push!(info.presLInf, presLInf)
end 

"""
    computePDMResidualsAndObjective!(info::AdaPDMIterationInfo, mbp::MultiblockProblem, param::AbstractAdaPDMParam)

Compute the primal and dual residuals and the objective value after an iteration.

# Arguments
- `info::AdaPDMIterationInfo`: The iteration information object to update
- `mbp::MultiblockProblem`: The multiblock problem being solved
- `param::AbstractAdaPDMParam`: Parameters for the algorithm

# Details
This function:
1. Computes and stores Ax^{k+1} in the buffer
2. Calculates primal residuals by calling `computeAdaPDMPrimalResiduals!`
3. Computes dual residuals by calling `computeAdaPDMDualResiduals!`
4. Calculates the Lagrangian objective value
5. Updates the iteration info with all computed values
"""
function computePDMResidualsAndObjective!(info::AdaPDMIterationInfo, mbp::MultiblockProblem, param::AbstractAdaPDMParam)
    # iteration 0 for the purpose of logging information 
    if isempty(info.presL2)
        # primal residuals require the dual variables to be updated; set initial valuesto Inf
        push!(info.presL2, Inf)
        push!(info.presLInf, Inf)

        # dual residuals require the primal variables to be updated; set initial values to Inf for Malisky-Pock
        if isa(param, MalitskyPockParam)
            push!(info.dresL2, Inf)
            push!(info.dresLInf, Inf)
        else 
            computeAdaPDMDualResiduals!(info, mbp, param)
        end 

        # objective values require the primal variables to be updated; set initial values to Inf
        push!(info.lagrangianObj, Inf)
        push!(info.objectiveValue, Inf)

        # primalObj= 0.0 
        # for block in mbp.blocks
        #     primalObj += (block.f)(block.val) + (block.g)(block.val)
        # end 
        # push!(info.objectiveValue, primalObj)

        # update bufferAx and bufferAxPrev 
        mappings = mbp.constraints[1].mappings 
        for block in mbp.blocks[1:end-1]
            mappings[block.id](info.primalSol[block.id], info.bufferAx, true)
            mappings[block.id](info.primalSolPrev[block.id], info.bufferAxPrev, true)
        end 
        return 
    end 
    
    # valid iterations 
    computeAdaPDMPrimalResiduals!(info, mbp, param)
    computeAdaPDMDualResiduals!(info, mbp, param)

    # compute the Lagrangian objective value f(x) + g(x) + <y, Ax> - h^*(y) 
    copyto!(info.bufferAxPrev, info.bufferAx)
    partialObj = computePartialObjective!(info, mbp) # Σ_i (f_i+g_i); info.bufferAx now contains Ax^{k+1}

    # Primal objective of the composite form: Σ_i (f_i(x_i)+g_i(x_i)) + h(Ax)
    push!(info.objectiveValue, partialObj + (mbp.blocks[end].g)(info.bufferAx))

    obj = partialObj + dot(info.dualSol, info.bufferAx)   # ... + <y^{k+1}, Ax^{k+1}>
    axpy!(1.0, info.bufferAx, info.dualBuffer) # dualBuffer <- Ax^{k+1} + primal residuals \in \partial h^*(y^{k+1})

    # TODO: check feasibility tolerance 
    gValue = (mbp.blocks[end].g)(info.dualBuffer)
    if isinf(gValue)
        proximalOracle!(info.dualBuffer, mbp.blocks[end].g, info.dualBuffer, 1.0)
        gValue = (mbp.blocks[end].g)(info.dualBuffer)
    end 
    conjugateValue = dot(info.dualSol, info.dualBuffer) - gValue   
    obj -= conjugateValue 
    push!(info.lagrangianObj, obj)
end 

""" 
    AdaPDMIterationInfo(mbp::MultiblockProblem, param::AbstractAdaPDMParam)

Initialize the iteration info object for AdaPDM and AdaPDMPlus algorithms.

# Arguments
- `mbp::MultiblockProblem`: The multiblock problem to solve
- `param::AbstractAdaPDMParam`: The parameters for the AdaPDM algorithm

# Returns
- `AdaPDMIterationInfo`: An initialized iteration info object

# Details
This function initializes the primal and dual variables, buffers, and computes
initial residuals needed to start the AdaPDM algorithm.
""" 
function AdaPDMIterationInfo(mbp::MultiblockProblem, param::AbstractAdaPDMParam)
    info = AdaPDMIterationInfo() 
   
    # prepare initial primal step size from param 
    setupInitialPrimalDualStepSize!(info, param)

    # initialize dual solution and dual buffers with zeros 
    info.dualSol = zero(mbp.constraints[1].rhs)
    info.dualSolPrev = zero(mbp.constraints[1].rhs)
    info.bufferAx = zero(mbp.constraints[1].rhs)
    info.bufferAxPrev = zero(mbp.constraints[1].rhs)
    info.dualBuffer = zero(mbp.constraints[1].rhs)
    info.lineSearchDualBuffer = zero(mbp.constraints[1].rhs)
    
    constraint = mbp.constraints[1]
    # initialize primal solution and primal buffers 
    for block in mbp.blocks[1:end-1]
        # initialize primal buffers 
        info.primalBuffer1[block.id] = similar(block.val)
        info.primalBuffer2[block.id] = similar(block.val)
        if isa(param, MalitskyPockParam)
            info.lineSearchPrimalBuffer[block.id] = similar(block.val)
        end 
        # initialize x^{-1}
        info.primalSolPrev[block.id] = deepcopy(block.val)
        # initialize x^0
        ## the poximal center is x^{-1} - gamma * nabla f(x^{-1}) - gamma A'y^0
        copyto!(info.primalBuffer1[block.id], info.primalSolPrev[block.id])
        axpy!(-info.primalStepSize, gradientOracle(block.f, info.primalSolPrev[block.id]), info.primalBuffer1[block.id])
        axpy!(-info.primalStepSize, adjoint(constraint.mappings[block.id], info.dualSol), info.primalBuffer1[block.id])
        
        info.primalSol[block.id] = similar(block.val)
        proximalOracle!(info.primalSol[block.id], block.g, info.primalBuffer1[block.id], info.primalStepSize)
    end 
    
    # set initial primal residuals and Lagrangian objective value to Inf
    # this function also updates bufferAx and bufferAxPrev
    computePDMResidualsAndObjective!(info, mbp, param)
    
    # prepare bufferAxPrev 
    for block in mbp.blocks[1:end-1]
        # bufferAx has been updated in computeAdaPDMObjective!
        constraint.mappings[block.id](info.primalSolPrev[block.id], info.bufferAxPrev, true)
    end 

    # prepare initial estimate of operator norm 
    try 
        info.opNormEstimate = param.initialNormEstimate 
    catch 
        info.opNormEstimate = computeNormEstimate(mbp)
    end 

    return info 
end 



