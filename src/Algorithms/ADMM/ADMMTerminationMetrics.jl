"""
    IsMetricConvergentToZero

Enumeration for classifying the convergence behavior of sequences to zero.

Used to track whether metric sequences (such as primal residuals or dual residuals) 
converge to zero during ADMM iterations. The classification is based on sophisticated
convergence tests using logarithmic scaling and ratio analysis.

# Values
- `CONVERGENT_UNDEFINED`: Convergence status has not been determined yet
- `CONVERGENT_YES`: Sequence converges to zero based on ratio tests or tolerance satisfaction
- `CONVERGENT_NO`: Sequence does not converge to zero or stagnates above tolerances
"""
@enum IsMetricConvergentToZero begin 
    CONVERGENT_UNDEFINED 
    CONVERGENT_YES
    CONVERGENT_NO
end 

"""
    IsMetricBounded

Enumeration for classifying the boundedness properties of sequences.

Used to determine whether sequences like dual variables, constraint violations, or 
objective functions are bounded during ADMM iterations. The classification helps
detect problematic optimization problems and infeasibility.

# Values
- `BOUNDED_UNDEFINED`: Boundedness has not been determined yet
- `BOUNDED_YES`: Sequence is bounded and well-behaved
- `UPPER_UNBOUNDED`: Sequence grows without bound (diverges to +∞)
- `LOWER_UNBOUNDED`: Sequence decreases without bound (diverges to -∞)
"""
@enum IsMetricBounded begin 
    BOUNDED_UNDEFINED
    BOUNDED_YES
    UPPER_UNBOUNDED 
    LOWER_UNBOUNDED 
end 

"""
    ProblemClassification

Enumeration for classifying optimization problems based on ADMM convergence behavior.

This classification system categorizes optimization problems into different cases based
on the convergence patterns observed during ADMM iterations. It helps identify whether
the problem is well-posed, infeasible, unbounded, or unsuitable for ADMM.

# Values
- `CLASSIFICATION_UNDEFINED`: Problem classification has not been determined yet
- `CLASSIFICATION_CASE_A`: Well-posed problem with optimal solution and ADMM convergence
- `CLASSIFICATION_CASE_B`: Problem has optimal solution but ADMM may not be applicable
- `CLASSIFICATION_CASE_C`: Problem is lower bounded but does not have optimal solution
- `CLASSIFICATION_CASE_D`: Problem is lower bounded but ADMM may not be applicable
- `CLASSIFICATION_CASE_E`: Problem is lower unbounded (objective can decrease without bound)
- `CLASSIFICATION_CASE_F`: Problem is infeasible (constraints are inconsistent)

# Classification Criteria
The classification is based on convergence behavior analysis of:
- Primal residual sequences and their convergence to zero
- Dual residual sequences and their convergence patterns
- Boundedness of constraint violation norms (Bz norm)
- Boundedness of dual variable norms
- Objective function boundedness properties
"""
@enum ProblemClassification begin 
    CLASSIFICATION_UNDEFINED
    CLASSIFICATION_CASE_A
    CLASSIFICATION_CASE_B
    CLASSIFICATION_CASE_C
    CLASSIFICATION_CASE_D
    CLASSIFICATION_CASE_E
    CLASSIFICATION_CASE_F
end 

"""
    ADMMTerminationMetrics

Advanced metrics collection system for ADMM termination analysis and problem classification.

This structure collects detailed metrics during ADMM iterations to perform sophisticated
convergence analysis, boundedness detection, and problem classification. It enables
the detection of infeasible problems, unbounded problems, and cases where ADMM may
not be applicable.

# Metric History Fields
- `presL2BetweenPrimalUpdates::Vector{Float64}`: Primal residual L2 norms between primal updates
- `dualDifferenceL2::Vector{Float64}`: L2 norm of dual variable differences between iterations
- `BzNorm::Vector{Float64}`: L2 norm of Bz (constraint violation) sequences
- `dualNorm::Vector{Float64}`: L2 norm of dual variables
- `dualObj::Vector{Float64}`: Dual objective function values

# Buffer Fields
- `dualSolBetweenPrimalUpdates::Dict{String, NumericVariable}`: Dual solution between primal updates
- `BzBuffer::Dict{String, NumericVariable}`: Buffer for Bz computations
- `dualBuffer::Dict{String, NumericVariable}`: Buffer for dual computations

# Convergence Counters
Each metric has associated counters for tracking convergence patterns:
- `count_presL2_1, count_presL2_2, count_presL2_3`: Primal residual convergence counters
- `count_dres_1, count_dres_2, count_dres_3`: Dual residual convergence counters
- `count_Bz_norm_1` through `count_Bz_norm_5`: Bz norm boundedness counters
- `count_y_norm_1` through `count_y_norm_5`: Dual norm boundedness counters
- `count_obj_1` through `count_obj_4`: Objective function boundedness counters
- `count_dualobj_1` through `count_dualobj_4`: Dual objective boundedness counters

# Classification Fields
- `presL2_convergent::IsMetricConvergentToZero`: Primal residual convergence classification
- `dres_convergent::IsMetricConvergentToZero`: Dual residual convergence classification
- `Bz_norm_bounded::IsMetricBounded`: Bz norm boundedness classification
- `y_norm_bounded::IsMetricBounded`: Dual norm boundedness classification
- `obj_bounded::IsMetricBounded`: Objective function boundedness classification
- `dualobj_bounded::IsMetricBounded`: Dual objective boundedness classification
- `problem_classification::ProblemClassification`: Overall problem classification

# Tolerance Fields
- `presDiffTol::Float64`: Tolerance for detecting primal residual stagnation (default: 1e-6)
- `dresDiffTol::Float64`: Tolerance for detecting dual residual stagnation (default: 1e-6)

# Mathematical Background
The convergence analysis uses sophisticated techniques including:
- **Logarithmic scaling**: Tests sequences like `residual * log(log(iter))` for convergence
- **Ratio tests**: Analyzes consecutive terms to determine convergence behavior
- **Series convergence analysis**: Uses mathematical series tests to detect boundedness
- **Statistical counting**: Tracks patterns over 1000+ iterations for robust detection

# Usage
This structure is primarily used with the `OriginalADMMSubproblemSolver` to enable
advanced termination criteria and problem classification. It provides much more
detailed analysis than basic primal/dual residual checking.
"""
mutable struct ADMMTerminationMetrics 
    presL2BetweenPrimalUpdates::Vector{Float64}  # pres_1
    dualDifferenceL2::Vector{Float64}            # dres 
    BzNorm::Vector{Float64}                      # Bz_norm
    dualNorm::Vector{Float64}                    # y_norm
    dualObj::Vector{Float64}                     # dualobj
   
    dualSolBetweenPrimalUpdates::Dict{String, NumericVariable} # dualSol_1
    BzBuffer::Dict{String, NumericVariable}                   # BzBuffer
    dualBuffer::Dict{String, NumericVariable}                 # yBuffer, Not used in Mingyu's code. Used in current code for intermediate computation.
    edgeIDs::Vector{String}

    # metrics for infeasibility and unboundedness detection 
    count_presL2_1::Int64
    count_presL2_2::Int64
    count_presL2_3::Int64
    count_dres_1::Int64
    count_dres_2::Int64
    count_dres_3::Int64
    count_Bz_norm_1::Int64
    count_Bz_norm_2::Int64
    count_Bz_norm_3::Int64
    count_Bz_norm_4::Int64
    count_Bz_norm_5::Int64
    count_y_norm_1::Int64
    count_y_norm_2::Int64
    count_y_norm_3::Int64
    count_y_norm_4::Int64
    count_y_norm_5::Int64
    count_obj_1::Int64
    count_obj_2::Int64
    count_obj_3::Int64
    count_obj_4::Int64
    count_dualobj_1::Int64
    count_dualobj_2::Int64
    count_dualobj_3::Int64
    count_dualobj_4::Int64
    presL2_convergent::IsMetricConvergentToZero
    dres_convergent::IsMetricConvergentToZero
    Bz_norm_bounded::IsMetricBounded
    y_norm_bounded::IsMetricBounded
    obj_bounded::IsMetricBounded
    dualobj_bounded::IsMetricBounded
    problem_classification::ProblemClassification

    # tolerance 
    presDiffTol::Float64
    dresDiffTol::Float64

    """
        ADMMTerminationMetrics(info::ADMMIterationInfo)

    Construct and initialize ADMMTerminationMetrics from ADMM iteration information.

    # Arguments
    - `info::ADMMIterationInfo`: Current iteration info containing dual solution structure

    # Returns
    - `ADMMTerminationMetrics`: Initialized metrics object with empty histories and undefined classifications

    # Notes
    - Initializes buffers to match the structure of `info.dualSol`
    - Sets all counters to zero and classifications to undefined
    - Uses default tolerances of 1e-6 for stagnation detection
    """
    function ADMMTerminationMetrics(info::ADMMIterationInfo)

        dualSolBetweenPrimalUpdates = Dict{String, NumericVariable}()
        BzBuffer = Dict{String, NumericVariable}()
        dualBuffer = Dict{String, NumericVariable}()

        for edgeID in keys(info.dualSol)
            dualSolBetweenPrimalUpdates[edgeID] = similar(info.dualSol[edgeID])
            BzBuffer[edgeID] = similar(info.dualSol[edgeID])
            dualBuffer[edgeID] = similar(info.dualSol[edgeID])
        end 

        return new(
            Vector{Float64}(), 
            Vector{Float64}(), 
            Vector{Float64}(), 
            Vector{Float64}(), 
            Vector{Float64}(), 
            dualSolBetweenPrimalUpdates, 
            BzBuffer, 
            dualBuffer, 
            collect(keys(info.dualSol)),
            0, 0, 0, 
            0, 0, 0, 
            0, 0, 0, 0, 0, 
            0, 0, 0, 0, 0, 
            0, 0, 0, 0, 
            0, 0, 0, 0, 
            CONVERGENT_UNDEFINED, 
            CONVERGENT_UNDEFINED, 
            BOUNDED_UNDEFINED, 
            BOUNDED_UNDEFINED, 
            BOUNDED_UNDEFINED, 
            BOUNDED_UNDEFINED, 
            CLASSIFICATION_UNDEFINED, 
            1e-6, 
            1e-6)
    end 
end 

"""
    collectTerminationMetricsBetweenPrimalUpdates!(metrics::ADMMTerminationMetrics, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph)

Collect termination metrics between primal variable updates in ADMM iterations.

This function computes and stores metrics that are needed for advanced termination
analysis, specifically focusing on primal residuals and intermediate dual variable
states. It prepares data for convergence analysis and problem classification.

# Arguments
- `metrics::ADMMTerminationMetrics`: Metrics object to update with collected data
- `info::ADMMIterationInfo`: Current iteration information including primal solutions
- `admmGraph::ADMMBipartiteGraph`: ADMM problem structure with constraint mappings

# Computed Metrics
- **Primal residuals**: Computes ||Ax^{k+1} + Bz^k - b||₂ and stores in `metrics.presL2BetweenPrimalUpdates`
- **Intermediate dual variables**: Computes y^k + ρ(Ax^{k+1} + Bz^k - b) and stores in `metrics.dualSolBetweenPrimalUpdates`

# Algorithm Details
1. **Primal residual computation**: 
   - Computes constraint violations: `metrics.dualBuffer[edgeID] = -rhs + A₁x₁ + A₂x₂`
   - Uses constraint mappings to evaluate linear constraint functions
   - Computes L2 norm across all constraint blocks

2. **Intermediate dual update**:
   - Computes updated dual variables: `y^k + ρ * (primal residual)`
   - Stores results for later use in dual residual computation

# Threading
- Uses `@threads` for parallel computation across constraint edges
- Thread-safe operations on separate edge buffers

# Notes
- This function is called after primal variable updates but before dual updates
- The intermediate dual variables are used later for dual residual computation
- Primal residuals collected here are used for convergence analysis in `updateTerminationMetrics!`
"""
function collectTerminationMetricsBetweenPrimalUpdates!(metrics::ADMMTerminationMetrics, 
    info::ADMMIterationInfo, 
    admmGraph::ADMMBipartiteGraph)

    edges = metrics.edgeIDs
    numberEdges = length(edges)
    addToBuffer = true 

    # collect Ax^{k+1} + Bz^k-b in metrics.dualBuffer
    nT = Threads.maxthreadid()
    presSquareByThread = zeros(Float64, nT)
    @threads for idx in 1:numberEdges
        tid = Threads.threadid()
        edgeID = edges[idx]
        nodeID1 = admmGraph.edges[edgeID].nodeID1
        nodeID2 = admmGraph.edges[edgeID].nodeID2

        metrics.dualBuffer[edgeID] .= -admmGraph.edges[edgeID].rhs
        admmGraph.edges[edgeID].mappings[nodeID1](info.primalSol[nodeID1], metrics.dualBuffer[edgeID], addToBuffer)
        admmGraph.edges[edgeID].mappings[nodeID2](info.primalSol[nodeID2], metrics.dualBuffer[edgeID], addToBuffer)

        presSquareByThread[tid] += dot(metrics.dualBuffer[edgeID], metrics.dualBuffer[edgeID])
    end 

    presSquare = sum(presSquareByThread)
    push!(metrics.presL2BetweenPrimalUpdates, sqrt(presSquare))

    # collect y^k + rho * (Ax^{k+1} + Bz^k-b)
    rho = info.rhoHistory[end][1]
    @threads for idx in 1:numberEdges
        edgeID = edges[idx]
        copyto!(metrics.dualSolBetweenPrimalUpdates[edgeID], info.dualSol[edgeID])
        axpy!(rho, metrics.dualBuffer[edgeID], metrics.dualSolBetweenPrimalUpdates[edgeID])
    end 

end 


"""
    collectTerminationMetricsAfterDualUpdates!(metrics::ADMMTerminationMetrics, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph)

Collect termination metrics after dual variable updates in ADMM iterations.

This function computes and stores metrics that are essential for advanced termination
analysis, including dual norms, constraint violation norms, dual differences, and
dual objective values. These metrics enable sophisticated convergence analysis and
problem classification.

# Arguments
- `metrics::ADMMTerminationMetrics`: Metrics object to update with collected data
- `info::ADMMIterationInfo`: Current iteration information including updated dual solutions
- `admmGraph::ADMMBipartiteGraph`: ADMM problem structure with constraint mappings

# Computed Metrics
- **Bz norm**: Computes ||Bz^{k+1}||₂ and stores in `metrics.BzNorm`
- **Dual norm**: Computes ||y^{k+1}||₂ and stores in `metrics.dualNorm`
- **Dual difference**: Computes ||y^{k+1} - (y^k + ρ(Ax^{k+1} + Bz^k - b))||₂ and stores in `metrics.dualDifferenceL2`
- **Dual objective**: Computes dual objective value and stores in `metrics.dualObj`

# Algorithm Details
1. **Bz norm computation**:
   - Identifies z-variables (those with assignment == 1)
   - Computes Bz using constraint mappings
   - Computes L2 norm across all constraint blocks

2. **Dual statistics**:
   - Computes current dual variable norm
   - Computes difference between current and intermediate dual variables
   - Evaluates dual objective as primal objective + Lagrangian gap

# Threading
- Uses `@threads` for parallel computation across constraint edges
- Thread-safe operations on separate edge buffers

# Mathematical Background
The dual objective is computed as:
```
dual_obj = primal_obj + Σᵢ yᵢᵀ(Aᵢx + Bᵢz - bᵢ)
```

The dual difference measures the change in dual variables:
```
dual_diff = ||y^{k+1} - (y^k + ρ(Ax^{k+1} + Bz^k - b))||₂
```

# Notes
- This function is called after dual variable updates
- The metrics collected here are used for convergence analysis in `updateTerminationMetrics!`
- Dual differences are key indicators of algorithm convergence
"""
function collectTerminationMetricsAfterDualUpdates!(metrics::ADMMTerminationMetrics, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph)
    
    edges = metrics.edgeIDs
    numberEdges = length(edges)

    # compute Bz^{k+1}
    nT = Threads.maxthreadid()
    BzSquareByThread = zeros(Float64, nT)
    @threads for idx in 1:numberEdges
        tid = Threads.threadid()
        edgeID = edges[idx]
        nodeID1 = admmGraph.edges[edgeID].nodeID1
        nodeID2 = admmGraph.edges[edgeID].nodeID2

        zNodeID = admmGraph.nodes[nodeID1].assignment == 1 ? nodeID1 : nodeID2 
        admmGraph.edges[edgeID].mappings[zNodeID](info.primalSol[zNodeID], metrics.BzBuffer[edgeID], false)

        BzSquareByThread[tid] += dot(metrics.BzBuffer[edgeID], metrics.BzBuffer[edgeID])
    end 
    BzSquare = sum(BzSquareByThread)
    push!(metrics.BzNorm, sqrt(BzSquare))

    # compute dual difference and dual norm 
    dualNormSquareByThread = zeros(Float64, nT)
    dualDifferenceSquareByThread = zeros(Float64, nT)
    @threads for idx in 1:numberEdges
        tid = Threads.threadid()
        edgeID = edges[idx]
        dualNormSquareByThread[tid] += dot(info.dualSol[edgeID], info.dualSol[edgeID])

        copyto!(metrics.dualBuffer[edgeID], info.dualSol[edgeID])
        axpy!(-1.0, metrics.dualSolBetweenPrimalUpdates[edgeID], metrics.dualBuffer[edgeID])
        dualDifferenceSquareByThread[tid] += dot(metrics.dualBuffer[edgeID], metrics.dualBuffer[edgeID])
    end 
    dualNormSquare = sum(dualNormSquareByThread)
    dualDifferenceSquare = sum(dualDifferenceSquareByThread)
    push!(metrics.dualNorm, sqrt(dualNormSquare))
    push!(metrics.dualDifferenceL2, sqrt(dualDifferenceSquare))

    # compute dualobj 
    lagrangeGap = 0.0 
    for idx in 1:numberEdges
        edgeID = edges[idx]
        lagrangeGap += dot(info.dualSol[edgeID], info.dualBuffer[edgeID])
    end 
    push!(metrics.dualObj, info.obj[end] + lagrangeGap)
end 


"""
    updateTerminationMetrics!(metrics::ADMMTerminationMetrics, info::ADMMIterationInfo, presTolL2::Float64, dresTolL2::Float64)

Analyze various properties of the ADMM optimization problem to detect convergence, boundedness, and feasibility.

This function performs advanced analysis of the ADMM iteration sequences to classify the optimization problem
and detect various convergence properties including:
- Primal residual convergence behavior using logarithmic scaling and trend analysis
- Dual residual convergence behavior using statistical trend detection
- Boundedness of Bz norm sequences using series convergence analysis
- Boundedness of dual variable norm sequences with sophisticated convergence tests
- Objective function boundedness properties and monotonicity analysis
- Dual objective function boundedness properties and convergence detection

# Arguments
- `metrics::ADMMTerminationMetrics`: Metrics object containing iteration history, counters, and convergence states
- `info::ADMMIterationInfo`: Current iteration information including residuals and objective values
- `presTolL2::Float64`: Tolerance for primal residual L2 norm convergence detection
- `dresTolL2::Float64`: Tolerance for dual residual L2 norm convergence detection

# Algorithm Details
The function uses sophisticated convergence tests including:
- **Ratio tests with logarithmic scaling**: Uses `residual * log(log(iter))` factors to detect convergence patterns
- **Series convergence analysis**: Applies convergence tests to determine if Bz and dual norm sequences are bounded
- **Consecutive step counting**: Tracks trends over 1000+ consecutive iterations for robust classification
- **Relative change analysis**: Uses `metrics.presDiffTol` and `metrics.dresDiffTol` for stagnation detection
- **Statistical thresholds**: Uses counters with thresholds (10, 1000 steps) to avoid premature classification

# Convergence Classification
The function classifies sequences into:
- `CONVERGENT_YES`: Sequence converges to zero or satisfies tolerance
- `CONVERGENT_NO`: Sequence does not converge or stagnates above tolerance
- `BOUNDED_YES`: Sequence is bounded and well-behaved
- `UPPER_UNBOUNDED`: Sequence grows without bound
- `LOWER_UNBOUNDED`: Sequence decreases without bound

# Special Cases
- **Optimality conditions**: If both primal and dual residuals satisfy tolerances, all sequences are marked as bounded
- **Joint boundedness**: If both Bz and dual norms are bounded, convergence is assumed for residuals and objectives

# Notes
- Requires at least 3-5 iterations of history for meaningful analysis
- Iteration number is computed internally as `length(info.presL2) - 1`
- Updates the metrics object in-place with detected properties
- Robust against numerical issues with bounds checking and empty array protection
"""
function updateTerminationMetrics!(metrics::ADMMTerminationMetrics, info::ADMMIterationInfo, 
    presTolL2::Float64, 
    dresTolL2::Float64)

    iter = length(info.presL2) - 1

    # === Primal Residual Convergence Analysis ===
    if iter > 2 && metrics.presL2_convergent == CONVERGENT_UNDEFINED && 
       metrics.count_presL2_1 <= 1000 && metrics.count_presL2_2 <= 1000
        
        # CORRECT - Use the special metrics you collected
        current_presL2 = info.presL2[end] * (log(log(iter + 2)))
        previous_presL2 = info.presL2[end-1] * (log(log(iter + 1))) 

        if current_presL2 < previous_presL2  
            metrics.count_presL2_1 += 1
            metrics.count_presL2_2 = 0
        else
            metrics.count_presL2_1 = 0
            metrics.count_presL2_2 += 1
        end
    elseif iter > 2 && metrics.presL2_convergent == CONVERGENT_UNDEFINED && metrics.count_presL2_1 > 1000  
        metrics.presL2_convergent = CONVERGENT_YES
    elseif iter > 2 && metrics.presL2_convergent == CONVERGENT_UNDEFINED && metrics.count_presL2_2 > 1000  
        metrics.presL2_convergent = CONVERGENT_NO 
    end

    # Direct tolerance check
    if iter > 2 && info.presL2[end] < presTolL2
        metrics.presL2_convergent = CONVERGENT_YES
    elseif iter > 2 && metrics.count_presL2_3 <= 10 
        if abs((info.presL2[end] - info.presL2[end-1]) / info.presL2[end]) < metrics.presDiffTol
           metrics.count_presL2_3 += 1
        else
           metrics.count_presL2_3 = 0  
        end
    elseif iter > 2 && metrics.count_presL2_3 > 10
        metrics.presL2_convergent = CONVERGENT_NO
    end 

    # === Dual Residual Convergence Analysis ===
    if iter > 2 && metrics.dres_convergent == CONVERGENT_UNDEFINED && 
       metrics.count_dres_1 <= 1000 && metrics.count_dres_2 <= 1000
        
        # CORRECT - Use the special metrics you collected
        current_dres = metrics.dualDifferenceL2[end] * (log(log(iter + 2)))
        previous_dres = metrics.dualDifferenceL2[end-1] * (log(log(iter + 1))) 

        if current_dres < previous_dres  
            metrics.count_dres_1 += 1
            metrics.count_dres_2 = 0
        else
            metrics.count_dres_1 = 0
            metrics.count_dres_2 += 1
        end
    elseif iter > 2 && metrics.dres_convergent == CONVERGENT_UNDEFINED && metrics.count_dres_1 > 1000  
        metrics.dres_convergent = CONVERGENT_YES
    elseif iter > 2 && metrics.dres_convergent == CONVERGENT_UNDEFINED && metrics.count_dres_2 > 1000  
        metrics.dres_convergent = CONVERGENT_NO 
    end

    # Direct tolerance check
    if iter > 2 && metrics.dualDifferenceL2[end] < dresTolL2
        metrics.dres_convergent = CONVERGENT_YES
    elseif iter > 2 && metrics.count_dres_3 <= 10 && metrics.dres_convergent == CONVERGENT_UNDEFINED
        if abs((metrics.dualDifferenceL2[end] - metrics.dualDifferenceL2[end-1]) / metrics.dualDifferenceL2[end]) < metrics.dresDiffTol
           metrics.count_dres_3 += 1
        else
           metrics.count_dres_3 = 0  
        end
    elseif iter > 2 && metrics.count_dres_3 > 10 && metrics.dres_convergent == CONVERGENT_UNDEFINED
        metrics.dres_convergent = CONVERGENT_NO
    end 

    # === Bz Norm Boundedness Analysis ===
    if iter > 4 && metrics.Bz_norm_bounded == BOUNDED_UNDEFINED && !isempty(metrics.BzNorm)
        if metrics.count_Bz_norm_1 <= 1000 && metrics.count_Bz_norm_2 <= 1000 && 
           metrics.count_Bz_norm_3 <= 10 && metrics.count_Bz_norm_4 <= 10
            
            # Series convergence analysis
            if length(metrics.BzNorm) >= 3
                # Safety check to prevent division by zero
                if abs(metrics.BzNorm[end-1] - metrics.BzNorm[end-2]) < 1e-12 || 
                   abs(metrics.BzNorm[end-2] - metrics.BzNorm[end-3]) < 1e-12
                    current_rabbe_Bz_norm = 0.0
                    previous_rabbe_Bz_norm = 0.0
                else
                    current_rabbe_Bz_norm = (iter-1) * (1 - (metrics.BzNorm[end] - metrics.BzNorm[end-1]) / 
                                                         (metrics.BzNorm[end-1] - metrics.BzNorm[end-2]))
                    previous_rabbe_Bz_norm = (iter-2) * (1 - (metrics.BzNorm[end-1] - metrics.BzNorm[end-2]) / 
                                                          (metrics.BzNorm[end-2] - metrics.BzNorm[end-3]))
                end

                if metrics.BzNorm[end] >= metrics.BzNorm[end-1] && 
                   metrics.BzNorm[end] / (log(log(iter + 2))) > metrics.BzNorm[end-1] / (log(log(iter + 1)))
                    
                    if current_rabbe_Bz_norm < previous_rabbe_Bz_norm <= 1
                       metrics.count_Bz_norm_1 += 1
                    else
                       metrics.count_Bz_norm_1 = 0
                    end    

                    if abs((current_rabbe_Bz_norm - previous_rabbe_Bz_norm) / current_rabbe_Bz_norm) < 1e-5 && 
                       current_rabbe_Bz_norm <= 1 + 1e-3
                       metrics.count_Bz_norm_3 += 1
                    else
                       metrics.count_Bz_norm_3 = 0
                    end
                    
                elseif metrics.BzNorm[end] < metrics.BzNorm[end-1] || 
                       metrics.BzNorm[end] / (log(log(iter + 2))) < metrics.BzNorm[end-1] / (log(log(iter + 1)))
                    
                    if current_rabbe_Bz_norm > previous_rabbe_Bz_norm > 1
                        metrics.count_Bz_norm_2 += 1
                    else
                        metrics.count_Bz_norm_2 = 0
                    end
     
                    if abs((current_rabbe_Bz_norm - previous_rabbe_Bz_norm) / current_rabbe_Bz_norm) < 1e-5 && 
                       current_rabbe_Bz_norm > 1 + 1e-3
                       metrics.count_Bz_norm_4 += 1
                    else
                       metrics.count_Bz_norm_4 = 0  # Fixed: was == 0
                    end
                else
                    metrics.count_Bz_norm_1 = 0
                    metrics.count_Bz_norm_2 = 0
                    metrics.count_Bz_norm_3 = 0
                    metrics.count_Bz_norm_4 = 0
                end    
            end
        elseif metrics.count_Bz_norm_1 > 1000 || metrics.count_Bz_norm_3 > 10
            metrics.Bz_norm_bounded = UPPER_UNBOUNDED
        elseif metrics.count_Bz_norm_2 > 1000 || metrics.count_Bz_norm_4 > 10
            metrics.Bz_norm_bounded = BOUNDED_YES   
        end 
    end

    # Simple boundedness check
    if iter > 2 && metrics.Bz_norm_bounded == BOUNDED_UNDEFINED && metrics.count_Bz_norm_5 <= 10 && !isempty(metrics.BzNorm)
        if metrics.BzNorm[end] <= 1e-5  
            metrics.Bz_norm_bounded = BOUNDED_YES
        elseif metrics.BzNorm[end] > 1e-5 && length(metrics.BzNorm) >= 2 &&
               abs((metrics.BzNorm[end] - metrics.BzNorm[end-1]) / metrics.BzNorm[end]) < 1e-5
            metrics.count_Bz_norm_5 += 1
        else
            metrics.count_Bz_norm_5 = 0
        end
    elseif iter > 2 && metrics.Bz_norm_bounded == BOUNDED_UNDEFINED && metrics.count_Bz_norm_5 > 10
        metrics.Bz_norm_bounded = BOUNDED_YES 
    end

    # === Dual Norm Boundedness Analysis ===
    if iter > 4 && metrics.y_norm_bounded == BOUNDED_UNDEFINED && !isempty(metrics.dualNorm)
        if metrics.count_y_norm_1 <= 1000 && metrics.count_y_norm_2 <= 1000 && 
           metrics.count_y_norm_3 <= 10 && metrics.count_y_norm_4 <= 10
            
            if length(metrics.dualNorm) >= 3
                # Safety check to prevent division by zero
                if abs(metrics.dualNorm[end-1] - metrics.dualNorm[end-2]) < 1e-12 || 
                   abs(metrics.dualNorm[end-2] - metrics.dualNorm[end-3]) < 1e-12
                    current_rabbe_y_norm = 0.0
                    previous_rabbe_y_norm = 0.0
                else
                    current_rabbe_y_norm = (iter-1) * (1 - (metrics.dualNorm[end] - metrics.dualNorm[end-1]) / 
                                                        (metrics.dualNorm[end-1] - metrics.dualNorm[end-2]))
                    previous_rabbe_y_norm = (iter-2) * (1 - (metrics.dualNorm[end-1] - metrics.dualNorm[end-2]) / 
                                                         (metrics.dualNorm[end-2] - metrics.dualNorm[end-3]))
                end

                if metrics.dualNorm[end] >= metrics.dualNorm[end-1] && 
                   metrics.dualNorm[end] / (log(log(iter + 2))) > metrics.dualNorm[end-1] / (log(log(iter + 1)))
                    
                    if current_rabbe_y_norm < previous_rabbe_y_norm <= 1
                       metrics.count_y_norm_1 += 1
                    else
                       metrics.count_y_norm_1 = 0
                    end    

                    if abs((current_rabbe_y_norm - previous_rabbe_y_norm) / current_rabbe_y_norm) < 1e-5 && 
                       current_rabbe_y_norm <= 1 + 1e-3
                       metrics.count_y_norm_3 += 1
                    else
                       metrics.count_y_norm_3 = 0
                    end
                    
                elseif metrics.dualNorm[end] < metrics.dualNorm[end-1] || 
                       metrics.dualNorm[end] / (log(log(iter + 2))) < metrics.dualNorm[end-1] / (log(log(iter + 1)))
                    
                    if current_rabbe_y_norm > previous_rabbe_y_norm > 1
                        metrics.count_y_norm_2 += 1
                    else
                        metrics.count_y_norm_2 = 0
                    end
     
                    if abs((current_rabbe_y_norm - previous_rabbe_y_norm) / current_rabbe_y_norm) < 1e-5 && 
                       current_rabbe_y_norm > 1 + 1e-3
                       metrics.count_y_norm_4 += 1
                    else
                       metrics.count_y_norm_4 = 0  # Fixed: was == 0
                    end
                else
                    metrics.count_y_norm_1 = 0
                    metrics.count_y_norm_2 = 0
                    metrics.count_y_norm_3 = 0
                    metrics.count_y_norm_4 = 0
                end    
            end
        elseif metrics.count_y_norm_1 > 1000 || metrics.count_y_norm_3 > 10
            metrics.y_norm_bounded = UPPER_UNBOUNDED
        elseif metrics.count_y_norm_2 > 1000 || metrics.count_y_norm_4 > 10 
            metrics.y_norm_bounded = BOUNDED_YES   
        end 
    end

    # Simple dual norm boundedness check
    if iter > 2 && metrics.y_norm_bounded == BOUNDED_UNDEFINED && metrics.count_y_norm_5 <= 10 && !isempty(metrics.dualNorm)
        if metrics.dualNorm[end] <= 1e-5  
            metrics.y_norm_bounded = BOUNDED_YES
        elseif metrics.dualNorm[end] > 1e-5 && length(metrics.dualNorm) >= 2 &&
               abs((metrics.dualNorm[end] - metrics.dualNorm[end-1]) / metrics.dualNorm[end]) < 1e-5
            metrics.count_y_norm_5 += 1
        else
            metrics.count_y_norm_5 = 0
        end
    elseif iter > 2 && metrics.y_norm_bounded == BOUNDED_UNDEFINED && metrics.count_y_norm_5 > 10
        metrics.y_norm_bounded = BOUNDED_YES 
    end

    # === Objective Function Boundedness Analysis ===
    if iter > 2 && metrics.obj_bounded == BOUNDED_UNDEFINED && 
       metrics.count_obj_1 <= 1000 && metrics.count_obj_2 <= 1000 && metrics.count_obj_3 <= 1000 &&
       length(info.obj) >= 3
        
        current_obj = abs((info.obj[end] - info.obj[2]) / (log(log(iter + 2))))  
        previous_obj = abs((info.obj[end-1] - info.obj[2]) / (log(log(iter + 1))))                     

        if current_obj >= previous_obj && info.obj[end] > info.obj[2]
            metrics.count_obj_1 += 1
            metrics.count_obj_2 = 0
            metrics.count_obj_3 = 0
        elseif current_obj >= previous_obj && info.obj[end] < info.obj[2]
            metrics.count_obj_1 = 0
            metrics.count_obj_2 = 0
            metrics.count_obj_3 += 1 
        else
            metrics.count_obj_1 = 0
            metrics.count_obj_2 += 1
            metrics.count_obj_3 = 0
        end
    elseif iter > 2 && metrics.obj_bounded == BOUNDED_UNDEFINED && metrics.count_obj_1 > 1000  
        metrics.obj_bounded = UPPER_UNBOUNDED   
    elseif iter > 2 && metrics.obj_bounded == BOUNDED_UNDEFINED && metrics.count_obj_2 > 1000  
        metrics.obj_bounded = BOUNDED_YES
    elseif iter > 2 && metrics.obj_bounded == BOUNDED_UNDEFINED && metrics.count_obj_3 > 1000  
        metrics.obj_bounded = LOWER_UNBOUNDED
    end

    # Objective convergence check
    if iter > 2 && metrics.obj_bounded == BOUNDED_UNDEFINED && metrics.count_obj_4 <= 10 && length(info.obj) >= 2
        if info.obj[end] != 0 && abs((info.obj[end] - info.obj[end-1]) / info.obj[end]) < 1e-5
            metrics.count_obj_4 += 1
        else
            metrics.count_obj_4 = 0
        end
    elseif iter > 2 && metrics.obj_bounded == BOUNDED_UNDEFINED && metrics.count_obj_4 > 10
        metrics.obj_bounded = BOUNDED_YES 
    end

    # === Dual Objective Function Boundedness Analysis ===
    if iter > 2 && metrics.dualobj_bounded == BOUNDED_UNDEFINED && 
       metrics.count_dualobj_1 <= 1000 && metrics.count_dualobj_2 <= 1000 && metrics.count_dualobj_3 <= 1000 &&
       length(metrics.dualObj) >= 3
        
        current_dualobj = abs((metrics.dualObj[end] - metrics.dualObj[2]) / (log(log(iter + 2))))  
        previous_dualobj = abs((metrics.dualObj[end-1] - metrics.dualObj[2]) / (log(log(iter + 1))))                     

        if current_dualobj >= previous_dualobj && metrics.dualObj[end] > metrics.dualObj[2]
            metrics.count_dualobj_1 += 1
            metrics.count_dualobj_2 = 0
            metrics.count_dualobj_3 = 0
        elseif current_dualobj >= previous_dualobj && metrics.dualObj[end] < metrics.dualObj[2]
            metrics.count_dualobj_1 = 0
            metrics.count_dualobj_2 = 0
            metrics.count_dualobj_3 += 1 
        else
            metrics.count_dualobj_1 = 0
            metrics.count_dualobj_2 += 1
            metrics.count_dualobj_3 = 0
        end
    elseif iter > 2 && metrics.dualobj_bounded == BOUNDED_UNDEFINED && metrics.count_dualobj_3 > 1000  
        metrics.dualobj_bounded = LOWER_UNBOUNDED
    elseif iter > 2 && metrics.dualobj_bounded == BOUNDED_UNDEFINED && metrics.count_dualobj_1 > 1000  
        metrics.dualobj_bounded = UPPER_UNBOUNDED   
    elseif iter > 2 && metrics.dualobj_bounded == BOUNDED_UNDEFINED && metrics.count_dualobj_2 > 1000  
        metrics.dualobj_bounded = BOUNDED_YES
    end

    # Dual objective convergence check
    if iter > 2 && metrics.dualobj_bounded == BOUNDED_UNDEFINED && metrics.count_dualobj_4 <= 10 && !isempty(metrics.dualObj)
        if length(metrics.dualObj) >= 2 && metrics.dualObj[end] != 0 && 
           abs((metrics.dualObj[end] - metrics.dualObj[end-1]) / metrics.dualObj[end]) < 1e-5
            metrics.count_dualobj_4 += 1
        else
            metrics.count_dualobj_4 = 0
        end
    elseif iter > 2 && metrics.dualobj_bounded == BOUNDED_UNDEFINED && metrics.count_dualobj_4 > 10
        metrics.dualobj_bounded = BOUNDED_YES 
    end
    
    # === Special Cases: Automatic Property Assignment ===
    # If optimality conditions are met, assume all sequences are well-behaved
    if iter > 2 && info.presL2[end] < presTolL2 && metrics.dualDifferenceL2[end] < dresTolL2
        metrics.Bz_norm_bounded = BOUNDED_YES
        metrics.y_norm_bounded = BOUNDED_YES
        metrics.obj_bounded = BOUNDED_YES
        metrics.dualobj_bounded = BOUNDED_YES
    end

    # If both key norms are bounded, assume convergence
    if iter > 2 && metrics.Bz_norm_bounded == BOUNDED_YES && metrics.y_norm_bounded == BOUNDED_YES
        metrics.presL2_convergent = CONVERGENT_YES
        metrics.dres_convergent = CONVERGENT_YES
        metrics.obj_bounded = BOUNDED_YES
        metrics.dualobj_bounded = BOUNDED_YES
    end

    # classification
    if metrics.presL2_convergent == CONVERGENT_YES &&  metrics.dres_convergent == CONVERGENT_YES && metrics.Bz_norm_bounded == BOUNDED_YES && metrics.y_norm_bounded == BOUNDED_YES && metrics.obj_bounded == BOUNDED_YES && metrics.dualobj_bounded == BOUNDED_YES
        metrics.problem_classification = CLASSIFICATION_CASE_A
    elseif metrics.presL2_convergent == CONVERGENT_YES &&  metrics.dres_convergent == CONVERGENT_YES && metrics.Bz_norm_bounded == BOUNDED_YES && metrics.y_norm_bounded == UPPER_UNBOUNDED && metrics.obj_bounded != BOUNDED_UNDEFINED &&  metrics.dualobj_bounded != BOUNDED_UNDEFINED 
        if metrics.obj_bounded == BOUNDED_YES && metrics.dualobj_bounded == BOUNDED_YES
            metrics.problem_classification = CLASSIFICATION_CASE_B
        else
            metrics.problem_classification = CLASSIFICATION_CASE_F
        end
    elseif metrics.presL2_convergent == CONVERGENT_YES &&  metrics.dres_convergent == CONVERGENT_YES && metrics.Bz_norm_bounded == UPPER_UNBOUNDED && metrics.y_norm_bounded == BOUNDED_YES && metrics.obj_bounded != BOUNDED_UNDEFINED &&  metrics.dualobj_bounded != BOUNDED_UNDEFINED
        if metrics.obj_bounded == BOUNDED_YES && metrics.dualobj_bounded == BOUNDED_YES
            metrics.problem_classification = CLASSIFICATION_CASE_C
        elseif metrics.obj_bounded == LOWER_UNBOUNDED && metrics.dualobj_bounded == LOWER_UNBOUNDED
            metrics.problem_classification = CLASSIFICATION_CASE_E
        else
            metrics.problem_classification = CLASSIFICATION_CASE_F
        end
    elseif metrics.presL2_convergent == CONVERGENT_YES &&  metrics.dres_convergent == CONVERGENT_YES && metrics.Bz_norm_bounded == UPPER_UNBOUNDED && metrics.y_norm_bounded == UPPER_UNBOUNDED && metrics.obj_bounded != BOUNDED_UNDEFINED &&  metrics.dualobj_bounded != BOUNDED_UNDEFINED
        if metrics.obj_bounded == BOUNDED_YES && metrics.dualobj_bounded == BOUNDED_YES
            metrics.problem_classification = CLASSIFICATION_CASE_D
        elseif metrics.obj_bounded == LOWER_UNBOUNDED && metrics.dualobj_bounded == LOWER_UNBOUNDED
            metrics.problem_classification = CLASSIFICATION_CASE_E
        else
            metrics.problem_classification = CLASSIFICATION_CASE_F
        end
    elseif metrics.presL2_convergent == CONVERGENT_YES &&  metrics.dres_convergent == CONVERGENT_NO 
            metrics.problem_classification = CLASSIFICATION_CASE_E
    elseif metrics.presL2_convergent == CONVERGENT_NO
            metrics.problem_classification = CLASSIFICATION_CASE_F           
    end
end