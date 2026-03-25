"""
    SRAAdapter <: AbstractADMMAdapter

Spectral Radius Adaptive (SRA) adapter for ADMM penalty parameter adaptation.

This adapter implements a sophisticated penalty parameter adaptation strategy based on
spectral radius estimation. It analyzes the convergence behavior of dual variables and
constraint violations to automatically adjust ρ for optimal convergence rates.

**Mathematical Background**

The SRA adapter is based on the observation that the optimal penalty parameter ρ is
related to the spectral radius of the iteration matrix. The adapter approximates this
by analyzing the relationship between consecutive iterates:

For dual variables: `Δy^k = y^{k+1} - y^k`
For constraint violations: `ΔBz^k = B(z^{k+1} - z^k)`

The ratio `‖Δy^k‖₂ / ‖ΔBz^k‖₂` provides an estimate of the appropriate penalty parameter.

**Algorithm Strategy**

The adapter computes:
1. **Dual difference norm**: `‖y^{k+1} - y^k‖₂`
2. **Constraint mapping difference norm**: `‖B(z^{k+1} - z^k)‖₂`  
3. **Spectral ratio**: `ρ_new = ‖Δy^k‖₂ / ‖ΔBz^k‖₂`

Special cases are handled when norms are near zero:
- If `‖Δy^k‖₂ ≈ 0` and `‖ΔBz^k‖₂ > 0`: Decrease ρ (dual optimality achieved)
- If `‖Δy^k‖₂ > 0` and `‖ΔBz^k‖₂ ≈ 0`: Increase ρ (constraint satisfaction achieved)
- If both ≈ 0: Keep ρ unchanged (convergence achieved)

**Parameters**

- `T::Int64`: Update frequency (default: 5) - Parameter updated every T iterations
- `increasingFactor::Float64`: Factor for increasing ρ in special cases (default: 2.0)
- `decreasingFactor::Float64`: Factor for decreasing ρ in special cases (default: 2.0)

**Advantages over RB Adapter**

1. **Theoretical Foundation**: Based on spectral analysis of ADMM convergence
2. **Adaptive Estimation**: Directly estimates optimal ρ rather than simple ratio balancing
3. **Sophisticated Handling**: Special case analysis for near-convergence scenarios
4. **Reduced Oscillations**: Updates less frequently to avoid parameter oscillations

**Performance Characteristics**

- **Computational Cost**: O(problem_size) per update (computing norms)
- **Memory Usage**: O(problem_size) for storing difference vectors
- **Update Frequency**: Every T iterations (default: 5)
- **Stability**: Generally more stable than frequent RB updates

**Parameter Selection Guidelines**

- **T (Update Frequency)**: 
  - Smaller values (3-5): More responsive adaptation
  - Larger values (8-15): More stable, less frequent updates
  - Very large values (>20): May miss important adaptations

- **increasingFactor/decreasingFactor**:
  - Smaller values (1.5-2.0): Conservative adjustments
  - Larger values (2.5-5.0): More aggressive adjustments


"""
struct SRAAdapter <: AbstractADMMAdapter 
    T::Int64 
    increasingFactor::Float64 
    decreasingFactor::Float64
    dualDiff::Dict{String, NumericVariable}      # y^{k+1} - y^{k}
    rightNodeDiff::Dict{String, NumericVariable} # z^{k+1} - z^{k}
    BzDiff::Dict{String, NumericVariable}        # Bz^{k+1} - Bz^{k}
    
    """
        SRAAdapter(;T::Int64=5, increasingFactor::Float64=2.0, decreasingFactor::Float64=2.0)

    Construct a Spectral Radius Adaptive adapter with specified parameters.

    **Arguments**
    - `T::Int64=5`: Update frequency (parameter updated every T iterations)
    - `increasingFactor::Float64=2.0`: Factor for increasing ρ in special cases
    - `decreasingFactor::Float64=2.0`: Factor for decreasing ρ in special cases

    **Parameter Selection Guidelines**

    **T (Update Frequency)**:
    - **Small values (3-5)**: More responsive to changes, higher computational cost
    - **Medium values (5-10)**: Balanced responsiveness and stability (recommended)
    - **Large values (10-20)**: More stable, may miss rapid changes
    - **Very large values (>20)**: Risk of missing important adaptations

    **increasingFactor**:
    - **Conservative (1.5-2.0)**: Gradual increases, more stable
    - **Moderate (2.0-3.0)**: Balanced performance (recommended)
    - **Aggressive (3.0-5.0)**: Rapid increases, risk of overshooting

    **decreasingFactor**:
    - **Conservative (1.5-2.0)**: Gradual decreases, more stable
    - **Moderate (2.0-3.0)**: Balanced performance (recommended)
    - **Aggressive (3.0-5.0)**: Rapid decreases, risk of undershooting

    **Recommended Combinations**
    - **Conservative**: `T=10, increasingFactor=1.5, decreasingFactor=1.5`
    - **Balanced**: `T=5, increasingFactor=2.0, decreasingFactor=2.0` (default)
    - **Aggressive**: `T=3, increasingFactor=3.0, decreasingFactor=3.0`


    """
    function SRAAdapter(;T::Int64=5, 
                        increasingFactor::Float64=2.0, 
                        decreasingFactor::Float64=2.0)
        return new(T, 
            increasingFactor, 
            decreasingFactor, 
            Dict{String, NumericVariable}(), 
            Dict{String, NumericVariable}(), 
            Dict{String, NumericVariable}())
    end 
end 

"""
    initialize!(adapter::SRAAdapter, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph)

Initialize the SRA adapter by setting up difference tracking variables.

This function sets up the internal state required for spectral radius estimation.
It creates storage for tracking differences in dual variables, primal variables
on right nodes, and constraint mapping differences.

**Arguments**
- `adapter::SRAAdapter`: The SRA adapter instance to initialize
- `info::ADMMIterationInfo`: Current ADMM iteration information
- `admmGraph::ADMMBipartiteGraph`: Bipartite graph structure of the ADMM problem

**Initialization Process**

1. **Dual Difference Setup**: Creates zero-valued storage for `y^{k+1} - y^k`
2. **Constraint Mapping Difference Setup**: Creates zero-valued storage for `B(z^{k+1} - z^k)`
3. **Right Node Difference Setup**: Creates zero-valued storage for `z^{k+1} - z^k`

**Memory Allocation**

The adapter allocates memory proportional to:
- Number of edges × dual variable size (for `dualDiff` and `BzDiff`)
- Number of right nodes × primal variable size (for `rightNodeDiff`)

**Post-Initialization State**
- All difference tracking variables are initialized to zero
- Adapter is ready for penalty parameter updates
- Memory is allocated for efficient difference computations


"""
function initialize!(adapter::SRAAdapter, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph)
    for edgeID in keys(admmGraph.edges)
        adapter.dualDiff[edgeID] = zero(admmGraph.edges[edgeID].rhs)
        adapter.BzDiff[edgeID] = zero(admmGraph.edges[edgeID].rhs)
    end 
    for nodeID in admmGraph.right 
        adapter.rightNodeDiff[nodeID] = zero(admmGraph.nodes[nodeID].val)
    end 
end 

"""
    updatePenalty(adapter::SRAAdapter, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph, iter::Int64)

Update the penalty parameter using Spectral Radius Adaptive strategy.

This function implements the core SRA algorithm that estimates the optimal penalty
parameter based on spectral radius analysis of the ADMM iteration matrix.

**Arguments**
- `adapter::SRAAdapter`: The SRA adapter with update parameters
- `info::ADMMIterationInfo`: Current iteration information with primal/dual solutions
- `admmGraph::ADMMBipartiteGraph`: Bipartite graph representation of the problem
- `iter::Int64`: Current iteration number

**Returns**
- `Bool`: `true` if ρ was updated, `false` otherwise

**Algorithm Steps**

1. **Update Timing**: Check if update should occur (every T iterations, starting from iteration 1)
2. **Dual Difference Computation**: Calculate `‖y^{k+1} - y^k‖₂`
3. **Constraint Difference Computation**: Calculate `‖B(z^{k+1} - z^k)‖₂`
4. **Spectral Ratio Analysis**: Determine new ρ based on the ratio:
   - If `‖Δy‖₂ ≈ 0` and `‖ΔBz‖₂ > 0`: Decrease ρ (dual optimality achieved)
   - If `‖Δy‖₂ > 0` and `‖ΔBz‖₂ ≈ 0`: Increase ρ (primal feasibility achieved)
   - If both ≈ 0: Keep ρ unchanged (convergence achieved)
   - Otherwise: Set `ρ = ‖Δy‖₂ / ‖ΔBz‖₂` (spectral estimate)
5. **Bounds Enforcement**: Clamp ρ to `[ADMM_MIN_RHO, ADMM_MAX_RHO]`
6. **History Update**: Update penalty history if change is significant

**Mathematical Details**

**Dual Difference Computation**:
```math
\\|Δy^k\\|_2 = \\sqrt{\\sum_{edges} \\|y_{edge}^{k+1} - y_{edge}^k\\|_2^2}
```

**Constraint Mapping Difference Computation**:
```math
\\|ΔBz^k\\|_2 = \\sqrt{\\sum_{edges} \\|B_{edge}(z_{right}^{k+1} - z_{right}^k)\\|_2^2}
```

**Spectral Ratio**:
```math
ρ_{new} = \\frac{\\|Δy^k\\|_2}{\\|ΔBz^k\\|_2}
```

**Special Cases Handling**

- **Near-dual-optimality**: `‖Δy‖₂ ≈ 0` suggests dual variables have converged
- **Near-primal-feasibility**: `‖ΔBz‖₂ ≈ 0` suggests constraint violations are stable
- **Near-convergence**: Both small suggests overall convergence

**Performance Characteristics**

- **Computational Cost**: O(problem_size) every T iterations
- **Memory Usage**: O(problem_size) for difference storage
- **Numerical Stability**: Handles zero-norm cases gracefully
- **Convergence**: Generally leads to well-balanced convergence


"""
function updatePenalty(adapter::SRAAdapter, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph, iter::Int64)
    if iter == 1 || iter % adapter.T != 1
        return false 
    end 

    # numerator: ||y^{k+1} - y^{k}||_2
    dualDiff = 0.0
    for edgeID in info.edgeIDs
        copyto!(adapter.dualDiff[edgeID], info.dualSol[edgeID])
        axpy!(-1.0, info.dualSolPrev[edgeID], adapter.dualDiff[edgeID])
        dualDiff += dot(adapter.dualDiff[edgeID], adapter.dualDiff[edgeID])
    end 
    dualDiff = sqrt(dualDiff)

    # denominator: ||Bz^{k+1} - Bz^{k}||_2
    BzDiff = 0.0 
    for nodeID in admmGraph.right 
        copyto!(adapter.rightNodeDiff[nodeID], info.primalSol[nodeID])
        axpy!(-1.0, info.primalSolPrev[nodeID], adapter.rightNodeDiff[nodeID])
        for edgeID in admmGraph.nodes[nodeID].neighbors 
            admmGraph.edges[edgeID].mappings[nodeID](adapter.rightNodeDiff[nodeID], adapter.BzDiff[edgeID], false)
            BzDiff += dot(adapter.BzDiff[edgeID], adapter.BzDiff[edgeID])
        end 
    end 
    BzDiff = sqrt(BzDiff)

    oldRho = info.rhoHistory[end][1]
    newRho = oldRho 
    if abs(dualDiff) < ZeroTolerance && BzDiff > ZeroTolerance 
        newRho = oldRho / adapter.decreasingFactor
    elseif abs(dualDiff) > ZeroTolerance && BzDiff < ZeroTolerance 
        newRho = oldRho * adapter.increasingFactor
    elseif abs(dualDiff) < ZeroTolerance && BzDiff < ZeroTolerance 
        newRho = oldRho 
    else 
        newRho = dualDiff / BzDiff 
    end 

    newRho = max(min(newRho, ADMM_MAX_RHO), ADMM_MIN_RHO)
    
    if (abs(newRho - oldRho) > FeasTolerance)    
        push!(info.rhoHistory, (newRho, iter))
        return true 
    end 
    
    return false 
end