"""
    SpecializedOriginalADMMSubproblemSolver

Abstract type for specialized original ADMM subproblem solvers.

This type represents the interface for specialized solvers that implement the
original ADMM subproblem formulation. Each specialized solver must provide the
following methods:
"""
abstract type SpecializedOriginalADMMSubproblemSolver end  

include("SpecializedOriginalADMMSubproblemSolvers/LinearSolver.jl")
include("SpecializedOriginalADMMSubproblemSolvers/ProximalMappingSolver.jl")
include("SpecializedOriginalADMMSubproblemSolvers/ConsensusSolver.jl")
include("SpecializedOriginalADMMSubproblemSolvers/OneDimensionalQuadraticSolver.jl")
include("SpecializedOriginalADMMSubproblemSolvers/JuMPSolver.jl")

"""
    OriginalADMMSubproblemSolver <: AbstractADMMSubproblemSolver

Original ADMM subproblem solver that computes exact solutions using specialized solvers.

This solver implements the standard ADMM approach where each primal subproblem is
solved exactly (or to high precision) using problem-specific specialized solvers.
It automatically detects subproblem structure and selects the most appropriate
solver from a hierarchy of specialized methods.

**Mathematical Background**

The ADMM subproblem for node `i` takes the form:
```math
\\min_{x_i} f_i(x_i) + g_i(x_i) + ⟨λ, A_i x_i⟩ + \\frac{ρ}{2}\\|A_i x_i + \\sum_{j≠i} A_j x_j - c\\|_2^2
```

This can be rewritten as:
```math
\\min_{x_i} f_i(x_i) + g_i(x_i) + ⟨λ, A_i x_i⟩ + \\frac{ρ}{2}(x_i^T A_i^T A_i x_i + 2⟨A_i^T(\\sum_{j≠i} A_j x_j - c), x_i⟩ + \\text{const})
```

**Specialized Solver Hierarchy**

The solver attempts to use specialized solvers in order of preference:

1. **LinearSolver**: For linear/quadratic subproblems (fastest)
2. **ProximalMappingSolver**: For problems with known proximal operators
3. **JuMPSolver**: For general nonlinear problems (most flexible)

**Key Features**

- **Automatic Detection**: Automatically selects the best solver for each subproblem
- **Exact Solutions**: Computes exact or high-precision solutions
- **Precomputed Adjoints**: Uses `EdgeData` for efficient repeated computations
- **Parallel Support**: Supports parallel solution of independent subproblems
- **Anderson Acceleration**: Special handling for Anderson acceleration schemes

**Performance Characteristics**

- **Convergence**: Typically requires fewer iterations due to exact solutions
- **Per-iteration Cost**: Higher than linearized methods but more reliable
- **Memory Usage**: Moderate (precomputed adjoint mappings)
- **Scalability**: Good for problems with structured subproblems

**Implementation Notes**

- Each node gets its own specialized solver instance
- Edge data is precomputed once during initialization
- Augmented Lagrangian coefficients are recomputed each iteration
- Special handling for Anderson acceleration requires modified linear terms
"""
mutable struct OriginalADMMSubproblemSolver <: AbstractADMMSubproblemSolver
    models::Dict{String, SpecializedOriginalADMMSubproblemSolver}
    edgeData::Dict{String, EdgeData}
    augmentedLagrangianLinearCoefficientsBuffer::Dict{String, NumericVariable} # buffer to store the linear coefficients of primal variables, i.e., A'y or B'y
    
    """
        OriginalADMMSubproblemSolver()

    Construct an original ADMM subproblem solver with empty state.

    **Initialization**
    - `models`: Dictionary of specialized solvers for each node (empty initially)
    - `edgeData`: Dictionary of precomputed edge information (empty initially)
    - `augmentedLagrangianLinearCoefficientsBuffer`: Working buffers for each node (empty initially)

    **Usage**
    After construction, call `initialize!` to set up the solver for a specific problem.

    **Example**
    ```julia
    solver = OriginalADMMSubproblemSolver()
    success = initialize!(solver, admmGraph, info)
    ```
    """
    OriginalADMMSubproblemSolver() = new(Dict{String, SpecializedOriginalADMMSubproblemSolver}(), 
        Dict{String, EdgeData}(), 
        Dict{String, NumericVariable}())
end

"""
    selectNodalSolver(solver::OriginalADMMSubproblemSolver, nodeID::String, 
                     admmGraph::ADMMBipartiteGraph, rho::Float64)

Select and initialize the most appropriate specialized solver for a given node.

This function attempts to create specialized solvers in order of preference,
starting with the fastest/most specialized and falling back to more general
solvers if needed.

**Arguments**
- `solver::OriginalADMMSubproblemSolver`: The main solver instance
- `nodeID::String`: Identifier of the node to create a solver for
- `admmGraph::ADMMBipartiteGraph`: The bipartite graph structure
- `rho::Float64`: Current penalty parameter value
- `logLevel::Int64`: Logging level
**Returns**
- `Bool`: `true` if a specialized solver was successfully created, `false` otherwise

**Solver Selection Priority**

1. **LinearSolver**: For linear/quadratic objectives with linear constraints
   - Fastest execution
   - Direct linear system solution
   - Handles most structured problems

2. **ProximalMappingSolver**: For problems with known proximal operators
   - Closed-form solutions when available
   - Efficient for regularized problems
   - Handles non-smooth objectives

3. **JuMPSolver**: For general nonlinear problems
   - Most flexible, handles arbitrary objectives
   - Uses numerical optimization solvers
   - Fallback for complex subproblems

**Error Handling**

Each solver constructor may throw an exception if it cannot handle the subproblem
structure. The function catches these exceptions and tries the next solver in the
hierarchy.

**Example**
```julia
success = selectNodalSolver(solver, "node1", admmGraph, 1.0)
if !success
    error("No suitable solver found for node1")
end
```
"""
function selectNodalSolver(solver::OriginalADMMSubproblemSolver, 
    nodeID::String, 
    admmGraph::ADMMBipartiteGraph, 
    rho::Float64, 
    logLevel::Int64)

    SpecializedOriginalADMMSubproblemSolverList = [
        LinearSolver, 
        ProximalMappingSolver,
        # ConsensusSolver,
        OneDimensionalQuadraticSolver,
        JuMPSolver 
    ]

    for nodalSolver in SpecializedOriginalADMMSubproblemSolverList
        try 
            solver.models[nodeID] = nodalSolver(nodeID, admmGraph, solver.edgeData, rho, logLevel)
            return true 
        catch e 
            @PDMOError logLevel "OriginalADMMSubproblemSolver: error catched: $e."
            continue
        end 
    end 
    return false
end 

"""
    initialize!(solver::OriginalADMMSubproblemSolver, admmGraph::ADMMBipartiteGraph, info::ADMMIterationInfo)

Initialize the original ADMM subproblem solver with problem-specific information.

This function sets up the solver by precomputing edge data, selecting specialized
solvers for each node, and allocating necessary buffers. It prepares the solver
for efficient iterative solving.

**Arguments**
- `solver::OriginalADMMSubproblemSolver`: The solver instance to initialize
- `admmGraph::ADMMBipartiteGraph`: The bipartite graph structure of the problem
- `info::ADMMIterationInfo`: Current iteration information containing penalty parameter

**Returns**
- `Bool`: `true` if initialization was successful, `false` if any node couldn't be handled

**Initialization Process**

1. **Edge Data Precomputation**: Creates `EdgeData` for each edge containing precomputed adjoints
2. **Solver Selection**: Attempts to create specialized solver for each node
3. **Buffer Allocation**: Allocates augmented Lagrangian coefficient buffers
4. **Validation**: Ensures all nodes have appropriate solvers

**Effects**
- Populates `solver.edgeData` with precomputed adjoint information
- Populates `solver.models` with specialized solvers for each node
- Allocates `solver.augmentedLagrangianLinearCoefficientsBuffer` for each node

**Error Handling**
- Warns if no specialized solver can be found for any node
- Returns `false` if initialization fails for any node
- Logs warnings for nodes that couldn't be handled

**Example**
```julia
solver = OriginalADMMSubproblemSolver()
success = initialize!(solver, admmGraph, info)
if !success
    error("Failed to initialize solver")
end
```
"""
function initialize!(solver::OriginalADMMSubproblemSolver,
    admmGraph::ADMMBipartiteGraph, 
    info::ADMMIterationInfo, 
    logLevel::Int64)
    rho = info.rhoHistory[end][1]

    for (edgeID, edge) in admmGraph.edges 
        solver.edgeData[edgeID] = EdgeData(edge)
    end 

    for (nodeID, node) in admmGraph.nodes 
        if selectNodalSolver(solver, nodeID, admmGraph, rho, logLevel) == false 
            @PDMOWarn logLevel "OriginalADMMSubproblemSolver: No specialized solver found for node $nodeID."
            return false 
        end 
        solver.augmentedLagrangianLinearCoefficientsBuffer[nodeID] = zero(node.val)
    end


    countLinearSolver = 0
    countJuMPSolver = 0
    countConsensusSolver = 0
    countProximaMappingSolver = 0
    countOneDimensionalQuadraticSolver = 0
    for (nodeID, model) in solver.models 
        if isa(model, LinearSolver)
            countLinearSolver += 1
        elseif isa(model, JuMPSolver)
            countJuMPSolver += 1
        elseif isa(model, ConsensusSolver)
            countConsensusSolver += 1
        elseif isa(model, ProximalMappingSolver)
            countProximaMappingSolver += 1
        elseif isa(model, OneDimensionalQuadraticSolver)
            countOneDimensionalQuadraticSolver += 1
        end 
    end 
    if countLinearSolver > 0
        @PDMOInfo logLevel  "OriginalADMMSubproblemSolver: number of LinearSolver = $countLinearSolver"
    end 
    if countJuMPSolver > 0
        @PDMOInfo logLevel  "OriginalADMMSubproblemSolver: number of JuMPSolver = $countJuMPSolver"
    end 
    if countConsensusSolver > 0
        @PDMOInfo logLevel  "OriginalADMMSubproblemSolver: number of ConsensusSolver = $countConsensusSolver"
    end 
    if countProximaMappingSolver > 0
        @PDMOInfo logLevel  "OriginalADMMSubproblemSolver: number of ProximalMappingSolver = $countProximaMappingSolver"
    end 
    if countOneDimensionalQuadraticSolver > 0
        @PDMOInfo logLevel  "OriginalADMMSubproblemSolver: number of OneDimensionalQuadraticSolver = $countOneDimensionalQuadraticSolver"
    end 
    return true 
end 

"""
    solve!(solver::OriginalADMMSubproblemSolver, nodeID::String, accelerator::AbstractADMMAccelerator,
           admmGraph::ADMMBipartiteGraph, info::ADMMIterationInfo, isLeft::Bool, enableParallel::Bool=false)

Solve the ADMM subproblem for a specific node using the exact approach.

This function prepares the augmented Lagrangian linear coefficients and delegates
the actual solving to the appropriate specialized solver. It handles the conversion
between the ADMM formulation and the specialized solver's expected format.

**Arguments**
- `solver::OriginalADMMSubproblemSolver`: The solver instance
- `nodeID::String`: Identifier of the node to solve for
- `accelerator::AbstractADMMAccelerator`: Acceleration scheme (affects linear coefficient computation)
- `admmGraph::ADMMBipartiteGraph`: The bipartite graph structure
- `info::ADMMIterationInfo`: Current iteration information
- `isLeft::Bool`: Whether the node is on the left side of the bipartite graph
- `enableParallel::Bool`: Whether to enable parallel computation

**Effects**
- Updates `info.primalSol[nodeID]` with the new solution
- Stores previous solution in `info.primalSolPrev[nodeID]`
- Uses `solver.augmentedLagrangianLinearCoefficientsBuffer[nodeID]` for intermediate computation

**Algorithm**

1. **Linear Coefficient Computation**: Computes `A^T(λ + ρ(∑A_jx_j - c))` for the augmented Lagrangian
2. **Anderson Acceleration Handling**: Special case for Anderson acceleration using converter output
3. **Specialized Solver Call**: Delegates to the appropriate specialized solver

**Mathematical Details**

The linear coefficient for node `i` is:
```math
A_i^T(λ + ρ(\\sum_{j≠i} A_j x_j - c))
```

For Anderson acceleration, the coefficient uses the accelerated dual variables:
```math
A_i^T(ζ)
```
where `ζ` is the output of the Anderson converter.

**Performance Notes**
- Linear coefficient computation is O(problem_size) per node
- Specialized solvers have problem-specific complexity
- Anderson acceleration requires special handling but no additional computational cost
"""
function solve!(solver::OriginalADMMSubproblemSolver,
    nodeID::String, 
    accelerator::AbstractADMMAccelerator,
    admmGraph::ADMMBipartiteGraph, 
    info::ADMMIterationInfo, 
    isLeft::Bool,
    enableParallel::Bool = false)

    rho = info.rhoHistory[end][1]

    # prepare the linear term of the augmented Lagrangian function 
    if isLeft == false && isa(accelerator, AndersonAccelerator)
        fill!(solver.augmentedLagrangianLinearCoefficientsBuffer[nodeID], 0.0)
        for edgeID in admmGraph.nodes[nodeID].neighbors 
            edge = admmGraph.edges[edgeID]
            adjoint!(edge.mappings[nodeID], accelerator.converter.outputBuffer[edgeID], solver.augmentedLagrangianLinearCoefficientsBuffer[nodeID], true)
        end 
    else 
        fill!(solver.augmentedLagrangianLinearCoefficientsBuffer[nodeID], 0.0)
        for edgeID in admmGraph.nodes[nodeID].neighbors
            edge = admmGraph.edges[edgeID]
            otherID = nodeID == edge.nodeID1 ? edge.nodeID2 : edge.nodeID1     
            # linear coefficient = A'(lmd + rho * (By-b))
            edge.mappings[otherID](info.primalSol[otherID], info.dualBuffer[edgeID])  # By
            axpy!(-1.0, edge.rhs, info.dualBuffer[edgeID]) # By - b
            axpby!(1.0, info.dualSol[edgeID], rho, info.dualBuffer[edgeID]) # lmd + rho * (By - b)
            adjoint!(edge.mappings[nodeID], info.dualBuffer[edgeID], solver.augmentedLagrangianLinearCoefficientsBuffer[nodeID], true)
        end 
    end 

    # call solve! of the specialized solver 
    solve!(solver.models[nodeID], 
        nodeID, 
        admmGraph, 
        info, 
        solver.edgeData, 
        solver.augmentedLagrangianLinearCoefficientsBuffer, 
        enableParallel)
end 

"""
    updateDualResidualsInBuffer!(solver::OriginalADMMSubproblemSolver, info::ADMMIterationInfo, 
                                admmGraph::ADMMBipartiteGraph, accelerator::AbstractADMMAccelerator)

Compute and store dual residuals for convergence monitoring.

This function computes the dual residuals for the original ADMM method, which
measure the optimality condition violations. The residuals are computed using
the standard ADMM dual residual formula.

**Arguments**
- `solver::OriginalADMMSubproblemSolver`: The solver instance
- `info::ADMMIterationInfo`: Current iteration information
- `admmGraph::ADMMBipartiteGraph`: The bipartite graph structure
- `accelerator::AbstractADMMAccelerator`: Acceleration scheme (affects residual computation)

**Effects**
- Updates `info.dresL2` with L2 norm of dual residuals
- Updates `info.dresLInf` with L∞ norm of dual residuals
- Uses `info.primalBuffer` to store computed residuals per node

**Algorithm**

1. **Constraint Difference Computation**: Computes `B(z^{k+1} - z^k)` for right nodes
2. **Adjoint Application**: Applies `A^T` to get dual residuals for left nodes
3. **Anderson Acceleration Correction**: Adds correction terms for Anderson acceleration
4. **Norm Computation**: Computes L2 and L∞ norms of the dual residuals

**Mathematical Details**

The dual residual for the original ADMM method is:
```math
r_{dual}^k = ρA^TB(z^{k+1} - z^k)
```

For Anderson acceleration, an additional term is added:
```math
r_{dual}^k = ρA^TB(z^{k+1} - z^k) + A^T(ζ^{k+1} - ζ^k)
```

where `ζ` represents the Anderson acceleration variables.

**Performance Notes**
- Computational cost is O(problem_size) 
- Uses parallel computation with `@threads` for independent calculations
- Efficient buffer reuse minimizes memory allocations
"""
function updateDualResidualsInBuffer!(solver::OriginalADMMSubproblemSolver, 
    info::ADMMIterationInfo, 
    admmGraph::ADMMBipartiteGraph, 
    accelerator::AbstractADMMAccelerator)

    rho = info.rhoHistory[end][1]
    
    @threads for nodeID in admmGraph.left 
        info.primalBuffer[nodeID] .= 0.0 
    end 

    for nodeID in admmGraph.right
        # Calculate z^{k+1} - z^{k} and store in info.primalBuffer[nodeID]
        copyto!(info.primalBuffer[nodeID], info.primalSol[nodeID])
        axpy!(-1.0, info.primalSolPrev[nodeID], info.primalBuffer[nodeID])

        for edgeID in admmGraph.nodes[nodeID].neighbors
            # get ID of the other node for the current edge 
            otherID = admmGraph.edges[edgeID].nodeID1 == nodeID ? admmGraph.edges[edgeID].nodeID2 : admmGraph.edges[edgeID].nodeID1
            # store B(z^{k+1}-z^{k}) in info.dualBuffer[edgeID]
            admmGraph.edges[edgeID].mappings[nodeID](info.primalBuffer[nodeID], info.dualBuffer[edgeID], false)
            # store A'B(z^{k+1}-z^{k}) in info.primalBuffer[otherID], accumulating for multiple edges
            adjoint!(admmGraph.edges[edgeID].mappings[otherID], info.dualBuffer[edgeID], info.primalBuffer[otherID], true)
        end 
        
        # right node does not have dual residuals
        info.primalBuffer[nodeID] .=0.0
    end 

    for nodeID in admmGraph.left 
        rmul!(info.primalBuffer[nodeID], rho)
    end 

    # For anderson acceleration, add the term A'(zeta^{k+1} - zeta^{k}) to info.primalBuffer, 
    # where zeta^{k} and zeta^{k+1} are input and output of the accelerator, respectively.
    if isa(accelerator, AndersonAccelerator)
        for nodeID in admmGraph.left 
            for edgeID in admmGraph.nodes[nodeID].neighbors
                edge = admmGraph.edges[edgeID]
                adjoint!(edge.mappings[nodeID], -accelerator.converter.outputBuffer[edgeID], info.primalBuffer[nodeID], true)
                adjoint!(edge.mappings[nodeID], accelerator.converter.inputBuffer[edgeID], info.primalBuffer[nodeID], true)
            end 
        end 
    end 

    # if abs(solver.dualStepsize - 1.0) > ZeroTolerance 
    #     # add additional term: (1-dualStepsize) * rho * A'(Ax^{k+1} + Bz^k -b)
    #     @threads for nodeID in admmGraph.left 
    #         solver.primalBuffer[nodeID] .= 0.0
    #         for edgeID in admmGraph.nodes[nodeID].neighbors 
    #             nodeID1 = admmGraph.edges[edgeID].nodeID1
    #             nodeID2 = admmGraph.edges[edgeID].nodeID2
    #             solver.dualBuffer[edgeID] .= -admmGraph.edges[edgeID].rhs
    #             admmGraph.edges[edgeID].mappings[nodeID1](info.primalSolPrev[nodeID1], solver.dualBuffer[edgeID], true)
    #             admmGraph.edges[edgeID].mappings[nodeID2](info.primalSolPrev[nodeID2], solver.dualBuffer[edgeID], true )
    #             adjoint!(admmGraph.edges[edgeID].mappings[nodeID], solver.dualBuffer[edgeID], solver.primalBuffer[nodeID], true)
    #         end 
    #         axpy!((1-solver.dualStepsize) * rho, solver.primalBuffer[nodeID], info.primalBuffer[nodeID])
    #     end 
    # end 

    dresL2Square = 0.0 
    dresLInf = 0.0 
    for nodeID in admmGraph.left 
        dresL2Square += dot(info.primalBuffer[nodeID], info.primalBuffer[nodeID])
        dresLInf = max(dresLInf, norm(info.primalBuffer[nodeID], Inf))
    end 

    push!(info.dresL2, sqrt(dresL2Square))
    push!(info.dresLInf, dresLInf)
end 

"""
    update!(solver::OriginalADMMSubproblemSolver, info::ADMMIterationInfo, 
           admmGraph::ADMMBipartiteGraph, rhoUpdated::Bool)

Update solver state when algorithm parameters change.

This function updates the internal state of all specialized solvers when the
penalty parameter ρ or other algorithm parameters change. It ensures that
all solvers remain consistent with the current algorithm state.

**Arguments**
- `solver::OriginalADMMSubproblemSolver`: The solver instance
- `info::ADMMIterationInfo`: Current iteration information with updated parameters
- `admmGraph::ADMMBipartiteGraph`: The bipartite graph structure
- `rhoUpdated::Bool`: Flag indicating whether the penalty parameter was updated

**Effects**
- Calls `update!` on all specialized solvers if `rhoUpdated` is true
- No operations if `rhoUpdated` is false

**Implementation Notes**
- Uses parallel computation with `@threads` for independent solver updates
- Each specialized solver handles its own parameter updates
- The update is typically needed when ADMM adapters change the penalty parameter

**Example**
```julia
# After penalty parameter adaptation
if adapter_changed_rho
    update!(solver, info, admmGraph, true)
end
```
"""
function update!(solver::OriginalADMMSubproblemSolver, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph, rhoUpdated::Bool)
    if rhoUpdated == false 
        return 
    end 

    pairs = collect(solver.models)
    @threads for idx in eachindex(pairs)
        nodeID, model = pairs[idx]
        update!(model, info, admmGraph, rhoUpdated)
    end 
end 
