"""
    LinearSolver <: SpecializedOriginalADMMSubproblemSolver

Specialized solver for ADMM subproblems that reduce to linear systems.

This solver handles ADMM subproblems with quadratic or linear objective functions
and unconstrained domains, which can be solved exactly by solving linear systems
rather than using iterative optimization methods. The solver uses sparse matrix
factorizations for efficient solution of the resulting linear systems.

# Mathematical Formulation
The solver handles subproblems of the form:
    min f(x) + ⟨λ, Ax + By^k - b⟩ + (ρ/2)||Ax + By^k - b||²

where f(x) is quadratic, linear, or zero:
- **Quadratic**: f(x) = (1/2)x^T Q x + q^T x + r
- **Linear**: f(x) = q^T x + r  
- **Zero**: f(x) = 0

# Optimality Conditions
The first-order optimality conditions yield linear systems:
- **With Q**: (ρA^T A + Q)x = -A^T λ - ρA^T(By^k - b) - q
- **Without Q**: A^T A x = -A^T λ/ρ - A^T(By^k - b) - q/ρ

# Supported Problem Types
- **Objective Functions**: Zero, AffineFunction, QuadraticFunction
- **Domain Constraints**: Unconstrained or unbounded box constraints
- **Variable Types**: Vector variables only (no matrix variables)
- **Constraint Mappings**: Matrix, Identity, and Extraction mappings

# Factorization Strategy
The solver uses adaptive factorization based on matrix properties:
- **Cholesky**: For positive definite system matrices (preferred)
- **LDLT**: For positive semidefinite system matrices (fallback)
- **Precomputed A^T A**: Efficient handling of constraint structure

# Fields
- `Q::Union{SparseMatrixCSC{Float64, Int64}, Nothing}`: Quadratic term matrix
- `q::Union{Vector{Float64}, Nothing}`: Linear term vector
- `AAdjointSelf::SparseMatrixCSC{Float64, Int64}`: Precomputed A^T A matrix
- `currentRho::Float64`: Current penalty parameter ρ
- `systemMatrix::SparseMatrixCSC{Float64, Int64}`: System matrix for factorization
- `factorization::Union{SparseArrays.CHOLMOD.Factor{Float64}, Nothing}`: Matrix factorization
- `isPositiveDefinite::Bool`: Whether system matrix is positive definite
- `rhsBuffer::Vector{Float64}`: Working space for right-hand side vectors
- `logLevel::Int64`: Logging level

# Constructor Parameters
- `nodeID::String`: Node identifier in the ADMM graph
- `admmGraph::ADMMBipartiteGraph`: The bipartite graph structure
- `edgeData::Dict{String, EdgeData}`: Precomputed edge information
- `initialRho::Float64`: Initial penalty parameter

# Performance Characteristics
- **Computational Complexity**: O(n³) for factorization, O(n²) for solves
- **Memory Usage**: O(n²) for factorization storage
- **Convergence**: Exact solution in single step
- **Scalability**: Excellent for problems with sparse structure

# Implementation Notes
- Uses CHOLMOD for high-performance sparse factorizations
- Reuses factorizations when Q is present and ρ is fixed
- Handles both positive definite and semidefinite cases
- Efficient A^T A precomputation for constraint structure
"""
mutable struct LinearSolver <: SpecializedOriginalADMMSubproblemSolver 
    # quadratic term:: x'Qx + q'x; can be nothing    
    Q::Union{SparseMatrixCSC{Float64, Int64}, Nothing}
    q::Union{Vector{Float64}, Nothing}
    
    # A'A and A'b 
    AAdjointSelf::SparseMatrixCSC{Float64, Int64}

    currentRho::Float64

    # System matrix rho A'A + 2Q or A'A and its factorization
    systemMatrix::SparseMatrixCSC{Float64, Int64}
    factorization::Union{SparseArrays.CHOLMOD.Factor{Float64}, Nothing}
    isPositiveDefinite::Bool

    rhsBuffer::Vector{Float64}

    logLevel::Int64

    """
        LinearSolver(nodeID::String, edgeData::Dict{String, EdgeData}, admmGraph::ADMMBipartiteGraph, logLevel::Int64)

    Construct a linear solver for the specified ADMM node.

    This constructor validates the node structure for linear system compatibility,
    extracts quadratic and linear terms, and precomputes the A^T A matrix for
    efficient system solution.

    # Arguments
    - `nodeID::String`: Identifier of the ADMM node
    - `edgeData::Dict{String, EdgeData}`: Precomputed edge information
    - `admmGraph::ADMMBipartiteGraph`: Graph containing node and edge information
    - `logLevel::Int64`: Logging level

    # Validation Process
    1. **Variable Type Check**: Ensure variable is vector-valued
    2. **Constraint Check**: Verify domain is unconstrained or unbounded
    3. **Objective Check**: Validate objective is quadratic, linear, or zero
    4. **Structure Analysis**: Extract Q and q matrices/vectors

    # Effects
    - Creates solver with extracted problem structure
    - Precomputes A^T A matrix from all constraint mappings
    - Initializes buffers for linear system solution
    - Prepares for factorization setup

    # Error Conditions
    - Throws error if variable is not vector-valued
    - Throws error if domain has finite constraints
    - Throws error if objective is not supported type
    - Throws error for unsupported mapping types
    """
    function LinearSolver(nodeID::String, edgeData::Dict{String, EdgeData}, admmGraph::ADMMBipartiteGraph, logLevel::Int64)
        node = admmGraph.nodes[nodeID]
        @assert(length(size(node.val)) == 1, "LinearSolver only supports vector variables")
        @assert(isa(node.g, Zero) || 
                isa(node.g, IndicatorBox) && all(isinf.(-node.g.lb)) && all(isinf.(node.g.ub),), 
                isa(node.g, IndicatorSumOfNVariables) && node.g.numberVariables == 2, 
                "LinearSolver only supports unconstrained nodal domain: $(node.g)")
        @assert(isa(node.f, QuadraticFunction) || 
                isa(node.f, AffineFunction) || 
                isa(node.f, Zero), 
                "LinearSolver only supports quadratic or affine objective functions")
        
        # dim of the variable 
        n = length(node.val)
        
        # Extract Q and q if quadratic objective
        Q = nothing
        q = nothing
        if isa(node.f, QuadraticFunction)
            Q = node.f.Q
            q = node.f.q
        elseif isa(node.f, AffineFunction)
            q = node.f.A
        end

        # prepare A'A
        AAdjointSelf = spzeros(n,n)
        for edgeID in node.neighbors
            edge = admmGraph.edges[edgeID]
            isNode1 = edge.nodeID1 == nodeID

            adjointSelfMapping = isNode1 ? edgeData[edgeID].mapping1AdjSelf : edgeData[edgeID].mapping2AdjSelf
            if isa(adjointSelfMapping, LinearMappingMatrix)
                AAdjointSelf .+= adjointSelfMapping.A
            elseif isa(adjointSelfMapping, LinearMappingIdentity)
                AAdjointSelf .+= adjointSelfMapping.coe * I(n)
            elseif isa(adjointSelfMapping, LinearMappingExtraction)
                # More efficient to directly update the diagonal elements
                coe = adjointSelfMapping.coe
                for i in adjointSelfMapping.indexStart:adjointSelfMapping.indexEnd
                    AAdjointSelf[i,i] += coe
                end
            else
                error("LinearSolver: Unsupported mapping type to compute A'A.")
            end
        end

        # Initialize system matrix and factorization
        systemMatrix = spzeros(n,n)
        factorization = nothing
        isPositiveDefinite = false

        rhsBuffer = zeros(n)

        return new(Q, q, 
            AAdjointSelf, 
            0.0,  
            systemMatrix, 
            factorization, 
            isPositiveDefinite, 
            rhsBuffer, 
            logLevel)
    end
end

"""
    prepareLinearSolverData!(solver::LinearSolver, rho::Float64)

Prepare the linear solver for the given penalty parameter ρ.

This function constructs the system matrix and computes its factorization for
efficient linear system solution. It handles both quadratic and linear cases
with appropriate matrix construction and factorization strategies.

# Arguments
- `solver::LinearSolver`: The linear solver instance
- `rho::Float64`: Current penalty parameter value

# Effects
- Updates `solver.currentRho` with new penalty parameter
- Constructs system matrix: ρA^T A + Q (if Q exists) or A^T A
- Computes matrix factorization using Cholesky or LDLT
- Sets `solver.isPositiveDefinite` flag based on factorization success

# System Matrix Construction
- **With Q**: System matrix = ρA^T A + Q (emphasizing quadratic structure)
- **Without Q**: System matrix = A^T A (pure constraint structure)

# Factorization Strategy
1. **Cholesky Attempt**: Try Cholesky factorization for positive definite case
2. **LDLT Fallback**: Use LDLT factorization for positive semidefinite case
3. **Error Handling**: Provide informative warnings for semidefinite matrices

# Performance Notes
- Factorization is the most expensive operation: O(n³) complexity
- Reuses factorization when Q exists and ρ is unchanged
- Optimized for sparse matrix structure using CHOLMOD
- Handles both definite and semidefinite cases robustly
"""
function prepareLinearSolverData!(solver::LinearSolver, rho::Float64)
    solver.currentRho = rho
    
    # Only update if Q exists or factorization doesn't exist yet
    if isnothing(solver.factorization) == false && isnothing(solver.Q)
        return
    end

    # Construct system matrix if Q exists
    if isnothing(solver.Q) == false
        copyto!(solver.systemMatrix, solver.AAdjointSelf)
        rmul!(solver.systemMatrix, rho)
        axpy!(2.0, solver.Q, solver.systemMatrix) # solver.systemMatrix .+= 2 * solver.Q
    end

    # Factorize system matrix
    try
        solver.factorization = cholesky(isnothing(solver.Q) ? solver.AAdjointSelf : solver.systemMatrix)
        solver.isPositiveDefinite = true
    catch e
        if isa(e, PosDefException)
            solver.factorization = ldlt(isnothing(solver.Q) ? solver.AAdjointSelf : solver.systemMatrix)
            solver.isPositiveDefinite = false
            @PDMOWarn solver.logLevel "LinearSolver: System matrix is not positive definite (likely positive semidefinite), using LDLT factorization"
        else
            rethrow(e)
        end
    end
end

"""
    LinearSolver(nodeID::String, admmGraph::ADMMBipartiteGraph, edgeData::Dict{String, EdgeData}, initialRho::Float64)

Construct and initialize a linear solver for the specified ADMM node.

This constructor creates a LinearSolver instance and immediately prepares it
for solving by computing the initial factorization. It combines the construction
and initialization steps for convenience.

# Arguments
- `nodeID::String`: Identifier of the ADMM node
- `admmGraph::ADMMBipartiteGraph`: Graph containing node and edge information
- `edgeData::Dict{String, EdgeData}`: Precomputed edge information
- `initialRho::Float64`: Initial penalty parameter

# Effects
- Creates LinearSolver instance with problem structure
- Prepares factorization for initial penalty parameter
- Logs initialization success message
- Returns ready-to-use solver instance

# Performance Notes
- Combines construction and initialization for efficiency
- Avoids duplicate factorization computations
- Provides immediate solver readiness
"""
function LinearSolver(nodeID::String,
    admmGraph::ADMMBipartiteGraph, 
    edgeData::Dict{String, EdgeData}, 
    initialRho::Float64, 
    logLevel::Int64)

    solver = LinearSolver(nodeID, edgeData, admmGraph, logLevel)
    prepareLinearSolverData!(solver, initialRho)
    @PDMODebug logLevel "OriginalADMMSubproblemSolve: ADMM node $nodeID initialized with LinearSolver."
    return solver
end

"""
    solve!(solver::LinearSolver, nodeID::String, admmGraph::ADMMBipartiteGraph, info::ADMMIterationInfo,
           edgeData::Dict{String, EdgeData}, augmentedLagrangianLinearCoefficientsBuffer::Dict{String, NumericVariable}, enableParallel::Bool=false)

Solve the ADMM subproblem using linear system solution.

This function constructs the right-hand side vector and solves the linear system
to find the exact optimal solution. It handles both quadratic and linear cases
with appropriate scaling and system construction.

# Arguments
- `solver::LinearSolver`: The linear solver instance
- `nodeID::String`: Node identifier to solve for
- `admmGraph::ADMMBipartiteGraph`: The bipartite graph structure
- `info::ADMMIterationInfo`: Current iteration information
- `edgeData::Dict{String, EdgeData}`: Precomputed edge data
- `augmentedLagrangianLinearCoefficientsBuffer::Dict{String, NumericVariable}`: Linear coefficients
- `enableParallel::Bool`: Parallel execution flag (unused)

# Effects
- Updates `info.primalSol[nodeID]` with optimal solution
- Stores previous solution in `info.primalSolPrev[nodeID]`
- Uses `solver.rhsBuffer` for right-hand side construction

# Algorithm Steps
1. **Right-Hand Side Construction**: Build RHS vector from augmented Lagrangian
   - Add linear coefficients: A^T(λ + ρ(By^k - b))
   - Add objective linear terms: q (if present)
   - Apply appropriate scaling for system type
2. **System Solution**: Solve linear system using precomputed factorization
   - For Q present: (ρA^T A + Q)x = -RHS
   - For Q absent: A^T A x = -RHS/ρ
3. **Solution Update**: Store result in ADMM iteration information

# Mathematical Details
The linear systems solved are:
- **Quadratic Case**: (ρA^T A + Q)x = -A^T(λ + ρ(By^k - b)) - q
- **Linear Case**: A^T A x = -A^T(λ + ρ(By^k - b))/ρ - q/ρ

The augmented Lagrangian linear coefficients contain A^T(λ + ρ(By^k - b)),
which are precomputed for efficiency.

# Performance Notes
- O(n²) complexity for system solution using precomputed factorization
- Efficient sparse matrix operations using CHOLMOD
- Reuses factorization across iterations for constant ρ
- In-place operations minimize memory allocation
"""
function solve!(solver::LinearSolver,
    nodeID::String, 
    admmGraph::ADMMBipartiteGraph, 
    info::ADMMIterationInfo,
    edgeData::Dict{String, EdgeData}, 
    augmentedLagrangianLinearCoefficientsBuffer::Dict{String, NumericVariable}, 
    enableParallel::Bool = false)

    rho = solver.currentRho
    node = admmGraph.nodes[nodeID]
    
    copyto!(solver.rhsBuffer, augmentedLagrangianLinearCoefficientsBuffer[nodeID])
    if isnothing(solver.q) == false
        axpy!(1.0, solver.q, solver.rhsBuffer)
    end

    if isnothing(solver.Q)
        # Case: A'A x = rhsBuffer = -(A'lmd/rho + A'By^k - A'b + q/rho)
        rmul!(solver.rhsBuffer, -1.0/rho)
    else
        # Case: (rho A'A + 2Q)x = rhsBuffer = -(A'lmd + rho A'By^k - rho A'b + q 
        rmul!(solver.rhsBuffer, -1.0)
    end
    
    copyto!(info.primalSolPrev[nodeID], info.primalSol[nodeID])
    info.primalSol[nodeID] = solver.factorization \ solver.rhsBuffer
end

"""
    update!(solver::LinearSolver, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph, rhoUpdated::Bool)

Update solver state when ADMM parameters change.

This function handles solver updates when the penalty parameter ρ is modified
by ADMM adapters. For linear solvers, ρ changes require factorization updates
when quadratic terms are present.

# Arguments
- `solver::LinearSolver`: The linear solver instance
- `info::ADMMIterationInfo`: Current iteration information
- `admmGraph::ADMMBipartiteGraph`: The bipartite graph structure (unused)
- `rhoUpdated::Bool`: Flag indicating if ρ was updated

# Effects
- Calls `prepareLinearSolverData!` with new ρ if `rhoUpdated` is true
- Updates system matrix and factorization when Q is present
- No effects if `rhoUpdated` is false

# Update Strategy
- **With Q**: System matrix = ρA^T A + Q requires refactorization
- **Without Q**: System matrix = A^T A unchanged, no refactorization needed
- **Scaling**: Right-hand side scaling accounts for ρ changes

# Performance Notes
- Refactorization is expensive: O(n³) when Q is present
- Efficient for linear objectives (Q = nothing): no refactorization
- Balances accuracy with computational cost
- Essential for convergence with adaptive penalty parameters
"""
function update!(solver::LinearSolver, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph, rhoUpdated::Bool)
    if rhoUpdated == false
        return
    end

    newRho = info.rhoHistory[end][1]
    prepareLinearSolverData!(solver, newRho)
end
