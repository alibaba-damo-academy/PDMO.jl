                                                                                                           """
    ProximalMappingSolver <: SpecializedOriginalADMMSubproblemSolver

Specialized solver for ADMM subproblems that reduce to proximal mappings.

This solver handles a specific class of ADMM subproblems where the objective function
is purely regularization (f(x) = 0) and the constraint mapping is a scaled identity.
Such subproblems can be efficiently solved using proximal operators rather than
general optimization methods.

# Mathematical Formulation
The solver handles subproblems of the form:
    min g(y) + ⟨λ, Ax^k + By - b⟩ + (ρ/2)||Ax^k + By - b||²

where:
- g(y): Proximal-friendly regularization function
- B = mI: Scaled identity mapping (m ≠ 0)
- f(y) = 0: Zero smooth objective function

# Transformation to Proximal Mapping
By substituting B = mI, the subproblem becomes:
    min g(y) + ⟨m·λ, y⟩ + (ρm²/2)||y - (b - Ax^k)/m||²

This is equivalent to the proximal mapping:
    min g(y) + (1/2γ)||y - z||²

where:
- z = (b - Ax^k - λ/ρ)/m: Proximal center point
- γ = 1/(ρm²): Proximal parameter

# Supported Mapping Types
- **LinearMappingIdentity**: B = cI with scaling coefficient c
- **LinearMappingExtraction**: B = cE with extraction operator E
- **LinearMappingMatrix**: B = cI where matrix A is c·I (validated)

# Consistency Validation
The constructor verifies that all constraint mappings for the node are consistent
scaled identity operations:
- All mappings must scale the same vector by the same coefficient
- The scaling coefficient must be non-zero
- Inconsistent scaling throws an error

# Fields
- `scalingCoefficient::Float64`: The scaling coefficient m in B = mI
- `proximalPoint::NumericVariable`: Working space for proximal center z
- `currentRho::Float64`: Current penalty parameter ρ
- `gamma::Float64`: Proximal parameter γ = 1/(ρm²)
- `logLevel::Int64`: Logging level
# Constructor Parameters
- `nodeID::String`: Node identifier in the ADMM graph
- `admmGraph::ADMMBipartiteGraph`: The bipartite graph structure
- `edgeData::Dict{String, EdgeData}`: Precomputed edge information
- `rho::Float64`: Initial penalty parameter

# Performance Characteristics
- **Computational Complexity**: O(n) where n is the variable dimension
- **Memory Usage**: O(n) for proximal point storage
- **Convergence**: Depends on proximal operator efficiency
- **Scalability**: Excellent for large-scale problems with appropriate structure

# Implementation Notes
- Requires f(x) = 0 (Zero objective function)
- Validates mapping consistency during construction
- Uses efficient proximal operator implementations
- Automatic scaling coefficient detection and validation
"""
mutable struct ProximalMappingSolver <: SpecializedOriginalADMMSubproblemSolver

    scalingCoefficient::Float64 # B = scalingCoefficient * I
    proximalPoint::NumericVariable 
    currentRho::Float64 
    gamma::Float64
    logLevel::Int64
    """
        ProximalMappingSolver(nodeID::String, admmGraph::ADMMBipartiteGraph, edgeData::Dict{String, EdgeData}, rho::Float64, logLevel::Int64)

    Construct a proximal mapping solver for the specified ADMM node.

    This constructor validates that the node structure is compatible with proximal mapping
    solution, determines the scaling coefficient, and initializes the solver state.

    # Arguments
    - `nodeID::String`: Identifier of the ADMM node
    - `admmGraph::ADMMBipartiteGraph`: Graph containing node and edge information
    - `edgeData::Dict{String, EdgeData}`: Precomputed edge data
    - `rho::Float64`: Initial penalty parameter

    # Validation Process
    1. **Objective Check**: Verify f(x) = 0 (Zero function)
    2. **Mapping Analysis**: Examine all constraint mappings for identity structure
    3. **Scaling Detection**: Determine common scaling coefficient across mappings
    4. **Consistency Validation**: Ensure all mappings use same scaling coefficient

    # Effects
    - Creates solver with validated scaling coefficient
    - Initializes proximal point workspace
    - Computes initial proximal parameter γ
    - Sets up solver for efficient proximal mapping computation

    # Error Conditions
    - Throws error if f(x) ≠ 0
    - Throws error if mappings are not identity-type
    - Throws error if scaling coefficients are inconsistent
    - Throws error if scaling coefficient is zero
    """
    function ProximalMappingSolver(nodeID::String, 
        admmGraph::ADMMBipartiteGraph, 
        edgeData::Dict{String, EdgeData}, 
        rho::Float64, 
        logLevel::Int64)

        node = admmGraph.nodes[nodeID]
        @assert(isa(node.f, Zero), "ProximalMappingSolver only supports Zero as f.")
    
        """
            isIdentityTypeMapping(A::SparseMatrixCSC{Float64, Int64}) -> Bool

        Check if a sparse matrix represents a scaled identity mapping.

        This function validates that a matrix has the structure A = cI for some
        scalar c, which is required for proximal mapping reduction.

        # Arguments
        - `A::SparseMatrixCSC{Float64, Int64}`: Sparse matrix to check

        # Returns
        - `Bool`: True if matrix is scaled identity, false otherwise

        # Algorithm
        1. **Dimension Check**: Verify matrix is square
        2. **Sparsity Check**: Ensure number of nonzeros equals dimension
        3. **Diagonal Check**: Verify all diagonal elements are equal and non-zero
        4. **Structure Validation**: Confirm no off-diagonal elements exist
        """
        function isIdentityTypeMapping(A::SparseMatrixCSC{Float64, Int64})
            m, n = size(A)
            # Check matrix is square
            if m != n || n != nnz(A)
                return false 
            end 

            # Get first diagonal element
            ele = A[1,1]
            if abs(ele) < ZeroTolerance
                return false 
            end

            # Check if all diagonal elements are equal
            for i in 1:n
                if abs(A[i,i] - ele) > ZeroTolerance
                    return false
                end
            end

            return true
        end
        
        # IMPORTANT:
        # Use dense buffers here regardless of `node.val` storage (which can be sparse),
        # because we only estimate/validate the common scaling coefficient of identity-type mappings.
        # Using sparse vectors here can keep a fixed sparsity pattern and break slice writes.
        # initialValue = fill(1.0, size(node.val))
        #s caledValue = zeros(Float64, size(node.val))

        # seems OK here with sparse vectors
        initialValue = similar(node.val)
        initialValue .= 1.0
        scaledValue = zero(node.val)

        for edgeID in node.neighbors
            mapping = admmGraph.edges[edgeID].mappings[nodeID]
            @assert(isa(mapping, LinearMappingIdentity) || 
                    isa(mapping, LinearMappingExtraction) || 
                    isa(mapping, LinearMappingMatrix) && isIdentityTypeMapping(mapping.A), 
                    "ProximalMappingSolver: only supports identity-type mappings.")
            if isa(mapping, LinearMappingExtraction)
                # Extraction reduces the first dimension. Compute into a correctly-sized buffer
                # and then add it back into the corresponding slice of `scaledValue`.
                expectedShape = (mapping.indexEnd - mapping.indexStart + 1, Base.tail(size(initialValue))...)
                ret = zeros(Float64, expectedShape)
                mapping(initialValue, ret, false)
                slice = (mapping.indexStart:mapping.indexEnd, ntuple(_ -> Colon(), ndims(initialValue) - 1)...)
                scaledValue[slice...] .+= ret
            elseif isa(mapping, LinearMappingMatrix)
                mapping(initialValue, scaledValue, true)
            else
                # LinearMappingIdentity: output has the same shape as input.
                mapping(initialValue, scaledValue, true)
            end
        end 

        # after the above loop, scaledValue contains the scaling coefficients of the identity-type mappings
        scalingCoefficient = sum(scaledValue)/sum(initialValue)
        @assert(scalingCoefficient != 0.0, "ProximalMappingSolver: scaling coefficient must be non-zero.")
        for i in eachindex(scaledValue)
            diff = scaledValue[i] - scalingCoefficient * initialValue[i]
            @assert(abs(diff) < FeasTolerance, "ProximalMappingSolver: inconsistent scaling coefficients.")
        end 
        
        initialValue .= 0.0
        @PDMOWarn logLevel "OriginalADMMSubproblemSolve: ADMM node $nodeID initialized with ProximalMappingSolver."
        return new(scalingCoefficient, initialValue, rho, 1/(rho * scalingCoefficient^2), logLevel)
    end 
end 

"""
    solve!(solver::ProximalMappingSolver, nodeID::String, admmGraph::ADMMBipartiteGraph, info::ADMMIterationInfo,
           edgeData::Dict{String, EdgeData}, augmentedLagrangianLinearCoefficientsBuffer::Dict{String, NumericVariable}, enableParallel::Bool=false)

Solve the ADMM subproblem using proximal mapping computation.

This function efficiently solves the subproblem by reducing it to a proximal mapping
evaluation. It constructs the proximal center point and applies the proximal operator
of the regularization function.

# Arguments
- `solver::ProximalMappingSolver`: The proximal mapping solver instance
- `nodeID::String`: Node identifier to solve for
- `admmGraph::ADMMBipartiteGraph`: The bipartite graph structure
- `info::ADMMIterationInfo`: Current iteration information
- `edgeData::Dict{String, EdgeData}`: Precomputed edge data
- `augmentedLagrangianLinearCoefficientsBuffer::Dict{String, NumericVariable}`: Linear coefficients
- `enableParallel::Bool`: Parallel execution flag (unused)

# Effects
- Updates `info.primalSol[nodeID]` with optimal solution
- Stores previous solution in `info.primalSolPrev[nodeID]`
- Uses `solver.proximalPoint` for intermediate computations

# Algorithm Steps
1. **Proximal Center Construction**: Compute z = (b - Ax^k - λ/ρ)/m
   - Use precomputed linear coefficients from augmented Lagrangian
   - Scale by proximal parameter γ to get proximal center
2. **Proximal Operator Application**: Compute prox_{γg}(z)
   - Apply proximal operator of regularization function g
   - Use optimal proximal parameter γ = 1/(ρm²)
3. **Solution Update**: Store result in ADMM iteration information

# Mathematical Details
The subproblem:
    min g(y) + ⟨m·λ, y⟩ + (ρm²/2)||y - (b - Ax^k)/m||²

is equivalent to:
    prox_{γg}(z) where z = (b - Ax^k - λ/ρ)/m, γ = 1/(ρm²)

The augmented Lagrangian linear coefficients contain A^T(λ + ρ(By^k - b)),
which for identity mappings simplifies to m(λ + ρ(my^k - b)).

# Performance Notes
- O(n) complexity for proximal center computation
- Proximal operator complexity depends on regularization function g
- No iterative optimization required - direct computation
- Efficient memory usage with in-place operations
"""
function solve!(solver::ProximalMappingSolver, 
    nodeID::String, 
    admmGraph::ADMMBipartiteGraph, 
    info::ADMMIterationInfo,
    edgeData::Dict{String, EdgeData}, 
    augmentedLagrangianLinearCoefficientsBuffer::Dict{String, NumericVariable},
    enableParallel::Bool = false)

    fill!(solver.proximalPoint, 0.0)
    copyto!(solver.proximalPoint, augmentedLagrangianLinearCoefficientsBuffer[nodeID])
    rmul!(solver.proximalPoint, -solver.gamma)
    # for edgeID in admmGraph.nodes[nodeID].neighbors 
    #     edge = admmGraph.edges[edgeID]

    #     otherID = nodeID == edge.nodeID1 ? edge.nodeID2 : edge.nodeID1 

    #     copy!(info.dualBuffer[edgeID], info.dualSol[edgeID])
    #     rmul!(info.dualBuffer[edgeID], 1/solver.currentRho)
    #     edge.mappings[otherID](info.primalSol[otherID], info.dualBuffer[edgeID], true)
    #     axpy!(-1.0, edge.rhs, info.dualBuffer[edgeID])
    #     axpy!(1.0, info.dualBuffer[edgeID], solver.proximalPoint)
    # end 

    # rmul!(solver.proximalPoint, -1.0/solver.scalingCoefficient)

    copyto!(info.primalSolPrev[nodeID], info.primalSol[nodeID])
    proximalOracle!(info.primalSol[nodeID], admmGraph.nodes[nodeID].g, solver.proximalPoint, solver.gamma)
end 

"""
    update!(solver::ProximalMappingSolver, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph, rhoUpdated::Bool)

Update solver parameters when the penalty parameter ρ changes.

This function recalculates the proximal parameter γ when the penalty parameter ρ
is updated by ADMM adapters. The relationship γ = 1/(ρm²) ensures that the
proximal mapping remains equivalent to the original subproblem.

# Arguments
- `solver::ProximalMappingSolver`: The proximal mapping solver instance
- `info::ADMMIterationInfo`: Current iteration information
- `admmGraph::ADMMBipartiteGraph`: The bipartite graph structure (unused)
- `rhoUpdated::Bool`: Flag indicating if ρ was updated

# Effects
- Updates `solver.currentRho` with new penalty parameter if `rhoUpdated` is true
- Recomputes `solver.gamma` using the formula γ = 1/(ρm²)
- No effects if `rhoUpdated` is false

# Mathematical Justification
The proximal parameter γ must satisfy γ = 1/(ρm²) to maintain equivalence:
- Original subproblem has penalty term (ρ/2)||By - (b - Ax^k)||²
- With B = mI, this becomes (ρm²/2)||y - (b - Ax^k)/m||²
- Proximal form requires (1/2γ)||y - z||², so γ = 1/(ρm²)

# Notes
- This is a lightweight operation involving only scalar computations
- Critical for maintaining problem equivalence across ρ updates
- Ensures convergence properties are preserved after adaptation
"""
function update!(solver::ProximalMappingSolver, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph, rhoUpdated::Bool)
    if rhoUpdated == false 
        return 
    end 
    solver.currentRho = info.rhoHistory[end][1]
    solver.gamma = 1/(solver.currentRho * solver.scalingCoefficient^2)
end
