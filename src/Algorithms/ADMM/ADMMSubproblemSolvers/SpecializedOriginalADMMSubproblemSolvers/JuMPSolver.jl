"""
    JuMPSolver <: SpecializedOriginalADMMSubproblemSolver

JuMP-based solver for exact ADMM subproblems requiring optimization modeling.

This solver handles complex ADMM subproblems that cannot be solved in closed form,
including those with nonlinear objectives, complex constraints, or specialized structure.
It constructs and solves optimization models using the JuMP mathematical programming
interface with the Ipopt nonlinear optimizer.

# Mathematical Formulation
The solver handles subproblems of the form:
    min f(x) + g(x) + ⟨λ, Ax + By^k - b⟩ + (ρ/2)||Ax + By^k - b||²

where:
- f(x): Smooth objective function (linear, quadratic, or nonlinear)
- g(x): Constraint function (handled via JuMP constraints)
- Ax + By^k - b: ADMM coupling constraints
- λ: Dual variables, ρ: Penalty parameter

# Supported Problem Types
- **Linear Objectives**: f(x) = c^T x
- **Quadratic Objectives**: f(x) = (1/2)x^T Q x + c^T x
- **Nonlinear Objectives**: Currently supports ComponentwiseExponentialFunction
- **Constraint Mappings**: Matrix, Identity, and Extraction mappings
- **Domain Constraints**: Handled through JuMP constraint system

# Solver Configuration
- **Primary Optimizer**: Ipopt (Interior Point Optimizer)
- **Linear Solver**: MA27 (if HSL library available) for enhanced performance
- **Tolerance Settings**: High precision for ADMM convergence requirements
- **Output**: Silent mode for integration with ADMM framework

# Fields
- `model::JuMP.AbstractModel`: JuMP optimization model
- `var::Dict{String, Vector{JuMP.VariableRef}}`: JuMP variables by node
- `objExpressions::Vector{Union{JuMP.AffExpr, JuMP.QuadExpr}}`: Linear/quadratic objective parts
- `nonlinearObjExpressions::Vector{Any}`: Nonlinear objective parts
- `currentRho::Float64`: Current penalty parameter value
- `logLevel::Int64`: Logging level
# Constructor Parameters
- `nodeID::String`: Node identifier in the ADMM graph
- `admmGraph::ADMMBipartiteGraph`: The bipartite graph structure
- `edgeData::Dict{String, EdgeData}`: Precomputed edge information
- `rho::Float64`: Initial penalty parameter
- `logLevel::Int64`: Logging level
# Performance Characteristics
- **Computational Complexity**: Depends on problem structure and Ipopt performance
- **Memory Usage**: JuMP model storage plus solver workspace
- **Convergence**: Depends on Ipopt convergence for each subproblem
- **Scalability**: Limited by nonlinear solver performance

# Implementation Notes
- Uses HSL MA27 linear solver when available for better performance
- Handles both linear/quadratic and nonlinear objectives seamlessly
- Integrates with ADMM framework through standardized interface
- Provides automatic model construction from problem structure
"""
mutable struct JuMPSolver <: SpecializedOriginalADMMSubproblemSolver 
    model::JuMP.AbstractModel
    var::Dict{String, Vector{JuMP.VariableRef}}
    objExpressions::Vector{Union{JuMP.AffExpr, JuMP.QuadExpr}} # Store linear/quadratic parts
    nonlinearObjExpressions::Vector{Any}                       # Store nonlinear parts
    currentRho::Float64 
    logLevel::Int64
    """
        JuMPSolver(nodeID::String, admmGraph::ADMMBipartiteGraph, edgeData::Dict{String, EdgeData}, rho::Float64, logLevel::Int64)

    Construct a JuMP-based solver for the specified ADMM node.

    This constructor analyzes the node's objective and constraint structure to build
    an appropriate JuMP model. It handles function decomposition, variable creation,
    and initial model setup with proper solver configuration.

    # Arguments
    - `nodeID::String`: Identifier of the ADMM node
    - `admmGraph::ADMMBipartiteGraph`: Graph containing node and edge information
    - `edgeData::Dict{String, EdgeData}`: Precomputed adjoint mappings
    - `rho::Float64`: Initial penalty parameter
    - `logLevel::Int64`: Logging level
    # Model Construction Process
    1. **Solver Configuration**: Initialize Ipopt with appropriate settings
    2. **Function Analysis**: Decompose objective into linear/quadratic and nonlinear parts
    3. **Variable Creation**: Add JuMP variables corresponding to node variables
    4. **Constraint Setup**: Handle domain constraints through JuMP constraint system
    5. **Objective Preparation**: Store linear/quadratic parts, flag nonlinear components

    # Effects
    - Creates JuMP model with Ipopt optimizer
    - Configures solver with HSL MA27 if available
    - Initializes variable and auxiliary variable dictionaries
    - Stores objective expressions for efficient updates
    - Sets up nonlinear objective flag for solving
    """
    function JuMPSolver(nodeID::String, admmGraph::ADMMBipartiteGraph, edgeData::Dict{String, EdgeData}, rho::Float64, logLevel::Int64) 
    
        node = admmGraph.nodes[nodeID]

        model = JuMP.Model(Ipopt.Optimizer)
        # JuMP.set_attribute(model, "tol", 1e-12)
        # JuMP.set_attribute(model, "constr_viol_tol", 1e-12)
        JuMP.set_silent(model)
        if HSL_FOUND 
            JuMP.set_attribute(model, "hsllib", HSL_jll.libhsl_path)
            JuMP.set_attribute(model, "linear_solver", "ma27")
        end 

        var = Dict{String, Vector{JuMP.VariableRef}}()
        objExpressions = Vector{Union{JuMP.AffExpr, JuMP.QuadExpr}}()
        nonlinearObjExpressions = Vector{Any}()
        
        # Use the same approach as in MultiblockProblem.jl
        addBlockVariableToJuMPModel!(model, 
            node.f, 
            node.g,
            node.val, 
            nodeID, 
            var, 
            objExpressions,
            nonlinearObjExpressions)

        @PDMODebug logLevel "OriginalADMMSubproblemSolve: ADMM node $nodeID initialized with JuMPSolver."
        return new(model, var, objExpressions, nonlinearObjExpressions, rho, logLevel)
    end 
end

"""
    addQuadraticTermToJuMPObjective!(ALQuadTerms::JuMP.QuadExpr, solver::JuMPSolver, nodeID::String, 
                                   edge::ADMMEdge, edgeData::EdgeData, augmentedLagrangianLinearCoefficientsBuffer::Dict{String, NumericVariable}, rho::Float64)

Add augmented Lagrangian quadratic terms to the JuMP objective function.

This function constructs the quadratic penalty terms (ρ/2)||Ax + By^k - b||² for the
augmented Lagrangian. It handles different mapping types (Matrix, Identity, Extraction)
and uses precomputed adjoint mappings for efficiency.

# Arguments
- `ALQuadTerms::JuMP.QuadExpr`: JuMP quadratic expression to modify
- `solver::JuMPSolver`: The JuMP solver instance
- `nodeID::String`: Current node identifier
- `edge::ADMMEdge`: Edge information containing mappings
- `edgeData::EdgeData`: Precomputed adjoint mappings
- `augmentedLagrangianLinearCoefficientsBuffer::Dict{String, NumericVariable}`: Linear coefficients
- `rho::Float64`: Current penalty parameter

# Effects
- Adds quadratic term (ρ/2)x^T A^T A x to the objective expression
- Handles different mapping types with appropriate formulations
- Uses precomputed A^T A matrices for efficiency

# Mathematical Details
For each edge mapping A, adds:
- **Matrix Mapping**: (ρ/2)x^T A^T A x using precomputed A^T A
- **Identity Mapping**: (ρ/2)c² x^T x where c is the scaling coefficient
- **Extraction Mapping**: (ρ/2)c² Σᵢ xᵢ² for extracted components

# Error Handling
- Validates mapping types and throws descriptive errors for unsupported types
- Ensures consistency between edge structure and precomputed data
"""
function addQuadraticTermToJuMPObjective!(ALQuadTerms::JuMP.QuadExpr, 
    solver::JuMPSolver, 
    nodeID::String, 
    edge::ADMMEdge, 
    edgeData::EdgeData, 
    augmentedLagrangianLinearCoefficientsBuffer::Dict{String, NumericVariable}, 
    rho::Float64)
    
    # quadratic term = 0.5 * rho * x'(A'A)x
    mapping = edge.mappings[nodeID]
    if isa(mapping, LinearMappingMatrix)
        if nodeID == edge.nodeID1
            @assert(isa(edgeData.mapping1AdjSelf, LinearMappingMatrix), "OriginalADMMSubproblemSolver: mapping1AdjSelf is not a LinearMappingMatrix")
            JuMP.add_to_expression!(ALQuadTerms, 0.5 * rho * solver.var[nodeID]' * edgeData.mapping1AdjSelf.A * solver.var[nodeID])
        else 
            @assert(isa(edgeData.mapping2AdjSelf, LinearMappingMatrix), "OriginalADMMSubproblemSolver: mapping2AdjSelf is not a LinearMappingMatrix")
            JuMP.add_to_expression!(ALQuadTerms, 0.5 * rho * solver.var[nodeID]' * edgeData.mapping2AdjSelf.A * solver.var[nodeID])
        end 
    elseif isa(mapping, LinearMappingIdentity)
        JuMP.add_to_expression!(ALQuadTerms, 0.5 * rho * mapping.coe^2 * solver.var[nodeID]' * solver.var[nodeID])
    elseif isa(mapping, LinearMappingExtraction)
        JuMP.add_to_expression!(ALQuadTerms, 0.5 * rho * mapping.coe^2 * 
            sum(solver.var[nodeID][i]^2 for i in mapping.indexStart:mapping.indexEnd))
    else 
        error("JuMPSolver: Cannot add quadratic term to objective. Unknown mapping type = $(typeof(mapping))")
    end
end 

"""
    solve!(solver::JuMPSolver, nodeID::String, admmGraph::ADMMBipartiteGraph, info::ADMMIterationInfo,
           edgeData::Dict{String, EdgeData}, augmentedLagrangianLinearCoefficientsBuffer::Dict{String, NumericVariable}, enableParallel::Bool=false)

Solve the ADMM subproblem using JuMP optimization.

This function constructs and solves the complete augmented Lagrangian subproblem for
the specified node. It builds the objective function from stored expressions and
current ADMM state, then optimizes using the configured solver.

# Arguments
- `solver::JuMPSolver`: The JuMP solver instance
- `nodeID::String`: Node identifier to solve for
- `admmGraph::ADMMBipartiteGraph`: The bipartite graph structure
- `info::ADMMIterationInfo`: Current iteration information
- `edgeData::Dict{String, EdgeData}`: Precomputed edge data
- `augmentedLagrangianLinearCoefficientsBuffer::Dict{String, NumericVariable}`: Linear coefficients
- `enableParallel::Bool`: Parallel execution flag (unused)

# Effects
- Updates `info.primalSol[nodeID]` with optimal solution
- Stores previous solution in `info.primalSolPrev[nodeID]`
- Modifies JuMP model objective for current iteration

# Algorithm Steps
1. **Objective Construction**: Build complete augmented Lagrangian objective
   - Add linear terms: ⟨A^T(λ + ρ(By^k - b)), x⟩
   - Add quadratic terms: (ρ/2)||Ax + By^k - b||²
   - Include original objective: f(x) + g(x)
2. **Objective Setting**: Configure JuMP model with complete objective
   - Use @objective for linear/quadratic objectives
   - Use @NLobjective for nonlinear objectives  
3. **Optimization**: Solve model using Ipopt
4. **Solution Extraction**: Extract optimal variables and update ADMM state

# Mathematical Details
The complete subproblem solved is:
    min f(x) + g(x) + ⟨λ, Ax + By^k - b⟩ + (ρ/2)||Ax + By^k - b||²

where the augmented Lagrangian terms are efficiently constructed using:
- Precomputed linear coefficients A^T(λ + ρ(By^k - b))
- Precomputed quadratic forms A^T A for penalty terms
- Stored objective expressions for f(x) components

# Performance Notes
- Reuses JuMP model structure across iterations
- Efficient objective updates without model reconstruction
- Leverages high-performance linear algebra in penalty term construction
"""
function solve!(solver::JuMPSolver, 
    nodeID::String, 
    admmGraph::ADMMBipartiteGraph, 
    info::ADMMIterationInfo,
    edgeData::Dict{String, EdgeData}, 
    augmentedLagrangianLinearCoefficientsBuffer::Dict{String, NumericVariable},
    enableParallel::Bool = false)

    rho = info.rhoHistory[end][1]
    node = admmGraph.nodes[nodeID]
    
    # Collect all relaxation and penalty terms in the AL objective
    ALQuadTerms = JuMP.QuadExpr()

    # add linear term = <A'(lmd + rho * (By-b)), x>
    JuMP.add_to_expression!(ALQuadTerms, dot(augmentedLagrangianLinearCoefficientsBuffer[nodeID], solver.var[nodeID]))

    for edgeID in node.neighbors
       addQuadraticTermToJuMPObjective!(ALQuadTerms, 
            solver, 
            nodeID, 
            admmGraph.edges[edgeID], 
            edgeData[edgeID], 
            augmentedLagrangianLinearCoefficientsBuffer, 
            rho)
    end
    
    # Add the linear/quadratic parts from objExpressions
    for expr in solver.objExpressions
        JuMP.add_to_expression!(ALQuadTerms, expr)
    end

    # Set the objective based on whether we have nonlinear terms
    if !isempty(solver.nonlinearObjExpressions)
        # For nonlinear objectives, combine all expressions
        nonlinearObj = ALQuadTerms  # Start with linear/quadratic terms
        for expr in solver.nonlinearObjExpressions
            nonlinearObj += expr
        end

        # Set nonlinear objective
        JuMP.@NLobjective(solver.model, Min, nonlinearObj)
    else
        # Set quadratic objective
        JuMP.@objective(solver.model, Min, ALQuadTerms)
    end

    # Solve the model
    JuMP.optimize!(solver.model)

    # Update solution
    copyto!(info.primalSolPrev[nodeID], info.primalSol[nodeID])
    info.primalSol[nodeID] .= JuMP.value.(solver.var[nodeID])
end 

"""
    update!(solver::JuMPSolver, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph, rhoUpdated::Bool)

Update solver state when ADMM parameters change.

This function handles solver updates when the penalty parameter ρ is modified by
ADMM adapters. For JuMP solvers, the main update is tracking the current ρ value
for use in objective construction.

# Arguments
- `solver::JuMPSolver`: The JuMP solver instance
- `info::ADMMIterationInfo`: Current iteration information
- `admmGraph::ADMMBipartiteGraph`: The bipartite graph structure (unused)
- `rhoUpdated::Bool`: Flag indicating if ρ was updated

# Effects
- Updates `solver.currentRho` with new penalty parameter if `rhoUpdated` is true
- No effects if `rhoUpdated` is false

# Notes
- JuMP models are reconstructed each iteration, so no model updates needed
- The stored ρ value is used for objective scaling in subsequent solve calls
- This is a lightweight operation compared to other solver update procedures
"""
function update!(solver::JuMPSolver, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph, rhoUpdated::Bool)
    if rhoUpdated == false 
        return 
    end 
    solver.currentRho = info.rhoHistory[end][1]
end 