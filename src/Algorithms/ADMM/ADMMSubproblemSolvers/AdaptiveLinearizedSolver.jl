"""
    safe_div(num::Float64, den::Float64) -> Float64

Safe division with special handling for degenerate cases.

Returns 0.0 when both numerator and denominator are zero to avoid undefined behavior
in ratio computations. This is essential for the adaptive algorithm when computing
ratios that may involve zero quantities in degenerate optimization scenarios.

# Arguments
- `num::Float64`: Numerator value
- `den::Float64`: Denominator value

# Returns
- `Float64`: 0.0 if both arguments are zero, otherwise num/den

# Mathematical Details
The function implements the limit convention:
- lim(x→0) f(x)/g(x) = 0 when both f(x) → 0 and g(x) → 0
- This prevents NaN propagation in adaptive step size computations
- Essential for handling edge cases in convergence rate analysis
"""
@inline function safe_div(num::Float64, den::Float64)
    den == 0.0 && return num == 0.0 ? 0.0 : num/den
    return num/den
end

"""
    solve_cubic_real(a::Float64, b::Float64, c::Float64, d::Float64) -> Float64

Solve the cubic equation ax³ + bx² + cx + d = 0 for the smallest positive real root.

This function is central to the adaptive linearized ADMM algorithm, where cubic equations
arise from optimizing the convergence rate with respect to the step size parameter γ.
The algorithm uses discriminant analysis and Cardano's formula to find real roots
analytically, then filters for positive solutions.

# Arguments
- `a::Float64`: Coefficient of x³ term
- `b::Float64`: Coefficient of x² term  
- `c::Float64`: Coefficient of x term
- `d::Float64`: Constant term

# Returns
- `Float64`: Smallest positive real root if exists, otherwise `Inf`

# Algorithm Steps
1. **Degenerate Case Handling**: If a = 0, solve as quadratic/linear equation
2. **Depressed Cubic Transformation**: Substitute y = x + b/(3a) to eliminate x² term
3. **Discriminant Computation**: Δ = (p/3)³ + (q/2)² where p, q are transformed coefficients
4. **Root Classification by Discriminant**:
   - Δ > 0: One real root using Cardano's formula
   - Δ = 0: Multiple real roots (two or three)
   - Δ < 0: Three distinct real roots using trigonometric method
5. **Filtering**: Return minimum positive root among all candidates

# Mathematical Background
The cubic equations in adaptive ADMM arise from:
- Optimizing convergence rate bounds with respect to γ
- Balancing primal and dual convergence rates
- Incorporating problem-specific geometry (Lipschitz constants, operator norms)

The positive root corresponds to the optimal step size that maximizes convergence speed
while maintaining stability constraints.

# Numerical Considerations
- Uses robust cube root function: cbrt(x) = sign(x) * |x|^(1/3)
- Handles numerical precision in discriminant near zero
- Filters roots using strict positivity check for stability
"""
function solve_cubic_real(a::Float64, b::Float64, c::Float64, d::Float64 ; tol = 1e-6)
    # degenerate cases
    if abs(a) < tol
        if abs(b) < tol 
            return abs(c) < tol  ? Inf : ( (x = -d/c) > 0 ? x : Inf )
        end
        # quadratic case
        Δq = c^2 - 4*b*d
        if Δq < 0
            return Inf
        end
        x1 = (-c + sqrt(Δq)) / (2*b)
        x2 = (-c - sqrt(Δq)) / (2*b)
        positive_roots = filter(x -> x > 0, (x1, x2))
        return isempty( positive_roots) ? Inf : minimum(positive_roots)
    end

    offset = b / (3a)
    p = (3a*c - b^2) / (3a^2)
    q = (2b^3 - 9a*b*c + 27a^2*d) / (27a^3)
    Δ = (p/3)^3 + (q/2)^2

    cbrt(x) = sign(x) * abs(x)^(1/3)

    y_roots = Float64[]
    if Δ > 0
        u1 = cbrt(-q/2 + sqrt(Δ))
        u2 = cbrt(-q/2 - sqrt(Δ))
        push!(y_roots, u1 + u2)
    elseif iszero(Δ)
        u = cbrt(-q/2)
        push!(y_roots, 2u)
        push!(y_roots, -u)
    else
        r = 2 * sqrt(-p/3)
        θ = acos( (-q/2) / sqrt(-(p/3)^3) )
        push!(y_roots, r * cos(θ/3))
        push!(y_roots, r * cos((θ + 2*π)/3))
        push!(y_roots, r * cos((θ + 4*π)/3))
    end

    x_roots = [y - offset for y in y_roots]
    positive_roots = filter(x -> x > 0 && isreal(x), x_roots)
    return isempty(positive_roots) ? Inf : minimum(positive_roots)
end

"""
    AdaptiveLinearizedSolver <: AbstractADMMSubproblemSolver

Adaptive Linearized ADMM Solver with dynamic step size optimization.

This solver implements a sophisticated adaptive variant of linearized ADMM where the
proximal step size γ is dynamically optimized based on real-time analysis of the
optimization landscape. The algorithm maintains separate gradient histories for left
and right nodes and uses geometric convergence analysis to balance convergence speed
with numerical stability.

# Mathematical Foundation
The adaptive linearized ADMM solves subproblems of the form:
- Left nodes: x^{k+1} = prox_{γg}(x^k - γ(∇f(x^k) + A^T u^{k+1}))
- Right nodes: y^{k+1} = prox_{γh}(y^k - γ(∇g(y^k) + B^T u^{k+1}))

where γ is adaptively updated by analyzing:
- Local Lipschitz constants L_f, L_g of objective gradients
- Strong convexity parameters ℓ_f, ℓ_g when available
- Operator norms ||A^T A||, ||B^T B|| of constraint mappings
- Convergence rate bounds and stability margins

# Adaptive Strategy
The algorithm optimizes γ by solving constrained optimization problems:
1. **Primal-Dual Balance**: Ensure balanced convergence rates between primal and dual variables
2. **Geometric Adaptation**: Incorporate problem-specific curvature information
3. **Stability Constraints**: Maintain numerical stability through step size bounds
4. **Convergence Acceleration**: Maximize convergence speed subject to stability

# Algorithm Phases
1. **Gradient History Tracking**: Maintain ∇f(x^k), ∇f(x^{k-1}), ∇g(y^k), ∇g(y^{k-1})
2. **Constraint Violation Analysis**: Compute Δu = φ(A(x^k - x^{k-1}) + B(y^k - y^{k-1})) + Ax^k + By^k - c
3. **Geometry Estimation**: Estimate local Lipschitz constants and operator norms
4. **Optimization**: Solve cubic equations for optimal γ candidates
5. **Selection**: Choose minimum valid γ among all candidates

# Fields
- `proximalStepsizeGamma::Float64`: Current adaptive proximal step size γ
- `primalBuffer::Dict{String, NumericVariable}`: Working space for primal computations
- `dualBuffer::Dict{String, NumericVariable}`: Working space for dual computations
- `φ::Float64`: Golden ratio parameter (≈1.618) for constraint violation weighting
- `r::Float64`: Primal-dual ratio parameter controlling balance between updates
- `primalprev::Dict{String, NumericVariable}`: Previous iteration primal variables
- `xgradPrev::Dict{String, NumericVariable}`: Previous gradients for left nodes
- `xgradCur::Dict{String, NumericVariable}`: Current gradients for left nodes
- `ygradPrev::Dict{String, NumericVariable}`: Previous gradients for right nodes
- `ygradCur::Dict{String, NumericVariable}`: Current gradients for right nodes
- `ifSimple::Bool`: Whether to use simplified adaptive scheme

# Constructor Parameters
- `gamma::Float64=1.0`: Initial proximal step size
- `r::Float64=1.0`: Primal-dual balancing parameter
- `ifSimple::Bool=false`: Use simplified adaptive scheme if true

# Performance Characteristics
- **Computational Complexity**: O(n) per iteration plus gradient evaluations
- **Memory Usage**: O(n) for gradient storage per node
- **Convergence Rate**: Adaptive between O(1/k) and linear depending on problem structure
- **Numerical Stability**: Automatic step size bounds prevent divergence

# Implementation Notes
- Uses golden ratio φ = (1 + √5)/2 for optimal constraint violation weighting
- Simplified scheme sets φ = 2 for reduced computational overhead
- Gradient buffering enables efficient multi-step analysis
- Parallel computation across nodes for scalability
"""
mutable struct AdaptiveLinearizedSolver <: AbstractADMMSubproblemSolver
    proximalStepsizeGamma::Float64
    # edgeData::Dict{String, EdgeData}
    primalBuffer::Dict{String, NumericVariable}    # working space for x,y
    dualBuffer::Dict{String, NumericVariable}      # working space for u 

    # === Adaptive scheme ===
    φ::Float64                                  # golden ratio
    r::Float64                                  # Primal-dual ratio
    primalprev::Dict{String, NumericVariable}     # x,y at k-1
    xgradPrev::Dict{String, NumericVariable}      #∇f(x^{k-1})
    xgradCur::Dict{String, NumericVariable}       #∇f(x^{k})
    ygradPrev::Dict{String, NumericVariable}      #∇g(y^{k-1})
    ygradCur::Dict{String, NumericVariable}       #∇g(y^{k})
    ifSimple::Bool                               
    
    logLevel::Int64
    function AdaptiveLinearizedSolver(;gamma::Float64=1.0, r::Float64=1.0, ifSimple::Bool=false)
        solver = new(gamma,
            # Dict{String, EdgeData}(),
            Dict{String, NumericVariable}(),
            Dict{String, NumericVariable}(),
            (1 + sqrt(5))/2,
            r,
            Dict{String, NumericVariable}(),
            Dict{String, NumericVariable}(),
            Dict{String, NumericVariable}(),
            Dict{String, NumericVariable}(),
            Dict{String, NumericVariable}(),
            ifSimple, 
            1
        )
        return solver
    end
end

"""
    initialize!(solver::AdaptiveLinearizedSolver, admmGraph::ADMMBipartiteGraph, info::ADMMIterationInfo) -> Bool

Initialize the adaptive linearized solver with problem-specific buffers and gradient computations.

This function sets up all necessary data structures for the adaptive algorithm, including
buffer allocation, gradient history initialization, and initial adaptive parameter setup.
It prepares the solver for sophisticated adaptive step size selection based on problem
geometry and convergence analysis.

# Arguments
- `solver::AdaptiveLinearizedSolver`: The solver instance to initialize
- `admmGraph::ADMMBipartiteGraph`: The bipartite graph structure of the problem
- `info::ADMMIterationInfo`: Current iteration information with initial solutions

# Returns
- `Bool`: Always returns `true` upon successful initialization

# Effects
- Allocates `primalBuffer` and `primalprev` for all nodes
- Allocates `dualBuffer` for all edges
- Initializes gradient buffers `xgradPrev`, `xgradCur`, `ygradPrev`, `ygradCur`
- Copies initial solutions to working buffers
- Computes initial gradients for adaptive algorithm startup
- Sets φ = 2 if simplified adaptive scheme is selected

# Initialization Process
1. **Buffer Allocation**: Create working spaces for primal and dual variables
2. **Gradient Buffer Setup**: Allocate separate gradient storage for left and right nodes
3. **Initial Solution Loading**: Copy current solutions to working buffers
4. **Gradient Computation**: Evaluate initial gradients for adaptive startup
5. **Scheme Configuration**: Set parameters for simple vs. full adaptive scheme

# Mathematical Preparation
The initialization computes:
- ∇f(x^0) for all left nodes: Initial gradient evaluation
- ∇g(y^0) for all right nodes: Initial gradient evaluation
- Previous and current gradient buffers for multi-step analysis
- Golden ratio parameter φ or simplified φ = 2

# Performance Notes
- Gradient evaluations are the most expensive initialization step
- Buffer allocation is O(n) for each node
- Parallel gradient computation across nodes when possible
- Efficient memory layout for cache-friendly access patterns
"""
function initialize!(solver::AdaptiveLinearizedSolver, admmGraph::ADMMBipartiteGraph, info::ADMMIterationInfo, logLevel::Int64)
    # Allocate primal and dual buffers
    for (nodeID,node) in admmGraph.nodes
        solver.primalBuffer[nodeID] = similar(node.val)
        solver.primalprev[nodeID] = similar(node.val)
    end

    for (nodeID,_) in admmGraph.nodes
        copyto!(solver.primalprev[nodeID], info.primalSol[nodeID])
    end

    for (edgeID,edge) in admmGraph.edges
        # solver.edgeData[edgeID] = EdgeData(edge)
        solver.dualBuffer[edgeID] = similar(edge.rhs)
    end

    # Buffers for adaptivity
    for nodeID in admmGraph.left
        solver.xgradPrev[nodeID]   = similar(solver.primalBuffer[nodeID])
        solver.xgradCur[nodeID]    = similar(solver.primalBuffer[nodeID])
    end
    for nodeID in admmGraph.right
        solver.ygradPrev[nodeID]   = similar(solver.primalBuffer[nodeID])
        solver.ygradCur[nodeID]    = similar(solver.primalBuffer[nodeID])
    end


    for (nodeID,_) in admmGraph.nodes
        copyto!(solver.primalBuffer[nodeID], info.primalSol[nodeID])
    end

   for nodeID in admmGraph.left
        gradientOracle!(solver.xgradPrev[nodeID],
                        admmGraph.nodes[nodeID].f,
                        solver.primalprev[nodeID])
        gradientOracle!(solver.xgradCur[nodeID],
                        admmGraph.nodes[nodeID].f,
                        solver.primalBuffer[nodeID])
    end
    for nodeID in admmGraph.right
        gradientOracle!(solver.ygradPrev[nodeID],
                        admmGraph.nodes[nodeID].f,
                        solver.primalprev[nodeID])
        gradientOracle!(solver.ygradCur[nodeID],
                        admmGraph.nodes[nodeID].f,
                        solver.primalBuffer[nodeID])
    end

    solver.logLevel = logLevel

    if solver.ifSimple
        solver.φ = 2
        @PDMOInfo solver.logLevel "AdaptiveLinearizedSolver: Implementing simple adaptive ADMM"
    end

    return true
end

"""
    updateGammaAndDual!(solver::AdaptiveLinearizedSolver, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph)

Update the adaptive step size γ and dual variables using geometric convergence analysis.

This is the core adaptive algorithm that analyzes the geometry of the optimization landscape
to determine an optimal step size. It computes various geometric quantities (Lipschitz constants,
operator norms, optimality gaps) and solves optimization subproblems to find the best γ.

# Arguments
- `solver::AdaptiveLinearizedSolver`: The solver instance
- `info::ADMMIterationInfo`: Current iteration information
- `admmGraph::ADMMBipartiteGraph`: The bipartite graph structure

# Effects
- Updates `solver.proximalStepsizeGamma` with new adaptive step size
- Updates dual variables in `info.dualSol`
- Copies current solutions to "previous" buffers for next iteration
- Updates gradient histories for next adaptive computation

# Algorithm Steps
1. **Constraint Violation Computation**: Δu = φ(A(x^k - x^{k-1}) + B(y^k - y^{k-1})) + Ax^k + By^k - c
2. **Operator Norm Estimation**: Compute a = ||A^T Δu||/||Δu||, b = ||B^T Δu||/||Δu||
3. **Geometry Analysis**: Estimate Lipschitz constants L, strong convexity ℓ, and optimality gaps δ
4. **Optimization Parameter Computation**: Solve for λ, μ parameters from convergence conditions
5. **Step Size Candidates**: Generate multiple γ candidates from different optimization criteria
6. **Candidate Selection**: Choose minimum valid γ among all candidates
7. **Dual Update**: Update dual variables using u^{k+1} = u^k + rγ_{k+1}Δu

# Mathematical Background
The adaptive rule optimizes the convergence rate by balancing:
- **Primal Convergence**: Controlled by Lipschitz constants and step size
- **Dual Convergence**: Controlled by operator norms and constraint violations
- **Stability**: Ensured by bounds on step size candidates

# Geometric Quantities
- **Lipschitz Constants**: L_x = ||∇f(x^k) - ∇f(x^{k-1})||/||x^k - x^{k-1}||
- **Strong Convexity**: ℓ_x = ⟨∇f(x^k) - ∇f(x^{k-1}), x^k - x^{k-1}⟩/||x^k - x^{k-1}||²
- **Optimality Gaps**: δ_x = γ²L_x² - 2γℓ_x
- **Operator Norms**: Estimated from constraint violation projections

# Step Size Candidates
1. **Expansion Candidate**: φγ_k (full scheme) or 1.5γ_k (simple scheme)
2. **Stability Candidate**: Based on convergence rate bounds
3. **Block Candidates**: Γ_x, Γ_y from cubic equation solutions

# Performance Notes
- Dominant cost is gradient difference computations
- Cubic equation solving is numerically stable
- Parallel computation across constraint edges
- Efficient in-place operations for memory conservation
"""
function updateGammaAndDual!(solver::AdaptiveLinearizedSolver, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph)
    γk = solver.proximalStepsizeGamma

    for (nodeID,_) in admmGraph.nodes
        copyto!(solver.primalBuffer[nodeID], info.primalSol[nodeID])  # double-check if the workspace is loaded with x^k, y^k
    end

    #  Step1) Get Δu
    norm2_du=0.0
    for (edgeID,edge) in admmGraph.edges
        # += ϕ ( A(x^k - x^{k-1}) + B(y^k - y^{k-1}) )
        edge.mappings[edge.nodeID1](solver.primalBuffer[edge.nodeID1] .- solver.primalprev[edge.nodeID1], solver.dualBuffer[edgeID], false)
        edge.mappings[edge.nodeID2](solver.primalBuffer[edge.nodeID2] .- solver.primalprev[edge.nodeID2], solver.dualBuffer[edgeID], true)
        rmul!(solver.dualBuffer[edgeID], solver.φ)
        # += Ax^k + By^k - c
        edge.mappings[edge.nodeID1](solver.primalBuffer[edge.nodeID1], solver.dualBuffer[edgeID], true)
        edge.mappings[edge.nodeID2](solver.primalBuffer[edge.nodeID2], solver.dualBuffer[edgeID], true)                 
        axpy!(-1.0, edge.rhs, solver.dualBuffer[edgeID])
        norm2_du += dot(solver.dualBuffer[edgeID],solver.dualBuffer[edgeID])
    end
    norm_du = sqrt(norm2_du)

    # Step2) Get a, b
    At_du = Dict{String,NumericVariable}()
    Bt_du = Dict{String,NumericVariable}()
    for nodeID in admmGraph.left
        At_du[nodeID] = fill!(similar(solver.primalBuffer[nodeID]), 0.0)
        for edgeID in admmGraph.nodes[nodeID].neighbors
            adjoint!(admmGraph.edges[edgeID].mappings[nodeID], solver.dualBuffer[edgeID], At_du[nodeID], true)
        end
    end
    for nodeID in admmGraph.right
        Bt_du[nodeID] = fill!(similar(solver.primalBuffer[nodeID]), 0.0)
        for edgeID in admmGraph.nodes[nodeID].neighbors
            adjoint!(admmGraph.edges[edgeID].mappings[nodeID], solver.dualBuffer[edgeID], Bt_du[nodeID], true)
        end
    end
    norm_At = sqrt(sum(dot(v,v) for v in values(At_du)))
    norm_Bt = sqrt(sum(dot(v,v) for v in values(Bt_du)))

    if norm_du == 0
        a = 0.0; b = 0.0
    else
        a = norm_At/norm_du
        b = norm_Bt/norm_du
    end

    # Step3) Get λ , μ , L, ℓ, δ
    numA = 0.0 ; denA = 0.0  ; numμA = 0.0;  denμA = 0.0
    dx_sq = 0.0; dxgrad_sq = 0.0; xcross = 0.0
    for nodeID in admmGraph.left
        dx =  similar(solver.primalprev[nodeID])
        copyto!(dx, solver.primalprev[nodeID])
        axpy!(-1.0, solver.primalBuffer[nodeID], dx) 
        dxgrad = similar(solver.xgradPrev[nodeID])    
        copyto!(dxgrad, solver.xgradPrev[nodeID])      
        axpy!(-1.0, solver.xgradCur[nodeID], dxgrad)
        # Get L, ℓ, δ
        dx_sq     += dot(dx, dx)
        dxgrad_sq += dot(dxgrad, dxgrad)
        xcross    += dot(dxgrad, dx)         
        Fk    = dx .- γk.* dxgrad                      
        v  = At_du[nodeID]       
        # numerator: ⟨AᵀΔu, x^k - x^{k-1}⟩ = dot(v, -dx)
        numA += dot(v, -dx)
        # denominator: ∥AᵀΔu∥²/(16 a²) + 4 a² ∥dx∥²
        denA += safe_div(dot(v, v), 16*a^2) + 4*a^2 * dot(dx, dx)
        # numerator: ⟨AᵀΔu, F_k(x^{k-1}) - F_k(x^{k})⟩ = dot(v, tmp)
        numμA += dot(v, Fk)
        # denominator: γk∥AᵀΔu∥²/(2a²) + a² ∥tmp∥² / (2γk)
        denμA += γk*dot(v,v)/(2*a^2) +
                 a^2*dot(Fk,Fk)/(2*γk)
    end

    numB = 0.0 ; denB = 0.0 ; numμB = 0.0;  denμB = 0.0
    dy_sq = 0.0; dygrad_sq = 0.0; ycross = 0.0
    for nodeID in admmGraph.right
        dy =  similar(solver.primalprev[nodeID])
        copyto!(dy, solver.primalprev[nodeID])
        axpy!(-1.0, solver.primalBuffer[nodeID], dy)
        dygrad = similar(solver.ygradPrev[nodeID])    
        copyto!(dygrad, solver.ygradPrev[nodeID])      
        axpy!(-1.0, solver.ygradCur[nodeID], dygrad) 
        dy_sq    +=  dot(dy, dy)
        dygrad_sq += dot(dygrad, dygrad)
        ycross    += dot(dygrad, dy)   
        Gk    =  dy .- γk.* dygrad                  
        w  = Bt_du[nodeID]
        numB += dot(w, -dy)
        denB += safe_div(dot(w, w), 16*b^2) + 4*b^2 * dot(dy, dy)
        numμB += dot(w, Gk)
        denμB += γk*dot(w,w)/(2*b^2) +
                 b^2*dot(Gk, Gk)/(2*γk)
    end

    
    λA = safe_div(numA, denA)
    μA = safe_div(numμA, denμA)
    μA = abs(μA) < 1e-2 ? 1e-2 : μA #if |μA|<eps  , then μA= eps. if |μA|> eps then μA = μA for numerical stability.
    λB = safe_div(numB, denB)
    μB = safe_div(numμB, denμB)
    μB = abs(μB) < 1e-2 ? 1e-2 : μB
    
    if dx_sq <= 0
        Lx = 0.0; ℓx = 0.0; δx = 0.0
    else
        Lx = sqrt(dxgrad_sq)/sqrt(dx_sq)
        ℓx = xcross/dx_sq
        δx = γk^2*Lx^2 - 2*γk*ℓx
    end
    if dy_sq <= 0
        Ly = 0.0; ℓy = 0.0; δy = 0.0
    else
        Ly = sqrt(dygrad_sq)/sqrt(dy_sq)
        ℓy = ycross/dy_sq
        δy = γk^2*Ly^2 - 2*γk*ℓy
    end

    # Step4) Solve cubics
    if solver.ifSimple
        Γx = (γk/2) / ((γk * ℓx) + sqrt(max((γk * ℓx)^2 + (2/3) * (δx + 6*solver.r*a^2*γk^2*λA) , 0 ))) 
        Γy = (γk/2) / ((γk * ℓy) + sqrt(max((γk * ℓy)^2 + (2/3) * (δy + 6*solver.r*b^2*γk^2*λB) , 0 ))) 
    else
        αx = (solver.r)*a^2*μA*(δx+1)/ γk^2; βx = 2*(solver.φ)^2*(solver.r)*a^2*λA + δx/γk^2; γcx= solver.φ*ℓx; δcx=-1/2
        Γx = solve_cubic_real(αx, βx , γcx , δcx)
        αy = (solver.r)*b^2*μB*(δy+1)/γk^2; βy = 2*(solver.φ)^2*(solver.r)*b^2*λB + δy/γk^2; γcy=  solver.φ*ℓy; δcy=-1/2
        Γy =  solve_cubic_real(αy, βy , γcy , δcy)
    end

    # Step5) Update γ
    cand1 = solver.ifSimple ?  (3/2) * γk  : solver.φ * γk
    cand2 = solver.ifSimple ?  sqrt((4-λA-λB)/(32*(solver.r)*(a^2+b^2)))  : ((4-λA-λB)/(4*(solver.r)))*(1/((μA+μB)/(solver.r) + sqrt((μA+μB)^2/(solver.r)^2 + (a^2 + b^2)*(4-λA-λB)/(2*(solver.r)))))
    
    cands = filter(x -> isfinite(x), (cand1, cand2, Γx, Γy ))
    
    if isempty(cands)
        error("All γ–candidates are NaN/Inf; something is wrong")
    end
    solver.proximalStepsizeGamma = minimum(cands)  

    # Update Dual and save u^{k+1} to info.Sol
    for edgeID in collect(keys(info.dualSol))
        axpy!(
            solver.r * solver.proximalStepsizeGamma,
            solver.dualBuffer[edgeID],
            info.dualSol[edgeID]
          )

    end

    # Prev <- Cur for next iteration
    for nodeID in keys(solver.primalprev)
        copyto!(solver.primalprev[nodeID], info.primalSol[nodeID])
    end
    for nodeID in keys(solver.xgradPrev)
        copyto!(solver.xgradPrev[nodeID], solver.xgradCur[nodeID])
    end
    for nodeID in keys(solver.ygradPrev)
        copyto!(solver.ygradPrev[nodeID], solver.ygradCur[nodeID])
    end
end

"""
    update!(solver::AdaptiveLinearizedSolver, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph, rhoUpdated::Bool)

Update solver state at the beginning of each ADMM iteration.

This function is called before solving primal subproblems to update the dual variable
history and perform the adaptive step size computation. It ensures the solver has
the most current information for optimal performance.

# Arguments
- `solver::AdaptiveLinearizedSolver`: The solver instance
- `info::ADMMIterationInfo`: Current iteration information
- `admmGraph::ADMMBipartiteGraph`: The bipartite graph structure
- `rhoUpdated::Bool`: Whether penalty parameter was updated (unused in this solver)

# Effects
- Copies current dual solutions to previous dual solution buffers
- Calls `updateGammaAndDual!` to perform adaptive step size update
- Updates all internal solver state for the new iteration

# Algorithm Flow
1. **Dual History Update**: Copy u^k to previous dual buffers
2. **Adaptive Update**: Call core adaptive algorithm to update γ and dual variables
3. **State Synchronization**: Ensure all solver state is consistent

# Notes
- This function bridges the gap between ADMM framework and adaptive solver
- The `rhoUpdated` parameter is unused since this solver adapts γ instead of ρ
- Parallel copying of dual variables for efficiency
- Critical for maintaining gradient history consistency
"""
function update!(solver::AdaptiveLinearizedSolver, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph, rhoUpdated::Bool)
    @threads for edgeID in collect(keys(info.dualSol))
        copyto!(info.dualSolPrev[edgeID], info.dualSol[edgeID])
    end 
    updateGammaAndDual!(solver, info, admmGraph)
end

"""
    solve!(solver::AdaptiveLinearizedSolver, nodeID::String, accelerator::AbstractADMMAccelerator,
           admmGraph::ADMMBipartiteGraph, info::ADMMIterationInfo, isLeft::Bool, enableParallel::Bool=false)

Solve the ADMM subproblem for a specific node using adaptive linearized method.

This function performs one proximal gradient step for the specified node using the
current adaptive step size γ. It uses pre-computed gradients and applies the
proximal operator to the regularization function.

# Arguments
- `solver::AdaptiveLinearizedSolver`: The solver instance
- `nodeID::String`: Identifier of the node to solve for
- `accelerator::AbstractADMMAccelerator`: Acceleration scheme (unused)
- `admmGraph::ADMMBipartiteGraph`: The bipartite graph structure
- `info::ADMMIterationInfo`: Current iteration information
- `isLeft::Bool`: Whether the node is on the left side of the bipartite graph
- `enableParallel::Bool`: Whether to enable parallel computation (unused)

# Effects
- Updates `info.primalSol[nodeID]` with new primal solution
- Stores previous solution in `info.primalSolPrev[nodeID]`
- Uses pre-computed gradients from adaptive update phase

# Algorithm Steps
1. **Gradient Loading**: Load pre-computed gradient ∇f(x^k) from appropriate buffer
2. **Dual Term Addition**: Add adjoint dual contributions A^T u^{k+1} from all edges
3. **Proximal Step**: Compute x^{k+1} = prox_{γg}(x^k - γ(∇f(x^k) + A^T u^{k+1}))

# Mathematical Details
The subproblem solved is:
    x^{k+1} = argmin g(x) + ⟨∇f(x^k), x - x^k⟩ + ⟨A^T u^{k+1}, x⟩ + (1/2γ)||x - x^k||²

This is equivalent to:
    x^{k+1} = prox_{γg}(x^k - γ(∇f(x^k) + A^T u^{k+1}))

# Implementation Notes
- Gradients are pre-computed in `updateGammaAndDual!` for efficiency
- The adaptive γ has been optimized for current problem geometry
- Proximal operator depends on the structure of regularization function g
- Efficient memory usage with buffer reuse
"""
function solve!(solver::AdaptiveLinearizedSolver, 
    nodeID::String,
    accelerator::AbstractADMMAccelerator,
    admmGraph::ADMMBipartiteGraph,
    info::ADMMIterationInfo, 
    isLeft::Bool,
    enableParallel::Bool = false)

    if isLeft
        copyto!(info.primalBuffer[nodeID], solver.xgradCur[nodeID])
    else
        copyto!(info.primalBuffer[nodeID], solver.ygradCur[nodeID])
    end

    for edgeID in admmGraph.nodes[nodeID].neighbors
        edge = admmGraph.edges[edgeID]
        adjoint!(edge.mappings[nodeID],info.dualSol[edgeID],info.primalBuffer[nodeID],true)
    end 

    # info buffer <- x - proxStepsize * (nabla f_i(x_i) + A'u)
    rmul!(info.primalBuffer[nodeID], -solver.proximalStepsizeGamma)
    axpy!(1.0, info.primalSol[nodeID], info.primalBuffer[nodeID])

    # Save x^k to info.primalSolPrev
    copyto!(info.primalSolPrev[nodeID], info.primalSol[nodeID])

    # Update primal and save it to info.Sol
    proximalOracle!(info.primalSol[nodeID], admmGraph.nodes[nodeID].g, info.primalBuffer[nodeID], solver.proximalStepsizeGamma)
end 

"""
    updateDualResidualsInBuffer!(solver::AdaptiveLinearizedSolver, info::ADMMIterationInfo, 
                                admmGraph::ADMMBipartiteGraph, accelerator::AbstractADMMAccelerator)

Compute and store dual residuals for convergence monitoring in adaptive linearized ADMM.

This function computes the dual residuals specific to the adaptive linearized method,
accounting for the gradient differences and proximal terms with the adaptive step size γ.
The residuals measure optimality condition violations and are used for convergence assessment.

# Arguments
- `solver::AdaptiveLinearizedSolver`: The solver instance
- `info::ADMMIterationInfo`: Current iteration information
- `admmGraph::ADMMBipartiteGraph`: The bipartite graph structure  
- `accelerator::AbstractADMMAccelerator`: Acceleration scheme (unused)

# Effects
- Updates `info.dresL2` with L2 norm of dual residuals
- Updates `info.dresLInf` with L∞ norm of dual residuals
- Computes gradients at new iterates for next adaptive update
- Stores residuals in `info.primalBuffer` for norm computation

# Algorithm Steps
1. **Residual Computation**: For each node, compute dual residual components
2. **Gradient Updates**: Compute ∇f(x^{k+1}) and store for next iteration
3. **Norm Computation**: Calculate L2 and L∞ norms of all residuals
4. **History Update**: Store residual norms in convergence monitoring

# Mathematical Details
For each node, the dual residual is:
    r^{dual} = (1/γ)(x^k - x^{k+1}) + ∇f(x^{k+1}) - ∇f(x^k)

This captures the violation of the optimality condition:
    0 ∈ ∂g(x^{k+1}) + ∇f(x^{k+1}) + A^T u^{k+1} + (1/γ)(x^{k+1} - x^k)

The residual measures how far the current iterate is from satisfying the
first-order optimality conditions of the linearized subproblem.

# Implementation Notes
- Gradient computations here also prepare for the next adaptive step
- Parallel computation across nodes for efficiency
- The dual residual formulation is specific to the linearized method
- Residuals are stored in `info.primalBuffer` temporarily for norm computation
"""
function updateDualResidualsInBuffer!(solver::AdaptiveLinearizedSolver, 
    info::ADMMIterationInfo, 
    admmGraph::ADMMBipartiteGraph, 
    accelerator::AbstractADMMAccelerator)
    """ 
        info.primalBufferr are filled with x^k , y^k
    """
    # Compute dual residuals for left nodes
    @threads for nodeID in admmGraph.left 
        node = admmGraph.nodes[nodeID]
        copyto!(info.primalBuffer[nodeID], info.primalSolPrev[nodeID])
        #(1/γ)x^k
        rmul!( info.primalBuffer[nodeID], 1.0/solver.proximalStepsizeGamma)
        # Compute ∇f(x^{k+1}) and save it for later
        gradientOracle!(solver.xgradCur[nodeID], node.f, info.primalSol[nodeID])
        #  += ∇f(x^{k+1}) - ∇f(x^k)
        axpy!(1.0, solver.xgradCur[nodeID], info.primalBuffer[nodeID])
        axpy!(-1.0, solver.xgradPrev[nodeID], info.primalBuffer[nodeID])
        #  -(1/γ)(x^{k+1})
        axpy!(-1.0/solver.proximalStepsizeGamma, info.primalSol[nodeID],  info.primalBuffer[nodeID])    
    end 

    # Compute dual residuals for right nodes
    @threads for nodeID in admmGraph.right  
        node = admmGraph.nodes[nodeID]
        copyto!(info.primalBuffer[nodeID], info.primalSolPrev[nodeID])
        #(1/γ)y^k
        rmul!( info.primalBuffer[nodeID], 1.0/solver.proximalStepsizeGamma)
        # Compute ∇g(y^{k+1}) and save it for later
        gradientOracle!(solver.ygradCur[nodeID], node.f, info.primalSol[nodeID])
        #  += ∇g(y^{k+1}) - ∇g(y^k)
        axpy!(1.0, solver.ygradCur[nodeID], info.primalBuffer[nodeID])
        axpy!(-1.0, solver.ygradPrev[nodeID], info.primalBuffer[nodeID])
        #  -(1/γ)(y^{k+1})
        axpy!(-1.0/solver.proximalStepsizeGamma, info.primalSol[nodeID],  info.primalBuffer[nodeID])   
    end 

    dresL2Square = 0.0 
    dresLInf = 0.0 
    for nodeID in keys(info.primalBuffer)
        dresL2Square += dot(info.primalBuffer[nodeID], info.primalBuffer[nodeID])
        dresLInf = max(dresLInf, norm(info.primalBuffer[nodeID], Inf))
    end 
    push!(info.dresL2, sqrt(dresL2Square))
    push!(info.dresLInf, dresLInf)
end 