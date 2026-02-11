# Dual SVM Benchmark Results

## Problem Formulation

The Dual Support Vector Machine problem is a quadratic programming problem with linear constraints:

```math
\begin{aligned}
\min_{x} \quad & \frac{1}{2}\langle Qx, x \rangle - \langle e, x \rangle \\
\mathrm{s.t.} \quad & \langle b, x \rangle = 0 \\
& 0 \leq x_i \leq C \quad \forall i
\end{aligned}
```

where:
- `Q` is an `n × n` positive definite matrix (kernel matrix)
- `b` is an `n`-dimensional vector (class labels)
- `e` is the vector of all ones
- `x` is the `n`-dimensional optimization variable (dual variables)
- `C` is the regularization parameter

For ADMM decomposition, this is reformulated as a two-block problem:

```math
\begin{aligned}
\min_{x, z} \quad & \frac{1}{2}\langle Qx, x \rangle - \langle e, x \rangle \\
\mathrm{s.t.} \quad & x - z = 0, \quad 0 \leq z_i \leq C \quad \forall i \\
& x \in \{x: \langle b, x \rangle = 0\}
\end{aligned}
```

This reformulation separates the equality constraint from the box constraints, enabling efficient distributed solving using bipartite ADMM.

## Implementation Examples

### JuMP Implementation

Here's how to implement the Dual SVM problem using JuMP:

```julia
function solveDualSVMByJuMP(Q::SparseMatrixCSC{Float64, Int64}, b::Vector{Float64}, C::Float64, solver_optimizer=SCS.Optimizer; warmup=false)

    n = length(b)

    # Create JuMP model with specified optimizer
    model = Model(solver_optimizer)
    set_silent(model)
    
    # Define optimization variable
    @variable(model, x[1:n])
    
    # Define objective: 0.5 * <Qx, x> - <e, x>
    @objective(model, Min, 0.5 * x' * Q * x - sum(x))
    
    # Define constraints: <b, x> = 0, 0 <= x <= C
    @constraint(model, dot(b, x) == 0)
    @constraint(model, x .>= 0)
    @constraint(model, x .<= C)
    
    # Solve the problem and get solver time directly from JuMP
    optimize!(model)
    solver_time = solve_time(model)
    
    status = termination_status(model)
    optimal_value = objective_value(model)

    return optimal_value, solver_time
end
```

### Multiblock Problem Implementation

Here's how to formulate the same problem as a multiblock problem using PDMO:

```julia
function generateDualSVM(Q::SparseMatrixCSC{Float64, Int64}, b::Vector{Float64}, C::Float64)
    
    numberVars = length(b)
    @assert(numberVars == size(Q,1) == size(Q, 2), "DualSVM: input dimension mismatch. ")

    mbp = MultiblockProblem() 

    # x block
    block_x = BlockVariable() 
    block_x.f = QuadraticFunction(0.5 * Q, -ones(numberVars), 0.0)
    block_x.g = IndicatorHyperplane(b, 0.0)
    block_x.val = zeros(numberVars) # initial point
    xID = addBlockVariable!(mbp, block_x)

    # z block
    block_z = BlockVariable() 
    block_z.g = IndicatorBox(zeros(numberVars), ones(numberVars) * C)
    block_z.val = zeros(numberVars) # initial point
    zID = addBlockVariable!(mbp, block_z)

    # constraint: x - z = 0 
    constr = BlockConstraint() 
    addBlockMappingToConstraint!(constr, xID, LinearMappingIdentity(1.0))
    addBlockMappingToConstraint!(constr, zID, LinearMappingIdentity(-1.0))
    constr.rhs = spzeros(numberVars)
    addBlockConstraint!(mbp, constr)
    
    return mbp 
end
```

### Running ADMM with Different Parameters

After generating the multiblock problem, you can solve it using various ADMM configurations:

```julia
# ADMM with RB Adapter
param = ADMMParam(
    adapter = RBAdapter(testRatio=10.0, adapterRatio=2.0),
    presTolL2 = Inf,
    dresTolL2 = Inf,
    presTolLInf = 1e-4,
    dresTolLInf = 1e-4
)
result = runBipartiteADMM(mbp, param)
```

## Performance Results

### Small Scale (n = 500)

**ADMM Performance (Top 5):**

| Method | Time (s) | Iterations | Status |
|--------|----------|------------|---------|
| Doubly Linearized (baseline) | 1.35 | 879 | OPTIMAL |
| Adaptive Linearized (γ=1.0, r=1000) | 1.45 | 2753 | OPTIMAL |
| Adaptive Linearized Simple (γ=1.0, r=1000) | 1.58 | 2625 | OPTIMAL |
| Adaptive Linearized (γ=2.0, r=1000) | 1.60 | 3030 | OPTIMAL |
| Adaptive Linearized (γ=0.5, r=1000) | 1.78 | 2756 | OPTIMAL |

### Medium Scale (n = 1000)

**ADMM Performance (Top 5):**

| Method | Time (s) | Iterations | Status |
|--------|----------|------------|---------|
| Doubly Linearized (baseline) | 5.86 | 1357 | OPTIMAL |
| Original ADMM + SRA Adapter | 15.76 | 61 | OPTIMAL |
| Original ADMM + SRA + Halpern | 15.84 | 61 | OPTIMAL |
| Adaptive Linearized (γ=0.5, r=1000) | 17.95  | 9568 | OPTIMAL |
| Adaptive Linearized Simple (γ=1.0, r=1000) | 18.38  | 10030 | OPTIMAL |

### Large Scale (n = 4000)

**ADMM Performance (Top 5):**

| Method | Time (s) | Iterations | Status |
|--------|----------|------------|---------|
| Doubly Linearized (baseline) | 285.60 | 3348 | OPTIMAL |
| Original ADMM + SRA Adapter | 1034.30 | 69 | OPTIMAL |
| Original ADMM + SRA + Halpern | 1050.25  | 69 | OPTIMAL |
| Adaptive Linearized Simple (γ=1.0, r=1000) | 3600.01  | 98037 | TIME_LIMIT |
| Adaptive Linearized (γ=2.0, r=1000) | 3600.01  | 96944 | TIME_LIMIT |
