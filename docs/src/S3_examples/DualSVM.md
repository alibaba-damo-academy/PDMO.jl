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

## Solver References

### [JuMP](https://github.com/jump-dev/JuMP.jl) Solvers
- **[SCS](https://github.com/cvxgrp/scs)**: Splitting Conic Solver
- **[COSMO.jl](https://github.com/oxfordcontrol/COSMO.jl)**: Conic Operator Splitting Method  
- **[Clarabel.jl](https://github.com/oxfordcontrol/Clarabel.jl)**: Interior Point Conic Solver
- **[MadNLP.jl](https://github.com/MadNLP/MadNLP.jl)**: Nonlinear Programming Solver
- **[Ipopt](https://github.com/coin-or/Ipopt)**: Interior Point Optimizer

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

## Benchmark Methodology

The benchmark compares multiple solution approaches:

### JuMP Solvers
- **SCS**: Splitting Conic Solver
- **COSMO**: Conic Operator Splitting Method  
- **Clarabel**: Interior Point Conic Solver
- **MadNLP**: Nonlinear Programming Solver
- **Ipopt**: Interior Point Optimizer

### ADMM Variants
- **Original ADMM**: Basic consensus ADMM with no acceleration
- **Anderson Accelerator**: History-based acceleration method
- **Auto-Halpern Accelerator**: Adaptive step-size acceleration
- **RB Adapter**: Residual Balancing for automatic parameter tuning
- **SRA Adapter**: Spectral Residual Adaptation
- **Linearized Solvers**: Adaptive and doubly linearized variants

### Problem Scales
Three different problem sizes are tested:
- **Small**: n = 500 (number of variables)
- **Medium**: n = 1000  
- **Large**: n = 4000

## Performance Results

### Small Scale (n = 500)

**JuMP Solver Performance:**

| Solver | Time (s) | Objective | Status |
|--------|----------|-----------|---------|
| COSMO | 0.0407 | -2.0304 | OPTIMAL |
| SCS | 0.0975 | -2.0304 | OPTIMAL |
| Clarabel | 0.1939 | -2.0304 | OPTIMAL |
| Ipopt | 0.2473 | -2.0305 | LOCALLY_SOLVED |
| MadNLP | 8.9777 | -2.0305 | LOCALLY_SOLVED |

**ADMM Performance (Top 5):**

| Method | Time (s) | Speedup | Iterations | Status |
|--------|----------|---------|------------|---------|
| Doubly Linearized (baseline) | 1.35 | **6.67x** | 879 | OPTIMAL |
| Adaptive Linearized (γ=1.0, r=1000) | 1.45 | 6.20x | 2753 | OPTIMAL |
| Adaptive Linearized Simple (γ=1.0, r=1000) | 1.58 | 5.68x | 2625 | OPTIMAL |
| Adaptive Linearized (γ=2.0, r=1000) | 1.60 | 5.63x | 3030 | OPTIMAL |
| Adaptive Linearized (γ=0.5, r=1000) | 1.78 | 5.03x | 2756 | OPTIMAL |

### Medium Scale (n = 1000)

**JuMP Solver Performance:**

| Solver | Time (s) | Objective | Status |
|--------|----------|-----------|---------|
| SCS | 0.2463 | -2.1057 | OPTIMAL |
| Ipopt | 0.5411 | -2.1057 | LOCALLY_SOLVED |
| COSMO | 0.6065 | -2.1057 | OPTIMAL |
| Clarabel | 1.5996 | -2.1057 | OPTIMAL |
| MadNLP | 3.8568 | -2.1057 | LOCALLY_SOLVED |

**ADMM Performance (Top 5):**

| Method | Time (s) | Speedup | Iterations | Status |
|--------|----------|---------|------------|---------|
| Doubly Linearized (baseline) | 5.86 | **0.66x** | 1357 | OPTIMAL |
| Original ADMM + SRA Adapter | 15.76 | 0.24x | 61 | OPTIMAL |
| Original ADMM + SRA + Halpern | 15.84 | 0.24x | 61 | OPTIMAL |
| Adaptive Linearized (γ=0.5, r=1000) | 17.95 | 0.21x | 9568 | OPTIMAL |
| Adaptive Linearized Simple (γ=1.0, r=1000) | 18.38 | 0.21x | 10030 | OPTIMAL |

### Large Scale (n = 4000)

**JuMP Solver Performance:**

| Solver | Time (s) | Objective | Status |
|--------|----------|-----------|---------|
| SCS | 12.1316 | -2.2632 | OPTIMAL |
| COSMO | 38.3884 | -2.2632 | OPTIMAL |
| Clarabel | 156.3004 | -2.2632 | OPTIMAL |
| MadNLP | 258.7855 | -2.2632 | LOCALLY_SOLVED |
| Ipopt | 306.6743 | -2.2633 | LOCALLY_SOLVED |

**ADMM Performance (Top 5):**

| Method | Time (s) | Speedup | Iterations | Status |
|--------|----------|---------|------------|---------|
| Doubly Linearized (baseline) | 285.60 | **1.07x** | 3348 | OPTIMAL |
| Original ADMM + SRA Adapter | 1034.30 | 0.30x | 69 | OPTIMAL |
| Original ADMM + SRA + Halpern | 1050.25 | 0.29x | 69 | OPTIMAL |
| Adaptive Linearized Simple (γ=1.0, r=1000) | 3600.01 | 0.09x | 98037 | TIME_LIMIT |
| Adaptive Linearized (γ=2.0, r=1000) | 3600.01 | 0.09x | 96944 | TIME_LIMIT |

## Multi-Scale Performance Summary

| Scale | JuMP Reference | Best ADMM Method | ADMM Time | Speedup | Status |
|-------|----------------|------------------|-----------|---------|---------|
| Small (n=500) | MadNLP: 8.98s | Doubly Linearized (baseline) | 1.35s | 6.67x | OPTIMAL |
| Medium (n=1000) | SCS: 0.25s | Doubly Linearized (baseline) | 5.86s | 0.66x | OPTIMAL |
| Large (n=4000) | SCS: 12.13s | Doubly Linearized (baseline) | 285.60s | 1.07x | OPTIMAL |
