# Dual Lasso Benchmark Results

## Problem Formulation

The Dual Lasso problem is a quadratic programming problem with infinity norm constraints:

```math
\begin{aligned}
\min_{x} \quad & \frac{1}{4}\|x\|^2 - b^\top x \\
\mathrm{s.t.} \quad &  \|Ax\|_{\infty} \leq \lambda
\end{aligned}
```

where:
- `A` is an `m × n` matrix
- `b` is an `n`-dimensional vector
- `x` is the `n`-dimensional optimization variable  
- `λ` is the regularization parameter

For ADMM decomposition, this is reformulated as a two-block problem:
```math
\begin{aligned}
\min_{x, z} \quad & \frac{1}{4}\|x\|^2 - b^\top x \\
\mathrm{s.t.} \quad &  Ax - z = 0, \|z\|_{\infty} \leq \lambda
\end{aligned}
```


This reformulation enables efficient distributed solving using bipartite ADMM.

## Implementation Examples

### JuMP Implementation

Here's how to implement the Dual Lasso problem using JuMP:

```julia
using JuMP, SCS

function solveDualLassoByJuMP(A::SparseMatrixCSC{Float64, Int64}, b::Vector{Float64}, lambda::Float64, solver_optimizer=SCS.Optimizer; warmup=false)

    m, n = size(A)

    # Create JuMP model with specified optimizer
    model = Model(solver_optimizer)
    set_silent(model)
    
    # Define optimization variable
    @variable(model, x[1:n])
    
    # Define objective: (1/4) ||x||^2 - b'x
    @objective(model, Min, 0.25 * sum(x[i]^2 for i in 1:n) - dot(b, x))
    
    # Define constraint: ||Ax||_{inf} <= lambda
    # This is equivalent to: -lambda <= (Ax)_i <= lambda for all i
    Ax = A * x
    @constraint(model, -lambda .<= Ax .<= lambda)
    
    # Solve the problem and get solver time directly from JuMP
    optimize!(model)
    solver_time = solve_time(model)
    
    status = termination_status(model)
    optimal_value = objective_value(model)
    optimal_x = value.(x)

    return optimal_value, solver_time
end
```

### Multiblock Problem Implementation

Here's how to formulate the same problem as a multiblock problem for ADMM:

```julia
using PDMO

function generateDualLasso(A::SparseMatrixCSC{Float64, Int64}, b::Vector{Float64}, lambda::Float64)
    mbp = MultiblockProblem() 
    
    numberRows, numberCols = size(A)
    @assert(numberCols == length(b), "DualLasso: Dimension mismatch. ")

    # x block
    block_x = BlockVariable() 
    block_x.f = QuadraticFunction(0.25 * spdiagm(0 => ones(numberCols)), -b, 0.0)
    block_x.val = zeros(numberCols)
    xID = addBlockVariable!(mbp, block_x)

    # z block 
    block_z = BlockVariable() 
    block_z.g = IndicatorBox(-lambda * ones(numberRows), ones(numberRows) * lambda)
    block_z.val = zeros(numberRows)
    zID = addBlockVariable!(mbp, block_z)

    # constraint 
    constr = BlockConstraint() 
    addBlockMappingToConstraint!(constr, xID, LinearMappingMatrix(A))
    addBlockMappingToConstraint!(constr, zID, LinearMappingIdentity(-1.0))
    constr.rhs = zeros(numberRows)
    addBlockConstraint!(mbp, constr)

    return mbp
end 
```
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

### [JuMP](https://github.com/jump-dev/JuMP.jl) Solvers
- **[SCS](https://github.com/cvxgrp/scs)**: Splitting Conic Solver
- **[COSMO.jl](https://jump.dev/JuMP.jl/stable/packages/COSMO/)**: Conic Operator Splitting Method  
- **[Clarabel.jl](https://github.com/oxfordcontrol/Clarabel.jl)**: Interior Point Conic Solver
- **[MadNLP.JL](https://github.com/MadNLP/MadNLP.jl)**: Nonlinear Programming Solver
- **[Ipopt](https://github.com/coin-or/Ipopt)**: Interior Point Optimizer

### ADMM Variants
- **Original ADMM**: Baseline implementation
- **Accelerated Methods**: Anderson and Auto-Halpern acceleration
- **Adaptive Methods**: RB and SRA adapters
- **Linearized Methods**: Adaptive and doubly linearized solvers

### Problem Scales
- **Small Scale**: 100 × 200 (λ = 2.0)
- **Medium Scale**: 2000 × 3000 (λ = 2.0)  
- **Large Scale**: 4000 × 8000 (λ = 2.0)

## Performance Results

### Small Scale (100 × 200)

**JuMP Solver Performance:**

| Solver | Time (s) | Status | Objective |
|--------|----------|---------|-----------|
| MadNLP | 7.66 | LOCALLY_SOLVED | -118.780846 |
| Ipopt | 0.12 | LOCALLY_SOLVED | -118.780846 |
| Clarabel | 0.08 | OPTIMAL | -118.780845 |
| SCS | 0.01 | OPTIMAL | -118.780846 |
| COSMO | 0.004 | OPTIMAL | -118.780846 |

**ADMM Performance (Top 5):**

| Method | Time (s) | Speedup | Iterations | Status |
|--------|----------|---------|------------|---------|
| Original ADMM + RB + Halpern | 0.02 | 458.87x | 77 | OPTIMAL |
| Original ADMM + SRA + Halpern | 0.02 | 314.57x | 61 | OPTIMAL |
| Original ADMM + SRA Adapter | 0.03 | 238.37x | 61 | OPTIMAL |
| Original ADMM + RB Adapter | 0.07 | 104.77x | 77 | OPTIMAL |
| Original ADMM + Halpern only | 1.91 | 4.01x | 26909 | OPTIMAL |

### Medium Scale (2000 × 3000)

**JuMP Solver Performance:**

| Solver | Time (s) | Status | Objective |
|--------|----------|---------|-----------|
| Clarabel | 252.44 | OPTIMAL | -1114.120645 |
| SCS | 31.25 | OPTIMAL | -1114.120651 |
| MadNLP | 30.26 | LOCALLY_SOLVED | -1114.120644 |
| Ipopt | 18.20 | LOCALLY_SOLVED | -1114.120647 |
| COSMO | 16.89 | OPTIMAL | -1114.120648 |

**ADMM Performance (Top 5):**

| Method | Time (s) | Speedup | Iterations | Status |
|--------|----------|---------|------------|---------|
| Original ADMM + RB Adapter | 15.24 | 16.57x | 158 | OPTIMAL |
| Original ADMM + SRA Adapter | 15.41 | 16.38x | 169 | OPTIMAL |
| Original ADMM + RB + Halpern | 17.04 | 14.82x | 248 | OPTIMAL |
| Original ADMM + SRA + Halpern | 18.17 | 13.90x | 283 | OPTIMAL |
| Original ADMM + Anderson only | 356.66 | 0.71x | 9061 | OPTIMAL |

### Large Scale (4000 × 8000)

**JuMP Solver Performance:**

| Solver | Time (s) | Status | Objective |
|--------|----------|---------|-----------|
| Clarabel | 2387.48 | OPTIMAL | -4156.277839 |
| SCS | 288.69 | OPTIMAL | -4156.277722 |
| MadNLP | 228.98 | LOCALLY_SOLVED | -4156.277834 |
| Ipopt | 211.54 | LOCALLY_SOLVED | -4156.277834 |
| COSMO | 158.64 | OPTIMAL | -4156.277843 |

**ADMM Performance (Top 5):**

| Method | Time (s) | Speedup | Iterations | Status |
|--------|----------|---------|------------|---------|
| Original ADMM + SRA Adapter | 103.91 | 22.98x | 350 | OPTIMAL |
| Original ADMM + SRA + Halpern | 140.49 | 16.99x | 644 | OPTIMAL |
| Original ADMM + RB Adapter | 147.68 | 16.17x | 347 | OPTIMAL |
| Original ADMM + RB + Halpern | 186.31 | 12.81x | 638 | OPTIMAL |
| Adaptive Linearized Simple (γ=1.0, r=1000) | 3600.03 | 0.66x | 31305 | TIME_LIMIT |

## Multi-Scale Performance Summary

| Scale | JuMP Reference | Best ADMM Method | ADMM Time | Speedup | Status |
|-------|----------------|------------------|-----------|---------|---------|
| Small (100×200) | MadNLP: 7.66s | Original ADMM + RB + Halpern | 0.02s | 458.87x | OPTIMAL |
| Medium (2000×3000) | Clarabel: 252.44s | Original ADMM + RB Adapter | 15.24s | 16.57x | OPTIMAL |
| Large (4000×8000) | Clarabel: 2387.48s | Original ADMM + SRA Adapter | 103.91s | 22.98x | OPTIMAL |

