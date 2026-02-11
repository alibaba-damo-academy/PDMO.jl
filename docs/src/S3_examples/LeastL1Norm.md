# Least L1 Norm Benchmark Results

## Problem Formulation

The Least L1 Norm problem aims to minimize the L1 norm of the residual:

```math 
\min_{x} \|Ax - b\|_1
```

where:
- `A` is an `m × n` matrix 
- `b` is an `m`-dimensional vector
- `x` is the `n`-dimensional optimization variable

For ADMM decomposition, this is reformulated as a two-block problem:

```math
\begin{aligned}
\min_{x, z} \quad & \|z\|_1 \\
\mathrm{s.t.} \quad & Ax - z = b
\end{aligned}
```

This reformulation enables efficient distributed solving using bipartite ADMM.

## Solver References

### [JuMP](https://github.com/jump-dev/JuMP.jl) Solvers
- **[SCS](https://github.com/cvxgrp/scs)**: Splitting Conic Solver
- **[COSMO.jl](https://github.com/oxfordcontrol/COSMO.jl)**: Conic Operator Splitting Method  
- **[Clarabel.jl](https://github.com/oxfordcontrol/Clarabel.jl)**: Interior Point Conic Solver
- **[MadNLP.jl](https://github.com/MadNLP/MadNLP.jl)**: Nonlinear Programming Solver
- **[Ipopt](https://github.com/coin-or/Ipopt)**: Interior Point Optimizer

## Implementation Examples

### JuMP Implementation

Here's how to implement the Least L1 Norm problem using JuMP with the epigraph formulation:

```julia
function solveLeastL1NormByJuMP(A::SparseMatrixCSC{Float64, Int64}, b::Vector{Float64}, solver_optimizer=SCS.Optimizer; warmup=false)

    m, n = size(A)

    # Create JuMP model with specified optimizer
    model = Model(solver_optimizer)
    set_silent(model)
    
    # Define optimization variable
    @variable(model, x[1:n])
    
    # Define auxiliary variables for L1 norm ||Ax - b||_1
    # We use the epigraph formulation: ||Ax - b||_1 <= sum(t), -t <= Ax - b <= t
    @variable(model, t[1:m] >= 0)
    
    # Define objective: sum(t) which represents ||Ax - b||_1
    @objective(model, Min, sum(t))
    
    # Define constraints for L1 norm: -t <= Ax - b <= t (split into two constraints)
    residual = A * x - b
    @constraint(model, residual .<= t)
    @constraint(model, residual .>= -t)
    
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

Here's how to formulate the same problem as a multiblock problem for ADMM using the PDMO framework:

```julia
function generateLeastL1Norm(A::SparseMatrixCSC{Float64, Int64}, b::Vector{Float64})
    mbp = MultiblockProblem() 

    numberRows numberCols = size(A)

    # x block 
    block_x = BlockVariable()  
    block_x.val = randn(numberCols)
    xID = addBlockVariable!(mbp, block_x)

    # z block 
    block_z = BlockVariable()
    block_z.g = ElementwiseL1Norm()
    block_z.val = A * mbp.blocks[1].val - b 
    zID = addBlockVariable!(mbp, block_z)

    # constraint: Ax-z = b
    constr = BlockConstraint() 
    addBlockMappingToConstraint!(constr, xID, LinearMappingMatrix(A))
    addBlockMappingToConstraint!(constr, zID, LinearMappingIdentity(-1.0))
    constr.rhs = b
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

### JuMP Solvers
- **SCS**: Splitting Conic Solver
- **COSMO**: Conic Operator Splitting Method
- **Clarabel**: Interior Point Conic Solver  
- **MadNLP**: Nonlinear Programming Solver
- **Ipopt**: Interior Point Optimizer

### ADMM Variants
- **Original ADMM**: Standard ADMM with different configurations
- **Accelerated ADMM**: With Anderson and Auto-Halpern accelerators
- **Adaptive ADMM**: With RB and SRA adapters
- **Linearized ADMM**: Adaptive and doubly linearized variants

### Problem Scales
- **Small**: 500 × 250 (500 constraints, 250 variables)
- **Medium**: 1000 × 500 (1000 constraints, 500 variables)
- **Large**: 2000 × 1000 (2000 constraints, 1000 variables)

## Performance Results

### Small Scale (500 × 250)

**JuMP Solver Performance:**

| Solver | Time (s) | Status | Speedup vs Slowest |
|--------|----------|--------|---------------------|
| MadNLP | 9.83 | LOCALLY_SOLVED | 1.00x (reference) |
| SCS | 6.89 | OPTIMAL | 1.43x |
| COSMO | 2.34 | ITERATION_LIMIT | 4.20x |
| Ipopt | 1.39 | LOCALLY_SOLVED | 7.08x |
| Clarabel | 0.63 | OPTIMAL | 15.51x |

**ADMM Performance (Top 5):**

| Method | Time (s) | Speedup | Iterations | Status |
|--------|----------|---------|------------|--------|
| Original ADMM + Halpern only | 1.77 | 5.54x | 6,350 | OPTIMAL |
| Original ADMM + RB + Halpern | 2.10 | 4.69x | 8,552 | OPTIMAL |
| Original ADMM + Anderson only | 2.46 | 3.99x | 3,186 | OPTIMAL |
| Original ADMM + RB Adapter | 4.07 | 2.41x | 16,697 | OPTIMAL |
| Original ADMM (baseline) | 6.66 | 1.48x | 21,774 | OPTIMAL |

### Medium Scale (1000 × 500)

**JuMP Solver Performance:**

| Solver | Time (s) | Status | Speedup vs Slowest |
|--------|----------|--------|---------------------|
| SCS | 36.94 | OPTIMAL | 1.00x (reference) |
| MadNLP | 13.28 | LOCALLY_SOLVED | 2.78x |
| COSMO | 12.08 | ITERATION_LIMIT | 3.06x |
| Ipopt | 11.59 | LOCALLY_SOLVED | 3.19x |
| Clarabel | 5.20 | OPTIMAL | 7.11x |

**ADMM Performance (Top 5):**

| Method | Time (s) | Speedup | Iterations | Status |
|--------|----------|---------|------------|--------|
| Original ADMM + RB + Halpern | 7.70 | 4.80x | 8,143 | OPTIMAL |
| Original ADMM + Halpern only | 11.79 | 3.13x | 12,412 | OPTIMAL |
| Original ADMM + Anderson only | 14.38 | 2.57x | 4,864 | OPTIMAL |
| Original ADMM + RB Adapter | 22.03 | 1.68x | 23,514 | OPTIMAL |
| Original ADMM (baseline) | 24.91 | 1.48x | 26,234 | OPTIMAL |

### Large Scale (2000 × 1000)

**JuMP Solver Performance:**

| Solver | Time (s) | Status | Speedup vs Slowest |
|--------|----------|--------|---------------------|
| SCS | 367.16 | OPTIMAL | 1.00x (reference) |
| Ipopt | 188.52 | LOCALLY_SOLVED | 1.95x |
| MadNLP | 164.25 | LOCALLY_SOLVED | 2.24x |
| COSMO | 113.49 | ITERATION_LIMIT | 3.24x |
| Clarabel | 56.42 | OPTIMAL | 6.51x |

**ADMM Performance (Top 5):**

| Method | Time (s) | Speedup | Iterations | Status |
|--------|----------|---------|------------|--------|
| Original ADMM + RB Adapter | 54.91 | 6.69x | 9,742 | OPTIMAL |
| Original ADMM + RB + Halpern | 63.96 | 5.74x | 11,493 | OPTIMAL |
| Original ADMM + Anderson only | 159.97 | 2.30x | 8,127 | OPTIMAL |
| Original ADMM (baseline) | 282.29 | 1.30x | 51,226 | OPTIMAL |
| Original ADMM + Halpern only | 283.16 | 1.30x | 46,426 | OPTIMAL |

## Multi-Scale Performance Summary

| Scale | JuMP Reference | Best ADMM Method | ADMM Time | Speedup | Status |
|-------|----------------|------------------|-----------|---------|---------|
| Small (500×250) | MadNLP: 9.83s | Original ADMM + Halpern only | 1.77s | 5.54x | OPTIMAL |
| Medium (1000×500) | SCS: 36.94s | Original ADMM + RB + Halpern | 7.70s | 4.80x | OPTIMAL |
| Large (2000×1000) | SCS: 367.16s | Original ADMM + RB Adapter | 54.91s | 6.69x | OPTIMAL |

