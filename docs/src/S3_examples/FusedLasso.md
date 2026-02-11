# Fused Lasso Benchmark Results

## Problem Formulation

The Fused Lasso problem combines a quadratic data fidelity term with an L1 penalty on the differences between adjacent variables:

```math
\min_{x} \quad \frac{1}{2}\|Ax - b\|^2 + \lambda\|Dx\|_1
```

where:
- `A` is an `m × n` matrix
- `b` is an `m`-dimensional vector  
- `x` is the `n`-dimensional optimization variable
- `λ` is the regularization parameter
- `D` is the difference matrix:
  ```
  D = [-1  1  0 ...  0  0
        0 -1  1 ...  0  0
        ...
        0  0  0 ... -1  1]
  ```

For ADMM decomposition, this is reformulated as a two-block problem:

```math
\begin{aligned}
\min_{x, z} \quad & \frac{1}{2}\|Ax - b\|^2 + \lambda\|z\|_1 \\
\mathrm{s.t.} \quad & Dx - z = 0
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

Here's how to implement the Fused Lasso problem using JuMP with the epigraph formulation:

```julia
function solveFusedLassoByJuMP(A::SparseMatrixCSC{Float64, Int64}, b::Vector{Float64}, D::SparseMatrixCSC{Float64, Int64}, lambda::Float64, solver_optimizer=SCS.Optimizer; warmup=false)

    m, n = size(A)

    # Create JuMP model with specified optimizer
    model = Model(solver_optimizer)
    set_silent(model)
    
    # Define optimization variable
    @variable(model, x[1:n])
    
    # Define auxiliary variables for L1 norm ||Dx||_1
    # We use the epigraph formulation: ||Dx||_1 <= sum(t), -t <= Dx <= t
    d_size = size(D, 1)
    @variable(model, t[1:d_size] >= 0)
    
    # Define objective: (1/2) ||Ax - b||^2 + lambda * sum(t)
    residual = A * x - b
    @objective(model, Min, 0.5 * sum(residual[i]^2 for i in 1:length(residual)) + lambda * sum(t))
    
    # Define constraints for L1 norm: -t <= Dx <= t (split into two constraints)
    Dx = D * x
    @constraint(model, Dx .<= t)
    @constraint(model, Dx .>= -t)
    
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
function generateFusedLasso(A::SparseMatrixCSC{Float64, Int64}, b::Vector{Float64}, lambda::Float64)
    mbp = MultiblockProblem() 
    
    numberRows, numberCols = size(A)
    @assert(numberRows == length(b), "FusedLasso: Dimension mismatch. ")

    # x block
    block_x = BlockVariable() 
    # block_x.f = FrobeniusNormSquare(Matrix(A), b, numberCols, 1, 0.5)
    block_x.f = QuadraticFunction(0.5 * A' * A, -1.0 * A' * b, 0.5 * dot(b, b))
    block_x.val = zeros(numberCols)
    xID = addBlockVariable!(mbp, block_x)

    # z block 
    block_z = BlockVariable() 
    block_z.g = ElementwiseL1Norm(lambda)
    block_z.val = zeros(numberCols-1)
    zID = addBlockVariable!(mbp, block_z)

    # constraint 
    constr = BlockConstraint() 
    D = spdiagm(0 => -ones(numberCols - 1), 1 => ones(numberCols - 1))
    D = D[1:end-1, :]
    addBlockMappingToConstraint!(constr, xID, LinearMappingMatrix(D))
    addBlockMappingToConstraint!(constr, zID, LinearMappingIdentity(-1.0))
    constr.rhs = zeros(numberCols-1)
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
- **Original ADMM**: Basic consensus ADMM with no acceleration
- **Anderson Accelerator**: History-based acceleration method
- **Auto-Halpern Accelerator**: Adaptive step-size acceleration
- **RB Adapter**: Residual Balancing for automatic parameter tuning
- **SRA Adapter**: Spectral Residual Adaptation
- **Linearized Solvers**: Adaptive and doubly linearized variants

### Problem Scales
Three different problem sizes are tested:
- **Small**: 500 × 250 (A matrix dimensions)
- **Medium**: 1000 × 500  
- **Large**: 4000 × 2000

## Performance Results

### Small Scale (500 × 250)

**JuMP Solver Performance:**

| Solver | Time (s) | Objective | Status |
|--------|----------|-----------|---------|
| SCS | 0.0151 | 205.4149 | OPTIMAL |
| COSMO | 0.0272 | 205.4147 | OPTIMAL |
| Clarabel | 0.0265 | 205.4148 | OPTIMAL |
| Ipopt | 0.1369 | 205.4148 | LOCALLY_SOLVED |
| MadNLP | 7.8851 | 205.4148 | LOCALLY_SOLVED |

**ADMM Performance (Top 5):**

| Method | Time (s) | Speedup | Iterations | Status |
|--------|----------|---------|------------|---------|
| Original ADMM + RB + Halpern | 0.08 | **96.62x** | 856 | OPTIMAL |
| Original ADMM + RB Adapter | 0.10 | 81.72x | 454 | OPTIMAL |
| Original ADMM + SRA + Halpern | 0.11 | 69.80x | 63 | OPTIMAL |
| Original ADMM + SRA Adapter | 0.15 | 52.75x | 63 | OPTIMAL |
| Original ADMM + Anderson only | 0.23 | 34.21x | 159 | OPTIMAL |

### Medium Scale (1000 × 500)

**JuMP Solver Performance:**

| Solver | Time (s) | Objective | Status |
|--------|----------|-----------|---------|
| SCS | 0.0654 | 394.9359 | OPTIMAL |
| Ipopt | 0.1214 | 394.9358 | LOCALLY_SOLVED |
| COSMO | 0.1286 | 394.9358 | OPTIMAL |
| Clarabel | 0.1522 | 394.9358 | OPTIMAL |
| MadNLP | 0.4297 | 394.9358 | LOCALLY_SOLVED |

**ADMM Performance (Top 5):**

| Method | Time (s) | Speedup | Iterations | Status |
|--------|----------|---------|------------|---------|
| Original ADMM + Anderson only | 0.13 | **3.23x** | 229 | OPTIMAL |
| Original ADMM (baseline) | 0.13 | 3.18x | 880 | OPTIMAL |
| Original ADMM + Halpern only | 0.28 | 1.54x | 1707 | OPTIMAL |
| Original ADMM + RB Adapter | 0.45 | 0.96x | 867 | OPTIMAL |
| Original ADMM + RB + Halpern | 0.60 | 0.72x | 1681 | OPTIMAL |

### Large Scale (4000 × 2000)

**JuMP Solver Performance:**

| Solver | Time (s) | Objective | Status |
|--------|----------|-----------|---------|
| SCS | 2.3063 | 1346.478 | OPTIMAL |
| Ipopt | 3.0954 | 1346.480 | LOCALLY_SOLVED |
| COSMO | 4.9331 | 1346.480 | OPTIMAL |
| Clarabel | 10.3667 | 1346.480 | OPTIMAL |
| MadNLP | 47.4689 | 1346.480 | LOCALLY_SOLVED |

**ADMM Performance (Top 5):**

| Method | Time (s) | Speedup | Iterations | Status |
|--------|----------|---------|------------|---------|
| Original ADMM + Anderson only | 7.32 | **6.49x** | 474 | OPTIMAL |
| Original ADMM (baseline) | 8.21 | 5.78x | 1949 | OPTIMAL |
| Original ADMM + Halpern only | 14.97 | 3.17x | 3839 | OPTIMAL |
| Original ADMM + RB Adapter | 21.57 | 2.20x | 968 | OPTIMAL |
| Original ADMM + RB + Halpern | 25.05 | 1.89x | 1873 | OPTIMAL |

## Multi-Scale Performance Summary

| Scale | JuMP Reference | Best ADMM Method | ADMM Time | Speedup | Status |
|-------|----------------|------------------|-----------|---------|---------|
| Small (500×250) | MadNLP: 7.89s | Original ADMM + RB + Halpern | 0.08s | 96.62x | OPTIMAL |
| Medium (1000×500) | MadNLP: 0.43s | Original ADMM + Anderson only | 0.13s | 3.23x | OPTIMAL |
| Large (4000×2000) | MadNLP: 47.47s | Original ADMM + Anderson only | 7.32s | 6.49x | OPTIMAL |

