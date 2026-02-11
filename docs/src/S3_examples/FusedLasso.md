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

## Performance Results

### Small Scale (500 × 250)

**ADMM Performance (Top 5):**

| Method | Time (s) | Iterations | Status |
|--------|----------|------------|---------|
| Original ADMM + RB + Halpern | 0.08 |856 | OPTIMAL |
| Original ADMM + RB Adapter | 0.10 | 454 | OPTIMAL |
| Original ADMM + SRA + Halpern | 0.11| 63 | OPTIMAL |
| Original ADMM + SRA Adapter | 0.15 | 63 | OPTIMAL |
| Original ADMM + Anderson only | 0.23 | 159 | OPTIMAL |

### Medium Scale (1000 × 500)

**ADMM Performance (Top 5):**

| Method | Time (s) | Iterations | Status |
|--------|----------|------------|---------|
| Original ADMM + Anderson only | 0.13 |  229 | OPTIMAL |
| Original ADMM (baseline) | 0.13 | 880 | OPTIMAL |
| Original ADMM + Halpern only | 0.28 | 1707 | OPTIMAL |
| Original ADMM + RB Adapter | 0.45  | 867 | OPTIMAL |
| Original ADMM + RB + Halpern | 0.60 | 1681 | OPTIMAL |

### Large Scale (4000 × 2000)

**ADMM Performance (Top 5):**

| Method | Time (s) | Iterations | Status |
|--------|----------|------------|---------|
| Original ADMM + Anderson only | 7.32 |  474 | OPTIMAL |
| Original ADMM (baseline) | 8.21 | 1949 | OPTIMAL |
| Original ADMM + Halpern only | 14.97 |  3839 | OPTIMAL |
| Original ADMM + RB Adapter | 21.57 |  968 | OPTIMAL |
| Original ADMM + RB + Halpern | 25.05 |  1873 | OPTIMAL |
