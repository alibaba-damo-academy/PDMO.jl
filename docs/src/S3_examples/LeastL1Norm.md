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

## Performance Results

### Small Scale (500 × 250)
**ADMM Performance (Top 5):**

| Method | Time (s) |  Iterations | Status |
|--------|----------|------------|--------|
| Original ADMM + Halpern only | 1.77 |6,350 | OPTIMAL |
| Original ADMM + RB + Halpern | 2.10 |  8,552 | OPTIMAL |
| Original ADMM + Anderson only | 2.46 | 3,186 | OPTIMAL |
| Original ADMM + RB Adapter | 4.07 |  16,697 | OPTIMAL |
| Original ADMM (baseline) | 6.66 | 21,774 | OPTIMAL |

### Medium Scale (1000 × 500)



**ADMM Performance (Top 5):**

| Method | Time (s) |  Iterations | Status |
|--------|----------|------------|--------|
| Original ADMM + RB + Halpern | 7.70 |  8,143 | OPTIMAL |
| Original ADMM + Halpern only | 11.79 | 12,412 | OPTIMAL |
| Original ADMM + Anderson only | 14.38 | 4,864 | OPTIMAL |
| Original ADMM + RB Adapter | 22.03 | 23,514 | OPTIMAL |
| Original ADMM (baseline) | 24.91 | 26,234 | OPTIMAL |

### Large Scale (2000 × 1000)

**ADMM Performance (Top 5):**

| Method | Time (s) | Speedup | Status |
|--------|----------|---------|--------|
| Original ADMM + RB Adapter | 54.91 |  9,742 | OPTIMAL |
| Original ADMM + RB + Halpern | 63.96 |  11,493 | OPTIMAL |
| Original ADMM + Anderson only | 159.97 |  8,127 | OPTIMAL |
| Original ADMM (baseline) | 282.29 |  51,226 | OPTIMAL |
| Original ADMM + Halpern only | 283.16 |  46,426 | OPTIMAL |
