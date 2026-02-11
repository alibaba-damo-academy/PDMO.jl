"""
    ADMMParam

Parameters for the Alternating Direction Method of Multipliers (ADMM) algorithm.

ADMM solves constrained optimization problems of the form:
```
minimize    f(x) + g(y)
subject to  Ax + By = c
```

The algorithm iteratively updates primal variables x, y and dual variable λ (Lagrange multiplier)
using the augmented Lagrangian with penalty parameter ρ.

# Fields

## Convergence Parameters
- `initialRho::Float64`: Initial penalty parameter ρ > 0. Controls the weight of the constraint 
  violation penalty in the augmented Lagrangian. Larger values enforce constraints more strictly 
  but may slow convergence. Default: 10.0

- `maxIter::Int64`: Maximum number of ADMM iterations before termination. Default: 100000

- `presTolL2::Float64`: Primal residual tolerance in L2 norm. Terminates when 
  ||Ax + By - c||₂ ≤ presTolL2. Default: 1e-4

- `dresTolL2::Float64`: Dual residual tolerance in L2 norm. Terminates when 
  ||ρAᵀ(λᵏ⁺¹ - λᵏ)||₂ ≤ dresTolL2. Default: 1e-4

- `presTolLInf::Float64`: Primal residual tolerance in L∞ norm. Alternative termination 
  criterion: ||Ax + By - c||∞ ≤ presTolLInf. Default: 1e-6

- `dresTolLInf::Float64`: Dual residual tolerance in L∞ norm. Alternative termination 
  criterion for dual residual. Default: 1e-6

## Algorithm Components
- `solver::AbstractADMMSubproblemSolver`: Strategy for solving the x and y subproblems:
  - `DoublyLinearizedSolver()`: Linearizes both subproblems for faster iterations
  - `OriginalADMMSubproblemSolver()`: Solves subproblems exactly (slower but more accurate)
  - `AdaptiveLinearizedSolver()`: Adaptive linearization with dynamic step size control
  Default: DoublyLinearizedSolver()

- `adapter::AbstractADMMAdapter`: Strategy for dynamically adjusting ρ during iterations:
  - `NullAdapter()`: Fixed ρ throughout iterations
  - `ResidualBalancingAdapter()`: Adjusts ρ to balance primal and dual residual magnitudes
  Default: NullAdapter()

- `accelerator::AbstractADMMAccelerator`: Acceleration scheme for faster convergence:
  - `NullAccelerator()`: Standard ADMM without acceleration
  - `AndersonAccelerator()`: Anderson acceleration using previous iterates
  - `AutoHalpernAccelerator()`: Automatic Halpern acceleration
  Default: NullAccelerator()

## Practical Settings
- `logInterval::Int64`: Print progress information every logInterval iterations. 
  Set to 0 to disable logging. Default: 1000

- `timeLimit::Float64`: Maximum wall-clock time in seconds. Algorithm terminates if 
  time limit is exceeded. Default: 3600.0 (1 hour)

- `applyScaling::Bool`: Whether to apply problem scaling for better numerical conditioning.
  Default: false

- `enablePathologyCheck::Bool`: Whether to enable checks for pathological behavior 
  (e.g., divergence, numerical instability). Default: false

# Constructor
The constructor `ADMMParam(; kwargs...)` creates a parameter set with default values that can be
customized via keyword arguments. All parameters are optional and have sensible defaults.

# Examples
```julia
# Default parameters
params = ADMMParam()

# Custom parameters via keyword arguments
params = ADMMParam(
    initialRho = 1.0,
    maxIter = 50000,
    presTolL2 = 1e-6,
    solver = OriginalADMMSubproblemSolver(),
    applyScaling = true
)

# Modify after construction
params = ADMMParam()
params.initialRho = 1.0
params.adapter = RBAdapter()
```
"""
mutable struct ADMMParam 
    initialRho::Float64
    maxIter::Int64 
    presTolL2::Float64 
    dresTolL2::Float64
    presTolLInf::Float64
    dresTolLInf::Float64
    solver::AbstractADMMSubproblemSolver
    adapter::AbstractADMMAdapter
    accelerator::AbstractADMMAccelerator
    logInterval::Int64
    timeLimit::Float64
    applyScaling::Bool
    enablePathologyCheck::Bool
    logLevel::Int64
    dualDescent::Bool 
    ADMMParam(; 
      initialRho::Float64 = 10.0, 
      maxIter::Int64 = 100000, 
      presTolL2::Float64 = 1e-4, 
      dresTolL2::Float64 = 1e-4, 
      presTolLInf::Float64 = 1e-6, 
      dresTolLInf::Float64 = 1e-6, 
      solver::AbstractADMMSubproblemSolver = DoublyLinearizedSolver(), 
      adapter::AbstractADMMAdapter = NullAdapter(), 
      accelerator::AbstractADMMAccelerator = NullAccelerator(), 
      logInterval::Int64 = 1000, 
      timeLimit::Float64 = 3600.0, 
      applyScaling::Bool = false, 
      enablePathologyCheck::Bool = false,
      logLevel::Int64 = 1, 
      dualDescent::Bool = false, 
    ) = new(
        initialRho, 
        maxIter, 
        presTolL2, 
        dresTolL2, 
        presTolLInf, 
        dresTolLInf, 
        solver, 
        adapter, 
        accelerator, 
        logInterval, 
        timeLimit, 
        applyScaling, 
        enablePathologyCheck,
        logLevel, 
        dualDescent)
end
