# Getting Started
Before official release, we recommend the following practice to download and use ```PDMO.jl```. 

## Installation
Download the project. From where ```PDMO.jl``` is located, run:
```bash 
julia PDMO.jl/warmup.jl
```
This one-time step will:
- Set up all required dependencies
- Configure HSL if available
- Report HSL detection status

### HSL Setup (Optional)
```PDMO.jl``` relies on [Ipopt.jl](https://github.com/jump-dev/Ipopt.jl) to solve certain ADMM subproblems. Linear solvers can significantly affect the performance of `Ipopt`. For enhanced performance, you can optionally use linear solvers from [HSL](https://www.hsl.rl.ac.uk). If `HSL` is available, linear solver `MA27` will be used for `Ipopt` by default.

**Default setup (HSL not required):** No additional setup needed. `PDMO.jl` will use Ipopt's default linear solver.

**Enhanced performance with HSL:** 
1. Obtain HSL library from [https://www.hsl.rl.ac.uk/](https://www.hsl.rl.ac.uk/)
2. Set up your HSL_jll directory with the following structure:
   ```
   HSL_jll/
   └── override/
       └── lib/
           └── {SYSTEM_ARCHITECTURE}/
               └── libhsl.{so|dylib}
   ```
   Example architectures: `x86_64-linux-gnu-libgfortran5`, `aarch64-apple-darwin-libgfortran5`

3. Edit `warmup.jl` and update the HSL path:
   ```julia
   # Uncomment and modify this line:
   HSL_PATH = "/path/to/your/HSL_jll"
   ```

After successful setup, ```PDMO.jl``` is ready for use:
```julia 
using PDMO
``` 

## Quick Start
### Dual Square Root LASSO
We use the Dual Square Root LASSO as a beginning example: 
```math
\begin{aligned}
    \min_{x, z}\quad & \langle b, x\rangle \\
    \mathrm{s.t.} \quad & Ax - z = 0 \\
    & \|x\|_2 \leq 1, \|z\|_{\infty} \leq \lambda,
\end{aligned}
```
where $(A, b, \lambda)$ are given problem data of proper dimensions. 

To begin with, load ```PDMO.jl``` and other necessary packages.
```julia
using PDMO
using LinearAlgebra
using SparseArrays
using Random 
```
Next generate or load your own problem data. We use synthetic data here. 
```julia
numberRows = 10 
numberColumns = 20 
A = sparse(randn(numberRows, numberColumns))
b = randn(numberColumns)
lambda = 1.0
```
Then we can generate a ```MultiblockProblem``` for the Dual Square Root LASSO problem.
```julia
mbp = MultiblockProblem()

# add x block
block_x = BlockVariable() 
block_x.f = AffineFunction(b, 0.0)    # f_1(x) = <b, x>
block_x.g = IndicatorBallL2(1.0)      # g_1(x) = indicator of L2 ball 
block_x.val = zeros(numberColumns)    # initial value
xID = addBlockVariable!(mbp, block_x) # add x block to mbp; an ID is assigned

# add z block 
block_z = BlockVariable()                              
block_z.g = IndicatorBox(-lambda * ones(numberRows), # f_2(z) = Zero() by default
    ones(numberRows) * lambda)                       # g_2(x) = indicator of box
block_z.val = zeros(numberRows)                      # initial value
zID = addBlockVariable!(mbp, block_z)                # add z block to mbp; an ID is assigned

# add constraint: Ax-z=0
constr = BlockConstraint() 
addBlockMappingToConstraint!(constr, xID, LinearMappingMatrix(A))      # specify the mapping of x
addBlockMappingToConstraint!(constr, zID, LinearMappingIdentity(-1.0)) # specify the mapping of z 
constr.rhs = zeros(numberRows)                                         # specify RHS
addBlockConstraint!(mbp, constr)                                       # add constraint to mbp
```
Next we can run different variants of ADMM: 
```julia 
# run ADMM 
param = ADMMParam() 
param.solver = OriginalADMMSubproblemSolver()
param.adapter = RBAdapter(testRatio=10.0, adapterRatio=2.0)
param.accelerator = AndersonAccelerator()
result = runBipartiteADMM(mbp, param)
```
```julia
# run Doubly Linearized ADMM
param = ADMMParam() 
param.solver = DoublyLinearizedSolver() 
result = runBipartiteADMM(mbp, param)
```
```julia
# run Adaptive Linearized ADMM
param = ADMMParam() 
param.solver = AdaptiveLinearizedSolver()
result = runBipartiteADMM(mbp, param)
```
or different adaptive primal-dual methods: 
```julia
# run AdaPDM 
paramAdaPDM = AdaPDMParam(mbp)
result = runAdaPDM(mbp, paramAdaPDM)
```
```julia
# run AdaPDMPlus 
paramAdaPDMPlus = AdaPDMPlusParam(mbp)
result = runAdaPDM(mbp, paramAdaPDMPlus)
```
```julia
# run Malitsky-Pock 
paramMalitskyPock = MalitskyPockParam(mbp)
result = runAdaPDM(mbp, paramMalitskyPock)
```
```julia
# run Condat-Vu 
paramCondatVu = CondatVuParam(mbp)
result = runAdaPDM(mbp, paramCondatVu)
```

Upon termination of the selected algorithm, one can look for primal solution and iteration information through `result.solution` and `result.iterationInfo`, respectively. 


## User Defined Smooth and Proximable Functions
In addition to a set of built-in functions whose gradient or proximal oracles have been implemented, `PDMO.jl` supports user defined smooth and proximable functions. Consider the function 
```math
    F(x) = x_1 + |x_2| + x_3^4, ~x = [x_1, x_2, x_3]^\top \in \R^3,
```
which can be expressed as the sum of a smooth $f$ and a proximable $g$: 
```math 
    f(x) = x_1 + x_3^4, \quad g(x) = |x_2|.
```
In `PDMO.jl`, this block can be constructed as follows:
```julia
block = BlockVariable()
block.f = UserDefinedSmoothFunction(
    x -> x[1] + x[3]^4,                  # f
    x -> [1.0, 0.0, 4*x[3]^3])           # ∇f
block.g = UserDefinedProximalFunction(
    x -> abs(x[2]),                      # g
    (x, gamma) -> [                      # prox_{gamma g} 
        x[1], 
        sign(x[2]) * max(abs(x[2]) - gamma, 0.0),
        x[3]])
block.val = zeros(3)                     # initial value
```