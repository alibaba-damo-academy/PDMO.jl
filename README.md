# Codes for *Automating Reformulation for Parallel ADMM*
This branch provides codes for numerical experiments in the paper *Automating Reformulation for Parallel ADMM* by Sun et al. (2026). To begin with, download the project folder and checkout the current branch. 
```
cd PDMO.jl 
git checkout test/reformulation 
cd ..
```
The from where `PDMO.jl` is located, run 
```
julia PDMO.jl/warmup.jl
```
and 
```
julia PDMO.jl/advanced/warmup.jl
```
for a one-tine compilation. 

## Section 1: Circuit Demo
Run the following commands to generate three plots in Fig. 1. 
```
julia PDMO.jl/applications/Demo/demo.jl --rho 1 --k 10000
julia PDMO.jl/applications/Demo/demo.jl --rho 10 --k 1000
julia PDMO.jl/applications/Demo/demo.jl --rho 100 --k 100
```
Plots will be saved under `PDMO.jl/applications/Demo/`. 

## Section 3.1 Linear Program: A Case Study 
Run the following command to generate plots in Section 3.1
```
julia PDMO.jl/applications/GenericLP/enlight_hard_demo.jl /PATH/TO/enlight_hard.mps.gz 
```

## Section 3.2 Linear Program: The Network Flow Problem 
We use the following script `PDMO.jl/advanced/src/NetworkFlow/runNetworkFlowProblem.jl` to produce results presented in Section 3.2. Example usage: 

```
julia -t 16 PDMO.jl/advanced/src/NetworkFlow/runNetworkFlowProblem.jl  --solver original --maxIter 100000 --initialRho 1.0 --timeLimit 3600.0 --seed 1 --logInterval 1000 --random 300 2000 
```
where 
- `--random <nodes> <arcs>`: Generate a random feasible instance
- `--solver <original|doubly|adaptive>`: ADMM subproblem solver (default: `original`)
- `--maxIter <int>`: ADMM max iterations (default: `10000`)
- `--initialRho <float>`: Initial rho (default: `10.0`)
- `--timeLimit <float>`: Max wall-clock time in seconds (default: `3600.0`)

The script used to generate Fig. 11 in the paper is available upon request. 

## Section 3.3 Distributed DC Optimal Power Flow
We use the script `PDMO.jl/advanced/src/OPF/runDistributedOPF.jl` to produce results presented in Section 3.3. Example usage: 
```
julia -t 16 PDMO.jl/advanced/src/OPF/runDistributedOPF.jl /PATH/TO/case30.m 3 original 100.0 
```
where the arguments are: 
- `ARGS[1]` -> `matpower_path` (required): Absolute path to the MATPOWER case file.
- `ARGS[2]` -> `numberPartitions` (default: `3`): Number of partitions.
- `ARGS[3]` -> `admmSolver` (default: `original`): ADMM solver variant.
- `ARGS[4]` -> `initialRho` (default: `10.0`): Initial penalty parameter.
- `ARGS[5]` -> `tol` (default: `1.0e-4`): Convergence tolerance.
- `ARGS[6]` -> `maxIter` (default: `1000000`): Maximum number of iterations.
- `ARGS[7]` -> `timeLimit` (default: `7200.0`): Time limit in seconds.
- `ARGS[8]` -> `logInterval` (default: `100`): Iteration interval for logging.
- `ARGS[9]` -> `r_value` (default: `1.0e4`): `r` parameter value.
- `ARGS[10]` -> `seed` (default: `126`): Random seed.

The script used to generate Fig. 13 and Fig. 14 in the paper is available upon request. 

## Section 3.4 Decentralized Consensus Optimization 
We use the script  `PDMO.jl/advanced/src/DistributedOpt/runDistributedOpt.jl` to produce results presented in Section 3.4. Example usage:
```
julia -t 16 PDMO.jl/advanced/src/DistributedOpt/runDistributedOpt.jl 100 500 250 original 10.0 100000 1000 111
```
where the arguments are 
- `args[1]` -> `numberNodes` (required): Number of nodes.
- `args[2]` -> `n` (required): Problem size parameter `n`.
- `args[3]` -> `m` (required): Problem size parameter `m`.
- `args[4]` -> `solver_name` (default: `original`): Solver variant.
- `args[5]` -> `initialRho` (default: `10.0`): Initial penalty parameter.
- `args[6]` -> `maxIter` (default: `100000`): Maximum ADMM iterations.
- `args[7]` -> `logInterval` (default: `1000`): Logging interval (in iterations).
- `args[8]` -> `seed` (default: `126`): Random seed.
- `args[9]` -> `mipRelGap` (default: `0.01`): Relative MIP optimality gap.
- `args[10]` -> `mipHeuristicEffort` (default: `0.2`): MIP heuristic effort.
- `args[11]` -> `mipTimeLimit` (default: `60.0`): Time limit for MIP solve (seconds).

The script used to generate Table 1, Fig. 15, and Fig. 16 in the paper is available upon request. 

## Other Notes
If you encounter the error `libgobject-2.0.so: undefined symbol: g_dir_unref`, plus Cairo/FFMPEG/Plots failing, try to run Julia without Conda’s environment variables.
```
conda deactivate
unset LD_LIBRARY_PATH
```

# PDMO.jl - **Primal-Dual Methods for Optimization**

## Overview
`PDMO.jl` is a powerful Julia framework for primal-dual multiblock optimization, built for **rapid prototyping** and **high-performance computing**.

- **Solve Complex Problems**: Model and solve problems with multiple variable blocks and linear coupling constraints. 
- **Highly Customizable**: An open-source toolkit that is easy to adapt for your applications and specific algorithms.
- **Accelerate Research**: Benchmark your methods against classic and state-of-the-art solvers.

## Problem Formulation
`PDMO.jl` presents a unified framework for formulating and solving a ```MultiblockProblem``` of the form: 

```math 
\begin{aligned}
\min_{\mathbf{x}} \quad & F(\mathbf{x}) +  \sum_{j=1}^n \left( f_j(x_j) + g_j(x_j) \right)\\ 
\mathrm{s.t.} \quad  & \mathbf{A} \mathbf{x} = \mathbf{b},
\end{aligned}
```
where we have the following problem variables and data:

```math
\begin{array}{ccc}
n~\textbf{Block Variables} \quad & m~\textbf{ Block Constraints} \quad & \textbf{Block Matrix}~ (m \times n ~ \textbf{linear operators}) \\
\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} \quad & \mathbf{b} = \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_m \end{bmatrix} \quad & \mathbf{A} = \begin{bmatrix} \mathbf{A}_{1,1} & \mathbf{A}_{1,2} & \cdots & \mathbf{A}_{1,n} \\ \mathbf{A}_{2,1} & \mathbf{A}_{2,2} & \cdots & \mathbf{A}_{2,n} \\ \vdots & \vdots & \ddots & \vdots \\ \mathbf{A}_{m,1} & \mathbf{A}_{m,2} & \cdots & \mathbf{A}_{m,n} \end{bmatrix} \\
\end{array}
```

More specifically, 
- For each $j = 1,\cdots,n$, a `BlockVariable` $x_j$ represents a numeric array (i.e., scalar, vector, matrix, etc.), and is associated with two objective functions: 
    - each $f_j$ is differentiable, and $f_j(\cdot)$ and $\nabla f_j(\cdot)$ are available; 
    - each $g_j$ is proximable, and $g_j(\cdot)$ and $\text{prox}_{\gamma g_j}(\cdot)$ are available.
- For each $i = 1,\cdots,m$, a `BlockConstraint` is defined by some linear operators and a right-hand side array: 
    - the linear operator $\mathbf{A}_{i,j}$ is **non-zero** if and only if constraint $i$ involves blocks $x_j$;
    - the adjoint operator of $\mathbf{A}_{i,j}$ is available;
    - the right-hand side $b_i$ can be a numeric array of any shape. 
- Additionally, there might exist a smooth function $F$ that couples all BlockVariables:
    - we assume that $F(\cdot)$, $\nabla F(\cdot)$,  and $\nabla_j F(\cdot)$'s are available.
    
## Available Algorithms

`PDMO.jl` provides various algorithms to solve problems of the above form.

- **Alternating Direction Method of Multipliers (ADMM)**
  - Graph-based bipartization methods automatically generate ADMM-ready reformulations of `MultiblockProblem` when $F=0$.
  - Various ADMM variants are available: 
    - Original ADMM 
    - Doubly linearized ADMM 
    - Adaptive linearized ADMM 
  - Various algorithmic component can be selected: 
    - Penalty adapters, e.g., Residual Balancing, Spectral Radius Approximation
    - Accelerators, e.g., Halpern (with or without restart), Filtered Anderson

- **Adaptive Primal-Dual Method (AdaPDM)**
  - A suite of efficient and adaptive methods for problems with simpler coupling.
  - Various methods can be selected : 
    - Original Condat-Vũ Method (Condat 2013, Vũ 2013)
    - Adaptive Primal-Dual Method & Plus (Latafat et al. 2024)
    - Malitsky-Pock Methd (Malitsky and Pock, 2018)
 - **Block Coordinate Descent (BCD)** 
    - A suite of classic methods for problems without constraints, i.e., $m=0$. 
    - Various subproblem solvers can be selected:
      - Original BCD Subproblem Solver
      - Proximal BCD Subproblem Solver
      - Prox-linear BCD Subproblem Solver
## Key Features 
- 🧱 **Unified Modeling**: A versatile interface for structured problems.
- 🔄 **Automatic Decomposition**: Intelligently analyzes and reformulates problems for supported algorithms.
- 🧩 **Extensible by Design**: Easily add custom functions, constraints, and algorithms.
- 📊 **Modular Solvers**: A rich library of classic and modern algorithms.
- ⚡  **Non-Convex Ready**: Equipped with features to tackle non-convexity.


## Installation
Before official release, we recommend the following practice to download and use ```PDMO.jl```. 

### Project Setup
Download the project. From where ```PDMO.jl``` is located, run:
```bash 
julia PDMO.jl/warmup.jl
```
For enhanced performance, you can optionally use linear solvers from [HSL](https://www.hsl.rl.ac.uk):

1. Obtain HSL library from [https://www.hsl.rl.ac.uk/](https://www.hsl.rl.ac.uk/)
2. Set up your HSL_jll directory structure
3. Edit `warmup.jl` and update the HSL path before running the above command

This will set up all required dependencies and configure HSL if available.

After successful setup, activate the project
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


### User Defined Smooth and Proximable Functions
In addition to a set of built-in functions whose gradient or proximal oracles have been implemented, `PDMO.jl` supports user-defined smooth and proximable functions. Consider the function 
```math
    F(x) = x_1 + |x_2| + x_3^4, ~x = [x_1, x_2, x_3]^\top \in \mathbb{R}^3,
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

## Documentation

For comprehensive documentation, examples, and API references, visit our [full documentation](https://alibaba-damo-academy.github.io/PDMO.jl).

## Roadmap (Work in Progress)
- 🔍 Classification and detection for pathological problems
- 🚀 Advanced acceleration techniques for first-order methods 
- 🤖 AI coding assistant for user-defined functions
- 🛣️ Parallel, distributed, and GPU support

## Contributing

`PDMO.jl` is open source and welcomes contributions! Please contact [**info@mindopt.tech**](mailto:info@mindopt.tech) for more details.

## License

`PDMO.jl` is licensed under the MIT License. 