# PDMO.jl - **Primal-Dual Methods for Optimization**

## Overview
`PDMO.jl` is a powerful Julia framework for primal-dual multiblock optimization, built for **rapid prototyping** and **high-performance computing**.

- **Solve Complex Problems**: Model and solve problems with multiple variable blocks and linear coupling constriants. 
- **Highly Customizable**: An open-source toolkit that is easy to adapt for your applications and specific algorithms.
- **Accelerate Research**: Benchmark your methods against classic and
state-of-the-art solvers.

### Problem Formulation
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
- For each $j\in \{1,\cdots,n\}$, a `BlockVariable` $x_j$ represents a numeric array (i.e., scalar, vector, matrix, etc.), and is associated with two objective functions: 
    - each $f_j$ is differentiable, and $f_j(\cdot)$ and $\nabla f_j(\cdot)$ are available; 
    - each $g_j$ is proximable, and $g_j(\cdot)$ and $\text{prox}_{\gamma g_j}(\cdot)$ are available.
- For each $i \in \{1,\cdots,m\}$, a `BlockConstraint` is defined by some linear operators and a right-hand side array: 
    - the linear operator $\mathbf{A}_{i,j}$ is **non-zero** if and only if constraint $i$ involves blocks $x_j$;
    - the adjoint operator of $\mathbf{A}_{i,j}$ is available;
    - the right-hand side $b_i$ can be a numeric array of any shape. 
- Additionally, there might exist a smooth function $F$ that couples all BlockVariables:
    - we assume that $F(\cdot)$, $\nabla F(\cdot)$,  and $\nabla_j F(\cdot)$'s for $j\in \{1,\cdots, n\}$ are available.

### Algorithms

`PDMO.jl` provides various algorithms to solve problems of the above form.

- **Alternating Direction Method of Multipliers (ADMM)**
  - Graph-based bipartization methods automatically generate ADMM-ready reformulations of `MultiblockProblem` when $F = 0$. 
  - Various ADMM variants are available: 
    - Original ADMM 
    - Doubly linearized ADMM 
    - Adaptive linearized ADMM 
  - Various algorithmic component can be selected: 
    - Penalty adapters, e.g., Residual Balancing, Spectral Radius Approximation
    - Accelerators, e.g., Halpern (with or without restart), Filtered Anderson

- **Adaptive Primal-Dual Method (AdaPDM)**
  - A suite of efficient and adaptive methods for problems with simpler coupling:
  ```math 
    \begin{aligned}
    \min_{\mathbf{x}} \quad & \sum_{j=1}^{n-1} \left( f_j(x_j) + g_j(x_j) \right) + g_n(\mathbf{A}_{1,1}x_1 + \cdots + \mathbf{A}_{1,n-1}x_{n-1})
    \end{aligned}
  ```
  - Various methods can be selected: 
    - Original Condat-Vu
    - AdaPDM 
    - AdaPDM+
    - Malitsky-Pock
    
- **Block Coordinate Descent (BCD)**
  - A suite of classic methods for problems without constraints, i.e., $m=0$.
  ```math 
    \begin{aligned}
    \min_{\mathbf{x}} \quad & F\mathbf(\mathbf{x})  + \sum_{j=1}^{n} \left( f_j(x_j) + g_j(x_j) \right)
    \end{aligned}
  ```
  - Various subproblem solvers can be selected: 
    - Original BCD Subproblem Solver
    - Proximal BCD Subproblem Solver
    - Prox-linear BCD Subproblem Solver

### Key Features 
- 🧱 **Unified Modeling**: A versatile interface for structured problems.
- 🔄 **Automatic Decomposition**: Intelligently analyzes and reformulates problems for supported algorithms.
- 🧩 **Extensible by Design**: Easily add custom functions, constraints, and algorithms.
- 📊 **Modular Solvers**: A rich library of classic and modern algorithms.
- ⚡  **Non-Convex Ready**: Equipped with features to tackle non-convexity.


#### Roadmap (Work in Progress)
- 🔍 Classification and detection for pathological problems
- 🚀 Advanced acceleration techniques for first-order methods 
- 🤖 AI coding assistant for user-defined functions
- 🛣️ Parallel, distributed, and GPU support.

### Use Cases
- For Researchers Developing New Methods:
  - A testbed for easily experimenting with your methods against existing ones 
  - A platform to track academic advances in first-order methods

- For Optimization Users:
  - An open-sourced producet that is higly competitive
  - Easy to use: unifying formulation, intuitive APIs, and comprehensive documentation
  - Modular design: minimal effort required to customize applications and algorithms

## This Documentation 

- Check out [**Getting Started**](S1_getting_started.md) for installation guide and your first optimization problem with ```PDMO.jl```.
- Learn more about the theoretical foundations of [**ADMM**](S2_algorithms/ADMM.md), [**AdaPDM**](S2_algorithms/AdaPDM.md) and [**BCD**](S2_algorithms/BCD.md), and how to explore different algorithmic components for better performance. 
- See [**Examples**](S3_examples/LeastL1Norm.md) for pre-defined templates of some classic applications and benchmark results.
- Check out [**API References**](S4_api/main.md) to implement and customize your own algorithms


## Contributing

```PDMO.jl``` is open source and welcomes contributions! Please contact [**info@mindopt.tech**](mailto:info@mindopt.tech) for more details.

## License
Currently ```PDMO.jl``` is under the MIT License.


