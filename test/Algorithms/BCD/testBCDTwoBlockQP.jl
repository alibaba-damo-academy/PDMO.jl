"""
Test BCD Algorithm with Two-Block Quadratic Programming Problem

This test demonstrates how to set up and solve a two-block quadratic programming problem
using the Block Coordinate Descent (BCD) algorithm in PDMO.jl.

Problem formulation:
    min_{x₁,x₂} (1/2) * [x₁; x₂]ᵀ Q [x₁; x₂] + qᵀ [x₁; x₂] + f₁(x₁) + f₂(x₂)
    subject to: g₁(x₁) ∈ [-2,2]², g₂(x₂) ∈ [-1,3]³

Where:
- x₁ ∈ ℝ² (first block)
- x₂ ∈ ℝ³ (second block) 
- Q is a 5×5 positive definite coupling matrix
- q is a linear coefficient vector
- f₁, f₂ are individual quadratic block functions
- g₁, g₂ are box constraint indicator functions
"""

using Test
using PDMO
using LinearAlgebra
using SparseArrays
using Random

@testset "BCD Two-Block Quadratic Programming Tests" begin
    
    # Set random seed for reproducibility
    Random.seed!(42)
    
    @testset "Problem Setup and Construction" begin
        # Block dimensions
        block_dims = [2, 3]  # x₁ ∈ ℝ², x₂ ∈ ℝ³
        total_dim = sum(block_dims)
        
        @test length(block_dims) == 2
        @test total_dim == 5
        
        # Create a positive semidefinite coupling matrix Q
        Q_base = randn(total_dim, total_dim)
        Q = Q_base' * Q_base + 0.1 * I  # Ensure positive definiteness
        Q = (Q + Q') / 2  # Ensure symmetry
        
        # Linear coefficient vector
        q = randn(total_dim)
        r = 0.0
        
        @test size(Q) == (total_dim, total_dim)
        @test length(q) == total_dim
        @test isapprox(Q, Q')  # Check symmetry
        @test all(eigvals(Q) .> 0)  # Check positive definiteness
        
        # Create the quadratic multiblock coupling function
        coupling_function = QuadraticMultiblockFunction(Q, q, r, block_dims)
        @test isa(coupling_function, QuadraticMultiblockFunction)
        @test getNumberOfBlocks(coupling_function) == 2
        
        # Create MultiblockProblem
        mbp = MultiblockProblem()
        mbp.couplingFunction = coupling_function
        @test isa(mbp.couplingFunction, QuadraticMultiblockFunction)
    end
    
    @testset "Block Variables Creation" begin
        # Recreate problem for this test
        block_dims = [2, 3]
        total_dim = sum(block_dims)
        Q_base = randn(total_dim, total_dim)
        Q = Q_base' * Q_base + 0.1 * I
        Q = (Q + Q') / 2
        q = randn(total_dim)
        r = 0.0
        coupling_function = QuadraticMultiblockFunction(Q, q, r, block_dims)
        mbp = MultiblockProblem()
        mbp.couplingFunction = coupling_function
        
        # Block 1: x₁ ∈ ℝ² with quadratic individual function and box constraints
        block1 = BlockVariable("x1")
        H1 = [1.0 0.0; 0.0 2.0]  # Individual Hessian for block 1
        c1 = [0.5, -0.5]         # Individual linear coefficients for block 1
        block1.f = QuadraticFunction(sparse(H1), c1, 0.0)
        block1.g = IndicatorBox([-2.0, -2.0], [2.0, 2.0])
        block1.val = [0.1, -0.1]
        
        @test isa(block1.f, QuadraticFunction)
        @test isa(block1.g, IndicatorBox)
        @test length(block1.val) == block_dims[1]
        
        # Block 2: x₂ ∈ ℝ³ with quadratic individual function and box constraints  
        block2 = BlockVariable("x2")
        H2 = [1.5 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 2.5]  # Individual Hessian for block 2
        c2 = [0.2, -0.3, 0.1]                          # Individual linear coefficients for block 2
        block2.f = QuadraticFunction(sparse(H2), c2, 0.0)
        block2.g = IndicatorBox([-1.0, -1.0, -1.0], [3.0, 3.0, 3.0])
        block2.val = [0.5, 0.2, -0.3]
        
        @test isa(block2.f, QuadraticFunction)
        @test isa(block2.g, IndicatorBox)
        @test length(block2.val) == block_dims[2]
        
        # Add blocks to the multiblock problem
        addBlockVariable!(mbp, block1)
        addBlockVariable!(mbp, block2)
        
        @test length(mbp.blocks) == 2
        @test mbp.blocks[1].id == "x1"
        @test mbp.blocks[2].id == "x2"
    end
    
    @testset "BCD Algorithm with BCDProximalLinearSubproblemSolver" begin
        # Setup complete problem
        block_dims = [2, 3]
        total_dim = sum(block_dims)
        Random.seed!(42)  # Ensure reproducibility
        
        Q_base = randn(total_dim, total_dim)
        Q = Q_base' * Q_base + 0.1 * I
        Q = (Q + Q') / 2
        q = randn(total_dim)
        r = 0.0
        
        coupling_function = QuadraticMultiblockFunction(Q, q, r, block_dims)
        mbp = MultiblockProblem()
        mbp.couplingFunction = coupling_function
        
        # Block 1
        block1 = BlockVariable("x1")
        H1 = [1.0 0.0; 0.0 2.0]
        c1 = [0.5, -0.5]
        block1.f = QuadraticFunction(sparse(H1), c1, 0.0)
        block1.g = IndicatorBox([-2.0, -2.0], [2.0, 2.0])
        block1.val = [0.1, -0.1]
        
        # Block 2
        block2 = BlockVariable("x2")
        H2 = [1.5 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 2.5]
        c2 = [0.2, -0.3, 0.1]
        block2.f = QuadraticFunction(sparse(H2), c2, 0.0)
        block2.g = IndicatorBox([-1.0, -1.0, -1.0], [3.0, 3.0, 3.0])
        block2.val = [0.5, 0.2, -0.3]
        
        addBlockVariable!(mbp, block1)
        addBlockVariable!(mbp, block2)
        
        # Calculate initial objective
        initial_solution = Vector{Union{Float64, AbstractArray{Float64}}}([block1.val, block2.val])
        initial_obj = coupling_function(initial_solution) + 
                      block1.f(block1.val) + block1.g(block1.val) +
                      block2.f(block2.val) + block2.g(block2.val)
        
        @test isfinite(initial_obj)
        
        # Configure BCD Algorithm with BCDProximalLinearSubproblemSolver
        param = BCDParam(
            blockOrderRule = CyclicRule(),
            solver = BCDProximalLinearSubproblemSolver(),
            dresTolL2 = 1e-4,      # Relaxed tolerance for testing
            dresTolLInf = 1e-4,    # Relaxed tolerance for testing
            maxIter = 100,         # Reduced iterations for testing
            timeLimit = 30.0,      # Reduced time limit for testing
            logInterval = 50,       # Less frequent logging
            logLevel = 0
        )
        
        @test isa(param.solver, BCDProximalLinearSubproblemSolver)
        @test param.maxIter == 100
        
        # Run BCD Algorithm
        result_info = runBCD(mbp, param)
        
        # Test algorithm completion
        @test isa(result_info.iterationInfo.terminationStatus, PDMO.BCDTerminationStatus)
        @test result_info.iterationInfo.terminationStatus != PDMO.BCD_TERMINATION_UNSPECIFIED || 
              length(result_info.iterationInfo.obj) >= 2  # Either converged or made progress
        
        # Test solution structure
        @test haskey(result_info.solution, "x1")
        @test haskey(result_info.solution, "x2")
        @test length(result_info.solution["x1"]) == 2
        @test length(result_info.solution["x2"]) == 3
        
        # Test constraint satisfaction
        x1_final = result_info.solution["x1"]
        x2_final = result_info.solution["x2"]
        
        @test all(-2.0 .<= x1_final .<= 2.0)  # Block 1 box constraints
        @test all(-1.0 .<= x2_final .<= 3.0)  # Block 2 box constraints
        
        # Test objective improvement (or at least no degradation for short runs)
        if length(result_info.iterationInfo.obj) > 1
            final_obj = result_info.iterationInfo.obj[end]
            @test final_obj <= initial_obj + 1e-10  # Allow small numerical errors
        end
        
        # Test iteration info structure
        @test length(result_info.iterationInfo.obj) >= 1
        @test result_info.iterationInfo.totalTime >= 0.0
    end
    
    @testset "BCD Algorithm with BCDProximalSubproblemSolver" begin
        # Setup complete problem (same as above)
        block_dims = [2, 3]
        total_dim = sum(block_dims)
        Random.seed!(42)  # Ensure reproducibility
        
        Q_base = randn(total_dim, total_dim)
        Q = Q_base' * Q_base + 0.1 * I
        Q = (Q + Q') / 2
        q = randn(total_dim)
        r = 0.0
        
        coupling_function = QuadraticMultiblockFunction(Q, q, r, block_dims)
        mbp = MultiblockProblem()
        mbp.couplingFunction = coupling_function
        
        # Block 1
        block1 = BlockVariable("x1")
        H1 = [1.0 0.0; 0.0 2.0]
        c1 = [0.5, -0.5]
        block1.f = QuadraticFunction(sparse(H1), c1, 0.0)
        block1.g = IndicatorBox([-2.0, -2.0], [2.0, 2.0])
        block1.val = [0.1, -0.1]
        
        # Block 2
        block2 = BlockVariable("x2")
        H2 = [1.5 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 2.5]
        c2 = [0.2, -0.3, 0.1]
        block2.f = QuadraticFunction(sparse(H2), c2, 0.0)
        block2.g = IndicatorBox([-1.0, -1.0, -1.0], [3.0, 3.0, 3.0])
        block2.val = [0.5, 0.2, -0.3]
        
        addBlockVariable!(mbp, block1)
        addBlockVariable!(mbp, block2)
        
        # Configure BCD Algorithm with BCDProximalSubproblemSolver
        param = BCDParam(
            blockOrderRule = CyclicRule(),
            solver = BCDProximalSubproblemSolver(originalSubproblem = false),
            dresTolL2 = 1e-4,      # Relaxed tolerance for testing
            dresTolLInf = 1e-4,    # Relaxed tolerance for testing
            maxIter = 50,          # Reduced iterations for testing
            timeLimit = 30.0,      # Reduced time limit for testing
            logInterval = 25       # Less frequent logging
        )
        
        @test isa(param.solver, BCDProximalSubproblemSolver)
        
        # Run BCD Algorithm
        result_info = runBCD(mbp, param)
        
        # Test algorithm completion
        @test isa(result_info.iterationInfo.terminationStatus, PDMO.BCDTerminationStatus)
        
        # Test solution structure
        @test haskey(result_info.solution, "x1")
        @test haskey(result_info.solution, "x2")
        @test length(result_info.solution["x1"]) == 2
        @test length(result_info.solution["x2"]) == 3
        
        # Test constraint satisfaction
        x1_final = result_info.solution["x1"]
        x2_final = result_info.solution["x2"]
        
        @test all(-2.0 .<= x1_final .<= 2.0)  # Block 1 box constraints
        @test all(-1.0 .<= x2_final .<= 3.0)  # Block 2 box constraints
    end
    
    @testset "Solver Comparison" begin
        # Test that both solvers can handle the same problem
        # This is more of an integration test to ensure both solvers work
        
        block_dims = [2, 2]  # Simplified for faster testing
        total_dim = sum(block_dims)
        Random.seed!(123)
        
        Q_base = randn(total_dim, total_dim)
        Q = Q_base' * Q_base + 0.5 * I  # More conditioning
        Q = (Q + Q') / 2
        q = randn(total_dim)
        r = 0.0
        
        coupling_function = QuadraticMultiblockFunction(Q, q, r, block_dims)
        
        # Create two identical problems
        mbp1 = MultiblockProblem()
        mbp1.couplingFunction = coupling_function
        
        mbp2 = MultiblockProblem()
        mbp2.couplingFunction = coupling_function
        
        # Add identical blocks to both problems
        for mbp in [mbp1, mbp2]
            block1 = BlockVariable("x1")
            block1.f = QuadraticFunction(sparse([1.0 0.0; 0.0 1.0]), [0.0, 0.0], 0.0)
            block1.g = IndicatorBox([-1.0, -1.0], [1.0, 1.0])
            block1.val = [0.0, 0.0]
            
            block2 = BlockVariable("x2")
            block2.f = QuadraticFunction(sparse([1.0 0.0; 0.0 1.0]), [0.0, 0.0], 0.0)
            block2.g = IndicatorBox([-1.0, -1.0], [1.0, 1.0])
            block2.val = [0.0, 0.0]
            
            addBlockVariable!(mbp, block1)
            addBlockVariable!(mbp, block2)
        end
        
        # Test both solvers
        solvers = [
            BCDProximalLinearSubproblemSolver(),
            BCDProximalSubproblemSolver(originalSubproblem = false)
        ]
        
        results = []
        for (i, solver) in enumerate(solvers)
            param = BCDParam(
                blockOrderRule = CyclicRule(),
                solver = solver,
                dresTolL2 = 1e-3,
                dresTolLInf = 1e-3,
                maxIter = 20,
                timeLimit = 10.0,
                logInterval = 10
            )
            
            mbp = i == 1 ? mbp1 : mbp2
            result = runBCD(mbp, param)
            push!(results, result)
            
            # Basic sanity checks for each solver
            @test haskey(result.solution, "x1")
            @test haskey(result.solution, "x2")
            @test all(-1.0 .<= result.solution["x1"] .<= 1.0)
            @test all(-1.0 .<= result.solution["x2"] .<= 1.0)
        end
        
        @test length(results) == 2
    end
    
    @testset "BCD Three-Block Quadratic Programming Tests" begin
        # Set random seed for reproducibility
        Random.seed!(456)
        
        # Three-block problem: x₁ ∈ ℝ², x₂ ∈ ℝ³, x₃ ∈ ℝ²
        block_dims = [2, 3, 2]
        total_dim = sum(block_dims)  # 7 total dimensions
        
        @testset "Three-Block Problem Setup" begin
            @test length(block_dims) == 3
            @test total_dim == 7
            
            # Create a well-conditioned positive definite coupling matrix
            Q_base = randn(total_dim, total_dim)
            Q = Q_base' * Q_base + 0.2 * I  # Well-conditioned
            Q = (Q + Q') / 2  # Ensure symmetry
            
            # Linear coefficient vector
            q = randn(total_dim)
            r = 0.0
            
            @test size(Q) == (total_dim, total_dim)
            @test length(q) == total_dim
            @test isapprox(Q, Q')  # Check symmetry
            @test all(eigvals(Q) .> 0)  # Check positive definiteness
            
            # Create the quadratic multiblock coupling function
            coupling_function = QuadraticMultiblockFunction(Q, q, r, block_dims)
            @test isa(coupling_function, QuadraticMultiblockFunction)
            @test getNumberOfBlocks(coupling_function) == 3
        end
        
        @testset "Three-Block BCD with BCDProximalLinearSubproblemSolver" begin
            # Setup complete three-block problem
            block_dims = [2, 3, 2]
            total_dim = sum(block_dims)
            Random.seed!(456)  # Ensure reproducibility
            
            Q_base = randn(total_dim, total_dim)
            Q = Q_base' * Q_base + 0.3 * I  # Well-conditioned
            Q = (Q + Q') / 2
            q = randn(total_dim)
            r = 0.0
            
            coupling_function = QuadraticMultiblockFunction(Q, q, r, block_dims)
            mbp = MultiblockProblem()
            mbp.couplingFunction = coupling_function
            
            # Block 1: x₁ ∈ ℝ²
            block1 = BlockVariable("x1")
            H1 = [1.2 0.1; 0.1 1.5]  # Individual Hessian for block 1
            c1 = [0.3, -0.2]         # Individual linear coefficients
            block1.f = QuadraticFunction(sparse(H1), c1, 0.0)
            block1.g = IndicatorBox([-1.5, -1.5], [1.5, 1.5])  # Box constraints
            block1.val = [0.1, -0.1]
            
            # Block 2: x₂ ∈ ℝ³  
            block2 = BlockVariable("x2")
            H2 = [1.0 0.0 0.0; 0.0 1.3 0.1; 0.0 0.1 1.1]  # Individual Hessian
            c2 = [0.1, -0.4, 0.2]                          # Individual linear coefficients
            block2.f = QuadraticFunction(sparse(H2), c2, 0.0)
            block2.g = IndicatorBox([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0])  # Box constraints
            block2.val = [0.2, 0.1, -0.2]
            
            # Block 3: x₃ ∈ ℝ²
            block3 = BlockVariable("x3")
            H3 = [1.4 0.0; 0.0 1.6]  # Individual Hessian for block 3
            c3 = [-0.1, 0.3]         # Individual linear coefficients
            block3.f = QuadraticFunction(sparse(H3), c3, 0.0)
            block3.g = IndicatorBox([-1.0, -1.0], [1.0, 1.0])  # Tighter box constraints
            block3.val = [0.0, 0.0]
            
            addBlockVariable!(mbp, block1)
            addBlockVariable!(mbp, block2)
            addBlockVariable!(mbp, block3)
            
            # Calculate initial objective
            initial_solution = Vector{Union{Float64, AbstractArray{Float64}}}([block1.val, block2.val, block3.val])
            initial_obj = coupling_function(initial_solution) + 
                          block1.f(block1.val) + block1.g(block1.val) +
                          block2.f(block2.val) + block2.g(block2.val) +
                          block3.f(block3.val) + block3.g(block3.val)
            
            @test isfinite(initial_obj)
            
            # Configure BCD Algorithm with BCDProximalLinearSubproblemSolver
            param = BCDParam(
                blockOrderRule = CyclicRule(),
                solver = BCDProximalLinearSubproblemSolver(),
                dresTolL2 = 1e-4,      # Relaxed tolerance for testing
                dresTolLInf = 1e-4,    # Relaxed tolerance for testing
                maxIter = 150,         # More iterations for 3-block problem
                timeLimit = 60.0,      # Longer time limit
                logInterval = 50       # Less frequent logging
            )
            
            @test isa(param.solver, BCDProximalLinearSubproblemSolver)
            @test param.maxIter == 150
            
            # Run BCD Algorithm
            result_info = runBCD(mbp, param)
            
            # Test algorithm completion
            @test isa(result_info.iterationInfo.terminationStatus, PDMO.BCDTerminationStatus)
            @test result_info.iterationInfo.terminationStatus != PDMO.BCD_TERMINATION_UNSPECIFIED || 
                  length(result_info.iterationInfo.obj) >= 2  # Either converged or made progress
            
            # Test solution structure
            @test haskey(result_info.solution, "x1")
            @test haskey(result_info.solution, "x2") 
            @test haskey(result_info.solution, "x3")
            @test length(result_info.solution["x1"]) == 2
            @test length(result_info.solution["x2"]) == 3
            @test length(result_info.solution["x3"]) == 2
            
            # Test constraint satisfaction
            x1_final = result_info.solution["x1"]
            x2_final = result_info.solution["x2"]
            x3_final = result_info.solution["x3"]
            
            @test all(-1.5 .<= x1_final .<= 1.5)  # Block 1 box constraints
            @test all(-2.0 .<= x2_final .<= 2.0)  # Block 2 box constraints  
            @test all(-1.0 .<= x3_final .<= 1.0)  # Block 3 box constraints
            
            # Test objective improvement
            if length(result_info.iterationInfo.obj) > 1
                final_obj = result_info.iterationInfo.obj[end]
                @test final_obj <= initial_obj + 1e-10  # Allow small numerical errors
            end
            
            # Test iteration info structure
            @test length(result_info.iterationInfo.obj) >= 1
            @test result_info.iterationInfo.totalTime >= 0.0
        end
        
        @testset "Three-Block BCD with BCDProximalSubproblemSolver" begin
            # Setup the same three-block problem
            block_dims = [2, 3, 2]
            total_dim = sum(block_dims)
            Random.seed!(456)  # Same seed for comparison
            
            Q_base = randn(total_dim, total_dim)
            Q = Q_base' * Q_base + 0.3 * I
            Q = (Q + Q') / 2
            q = randn(total_dim)
            r = 0.0
            
            coupling_function = QuadraticMultiblockFunction(Q, q, r, block_dims)
            mbp = MultiblockProblem()
            mbp.couplingFunction = coupling_function
            
            # Same blocks as above
            block1 = BlockVariable("x1")
            H1 = [1.2 0.1; 0.1 1.5]
            c1 = [0.3, -0.2]
            block1.f = QuadraticFunction(sparse(H1), c1, 0.0)
            block1.g = IndicatorBox([-1.5, -1.5], [1.5, 1.5])
            block1.val = [0.1, -0.1]
            
            block2 = BlockVariable("x2")
            H2 = [1.0 0.0 0.0; 0.0 1.3 0.1; 0.0 0.1 1.1]
            c2 = [0.1, -0.4, 0.2]
            block2.f = QuadraticFunction(sparse(H2), c2, 0.0)
            block2.g = IndicatorBox([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0])
            block2.val = [0.2, 0.1, -0.2]
            
            block3 = BlockVariable("x3")
            H3 = [1.4 0.0; 0.0 1.6]
            c3 = [-0.1, 0.3]
            block3.f = QuadraticFunction(sparse(H3), c3, 0.0)
            block3.g = IndicatorBox([-1.0, -1.0], [1.0, 1.0])
            block3.val = [0.0, 0.0]
            
            addBlockVariable!(mbp, block1)
            addBlockVariable!(mbp, block2)
            addBlockVariable!(mbp, block3)
            
            # Configure BCD Algorithm with BCDProximalSubproblemSolver
            param = BCDParam(
                blockOrderRule = CyclicRule(),
                solver = BCDProximalSubproblemSolver(originalSubproblem = false),
                dresTolL2 = 1e-4,      # Relaxed tolerance for testing
                dresTolLInf = 1e-4,    # Relaxed tolerance for testing
                maxIter = 100,         # Fewer iterations for JuMP-based solver
                timeLimit = 60.0,      # Longer time limit
                logInterval = 25,       # Less frequent logging
                logLevel = 0

            )
            
            @test isa(param.solver, BCDProximalSubproblemSolver)
            
            # Run BCD Algorithm
            result_info = runBCD(mbp, param)
            
            # Test algorithm completion
            @test isa(result_info.iterationInfo.terminationStatus, PDMO.BCDTerminationStatus)
            
            # Test solution structure
            @test haskey(result_info.solution, "x1")
            @test haskey(result_info.solution, "x2")
            @test haskey(result_info.solution, "x3")
            @test length(result_info.solution["x1"]) == 2
            @test length(result_info.solution["x2"]) == 3
            @test length(result_info.solution["x3"]) == 2
            
            # Test constraint satisfaction
            x1_final = result_info.solution["x1"]
            x2_final = result_info.solution["x2"]
            x3_final = result_info.solution["x3"]
            
            @test all(-1.5 .<= x1_final .<= 1.5)  # Block 1 box constraints
            @test all(-2.0 .<= x2_final .<= 2.0)  # Block 2 box constraints
            @test all(-1.0 .<= x3_final .<= 1.0)  # Block 3 box constraints
        end
        
        @testset "Three-Block Problem Scalability" begin
            # Test with a slightly larger problem to ensure scalability
            block_dims = [3, 4, 3]  # 10 total dimensions
            total_dim = sum(block_dims)
            Random.seed!(789)
            
            Q_base = randn(total_dim, total_dim)
            Q = Q_base' * Q_base + 0.5 * I  # Well-conditioned
            Q = (Q + Q') / 2
            q = randn(total_dim)
            r = 0.0
            
            coupling_function = QuadraticMultiblockFunction(Q, q, r, block_dims)
            mbp = MultiblockProblem()
            mbp.couplingFunction = coupling_function
            
            # Create blocks with simple individual functions
            for i in 1:3
                block = BlockVariable("x$i")
                dim = block_dims[i]
                
                # Simple individual quadratic function
                H = Matrix(1.0I, dim, dim)  # Identity matrix
                c = zeros(dim)
                block.f = QuadraticFunction(sparse(H), c, 0.0)
                
                # Box constraints
                block.g = IndicatorBox(-ones(dim), ones(dim))
                block.val = 0.1 * randn(dim)  # Random initial point
                
                addBlockVariable!(mbp, block)
            end
            
            # Test with BCDProximalLinearSubproblemSolver for scalability
            param = BCDParam(
                blockOrderRule = CyclicRule(),
                solver = BCDProximalLinearSubproblemSolver(),
                dresTolL2 = 1e-3,      # Slightly relaxed for larger problem
                dresTolLInf = 1e-3,
                maxIter = 50,          # Fewer iterations for scalability test
                timeLimit = 30.0,
                logInterval = 25, 
                logLevel = 0
            )
            
            # Run BCD Algorithm
            result_info = runBCD(mbp, param)
            
            # Basic tests for scalability
            @test isa(result_info.iterationInfo.terminationStatus, PDMO.BCDTerminationStatus)
            @test haskey(result_info.solution, "x1")
            @test haskey(result_info.solution, "x2")
            @test haskey(result_info.solution, "x3")
            @test length(result_info.solution["x1"]) == 3
            @test length(result_info.solution["x2"]) == 4
            @test length(result_info.solution["x3"]) == 3
            
            # Test constraint satisfaction
            for i in 1:3
                x_final = result_info.solution["x$i"]
                @test all(-1.0 .<= x_final .<= 1.0)  # Box constraints
            end
        end

        @testset "Solver Reuse Reinitialization" begin
            function build_small_bcd_problem(block_dims::Vector{Int}, seed::Int)
                Random.seed!(seed)
                total_dim = sum(block_dims)
                Q_base = randn(total_dim, total_dim)
                Q = Q_base' * Q_base + 0.2 * I
                Q = (Q + Q') / 2
                q = randn(total_dim)
                coupling_function = QuadraticMultiblockFunction(Q, q, 0.0, block_dims)

                mbp = MultiblockProblem()
                mbp.couplingFunction = coupling_function
                for i in 1:length(block_dims)
                    dim = block_dims[i]
                    block = BlockVariable("x$i")
                    block.f = QuadraticFunction(sparse(Matrix(1.0I, dim, dim)), zeros(dim), 0.0)
                    block.g = IndicatorBox(-ones(dim), ones(dim))
                    block.val = 0.1 * randn(dim)
                    addBlockVariable!(mbp, block)
                end
                return mbp
            end

            # Reuse the same linearized solver across problems with different block counts.
            linear_solver = BCDProximalLinearSubproblemSolver()
            param_linear = BCDParam(
                blockOrderRule = CyclicRule(),
                solver = linear_solver,
                dresTolL2 = 1e-3,
                dresTolLInf = 1e-3,
                maxIter = 10,
                timeLimit = 10.0,
                logInterval = 20,
                logLevel = 0
            )
            result_a = runBCD(build_small_bcd_problem([2, 2], 901), param_linear; tryJuMP=false)
            @test length(linear_solver.models) == 2
            @test result_a !== nothing
            result_b = runBCD(build_small_bcd_problem([2, 2, 2], 902), param_linear; tryJuMP=false)
            @test length(linear_solver.models) == 3
            @test result_b !== nothing

            # Reuse the same JuMP-based solver across problems with different block counts.
            proximal_solver = BCDProximalSubproblemSolver(originalSubproblem=false)
            param_prox = BCDParam(
                blockOrderRule = CyclicRule(),
                solver = proximal_solver,
                dresTolL2 = 1e-3,
                dresTolLInf = 1e-3,
                maxIter = 5,
                timeLimit = 10.0,
                logInterval = 20,
                logLevel = 0
            )
            result_c = runBCD(build_small_bcd_problem([2, 2], 903), param_prox; tryJuMP=false)
            @test length(proximal_solver.models) == 2
            @test result_c !== nothing
            result_d = runBCD(build_small_bcd_problem([2, 2, 2], 904), param_prox; tryJuMP=false)
            @test length(proximal_solver.models) == 3
            @test result_d !== nothing
        end
    end
end
