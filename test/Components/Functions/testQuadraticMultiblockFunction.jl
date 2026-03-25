using Test
using PDMO
using LinearAlgebra
using SparseArrays
using JuMP
include("../../test_helper.jl")

@testset "QuadraticMultiblockFunction Tests" begin
    
    @testset "Constructor Tests" begin
        # Test basic construction
        Q = sparse([2.0 1.0 0.5; 1.0 3.0 0.2; 0.5 0.2 1.5])
        q = [1.0, 2.0, 0.5]
        r = 3.0
        blockDims = [2, 1]
        
        f = QuadraticMultiblockFunction(Q, q, r, blockDims)
        @test f isa QuadraticMultiblockFunction
        @test f isa AbstractMultiblockFunction
        @test f isa AbstractFunction
        @test f.Q == Q
        @test f.q == q
        @test f.r == r
        @test f.blockDims == blockDims
        @test length(f.buffer) == sum(blockDims)
        @test length(f.blockIndices) == length(blockDims)
        
        # Test precomputed block indices
        @test f.blockIndices[1] == 1:2
        @test f.blockIndices[2] == 3:3
        
        # Test dimension validation
        Q_wrong = sparse([2.0 1.0; 1.0 3.0])  # 2x2 matrix
        q_wrong = [1.0, 2.0, 0.5]  # 3-element vector
        blockDims_wrong = [2, 1]  # total = 3, but Q is 2x2
        @test_throws AssertionError QuadraticMultiblockFunction(Q_wrong, q_wrong, r, blockDims_wrong)
        
        # Test Q symmetry validation - should warn but not error (it gets symmetrized)
        Q_nonsym = sparse([2.0 1.0 0.5; 1.5 3.0 0.2; 0.5 0.2 1.5])  # Not symmetric
        f_nonsym = QuadraticMultiblockFunction(Q_nonsym, q, r, blockDims)  # Should work but warn
        @test f_nonsym isa QuadraticMultiblockFunction
        
        # Test empty block dimensions
        @test_throws AssertionError QuadraticMultiblockFunction(Q, q, r, Int[])
    end
    
    @testset "Type Hierarchy and Traits" begin
        Q = sparse([2.0 1.0; 1.0 3.0])
        q = [1.0, 2.0]
        r = 3.0
        blockDims = [1, 1]
        
        f = QuadraticMultiblockFunction(Q, q, r, blockDims)
        
        # Test inheritance
        @test f isa QuadraticMultiblockFunction
        @test f isa AbstractMultiblockFunction
        @test f isa AbstractFunction
        
        # Test traits
        @test isSmooth(f) == true
        @test isSmooth(QuadraticMultiblockFunction) == true
        @test isConvex(f) == false  # Default from AbstractMultiblockFunction
        @test isSupportedByJuMP(f) == true
        @test isSupportedByJuMP(QuadraticMultiblockFunction) == true
        
        # Test utility functions
        @test getNumberOfBlocks(f) == 2
        test_blocks = NumericVariable[[1.0], [2.0]]
        @test validateBlockDimensions(f, test_blocks) === nothing
    end
    
    @testset "Function Evaluation Tests" begin
        # Test case 1: Simple 2x2 case
        Q = sparse([2.0 1.0; 1.0 3.0])
        q = [1.0, 2.0]
        r = 5.0
        blockDims = [1, 1]
        
        f = QuadraticMultiblockFunction(Q, q, r, blockDims)
        x_blocks = NumericVariable[[2.0], [3.0]]  # x = [2, 3]
        
        # Manual calculation: f(x) = x'Qx + q'x + r
        x_concat = [2.0, 3.0]
        manual_result = dot(x_concat, Q * x_concat) + dot(q, x_concat) + r
        
        result = f(x_blocks)
        @test result ≈ manual_result
        
        # Test case 2: Larger block structure
        Q = sparse([2.0 1.0 0.5 0.0; 1.0 3.0 0.0 0.2; 0.5 0.0 1.5 0.1; 0.0 0.2 0.1 2.0])
        q = [1.0, 2.0, 0.5, 1.5]
        r = 10.0
        blockDims = [2, 2]
        
        f2 = QuadraticMultiblockFunction(Q, q, r, blockDims)
        x_blocks2 = NumericVariable[[1.0, 2.0], [3.0, 4.0]]
        x_concat2 = [1.0, 2.0, 3.0, 4.0]
        
        result2 = f2(x_blocks2)
        manual_result2 = dot(x_concat2, Q * x_concat2) + dot(q, x_concat2) + r
        @test result2 ≈ manual_result2
        
        # Test edge case: single block
        Q_single = sparse([2.0 1.0; 1.0 3.0])
        q_single = [1.0, 2.0]
        blockDims_single = [2]
        
        f3 = QuadraticMultiblockFunction(Q_single, q_single, r, blockDims_single)
        x_blocks3 = NumericVariable[[1.0, 2.0]]
        result3 = f3(x_blocks3)
        manual_result3 = dot([1.0, 2.0], Q_single * [1.0, 2.0]) + dot(q_single, [1.0, 2.0]) + r
        @test result3 ≈ manual_result3
    end
    
    @testset "Partial Gradient Tests" begin
        Q = sparse([2.0 1.0 0.5; 1.0 3.0 0.2; 0.5 0.2 1.5])
        q = [1.0, 2.0, 0.5]
        r = 3.0
        blockDims = [2, 1]
        
        f = QuadraticMultiblockFunction(Q, q, r, blockDims)
        x_blocks = NumericVariable[[1.0, 2.0], [3.0]]
        x_concat = [1.0, 2.0, 3.0]
        
        # Test partial gradient for block 1
        partial_grad_1 = similar(x_blocks[1], Float64)
        partialGradientOracle!(partial_grad_1, f, x_blocks, 1)
        
        # Manual calculation for block 1: ∇₁f = 2 * Q[1:2, :] * x + q[1:2]
        manual_partial_1 = 2 * (Q[1:2, :] * x_concat) + q[1:2]
        @test partial_grad_1 ≈ manual_partial_1
        
        # Test partial gradient for block 2
        partial_grad_2 = similar(x_blocks[2], Float64)
        partialGradientOracle!(partial_grad_2, f, x_blocks, 2)
        
        # Manual calculation for block 2: ∇₂f = 2 * Q[3:3, :] * x + q[3:3]
        manual_partial_2 = 2 * (Q[3:3, :] * x_concat) + q[3:3]
        @test partial_grad_2 ≈ manual_partial_2
        
        # Test non-mutating versions
        partial_grad_1_direct = partialGradientOracle(f, x_blocks, 1)
        partial_grad_2_direct = partialGradientOracle(f, x_blocks, 2)
        @test partial_grad_1_direct ≈ partial_grad_1
        @test partial_grad_2_direct ≈ partial_grad_2
        
        # Test out-of-bounds block index
        @test_throws AssertionError partialGradientOracle!(partial_grad_1, f, x_blocks, 0)
        @test_throws AssertionError partialGradientOracle!(partial_grad_1, f, x_blocks, 3)
    end
    
    @testset "Full Gradient Tests - All 4 Signatures" begin
        Q = sparse([2.0 1.0 0.5 0.0; 1.0 3.0 0.0 0.2; 0.5 0.0 1.5 0.1; 0.0 0.2 0.1 2.0])
        q = [1.0, 2.0, 0.5, 1.5]
        r = 3.0
        blockDims = [2, 2]
        
        f = QuadraticMultiblockFunction(Q, q, r, blockDims)
        x_blocks = NumericVariable[[1.0, 2.0], [3.0, 4.0]]
        x_concat = [1.0, 2.0, 3.0, 4.0]
        
        # Manual full gradient calculation: ∇f = 2 * Q * x + q
        manual_grad_concat = 2 * (Q * x_concat) + q
        
        # Test 1: gradientOracle!(Vector{NumericVariable}, Vector{NumericVariable})
        grad_blocks_1 = NumericVariable[similar(x_blocks[1], Float64), similar(x_blocks[2], Float64)]
        gradientOracle!(grad_blocks_1, f, x_blocks)
        grad_concat_1 = vcat(grad_blocks_1...)
        @test grad_concat_1 ≈ manual_grad_concat
        
        # Test 2: gradientOracle(Vector{NumericVariable}) -> Vector{NumericVariable}
        grad_blocks_2 = gradientOracle(f, x_blocks)
        grad_concat_2 = vcat(grad_blocks_2...)
        @test grad_concat_2 ≈ manual_grad_concat
        @test grad_concat_2 ≈ grad_concat_1
        
        # Test 3: gradientOracle!(NumericVariable, NumericVariable)
        grad_concat_3 = similar(x_concat, Float64)
        gradientOracle!(grad_concat_3, f, x_concat)
        @test grad_concat_3 ≈ manual_grad_concat
        
        # Test 4: gradientOracle(NumericVariable) -> NumericVariable
        grad_concat_4 = gradientOracle(f, x_concat)
        @test grad_concat_4 ≈ manual_grad_concat
        @test grad_concat_4 ≈ grad_concat_3
        
        # Test consistency between all methods
        @test norm(grad_concat_1 - grad_concat_2) < 1e-12
        @test norm(grad_concat_1 - grad_concat_3) < 1e-12
        @test norm(grad_concat_1 - grad_concat_4) < 1e-12
        
        # Test that block-wise gradient matches concatenated
        @test grad_blocks_1[1] ≈ grad_concat_3[1:2]
        @test grad_blocks_1[2] ≈ grad_concat_3[3:4]
    end
    
    @testset "Gradient Consistency Tests" begin
        # Test that partial gradients sum to full gradient
        Q = sparse([2.0 1.0 0.5; 1.0 3.0 0.2; 0.5 0.2 1.5])
        q = [1.0, 2.0, 0.5]
        r = 3.0
        blockDims = [1, 2]
        
        f = QuadraticMultiblockFunction(Q, q, r, blockDims)
        x_blocks = NumericVariable[[2.0], [1.0, 3.0]]
        
        # Get full gradient
        full_grad = gradientOracle(f, x_blocks)
        
        # Get partial gradients
        partial_grad_1 = partialGradientOracle(f, x_blocks, 1)
        partial_grad_2 = partialGradientOracle(f, x_blocks, 2)
        
        # Check consistency
        @test partial_grad_1 ≈ full_grad[1]
        @test partial_grad_2 ≈ full_grad[2]
        
        # Test with concatenated format
        x_concat = [2.0, 1.0, 3.0]
        full_grad_concat = gradientOracle(f, x_concat)
        @test full_grad_concat[1:1] ≈ partial_grad_1
        @test full_grad_concat[2:3] ≈ partial_grad_2
    end
    
    @testset "JuMP Integration Tests" begin
        Q = sparse([2.0 1.0 0.5; 1.0 3.0 0.2; 0.5 0.2 1.5])
        q = [1.0, 2.0, 0.5]
        r = 3.0
        blockDims = [2, 1]
        
        f = QuadraticMultiblockFunction(Q, q, r, blockDims)
        
        # Test JuMPAddSmoothFunction
        model = Model()
        vars = [[@variable(model) for _ in 1:blockDims[i]] for i in 1:length(blockDims)]
        
        obj_expr = JuMPAddSmoothFunction(f, model, vars)
        @test obj_expr isa JuMP.AbstractJuMPScalar
        
        # Test JuMPAddPartialBlockFunction
        model2 = Model()
        var_block1 = [@variable(model2) for _ in 1:blockDims[1]]
        vals = NumericVariable[[1.0, 2.0], [3.0]]  # Values for all blocks
        
        partial_obj_expr = JuMPAddPartialBlockFunction(f, model2, 1, var_block1, vals)
        @test partial_obj_expr isa JuMP.AbstractJuMPScalar
        
        # Test with second block
        var_block2 = [@variable(model2) for _ in 1:blockDims[2]]
        partial_obj_expr2 = JuMPAddPartialBlockFunction(f, model2, 2, var_block2, vals)
        @test partial_obj_expr2 isa JuMP.AbstractJuMPScalar
    end
    
    @testset "Buffer and Efficiency Tests" begin
        Q = sparse([2.0 1.0 0.5 0.0; 1.0 3.0 0.0 0.2; 0.5 0.0 1.5 0.1; 0.0 0.2 0.1 2.0])
        q = [1.0, 2.0, 0.5, 1.5]
        r = 3.0
        blockDims = [2, 2]
        
        f = QuadraticMultiblockFunction(Q, q, r, blockDims)
        x_blocks = NumericVariable[[1.0, 2.0], [3.0, 4.0]]
        
        # Test that buffer is properly allocated
        @test length(f.buffer) == sum(blockDims)
        @test f.buffer isa Vector{Float64}
        
        # Test that block indices are precomputed correctly
        @test f.blockIndices[1] == 1:2
        @test f.blockIndices[2] == 3:4
        
        # Test multiple evaluations (buffer reuse)
        result1 = f(x_blocks)
        result2 = f(x_blocks)
        @test result1 ≈ result2
        
        # Test with different inputs
        x_blocks2 = NumericVariable[[2.0, 1.0], [4.0, 3.0]]
        result3 = f(x_blocks2)
        @test result3 != result1  # Should be different
        
        # Test gradient computations use buffer efficiently
        grad1 = gradientOracle(f, x_blocks)
        grad2 = gradientOracle(f, x_blocks)
        @test grad1 ≈ grad2
    end
    
    @testset "Edge Cases and Error Handling" begin
        Q = sparse([1.0 0.0; 0.0 1.0])
        q = [0.0, 0.0]
        r = 0.0
        blockDims = [1, 1]
        
        f = QuadraticMultiblockFunction(Q, q, r, blockDims)
        
        # Test zero function (should work)
        x_zero = NumericVariable[[0.0], [0.0]]
        @test f(x_zero) ≈ 0.0
        
        # Test dimension mismatch errors
        x_wrong_blocks = NumericVariable[[1.0, 2.0], [3.0]]  # First block wrong size
        @test_throws AssertionError f(x_wrong_blocks)
        
        x_wrong_num_blocks = NumericVariable[[1.0]]  # Wrong number of blocks
        @test_throws AssertionError f(x_wrong_num_blocks)
        
        # Test gradient dimension mismatches
        grad_wrong = NumericVariable[similar([1.0, 2.0], Float64), similar([3.0], Float64)]  # Wrong dimensions
        @test_throws AssertionError gradientOracle!(grad_wrong, f, NumericVariable[[1.0], [2.0]])
        
        # Test single element blocks
        Q_single = sparse([2.0;;])  # Make it a matrix, not a vector
        f_single = QuadraticMultiblockFunction(Q_single, [1.0], 0.0, [1])
        x_single = NumericVariable[[3.0]]
        result_single = f_single(x_single)
        @test result_single ≈ 2.0 * 3.0^2 + 1.0 * 3.0  # 18 + 3 = 21
        
        grad_single = gradientOracle(f_single, x_single)
        @test grad_single[1] ≈ [2.0 * 2.0 * 3.0 + 1.0]  # [12 + 1] = [13]

        # Scalar block should be rejected (this implementation expects array blocks)
        x_scalar_block = NumericVariable[1.0, [2.0]]
        @test_throws AssertionError f(x_scalar_block)

        # Concatenated gradient API dimension checks
        grad_bad = zeros(1)
        x_good = [1.0, 2.0]
        @test_throws AssertionError gradientOracle!(grad_bad, f, x_good)
        @test_throws AssertionError gradientOracle!(0.0, f, x_good)
        @test_throws AssertionError gradientOracle!(zeros(2), f, 1.0)
    end
    
    @testset "Performance and Type Stability Tests" begin
        Q = sparse([2.0 1.0 0.5; 1.0 3.0 0.2; 0.5 0.2 1.5])
        q = [1.0, 2.0, 0.5]
        r = 3.0
        blockDims = [2, 1]
        
        f = QuadraticMultiblockFunction(Q, q, r, blockDims)
        x_blocks = NumericVariable[[1.0, 2.0], [3.0]]
        x_concat = [1.0, 2.0, 3.0]
        
        # Test type stability of function evaluation
        # Note: @inferred may fail with Vector{NumericVariable} due to union type complexity
        result = f(x_blocks)
        @test result isa Float64
        @test @inferred(getNumberOfBlocks(f)) isa Int
        
        # Test type stability of gradient oracles
        grad_blocks = NumericVariable[similar(x_blocks[1], Float64), similar(x_blocks[2], Float64)]
        gradientOracle!(grad_blocks, f, x_blocks)
        @test grad_blocks isa Vector{NumericVariable}
        
        grad_concat = similar(x_concat, Float64)
        gradientOracle!(grad_concat, f, x_concat)
        @test grad_concat isa Vector{Float64}
        
        @test gradientOracle(f, x_blocks) isa Vector{NumericVariable}
        @test gradientOracle(f, x_concat) isa Vector{Float64}
        
        partial_grad = similar(x_blocks[1], Float64)
        partialGradientOracle!(partial_grad, f, x_blocks, 1)
        @test partial_grad isa Vector{Float64}
        @test partialGradientOracle(f, x_blocks, 1) isa Vector{Float64}
    end
end