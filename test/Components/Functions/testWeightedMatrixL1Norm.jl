using Test
using PDMO
using LinearAlgebra
using SparseArrays
include("../../test_helper.jl")

@testset "WeightedMatrixL1Norm Tests" begin
    @testset "Constructor" begin
        # Test valid construction - standard case
        A = sparse([1.0 2.0; 3.0 4.0])
        f = WeightedMatrixL1Norm(A)
        @test f isa WeightedMatrixL1Norm
        @test f isa AbstractFunction
        @test f.A == A
        @test f.numberRows == 2
        @test f.numberColumns == 2
        @test f.inNonnegativeOrthant == false
        
        # Test valid construction - nonnegative orthant case
        f_nonneg = WeightedMatrixL1Norm(A, inNonnegativeOrthant=true)
        @test f_nonneg.inNonnegativeOrthant == true
        
        # Test with different matrix sizes
        A_3x2 = sparse([1.0 2.0; 3.0 4.0; 5.0 6.0])
        f_3x2 = WeightedMatrixL1Norm(A_3x2)
        @test f_3x2.numberRows == 3
        @test f_3x2.numberColumns == 2
        
        # Test with sparse matrix with zeros
        A_sparse = sparse([1.0 0.0; 0.0 2.0])
        f_sparse = WeightedMatrixL1Norm(A_sparse)
        @test f_sparse.A == A_sparse
        
        # Test error cases
        A_negative = sparse([1.0 -1.0; 2.0 3.0])
        @test_throws ErrorException WeightedMatrixL1Norm(A_negative)  # Negative values
        
        A_very_negative = sparse([-1e-5 1.0; 2.0 3.0])
        @test_throws ErrorException WeightedMatrixL1Norm(A_very_negative)  # Below -FeasTolerance
    end

    @testset "Function Traits" begin
        @test isProximal(WeightedMatrixL1Norm) == true
        @test isConvex(WeightedMatrixL1Norm) == true
        @test isSmooth(WeightedMatrixL1Norm) == false
        @test isSet(WeightedMatrixL1Norm) == false
    end

    @testset "Function Evaluation - Standard Case" begin
        # Test with simple weights
        A = sparse([1.0 2.0; 3.0 4.0])
        f = WeightedMatrixL1Norm(A)
        
        # Test with positive matrix
        X = [1.0 2.0; 3.0 4.0]
        expected = 1.0*1.0 + 2.0*2.0 + 3.0*3.0 + 4.0*4.0  # 1 + 4 + 9 + 16 = 30
        @test f(X) ≈ expected
        
        # Test with mixed signs
        X_mixed = [1.0 -2.0; -3.0 4.0]
        expected_mixed = 1.0*1.0 + 2.0*2.0 + 3.0*3.0 + 4.0*4.0  # Same as above due to abs
        @test f(X_mixed) ≈ expected_mixed
        
        # Test with zero matrix
        X_zero = zeros(2, 2)
        @test f(X_zero) ≈ 0.0
        
        # Test error for dimension mismatch
        X_wrong = [1.0 2.0 3.0; 4.0 5.0 6.0]  # 2x3 instead of 2x2
        @test_throws AssertionError f(X_wrong)
    end

    @testset "Function Evaluation - Nonnegative Orthant Case" begin
        A = sparse([1.0 2.0; 3.0 4.0])
        f = WeightedMatrixL1Norm(A, inNonnegativeOrthant=true)
        
        # Test with nonnegative matrix
        X_nonneg = [1.0 2.0; 3.0 4.0]
        expected_nonneg = 1.0*1.0 + 2.0*2.0 + 3.0*3.0 + 4.0*4.0  # 30
        @test f(X_nonneg) ≈ expected_nonneg
        
        # Test with matrix containing negative values
        X_negative = [1.0 -2.0; 3.0 4.0]
        @test f(X_negative) == Inf
        
        # Test with matrix containing very small negative values (within tolerance)
        X_near = [1.0 -FeasTolerance/2; 3.0 4.0]
        expected_near = 1.0*1.0 + 2.0*(FeasTolerance/2) + 3.0*3.0 + 4.0*4.0
        @test f(X_near) ≈ expected_near
        
        # Test with matrix containing negative values clearly outside tolerance
        X_outside = [1.0 -1e-4; 3.0 4.0]  # -1e-5 is clearly outside FeasTolerance
        @test f(X_outside) == Inf
        
        # Test with zero matrix
        X_zero = zeros(2, 2)
        @test f(X_zero) ≈ 0.0
    end

    @testset "Proximal Oracle - Standard Case" begin
        # Test soft thresholding
        A = sparse([1.0 2.0; 3.0 4.0])
        f = WeightedMatrixL1Norm(A)
        
        X = [3.0 4.0; 5.0 6.0]
        γ = 1.0
        prox = proximalOracle(f, X, γ)
        
        # Expected: sign(x) * max(0, |x| - γ * A)
        # For X[1,1] = 3, A[1,1] = 1: sign(3) * max(0, 3 - 1*1) = 1 * 2 = 2
        # For X[1,2] = 4, A[1,2] = 2: sign(4) * max(0, 4 - 1*2) = 1 * 2 = 2
        # For X[2,1] = 5, A[2,1] = 3: sign(5) * max(0, 5 - 1*3) = 1 * 2 = 2
        # For X[2,2] = 6, A[2,2] = 4: sign(6) * max(0, 6 - 1*4) = 1 * 2 = 2
        expected = [2.0 2.0; 2.0 2.0]
        @test prox ≈ expected
        
        # Test in-place version
        prox_inplace = similar(X)
        proximalOracle!(prox_inplace, f, X, γ)
        @test prox_inplace ≈ expected
        
        # Test with negative values
        X_neg = [-3.0 4.0; -5.0 6.0]
        prox_neg = proximalOracle(f, X_neg, γ)
        # For X[1,1] = -3, A[1,1] = 1: sign(-3) * max(0, 3 - 1*1) = -1 * 2 = -2
        expected_neg = [-2.0 2.0; -2.0 2.0]
        @test prox_neg ≈ expected_neg
        
        # Test with small values that get thresholded to zero
        X_small = [0.5 1.5; 2.5 3.5]
        prox_small = proximalOracle(f, X_small, γ)
        # For X[1,1] = 0.5, A[1,1] = 1: sign(0.5) * max(0, 0.5 - 1*1) = 1 * 0 = 0
        # For X[1,2] = 1.5, A[1,2] = 2: sign(1.5) * max(0, 1.5 - 1*2) = 1 * 0 = 0
        # For X[2,1] = 2.5, A[2,1] = 3: sign(2.5) * max(0, 2.5 - 1*3) = 1 * 0 = 0
        # For X[2,2] = 3.5, A[2,2] = 4: sign(3.5) * max(0, 3.5 - 1*4) = 1 * 0 = 0
        expected_small = [0.0 0.0; 0.0 0.0]
        @test prox_small ≈ expected_small
    end

    @testset "Proximal Oracle - Nonnegative Orthant Case" begin
        A = sparse([1.0 2.0; 3.0 4.0])
        f = WeightedMatrixL1Norm(A, inNonnegativeOrthant=true)
        
        X = [3.0 4.0; 5.0 6.0]
        γ = 1.0
        prox = proximalOracle(f, X, γ)
        
        # Expected: max(0, x - γ * A)
        # For X[1,1] = 3, A[1,1] = 1: max(0, 3 - 1*1) = max(0, 2) = 2
        # For X[1,2] = 4, A[1,2] = 2: max(0, 4 - 1*2) = max(0, 2) = 2
        # For X[2,1] = 5, A[2,1] = 3: max(0, 5 - 1*3) = max(0, 2) = 2
        # For X[2,2] = 6, A[2,2] = 4: max(0, 6 - 1*4) = max(0, 2) = 2
        expected = [2.0 2.0; 2.0 2.0]
        @test prox ≈ expected
        
        # Test with negative values (should be projected to zero)
        X_neg = [-1.0 4.0; 5.0 -2.0]
        prox_neg = proximalOracle(f, X_neg, γ)
        # For X[1,1] = -1, A[1,1] = 1: max(0, -1 - 1*1) = max(0, -2) = 0
        # For X[1,2] = 4, A[1,2] = 2: max(0, 4 - 1*2) = max(0, 2) = 2
        # For X[2,1] = 5, A[2,1] = 3: max(0, 5 - 1*3) = max(0, 2) = 2
        # For X[2,2] = -2, A[2,2] = 4: max(0, -2 - 1*4) = max(0, -6) = 0
        expected_neg = [0.0 2.0; 2.0 0.0]
        @test prox_neg ≈ expected_neg
        
        # Test in-place version
        prox_inplace = similar(X)
        proximalOracle!(prox_inplace, f, X, γ)
        @test prox_inplace ≈ expected
    end

    @testset "Edge Cases" begin
        # Test with single element matrix
        A_single = sparse(reshape([1.0], 1, 1))  # 1x1 matrix
        f_single = WeightedMatrixL1Norm(A_single)
        x_single = reshape([1.0], 1, 1)
        @test f_single(x_single) ≈ 1.0
        
        # Test with zero matrix
        A_zero = spzeros(2, 2)
        f_zero = WeightedMatrixL1Norm(A_zero)
        x_zero = zeros(2, 2)
        @test f_zero(x_zero) ≈ 0.0
        
        # Test with sparse matrix having only one non-zero element
        A_sparse = sparse([0.0 1.0; 0.0 0.0])
        f_sparse = WeightedMatrixL1Norm(A_sparse)
        x_sparse = [0.0 1.0; 0.0 0.0]
        @test f_sparse(x_sparse) ≈ 1.0
        
        # Test with very small values
        A_small = sparse([1e-10 0.0; 0.0 1e-10])
        f_small = WeightedMatrixL1Norm(A_small)
        x_small = [1.0 0.0; 0.0 1.0]
        @test f_small(x_small) ≈ 2e-10
    end

    @testset "Mathematical Properties" begin
        A = sparse([1.0 2.0; 3.0 4.0])
        f = WeightedMatrixL1Norm(A)
        
        # Test that proximal operator is non-expansive
        X1 = [1.0 2.0; 3.0 4.0]
        X2 = [2.0 1.0; 4.0 3.0]
        γ = 1.0
        
        prox1 = proximalOracle(f, X1, γ)
        prox2 = proximalOracle(f, X2, γ)
        
        @test norm(prox1 - prox2) ≤ norm(X1 - X2) + 1e-10
        
        # Test that proximal operator reduces the objective function
        X_test = [5.0 6.0; 7.0 8.0]
        prox_test = proximalOracle(f, X_test, γ)
        
        obj_original = f(X_test) + (1/(2*γ)) * norm(X_test - X_test)^2
        obj_prox = f(prox_test) + (1/(2*γ)) * norm(prox_test - X_test)^2
        
        @test obj_prox ≤ obj_original + 1e-10
        
        # Test scaling property: prox_{λf}(x) = prox_f(x, λ)
        λ = 2.0
        A_scaled = λ * A
        f_scaled = WeightedMatrixL1Norm(A_scaled)
        
        prox_scaled_func = proximalOracle(f_scaled, X_test, 1.0)
        prox_scaled_gamma = proximalOracle(f, X_test, λ)
        
        @test prox_scaled_func ≈ prox_scaled_gamma atol=1e-10
        
        # Test with nonnegative orthant case
        f_nonneg = WeightedMatrixL1Norm(A, inNonnegativeOrthant=true)
        
        # Test that projection onto nonnegative orthant is idempotent
        X_nonneg = [1.0 2.0; 3.0 4.0]  # Already nonnegative
        prox_nonneg1 = proximalOracle(f_nonneg, X_nonneg, γ)
        prox_nonneg2 = proximalOracle(f_nonneg, prox_nonneg1, γ)
        
        # Should be close due to the nature of the proximal operator
        @test norm(prox_nonneg1 - prox_nonneg2) ≤ norm(prox_nonneg1) * 1e-10
        
        # Test that result is always nonnegative for nonnegative orthant case
        X_mixed = [-1.0 2.0; 3.0 -4.0]
        prox_mixed = proximalOracle(f_nonneg, X_mixed, γ)
        @test all(prox_mixed .≥ -1e-10)  # Allow for small numerical errors
        
        # Test convexity property: f(αx + (1-α)y) ≤ αf(x) + (1-α)f(y)
        α = 0.3
        X_conv1 = [1.0 2.0; 3.0 4.0]
        X_conv2 = [2.0 1.0; 4.0 3.0]
        
        f_combo = f(α * X_conv1 + (1-α) * X_conv2)
        f_convex_bound = α * f(X_conv1) + (1-α) * f(X_conv2)
        
        @test f_combo ≤ f_convex_bound + 1e-10
        
        # Test triangle inequality: ||A ⊙ (x + y)||₁ ≤ ||A ⊙ x||₁ + ||A ⊙ y||₁
        X_tri1 = [1.0 -1.0; 2.0 -2.0]
        X_tri2 = [-1.0 2.0; -2.0 3.0]
        
        f_sum = f(X_tri1 + X_tri2)
        f_triangle_bound = f(X_tri1) + f(X_tri2)
        
        @test f_sum ≤ f_triangle_bound + 1e-10
    end
end 