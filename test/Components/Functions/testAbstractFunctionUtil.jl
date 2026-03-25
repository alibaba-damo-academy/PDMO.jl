using Test
using PDMO
using LinearAlgebra
using SparseArrays
include("../../test_helper.jl")

@testset "AbstractFunctionUtil Tests" begin
    @testset "Lipschitz Constant Estimation - Special Cases" begin
        # Test with QuadraticFunction (exact case)
        Q = sparse([2.0 1.0; 1.0 3.0])  # 2x2 symmetric matrix
        quad_func = QuadraticFunction(Q)
        x_test = [1.0, 2.0]
        L_quad = estimateLipschitzConstant(quad_func, x_test)
        @test L_quad ≈ 2.0 * opnorm(Array(Q))  # Should be exact for quadratic functions
        
        # Test with AffineFunction (should be 0)
        A = sparse([1.0, 2.0])
        affine_func = AffineFunction(A)
        x_affine = [3.0, 4.0]
        L_affine = estimateLipschitzConstant(affine_func, x_affine)
        @test L_affine ≈ 0.0
        
        # Test with Zero function (should be 0)
        zero_func = Zero()
        x_zero = [1.0, 1.0]
        L_zero = estimateLipschitzConstant(zero_func, x_zero)
        @test L_zero ≈ 0.0
    end

    @testset "Lipschitz Constant Estimation - Scalar Functions" begin
        # Test with scalar quadratic function f(x) = x²
        Q_scalar = sparse(reshape([2.0], 1, 1))
        quad_scalar = QuadraticFunction(Q_scalar)
        x_scalar = [1.0]
        L_scalar = estimateLipschitzConstant(quad_scalar, x_scalar)
        @test L_scalar ≈ 4.0  # Lipschitz constant should be 2 * max eigenvalue
        
        # Test with different starting points
        x_scalar2 = [10.0]
        L_scalar2 = estimateLipschitzConstant(quad_scalar, x_scalar2)
        @test L_scalar2 ≈ L_scalar  # Should be independent of starting point
        
        # Test with different step sizes
        L_scalar3 = estimateLipschitzConstant(quad_scalar, x_scalar, maxStepSize=0.1)
        @test L_scalar3 ≈ L_scalar atol=1e-6  # Should be similar despite different step size
    end

    @testset "Lipschitz Constant Estimation - Vector Functions" begin
        # Test with high-dimensional quadratic function
        n = 100
        Q_large = sparse(1.0 * I(n))  # Identity matrix
        quad_large = QuadraticFunction(Q_large)
        x_large = randn(n)
        L_large = estimateLipschitzConstant(quad_large, x_large)
        @test L_large ≈ 2.0  # Should be 2 * largest eigenvalue = 2
        
        # Test with different maxTrials
        L_large2 = estimateLipschitzConstant(quad_large, x_large, maxTrials=10)
        @test L_large2 ≈ 2.0  # Should still be accurate with fewer trials
        
        # Test with different step sizes
        L_large3 = estimateLipschitzConstant(quad_large, x_large, minStepSize=1e-8, maxStepSize=0.1)
        @test L_large3 ≈ 2.0 atol=1e-6  # Should be accurate with different step sizes
    end

    @testset "Proximal Oracle of Conjugate - Basic Properties" begin
        # Test with Zero function
        zero_func = Zero()
        x_test = [1.0, 2.0, 3.0]
        γ = 1.0
        
        # For Zero function, prox_{γf*}(x) = x
        prox_conj = proximalOracleOfConjugate(zero_func, x_test, γ)
        @test prox_conj ≈ x_test
        
        # Test in-place version
        prox_conj_inplace = similar(x_test)
        proximalOracleOfConjugate!(prox_conj_inplace, zero_func, x_test, γ)
        @test prox_conj_inplace ≈ x_test
        
        # Test with different gamma values
        γ_large = 10.0
        prox_conj_large = proximalOracleOfConjugate(zero_func, x_test, γ_large)
        @test prox_conj_large ≈ x_test
    end

    @testset "Proximal Oracle of Conjugate - Error Cases" begin
        # Test error for non-proximal function
        struct NonProximalFunction <: AbstractFunction end
        isProximal(::Type{NonProximalFunction}) = false
        
        non_prox_func = NonProximalFunction()
        x_error = [1.0, 2.0]
        
        @test_throws ErrorException proximalOracleOfConjugate(non_prox_func, x_error)
        
        # Test error for negative gamma
        zero_func = Zero()
        @test_throws ErrorException proximalOracleOfConjugate(zero_func, x_error, -1.0)
        
        # Test error for scalar input
        @test_throws ErrorException proximalOracleOfConjugate(zero_func, 1.0)
        
        # Test error for dimension mismatch in in-place version
        y_wrong_size = [1.0]
        @test_throws ErrorException proximalOracleOfConjugate!(y_wrong_size, zero_func, x_error)
    end

    @testset "Proximal Oracle of Conjugate - Indicator Functions" begin
        # Test with IndicatorBox
        lower = [-1.0, -1.0]
        upper = [1.0, 1.0]
        box_func = IndicatorBox(lower, upper)
        
        # Test point inside box
        x_inside = [0.0, 0.5]
        γ = 1.0
        prox_box = proximalOracleOfConjugate(box_func, x_inside, γ)
        @test all(-1.0 .<= prox_box .<= 1.0)  # Result should be in box
        
        # Test point outside box
        x_outside = [2.0, -2.0]
        prox_box_outside = proximalOracleOfConjugate(box_func, x_outside, γ)
        @test all(-1.0 .<= prox_box_outside .<= 1.0)  # Result should be in box
        
        # Test with IndicatorNonnegativeOrthant
        nonneg_func = IndicatorNonnegativeOrthant()
        x_mixed = [1.0, -2.0, 3.0]
        prox_nonneg = proximalOracleOfConjugate(nonneg_func, x_mixed, γ)
        @test all(prox_nonneg .>= -1e-10)  # Result should be nonnegative
    end

    @testset "Proximal Oracle of Conjugate - Mathematical Properties" begin
        # Test Moreau's identity: x = prox_{γf}(x) + γ * prox_{f*/γ}(x/γ)
        # Using IndicatorBox as test function
        lower = [-1.0, -1.0]
        upper = [1.0, 1.0]
        box_func = IndicatorBox(lower, upper)
        
        x_test = [2.0, -0.5]
        γ = 1.0
        
        prox_f = proximalOracle(box_func, x_test, γ)
        prox_conj_f = proximalOracleOfConjugate(box_func, x_test, γ)
        
        # Verify Moreau's identity
        @test x_test ≈ prox_f + prox_conj_f atol=1e-10
        
        # Test scaling property
        γ2 = 2.0
        prox_conj_scaled = proximalOracleOfConjugate(box_func, x_test, γ2)
        # The proximal operator of the conjugate should scale appropriately
        @test norm(prox_conj_scaled) ≤ norm(prox_conj_f) * (γ2/γ + 1e-10)
        
        # Test non-expansiveness
        x1 = [1.0, 1.0]
        x2 = [-1.0, -1.0]
        prox_conj1 = proximalOracleOfConjugate(box_func, x1, γ)
        prox_conj2 = proximalOracleOfConjugate(box_func, x2, γ)
        @test norm(prox_conj1 - prox_conj2) ≤ norm(x1 - x2) + 1e-10
    end

end 