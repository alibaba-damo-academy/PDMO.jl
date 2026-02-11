using Test
using PDMO
using LinearAlgebra
using SparseArrays
include("../../test_helper.jl")

@testset "IndicatorBox Tests" begin
    @testset "Constructor" begin
        # Test scalar bounds
        lb_scalar = -2.0
        ub_scalar = 3.0
        f_scalar = IndicatorBox(lb_scalar, ub_scalar)
        @test f_scalar isa IndicatorBox
        @test f_scalar isa AbstractFunction
        @test f_scalar.lb == lb_scalar
        @test f_scalar.ub == ub_scalar
        
        # Test vector bounds
        lb_vec = [-1.0, -2.0, 0.0]
        ub_vec = [2.0, 1.0, 5.0]
        f_vec = IndicatorBox(lb_vec, ub_vec)
        @test f_vec.lb == lb_vec
        @test f_vec.ub == ub_vec
        
        # Test matrix bounds
        lb_mat = [-1.0 -2.0; 0.0 -3.0]
        ub_mat = [1.0 2.0; 3.0 1.0]
        f_mat = IndicatorBox(lb_mat, ub_mat)
        @test f_mat.lb == lb_mat
        @test f_mat.ub == ub_mat
        
        # Test error for mismatched sizes
        lb_wrong = [-1.0, -2.0]
        ub_wrong = [1.0, 2.0, 3.0]  # Different size
        @test_throws ErrorException IndicatorBox(lb_wrong, ub_wrong)
        
        # Test error for infeasible bounds (lb > ub)
        lb_infeas = [1.0, 2.0]
        ub_infeas = [0.0, 3.0]  # First element: 1.0 > 0.0
        @test_throws ErrorException IndicatorBox(lb_infeas, ub_infeas)
        
        # Test boundary case (lb = ub)
        lb_eq = [1.0, 2.0]
        ub_eq = [1.0, 2.0]
        f_eq = IndicatorBox(lb_eq, ub_eq)
        @test f_eq.lb == lb_eq
        @test f_eq.ub == ub_eq
    end

    @testset "Function Traits" begin
        @test isProximal(IndicatorBox) == true
        @test isSmooth(IndicatorBox) == false
        @test isConvex(IndicatorBox) == true
        @test isSet(IndicatorBox) == true
    end

    @testset "Function Evaluation" begin
        # Test scalar case
        f_scalar = IndicatorBox(-1.0, 2.0)
        
        # Inside the box
        @test f_scalar(0.0) ≈ 0.0
        @test f_scalar(-0.5) ≈ 0.0
        @test f_scalar(1.5) ≈ 0.0
        @test f_scalar(-1.0) ≈ 0.0  # On boundary
        @test f_scalar(2.0) ≈ 0.0   # On boundary
        
        # Outside the box
        @test f_scalar(-2.0) == Inf
        @test f_scalar(3.0) == Inf
        @test f_scalar(-1.1) == Inf
        @test f_scalar(2.1) == Inf
        
        # Test vector case
        lb_vec = [-1.0, 0.0, -2.0]
        ub_vec = [2.0, 3.0, 1.0]
        f_vec = IndicatorBox(lb_vec, ub_vec)
        
        # All elements inside
        x_inside = [0.0, 1.5, -0.5]
        @test f_vec(x_inside) ≈ 0.0
        
        # All elements on boundary
        x_boundary = [-1.0, 3.0, 1.0]
        @test f_vec(x_boundary) ≈ 0.0
        
        # Some elements outside
        x_outside1 = [3.0, 1.5, -0.5]  # First element outside
        @test f_vec(x_outside1) == Inf
        
        x_outside2 = [0.0, 4.0, -0.5]  # Second element outside
        @test f_vec(x_outside2) == Inf
        
        x_outside3 = [0.0, 1.5, 2.0]   # Third element outside
        @test f_vec(x_outside3) == Inf
        
        # Test matrix case
        lb_mat = [-1.0 0.0; -2.0 -1.0]
        ub_mat = [1.0 2.0; 1.0 3.0]
        f_mat = IndicatorBox(lb_mat, ub_mat)
        
        X_inside = [0.5 1.0; 0.0 2.0]
        @test f_mat(X_inside) ≈ 0.0
        
        X_outside = [2.0 1.0; 0.0 2.0]  # First element outside
        @test f_mat(X_outside) == Inf
    end

    @testset "Proximal Oracle (Projection)" begin
        # Test scalar case: projection onto [lb, ub]
        f_scalar = IndicatorBox(-1.0, 2.0)
        
        # Inside the box - should remain unchanged
        @test proximalOracle(f_scalar, 0.0) ≈ 0.0
        @test proximalOracle(f_scalar, -0.5) ≈ -0.5
        @test proximalOracle(f_scalar, 1.5) ≈ 1.5
        
        # On boundaries - should remain unchanged
        @test proximalOracle(f_scalar, -1.0) ≈ -1.0
        @test proximalOracle(f_scalar, 2.0) ≈ 2.0
        
        # Outside the box - should be projected to boundary
        @test proximalOracle(f_scalar, -2.0) ≈ -1.0  # Project to lower bound
        @test proximalOracle(f_scalar, 3.0) ≈ 2.0    # Project to upper bound
        @test proximalOracle(f_scalar, -10.0) ≈ -1.0
        @test proximalOracle(f_scalar, 10.0) ≈ 2.0
        
        # Test vector case: elementwise projection
        lb_vec = [-1.0, 0.0, -2.0]
        ub_vec = [2.0, 3.0, 1.0]
        f_vec = IndicatorBox(lb_vec, ub_vec)
        
        # Mixed case: some inside, some outside
        x_mixed = [-2.0, 1.5, 3.0]
        expected_proj = [-1.0, 1.5, 1.0]  # Project first and third elements
        prox_mixed = proximalOracle(f_vec, x_mixed)
        @test prox_mixed ≈ expected_proj
        @test size(prox_mixed) == size(x_mixed)
        
        # Test in-place proximal for vectors
        prox_inplace = similar(x_mixed)
        proximalOracle!(prox_inplace, f_vec, x_mixed)
        @test prox_inplace ≈ expected_proj
        
        # All elements outside
        x_all_outside = [-5.0, 10.0, -10.0]
        expected_all_proj = [-1.0, 3.0, -2.0]
        @test proximalOracle(f_vec, x_all_outside) ≈ expected_all_proj
        
        # Test matrix case
        lb_mat = [-1.0 0.0; -2.0 -1.0]
        ub_mat = [1.0 2.0; 1.0 3.0]
        f_mat = IndicatorBox(lb_mat, ub_mat)
        
        X_mixed = [-2.0 1.0; 0.5 5.0]
        expected_X_proj = [-1.0 1.0; 0.5 3.0]
        prox_X = proximalOracle(f_mat, X_mixed)
        @test prox_X ≈ expected_X_proj
        
        # Test in-place proximal for matrices
        prox_X_inplace = similar(X_mixed)
        proximalOracle!(prox_X_inplace, f_mat, X_mixed)
        @test prox_X_inplace ≈ expected_X_proj
        
        # Test error for scalar in-place
        @test_throws ErrorException proximalOracle!(0.0, f_scalar, 5.0)
        
        # Test that gamma parameter doesn't affect projection (indicator functions)
        @test proximalOracle(f_scalar, 3.0, 0.1) ≈ 2.0
        @test proximalOracle(f_scalar, 3.0, 10.0) ≈ 2.0
    end

    @testset "Edge Cases" begin
        # Test with very tight bounds (lb ≈ ub)
        ε = 1e-10
        f_tight = IndicatorBox(1.0, 1.0 + ε)
        
        @test f_tight(1.0) ≈ 0.0
        @test f_tight(1.0 + ε) ≈ 0.0
        @test f_tight(1.0 + ε/2) ≈ 0.0
        @test f_tight(0.9) == Inf
        @test f_tight(1.1) == Inf
        
        # Projection should work correctly
        @test proximalOracle(f_tight, 0.5) ≈ 1.0
        @test proximalOracle(f_tight, 1.5) ≈ 1.0 + ε
        
        # Test with zero bounds
        f_zero = IndicatorBox(0.0, 0.0)
        @test f_zero(0.0) ≈ 0.0
        @test f_zero(1e-4) == Inf  # Use value larger than FeasTolerance (1e-6)
        @test f_zero(-1e-4) == Inf  # Use value larger than FeasTolerance (1e-6)
        @test proximalOracle(f_zero, 5.0) ≈ 0.0
        @test proximalOracle(f_zero, -5.0) ≈ 0.0
        
        # Test with very large bounds
        f_large = IndicatorBox(-1e10, 1e10)
        large_val = 1e9
        @test f_large(large_val) ≈ 0.0
        @test f_large(-large_val) ≈ 0.0
        @test proximalOracle(f_large, 1e11) ≈ 1e10
        @test proximalOracle(f_large, -1e11) ≈ -1e10
        
        # Test with empty arrays
        lb_empty = Float64[]
        ub_empty = Float64[]
        f_empty = IndicatorBox(lb_empty, ub_empty)
        
        x_empty = Float64[]
        @test f_empty(x_empty) ≈ 0.0
        @test proximalOracle(f_empty, x_empty) ≈ Float64[]
        
        # Test with sparse matrices
        lb_sparse = sparse([-1.0 0.0; 0.0 -2.0])
        ub_sparse = sparse([1.0 2.0; 3.0 1.0])
        f_sparse = IndicatorBox(lb_sparse, ub_sparse)
        
        X_sparse_test = sparse([0.5 1.0; 2.0 0.5])
        @test f_sparse(X_sparse_test) ≈ 0.0
        
        X_sparse_outside = sparse([2.0 1.0; 2.0 0.5])
        @test f_sparse(X_sparse_outside) == Inf
    end

    @testset "Mathematical Properties" begin
        lb = [-2.0, -1.0, 0.0]
        ub = [1.0, 2.0, 3.0]
        f = IndicatorBox(lb, ub)
        
        # Test that projection is idempotent: proj(proj(x)) = proj(x)
        x_test = [-5.0, 0.5, 10.0]
        proj1 = proximalOracle(f, x_test)
        proj2 = proximalOracle(f, proj1)
        @test proj1 ≈ proj2
        
        # Test that projection is non-expansive: ||proj(x) - proj(y)|| ≤ ||x - y||
        x1 = randn(3)
        x2 = randn(3)
        proj_x1 = proximalOracle(f, x1)
        proj_x2 = proximalOracle(f, x2)
        
        @test norm(proj_x1 - proj_x2) ≤ norm(x1 - x2) + 1e-10
        
        # Test that projection minimizes distance to the set
        # For any point in the box, distance to projection should be ≤ distance to that point
        x_outside = [5.0, -5.0, 10.0]
        proj_x = proximalOracle(f, x_outside)
        
        # Check that proj_x is in the box
        @test all(proj_x .>= lb .- 1e-10)
        @test all(proj_x .<= ub .+ 1e-10)
        
        # For any point in the box, distance should be ≥ distance to projection
        x_in_box = [0.0, 1.0, 2.0]  # This point is in the box
        @test norm(x_outside - proj_x) ≤ norm(x_outside - x_in_box) + 1e-10
        
        # Test that projection onto a single point works
        point = [1.0, 2.0]
        f_point = IndicatorBox(point, point)
        
        test_points = [[0.0, 0.0], [2.0, 3.0], [1.0, 2.0]]
        for tp in test_points
            @test proximalOracle(f_point, tp) ≈ point
        end
    end
end 