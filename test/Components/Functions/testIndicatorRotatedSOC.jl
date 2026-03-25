using Test
using PDMO
include("../../test_helper.jl")

@testset "IndicatorRotatedSOC Tests" begin
    f = IndicatorRotatedSOC(4)

    @test isProximal(IndicatorRotatedSOC) == true
    @test isConvex(IndicatorRotatedSOC) == true
    @test isSet(IndicatorRotatedSOC) == true

    # In-cone example: u=1, v=2, ||w||^2 = 1 <= 2uv = 4
    x_in = [1.0, 2.0, 1.0, 0.0]
    @test f(x_in) == 0.0

    # Out-of-cone example
    x_out = [0.1, 0.1, 1.0, 1.0]
    @test f(x_out) == Inf

    # Projection should return a feasible point.
    y = proximalOracle(f, x_out)
    @test y[1] >= -FeasTolerance
    @test y[2] >= -FeasTolerance
    @test sum(abs2, y[3:end]) <= 2 * y[1] * y[2] + 1e-8

    y_inplace = similar(x_out)
    proximalOracle!(y_inplace, f, x_out)
    @test y_inplace ≈ y
end
