using Test
using PDMO
using JuMP
include("../../test_helper.jl")

@testset "IndicatorBinary Tests" begin
    f = IndicatorBinary()

    @test isProximal(IndicatorBinary) == true
    @test isConvex(IndicatorBinary) == false
    @test isSet(IndicatorBinary) == true
    @test isSupportedByJuMP(IndicatorBinary) == true

    @test f(0.0) == 0.0
    @test f(1.0) == 0.0
    @test f(0.25) == Inf
    @test f([0.0, 1.0, 0.0]) == 0.0
    @test f([0.0, 1.0, 0.2]) == Inf

    @test proximalOracle(f, 0.49) == 0.0
    @test proximalOracle(f, 0.51) == 1.0
    @test proximalOracle(f, [-0.3, 0.49, 0.5, 0.9]) == [0.0, 0.0, 1.0, 1.0]

    y = zeros(4)
    proximalOracle!(y, f, [-0.3, 0.49, 0.5, 0.9])
    @test y == [0.0, 0.0, 1.0, 1.0]
    @test_throws ErrorException proximalOracle!(0.0, f, 0.2)

    model = Model()
    vars = [@variable(model) for _ in 1:3]
    @test JuMPAddProximableFunction(f, model, vars) === nothing
    @test all(JuMP.is_binary, vars)
end
