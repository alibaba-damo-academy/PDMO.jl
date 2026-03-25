using Test
using PDMO
using JuMP
include("../../test_helper.jl")

@testset "IndicatorMixedInteger Tests" begin
    lb = [0.0, -2.0, 1.0]
    ub = [3.0, 2.0, 5.0]
    is_int = [true, false, true]
    f = IndicatorMixedInteger(lb, ub, is_int)

    @test isProximal(IndicatorMixedInteger) == true
    @test isConvex(IndicatorMixedInteger) == false
    @test isSet(IndicatorMixedInteger) == true
    @test isSupportedByJuMP(IndicatorMixedInteger) == true

    @test f([2.0, -1.2, 4.0]) == 0.0
    @test f([2.2, -1.2, 4.0]) == Inf
    @test f([2.0, -3.0, 4.0]) == Inf

    # Scalar behavior is defined for length-1 bounds.
    f_scalar = IndicatorMixedInteger([0.0], [5.0], [true])
    @test f_scalar(3.0) == 0.0
    @test f_scalar(3.2) == Inf
    @test proximalOracle(f_scalar, 2.6) == 3.0
    @test_throws ErrorException f(1.0)

    x = [2.2, -10.0, 4.7]
    y = proximalOracle(f, x)
    @test y == [2.0, -2.0, 5.0]

    y_inplace = zeros(3)
    proximalOracle!(y_inplace, f, x)
    @test y_inplace == [2.0, -2.0, 5.0]

    @test_throws ErrorException proximalOracle!(0.0, f_scalar, 2.6)
    @test_throws ErrorException proximalOracle(f, [1.0, 2.0])

    model = Model()
    vars = [@variable(model) for _ in 1:3]
    @test JuMPAddProximableFunction(f, model, vars) === nothing
    @test JuMP.lower_bound(vars[1]) == 0.0
    @test JuMP.upper_bound(vars[3]) == 5.0
    @test JuMP.is_integer(vars[1])
    @test !JuMP.is_integer(vars[2])
end
