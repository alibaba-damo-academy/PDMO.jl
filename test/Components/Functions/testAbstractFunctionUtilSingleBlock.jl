using Test
using PDMO
import PDMO: gradientOracle!, gradientOracle, isSmooth

struct SimpleOneBlockFunction <: AbstractMultiblockFunction end
isSmooth(::Type{SimpleOneBlockFunction}) = true

function gradientOracle!(grad::Vector{NumericVariable}, f::SimpleOneBlockFunction, x::Vector{NumericVariable})
    @test length(grad) == 1
    @test length(x) == 1
    grad[1] .= 2.0 .* x[1]
    return nothing
end

function gradientOracle(f::SimpleOneBlockFunction, x::Vector{NumericVariable})
    @test length(x) == 1
    return NumericVariable[2.0 .* x[1]]
end

@testset "AbstractFunctionUtil Single-Block Regression" begin
    f = SimpleOneBlockFunction()
    x = NumericVariable[[1.0, -2.0, 3.0]]
    @test_throws ArgumentError PDMO.estimateLipschitzConstantMultiblock(f, x; maxTrials=12)
end
