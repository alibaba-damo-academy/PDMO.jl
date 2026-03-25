using Test
using PDMO
using LinearAlgebra
using SparseArrays
include("../../test_helper.jl")

@testset "AbstractFunction Tests" begin
    
    @testset "Function Interface Tests" begin
        # Test if AbstractFunction type is defined and usable
        @test AbstractFunction isa Type
        
        # Test inheritance hierarchy
        if @isdefined(AbstractMultiblockFunction)
            @test AbstractMultiblockFunction <: AbstractFunction
        end
        
        # Test Zero function (which should exist)
        if @isdefined(Zero)
            zero_func = Zero()
            @test zero_func isa AbstractFunction
            
            # Test function evaluation
            x = randn(5)
            result = zero_func(x)
            @test result ≈ 0.0
            
            # Test trait checkers
            @test isSmooth(Zero) == true
            @test isProximal(Zero) == true
            @test isConvex(Zero) == true
        end
    end
    
    @testset "Function Traits Interface" begin
        # Test that trait functions exist and work
        @test isProximal isa Function
        @test isSmooth isa Function
        @test isConvex isa Function
        @test isSet isa Function
        
        # Test default trait values for AbstractFunction
        @test isProximal(AbstractFunction) == false
        @test isSmooth(AbstractFunction) == false
        @test isConvex(AbstractFunction) == false
        @test isSet(AbstractFunction) == false
    end
    
    @testset "Oracle Interface Tests" begin
        # Test that oracle functions are defined
        @test proximalOracle isa Function
        @test proximalOracle! isa Function
        @test gradientOracle isa Function
        @test gradientOracle! isa Function
        
        # Test NumericVariable type
        @test NumericVariable isa Type
        
        # Test that Float64 and arrays are NumericVariable
        @test 1.0 isa NumericVariable
        @test [1.0, 2.0] isa NumericVariable
        @test [1.0 2.0; 3.0 4.0] isa NumericVariable
    end

    # Include individual function tests
    @testset "Zero Function" begin
        include("testZero.jl")
    end
    
    @testset "QuadraticFunction" begin
        include("testQuadraticFunction.jl")
    end
    
    @testset "AffineFunction" begin
        include("testAffineFunction.jl")
    end
    
    @testset "ElementwiseL1Norm" begin
        include("testElementwiseL1Norm.jl")
    end
    
    @testset "IndicatorBox" begin
        include("testIndicatorBox.jl")
    end
    
    @testset "IndicatorPSD" begin
        include("testIndicatorPSD.jl")
    end
    
    @testset "FrobeniusNormSquare" begin
        include("testFrobeniusNormSquare.jl")
    end
    
    @testset "IndicatorSOC" begin
        include("testIndicatorSOC.jl")
    end

    @testset "IndicatorRotatedSOC" begin
        include("testIndicatorRotatedSOC.jl")
    end
    
    @testset "ComponentwiseExponentialFunction" begin
        include("testComponentwiseExponentialFunction.jl")
    end
    
    @testset "IndicatorBallL2" begin
        include("testIndicatorBallL2.jl")
    end
    
    @testset "IndicatorNonnegativeOrthant" begin
        include("testIndicatorNonnegativeOrthant.jl")
    end
    
    @testset "MatrixNuclearNorm" begin
        include("testMatrixNuclearNorm.jl")
    end
    
    @testset "IndicatorLinearSubspace" begin
        include("testIndicatorLinearSubspace.jl")
    end
    
    @testset "WeightedMatrixL1Norm" begin
        include("testWeightedMatrixL1Norm.jl")
    end
    
    @testset "IndicatorSumOfNVariables" begin
        include("testIndicatorSumOfNVariables.jl")
    end
    
    @testset "WrapperScalarInputFunction" begin
        include("testWrapperScalarInputFunction.jl")
    end

    @testset "AbstractFunctionUtil Single-Block Regression" begin
        include("testAbstractFunctionUtilSingleBlock.jl")
    end
    
    @testset "IndicatorHyperplane" begin
        include("testIndicatorHyperplane.jl")
    end

    @testset "IndicatorBinary" begin
        include("testIndicatorBinary.jl")
    end

    @testset "IndicatorMixedInteger" begin
        include("testIndicatorMixedInteger.jl")
    end
    
end 