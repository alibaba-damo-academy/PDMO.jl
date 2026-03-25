using Test
using PDMO
using SparseArrays

"""
Build a small composite problem for AdaPDM variants:
    min f1(x1) + g1(x1) + f2(x2) + g2(x2) + h(u)
    s.t. x1 + x2 - u = 0
where h(u) = 0.
"""
function _build_small_adapdm_problem()
    mbp = MultiblockProblem()

    block1 = BlockVariable("x1")
    block1.f = QuadraticFunction(sparse([2.0 0.0; 0.0 1.0]), [-0.8, 0.2], 0.0)
    block1.g = IndicatorBox([-2.0, -2.0], [2.0, 2.0])
    block1.val = [0.4, -0.3]
    addBlockVariable!(mbp, block1)

    block2 = BlockVariable("x2")
    block2.f = QuadraticFunction(sparse([1.5 0.0; 0.0 2.5]), [0.5, -0.6], 0.0)
    block2.g = IndicatorBox([-2.0, -2.0], [2.0, 2.0])
    block2.val = [-0.2, 0.5]
    addBlockVariable!(mbp, block2)

    block3 = BlockVariable("u")
    block3.f = Zero()
    block3.g = Zero()
    block3.val = zeros(2)
    addBlockVariable!(mbp, block3)

    constr = BlockConstraint("c1")
    addBlockMappingToConstraint!(constr, "x1", LinearMappingIdentity(1.0))
    addBlockMappingToConstraint!(constr, "x2", LinearMappingIdentity(1.0))
    addBlockMappingToConstraint!(constr, "u", LinearMappingIdentity(-1.0))
    constr.rhs = zeros(2)
    addBlockConstraint!(mbp, constr)

    return mbp
end

function _run_adapdm_with(param_builder)
    mbp = _build_small_adapdm_problem()
    initialPresL2, _ = checkMultiblockProblemFeasibility(mbp)

    param = param_builder(mbp)
    result = runAdaPDM(mbp, param; tryJuMP=false)

    @test result !== nothing
    @test haskey(result.solution, "x1")
    @test haskey(result.solution, "x2")
    @test haskey(result.solution, "u")
    @test length(result.solution["x1"]) == 2
    @test length(result.solution["x2"]) == 2
    @test length(result.solution["u"]) == 2

    info = result.iterationInfo
    @test !isempty(info.objectiveValue)
    @test !isempty(info.lagrangianObj)
    @test !isempty(info.presL2)
    @test !isempty(info.dresL2)
    @test isfinite(info.objectiveValue[end]) || isinf(info.objectiveValue[end])
    @test isfinite(info.lagrangianObj[end]) || isinf(info.lagrangianObj[end])
    @test isfinite(info.presL2[end])
    @test isfinite(info.dresL2[end])
    @test info.terminationStatus != PDMO.ADA_PDM_TERMINATION_UNSPECIFIED

    finalPresL2, finalPresLInf = checkMultiblockProblemFeasibility(mbp, result.solution)
    @test isfinite(finalPresL2)
    @test isfinite(finalPresLInf)
    @test finalPresL2 <= 5.0 * max(initialPresL2, 1e-8)
end

@testset "AdaPDM Method Tests" begin
    @testset "AdaPDMParam" begin
        _run_adapdm_with(mbp -> AdaPDMParam(
            mbp;
            maxIter=40,
            presTolL2=1e-4,
            dresTolL2=1e-4,
            presTolLInf=1e-5,
            dresTolLInf=1e-5,
            logInterval=1000,
            timeLimit=20.0,
            logLevel=0,
        ))
    end

    @testset "AdaPDMPlusParam" begin
        _run_adapdm_with(mbp -> AdaPDMPlusParam(
            mbp;
            lineSearchMaxIter=20,
            maxIter=40,
            presTolL2=1e-4,
            dresTolL2=1e-4,
            presTolLInf=1e-5,
            dresTolLInf=1e-5,
            logInterval=1000,
            timeLimit=20.0,
            logLevel=0,
        ))
    end

    @testset "MalitskyPockParam" begin
        _run_adapdm_with(mbp -> MalitskyPockParam(
            mbp;
            lineSearchMaxIter=20,
            maxIter=40,
            presTolL2=1e-4,
            dresTolL2=1e-4,
            presTolLInf=1e-5,
            dresTolLInf=1e-5,
            logInterval=1000,
            timeLimit=20.0,
            logLevel=0,
        ))
    end

    @testset "CondatVuParam" begin
        _run_adapdm_with(mbp -> CondatVuParam(
            mbp;
            maxIter=40,
            presTolL2=1e-4,
            dresTolL2=1e-4,
            presTolLInf=1e-5,
            dresTolLInf=1e-5,
            logInterval=1000,
            timeLimit=20.0,
            logLevel=0,
        ))
    end
end
