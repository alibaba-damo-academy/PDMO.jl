using Test
using PDMO
using SparseArrays

"""
Build a small 2-block constrained problem:
    min f1(x1) + f2(x2) + g1(x1) + g2(x2)
    s.t. x1 + x2 = rhs
"""
function _build_small_admm_problem()
    mbp = MultiblockProblem()

    block1 = BlockVariable("x1")
    block1.f = QuadraticFunction(sparse([2.0 0.0; 0.0 1.0]), [-1.0, 0.5], 0.0)
    block1.g = IndicatorBox([-2.0, -2.0], [2.0, 2.0])
    block1.val = [0.4, -0.2]
    addBlockVariable!(mbp, block1)

    block2 = BlockVariable("x2")
    block2.f = QuadraticFunction(sparse([1.5 0.0; 0.0 2.5]), [0.3, -0.4], 0.0)
    block2.g = IndicatorBox([-2.0, -2.0], [2.0, 2.0])
    block2.val = [-0.1, 0.3]
    addBlockVariable!(mbp, block2)

    constr = BlockConstraint("c1")
    addBlockMappingToConstraint!(constr, "x1", LinearMappingIdentity(1.0))
    addBlockMappingToConstraint!(constr, "x2", LinearMappingIdentity(1.0))
    constr.rhs = [0.2, -0.1]
    addBlockConstraint!(mbp, constr)

    return mbp
end

function _run_admm_with(solver, adapter)
    mbp = _build_small_admm_problem()
    initialPresL2, _ = checkMultiblockProblemFeasibility(mbp)

    param = ADMMParam(
        initialRho = 1.0,
        maxIter = 40,
        presTolL2 = 1e-4,
        dresTolL2 = 1e-4,
        presTolLInf = 1e-5,
        dresTolLInf = 1e-5,
        solver = solver,
        adapter = adapter,
        accelerator = NullAccelerator(),
        logInterval = 1000,
        timeLimit = 20.0,
        applyScaling = false,
        enablePathologyCheck = false,
        logLevel = 0,
    )

    result = runBipartiteADMM(
        mbp,
        param;
        bipartizationAlgorithm = BFS_BIPARTIZATION,
        tryJuMP = false,
    )

    @test result !== nothing
    @test haskey(result.solution, "x1")
    @test haskey(result.solution, "x2")
    @test length(result.solution["x1"]) == 2
    @test length(result.solution["x2"]) == 2

    info = result.iterationInfo
    @test !isempty(info.obj)
    @test !isempty(info.presL2)
    @test !isempty(info.dresL2)
    @test isfinite(info.obj[end])
    @test isfinite(info.presL2[end])
    @test isfinite(info.dresL2[end])
    @test info.terminationStatus != PDMO.ADMM_TERMINATION_UNSPECIFIED

    finalPresL2, finalPresLInf = checkMultiblockProblemFeasibility(mbp, result.solution)
    @test isfinite(finalPresL2)
    @test isfinite(finalPresLInf)
    # Allow mild non-monotonicity; just ensure the run does not diverge badly.
    @test finalPresL2 <= 5.0 * max(initialPresL2, 1e-8)

    return result
end

@testset "ADMM Bipartite Method Tests" begin
    @testset "DoublyLinearized + NullAdapter" begin
        _run_admm_with(DoublyLinearizedSolver(), NullAdapter())
    end

    @testset "OriginalADMM + NullAdapter" begin
        _run_admm_with(OriginalADMMSubproblemSolver(), NullAdapter())
    end

    @testset "AdaptiveLinearized + NullAdapter" begin
        _run_admm_with(AdaptiveLinearizedSolver(ifSimple=true), NullAdapter())
    end

    @testset "DoublyLinearized + RBAdapter" begin
        result = _run_admm_with(DoublyLinearizedSolver(), RBAdapter(testRatio=5.0, adapterRatio=2.0))
        @test length(result.iterationInfo.rhoHistory) >= 1
    end

    @testset "DoublyLinearized + SRAAdapter" begin
        result = _run_admm_with(DoublyLinearizedSolver(), SRAAdapter(T=2))
        @test length(result.iterationInfo.rhoHistory) >= 1
    end
end
