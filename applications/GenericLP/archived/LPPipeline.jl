import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using PDMO

using LinearAlgebra
using SparseArrays
using JuMP 
import MathOptInterface 
using Ipopt
using HiGHS

include("GenericLP.jl")


function runLPPipeline(case::String; 
    admmSolver::AbstractADMMSubproblemSolver = AdaptiveLinearizedSolver(gamma=1.0, r=1000.0, ifSimple=false),
    initialRho::Float64 = 1000.0, 
    maxIter::Int = 1000000,
    logInterval::Int = 1000)
    mps_dir = "/home/kaizhao-sun/$(case).mps.gz"
    println("runLPPipeline: mps_dir = $(mps_dir)")

    lp = GenericLP(mps_dir)
    mbp = generateGenericLP(lp)

    param = ADMMParam(
        initialRho = initialRho,
        maxIter = maxIter,
        logInterval = logInterval,
        solver = admmSolver,
        applyScaling = false, 
        timeLimit = 1800.0
    )
    result = runBipartiteADMM(mbp, param)

    return result
end 


function runLPPipelineCoCluster(case::String; 
    admmSolver::AbstractADMMSubproblemSolver = AdaptiveLinearizedSolver(gamma=1.0, r=1000.0, ifSimple=false),
    bipartizationAlgorithm::BipartizationAlgorithm = MILP_BIPARTIZATION,
    initialRho::Float64 = 1000.0, 
    maxIter::Int = 1000000,
    logInterval::Int = 1000
)
    mps_dir = "/home/kaizhao-sun/$(case).mps.gz"
    println("runLPPipeline: mps_dir = $(mps_dir)")

    lp = GenericLP(mps_dir)
    mbp = generateGenericLPWithCoClustering(lp)

    param = ADMMParam(
        initialRho = initialRho,
        maxIter = maxIter,
        logInterval = logInterval,
        solver = admmSolver,
        applyScaling = false, 
        timeLimit = 1800.0
    )
    result = runBipartiteADMM(mbp, param; 
        bipartizationAlgorithm = bipartizationAlgorithm)

    return result
end 