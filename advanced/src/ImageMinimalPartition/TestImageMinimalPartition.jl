import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

include("../include.jl")
include("ImageMinimalPartition.jl")
include("../../../src/plotting.jl")

"""
    testImageMinimalPartition(; kwargs...)

Run the Image Minimal Partition example with both:
- a multi-constraint ADMM formulation (`ImageMinimalPartition`)
- a single-constraint composite AdaPDM formulation (`ImageMinimalPartition_PDM`)

This file is intended as an executable experiment script.
"""
function testImageMinimalPartition(;
    imagePath::String = abspath(joinpath(@__DIR__, "image/NYU1398.jpg")),
    Q::Int64 = 20,
    lambda::Float64 = 20.0,
    color::Bool = false,
    testADMM::Bool = true,
    testAdaPDMPlus::Bool = true,
    testMalitskyPock::Bool = true,
    testCondatVu::Bool = true,
    pdmMaxIter::Int = 10000,
    admmMaxIter::Int = 10000,
    admmInitialRho::Float64 = 1.0,
    seed::Int = 44,
)
    @assert Q > 2 "Q must be > 2"

    Random.seed!(seed)

    folder_name = "ImageMinimalPartition"

    if testADMM
        mbp = ImageMinimalPartition(imagePath, Q, lambda)

        param = ADMMParam()
        param.maxIter = admmMaxIter
        param.solver = AdaptiveLinearizedSolver(gamma=1.0, r=1.0, ifSimple=false)
        info =  runBipartiteADMM(mbp, param)
        saveJSONL(info.iterationInfo , "Subroutine 2 (σ=1.0)" , folder_name)

        param = ADMMParam()
        param.maxIter = admmMaxIter
        param.solver = AdaptiveLinearizedSolver(gamma=1.0, r=1.0, ifSimple=true)
        info =  runBipartiteADMM(mbp, param)
        saveJSONL(info.iterationInfo  , "Subroutine 1 (σ=1.0)" , folder_name)

        param = ADMMParam()
        param.maxIter = admmMaxIter
        param.initialRho =  1.0
        param.solver = DoublyLinearizedSolver()
        info =  runBipartiteADMM(mbp, param)
        saveJSONL(info.iterationInfo  ,  "FLiP ADMM (ρ=1.0)" , folder_name)
    end

    if testAdaPDM || testAdaPDMPlus || testMalitskyPock || testCondatVu
        mbp_pdm = ImageMinimalPartition_PDM(imagePath, Q, lambda)

        if testAdaPDMPlus
            t_value = steps
            paramPDMPlus = AdaPDMPlusParam(mbp_pdm; t=10.0)
            paramPDMPlus.maxIter = pdmMaxIter
            info = runAdaPDM(mbp_pdm, paramPDMPlus)
            saveJSONL_adapdm(info.iterationInfo, "adaPDM+ (t=10.0)", folder_name)
        end
        if testMalitskyPock
            t_value = steps
            paramMalitskyPock = MalitskyPockParam(mbp_pdm; t=0.1)
            paramMalitskyPock.maxIter = pdmMaxIter
            info = runAdaPDM(mbp_pdm, paramMalitskyPock)
            saveJSONL_adapdm(info.iterationInfo, "Malitsky–Pock (β=0.1)", folder_name)
        end
        if testCondatVu
            paramCondatVu = CondatVuParam(mbp_pdm)
            paramCondatVu.maxIter = pdmMaxIter
            info = runAdaPDM(mbp_pdm, paramCondatVu)
            saveJSONL_adapdm(info.iterationInfo, "Condat-Vu", folder_name)
        end
    end
    plotting(folder_name , "max" ; smoothing = true) 
end


# Default run when executed as a script
testImageMinimalPartition(testADMM = true)
