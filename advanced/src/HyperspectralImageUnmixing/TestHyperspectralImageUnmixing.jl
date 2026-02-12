import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using CSV, DataFrames

include("../include.jl")
include("HyperspectralImageUnmixing.jl")
include("../../../src/plotting.jl")

L = 224 # L-spectral band
N = 50  # number of endmemebrs 
K = 9   # number of adjacent pixels


# Read Q from CSV file instead of generating it randomly
# Use absolute path to ensure the file can be found
Q_path = abspath(joinpath(@__DIR__, "usgs_matrix.csv"))
println("Looking for Q matrix at: $(Q_path)")
if isfile(Q_path)
    # Read the CSV file
    Q_df = CSV.read(Q_path, DataFrame)
    
    # Ignore the first column (convert to matrix and take columns 2:end)
    Q = Matrix(Q_df[:, 2:end])
    println("Loaded Q matrix from CSV (ignoring first column): $(size(Q))")
else # synthetic data 
    println("Warning: Could not find $(Q_path)")
    println("Generating random Q matrix instead")
    Q = rand(L, 2*N) # Fallback to random generation if file not found
end

useConvexFormulation = true # if true, use same weights for nuclear norm
Phi, Y, A, b = hyperSpectralImageData(Q, N, K, useConvexFormulation)
coefL1Norm = 0.1
coefNuclearNorm = 0.1


function testHyperspectralImageUnmixing(;
    nBlocks::Int64,
    testADMM::Bool = true,
    testAdaPDM::Bool = true,
    testMalitskyPock::Bool = true,
    testCondatVu::Bool = true,
    admmInitialRho::Float64 = 50.0,
)
    @assert nBlocks in (2, 3, 4, 5) "nBlocks must be one of (2, 3, 4, 5)"

    folder_name = "HyperspectralImageUnmixing"

    if testADMM
        # Build formulations
        mbp_admm = if nBlocks == 2
            HyperspectralImageUnmixing_2Blocks(Phi, Y, A, b, coefL1Norm, coefNuclearNorm)
        elseif nBlocks == 3
            HyperspectralImageUnmixing_3Blocks(Phi, Y, A, b, coefL1Norm, coefNuclearNorm)
        elseif nBlocks == 4
            HyperspectralImageUnmixing_4Blocks(Phi, Y, A, b, coefL1Norm, coefNuclearNorm)
        else
            HyperspectralImageUnmixing_5Blocks(Phi, Y, A, b, coefL1Norm, coefNuclearNorm)
        end
        
        param = ADMMParam()
        param.maxIter = 20000
        param.initialRho = admmInitialRho
        param.solver = AdaptiveLinearizedSolver(gamma=1.0, r=1.0, ifSimple=false)
        info =  runBipartiteADMM(mbp_admm, param; saveSolutionInMultiblockProblem=false)
        saveJSONL(info.iterationInfo , "Subroutine 2 (σ=1.0)" , folder_name)

        param = ADMMParam()
        param.maxIter = 20000
        param.initialRho = admmInitialRho
        param.solver = AdaptiveLinearizedSolver(gamma=1.0, r=1.0, ifSimple=true)
        info =  runBipartiteADMM(mbp_admm, param; saveSolutionInMultiblockProblem=false)
        saveJSONL(info.iterationInfo , "Subroutine 1 (σ=1.0)" , folder_name)

        param = ADMMParam()
        param.maxIter = 20000
        param.initialRho =  1.0
        param.solver = DoublyLinearizedSolver()
        info =  runBipartiteADMM(mbp_admm, param; saveSolutionInMultiblockProblem=false)
        saveJSONL(info.iterationInfo ,  "FLiP ADMM (σ=1.0)" , folder_name)
    end

    if testAdaPDM || testAdaPDMPlus || testMalitskyPock || testCondatVu
        mbp_pdm = if nBlocks == 2
            HyperspectralImageUnmixing_2Blocks(Phi, Y, A, b, coefL1Norm, coefNuclearNorm)
        elseif nBlocks == 3
            HyperspectralImageUnmixing_3Blocks_PDM(Phi, Y, A, b, coefL1Norm, coefNuclearNorm)
        elseif nBlocks == 4
            HyperspectralImageUnmixing_4Blocks_PDM(Phi, Y, A, b, coefL1Norm, coefNuclearNorm)
        else
            nothing
        end
    
        if mbp_pdm === nothing
            @warn "No PDM formulation is defined for nBlocks=$nBlocks; skipping AdaPDM-family runs."
            return
        end

        if testAdaPDM
            paramPDM = AdaPDMParam(mbp_pdm; t = 1.0)
            paramPDM.maxIter = 20000
            info = runAdaPDM(mbp_pdm, paramPDM; saveSolutionInMultiblockProblem=false)
            saveJSONL_adapdm(info.iterationInfo, "adaPDM (t=1.0)", folder_name)
        end
        if testMalitskyPock
            paramMalitskyPock = MalitskyPockParam(mbp_pdm; t = 0.5)
            paramMalitskyPock.maxIter = 20000
            info =runAdaPDM(mbp_pdm, paramMalitskyPock; saveSolutionInMultiblockProblem=false)
            saveJSONL_adapdm(info.iterationInfo, "Malitsky–Pock (β=0.5)", folder_name)
        end
        if testCondatVu
            paramCondatVu = CondatVuParam(mbp_pdm)
            paramCondatVu.maxIter = 20000
            info =runAdaPDM(mbp_pdm, paramCondatVu; saveSolutionInMultiblockProblem=false)
            saveJSONL_adapdm(info.iterationInfo, "Condat–Vu", folder_name)
        end
    end 
    plotting(folder_name , "max" ; smoothing = true) 
end 

# Default run when executed as a script
testHyperspectralImageUnmixing(testADMM=true, nBlocks=3)