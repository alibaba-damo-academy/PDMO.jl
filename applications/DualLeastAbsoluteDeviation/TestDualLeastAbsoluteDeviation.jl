import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using PDMO

using LinearAlgebra
using SparseArrays
using Random 
Random.seed!(77)

include("DualLeastAbsoluteDeviation.jl")
include("../datasets/libsvm.jl")
include("../../src/plotting.jl")

folder_name = "DualLeastAbsoluteDeviation"

X, y = load_libsvm_dataset( joinpath(@__DIR__, "../", "datasets/downloaded", "cpusmall_scale"), Float64)  
m, n = size(X)
X_aug    = sparse(transpose(hcat(Matrix(X),  ones(m))))
λ = 0.1
maxIter = 10000

mbp = generateDualLeastAbsoluteDeviation(X_aug, -y, λ)
# setup ADMM parameters 
param = ADMMParam() 
param.solver = AdaptiveLinearizedSolver(gamma=1.0, r=100.0 , ifSimple=true)
param.logInterval = 100
param.initialRho = 50.0
param.maxIter = maxIter
# run ADMM 
info = runBipartiteADMM(mbp, param)
saveJSONL(info.iterationInfo , "Subroutine 1 (σ=100.0)", folder_name)


mbp = generateDualLeastAbsoluteDeviation(X_aug, -y, λ)
# setup ADMM parameters 
param = ADMMParam() 
param.solver = AdaptiveLinearizedSolver(gamma=1.0, r=1.0 , ifSimple=false)
param.logInterval = 100
param.initialRho = 50.0
param.maxIter = maxIter
# run ADMM 
info = runBipartiteADMM(mbp, param)
saveJSONL(info.iterationInfo , "Subroutine 2 (σ=1.0)", folder_name)



mbp = generateDualLeastAbsoluteDeviation(X_aug, -y, λ)
param = AdaPDMPlusParam(mbp; t = 1.0)
param.maxIter = maxIter
info = runAdaPDM(
mbp, param
)
saveJSONL_adapdm(info.iterationInfo, "adaPDM+ (t=1.0)", folder_name)


mbp = generateDualLeastAbsoluteDeviation(X_aug, -y, λ)
param = MalitskyPockParam(mbp; t = 1.0)
param.maxIter = maxIter
info = runAdaPDM(
mbp, param
)
saveJSONL_adapdm(info.iterationInfo, "Malitsky–Pock (β=1.0)", folder_name)
    

plotting(folder_name , "max" ; smoothing = true) 



