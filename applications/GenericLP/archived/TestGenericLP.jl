import Pkg
Pkg.activate(".")

using PDMO

using LinearAlgebra
using SparseArrays
using JuMP 
import MathOptInterface 


include("GenericLP.jl")

mps_dir = "/home/kaizhao-sun/collection/stein45inf.mps.gz"
lp = GenericLP(mps_dir)
# mbp = generateGenericLP(lp)
mbp = generateGenericLP2(lp)

# setup ADMM parameters 
param = ADMMParam() 
param.solver = DoublyLinearizedSolver()
param.logInterval = 1000
param.initialRho = 10.0
# param.maxIter = 10000

# run ADMM 
runBipartiteADMM(mbp, param)