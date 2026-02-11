module PDMO

using LinearAlgebra
using SparseArrays
using JSON 
using Dates 
using FilePathsBase
using Logging 
using Base.Threads 
using Test
using Ipopt
using DataStructures
using Random
Random.seed!(126)

import Printf
import HiGHS
import JuMP

const norm = LinearAlgebra.norm 
const opnorm = LinearAlgebra.opnorm
const nnz = SparseArrays.nnz
const sparse = SparseArrays.sparse
const spzeros = SparseArrays.spzeros
const SparseVector = SparseArrays.SparseVector
const SparseMatrixCSC = SparseArrays.SparseMatrixCSC

const ZeroTolerance = 1.0e-12
const MaxFiniteValue = 1.0e30
const FeasTolerance = 1.0e-6

# Check if HSL_jll is available
const HSL_FOUND = try
    import HSL_jll
    # Check if HSL is properly linked
    occursin(r"override/lib/[^/]+/libhsl\.(so|dylib)", HSL_jll.libhsl_path)
catch
    false
end

# Print HSL status
if HSL_FOUND == false 
    @warn "PDMO: HSL is not properly linked or not available. Using default Ipopt linear solver."
end

export ZeroTolerance, MaxFiniteValue, FeasTolerance

include("main.jl")


# Export core functions 
export NumericVariable 
export AbstractFunction 
export isProximal, isSmooth, isConvex, isSet
export proximalOracle, proximalOracle!
export gradientOracle, gradientOracle!

# realizations of AbstractFunction 
export AffineFunction 
export ComponentwiseExponentialFunction 
export ElementwiseL1Norm 
export FrobeniusNormSquare 
export IndicatorBallL2 
export IndicatorBinary 
export IndicatorBox 
export IndicatorHyperplane 
export IndicatorMixedInteger 
export IndicatorLinearSubspace 
export IndicatorNonnegativeOrthant 
export IndicatorPSD 
export IndicatorRotatedSOC 
export IndicatorSOC 
export IndicatorSumOfNVariables 
export MatrixNuclearNorm 
export QuadraticFunction 
export UserDefinedSmoothFunction 
export UserDefinedProximalFunction 
export StackingProximalFunctions
export WeightedMatrixL1Norm 
export WrapperScalarInputFunction 
export WrapperScalingTranslationFunction 
export Zero 

# Export the conjugate proximal oracle functions
export proximalOracleOfConjugate, proximalOracleOfConjugate!

# Export utility functions
export estimateLipschitzConstant
export DEC, parseDEC

# Export multiblock function types and APIs
export AbstractMultiblockFunction, QuadraticMultiblockFunction
export partialGradientOracle!, partialGradientOracle
export getNumberOfBlocks, validateBlockDimensions
export JuMPAddPartialBlockFunction


# Export core mappings and abstract types
export AbstractMapping 
export NullMapping

# Export core interface functions
export adjoint
export adjoint!
export createAdjointMapping
export operatorNorm2

export LinearMappingIdentity 
export LinearMappingExtraction 
export LinearMappingMatrix 
export LinearMappingIdentityStacking
export LinearMappingStacking


# Export core types and functions for MultiblockGraph functionality
export BlockID 
export BlockVariable, checkBlockVariableValidity 
export BlockConstraint, checkBlockConstraintValidity 
export MultiblockProblem, addBlockVariable!, addBlockConstraint!
export checkMultiblockProblemValidity
export checkMultiblockProblemFeasibility
export checkCompositeProblemValidity!
export createFeasibilityProblem, transformConstraintsToQuadraticPenalty

export solveMultiblockProblemByJuMP
export MultiblockGraph, numberNodes, numberEdges, numberEdgesByTypes, getNodelNeighbors
export isMultiblockGraphBipartite, isMultiblockGraphConnected
export BfsBipartization, MilpBipartization, GNN_BIPARTIZATION, ADMMBipartiteGraph
export BipartizationAlgorithm
export BFS_BIPARTIZATION, MILP_BIPARTIZATION, DFS_BIPARTIZATION, SPANNING_TREE_BIPARTIZATION
export getBipartizationAlgorithmName
export DfsBipartization, SpanningTreeBipartization

# Export MultiblockGraph internal types and functions
export NodeType, EdgeType, Node, Edge
export createNodeID, createEdgeID

# Export ADMMBipartiteGraph types and functions  
export ADMMNode, ADMMEdge
export createADMMNodeID, createADMMEdgeID

# Export scaling functionality - commented out per user request
# export ScalingStrategy, ScalingOptions
# export MultiblockScalingInfo, ConstraintScalingAnalysis
# export scaleMultiblockProblem!, unscaleMultiblockProblem!
# export CONSERVATIVE, MODERATE, AGGRESSIVE
# export conservativeScaling, moderateScaling, aggressiveScaling, customScaling
# export reportScalingResults, getScalingStrategyInfo, allScalingFactorsAreOne

# Export JuMP interface functions  
export isSupportedByJuMP, JuMPAddProximableFunction, JuMPAddSmoothFunction
export addBlockVariableToJuMPModel!

# Export summary functions
export summary

# Export ADMM algorithm components
export ADMMNode, ADMMEdge, createADMMNodeID, createADMMEdgeID
export ADMMParam
export ADMMIterationInfo
export AbstractADMMAccelerator, AndersonAccelerator, AutoHalpernAccelerator, NullAccelerator
export AbstractADMMAdapter, RBAdapter, SRAAdapter, NullAdapter
export OriginalADMMSubproblemSolver, DoublyLinearizedSolver, AdaptiveLinearizedSolver
export AbstractADMMSubproblemSolver, SpecializedOriginalADMMSubproblemSolver
export LinearSolver, JuMPSolver, ProximalMappingSolver, ConsensusSolver

# export initialize!
# export update!
# export solve!
# export updateDualResidualsInBuffer!
# export getADMMSubproblemSolverName


# Export AdaPDM algorithm components
export AbstractAdaPDMParam, AdaPDMParam, AdaPDMPlusParam, MalitskyPockParam, CondatVuParam
export AdaPDMIterationInfo, AdaPDMTerminationStatus
export computeNormEstimate, getAdaPDMName
export updateDualSolution!, updatePrimalSolution!, setupInitialPrimalDualStepSize!
export computeAdaPDMDualResiduals!, computeAdaPDMPrimalResiduals!, computePDMResidualsAndObjective!
export computePartialObjective!, computeLipschitzAndCocoercivityEstimate
export prepareProximalCenterForConjugateProximalOracle!, prepareProximalCenterForPrimalUpdate!
export AdaPDMTerminationCriteria
export checkTerminationCriteria, checkOptimalTermination, checkIterationLimit
export checkTimeLimit, checkNumericalError, checkUnboundedness, getTerminationStatus
export AdaPDMLog

# Export BCD algorithm components
export AbstractBlockUpdateOrder, CyclicRule, updateBlockOrder!
export BCDParam, BCDIterationInfo, BCDTerminationStatus, BCDTerminationCriteria
export AbstractBCDSubproblemSolver, BCDProximalSubproblemSolver, BCDProximalLinearSubproblemSolver
export initialize!, solve!, updateDualResidual!, getBCDSubproblemSolverName
export BCDLog

# Export core algorithm functions
export runAdaPDM
export runBipartiteADMM
export runBCD

# Export PDMO logging macros
export @PDMOInfo, @PDMOWarn, @PDMOError, @PDMODebug

end # module PDMO
