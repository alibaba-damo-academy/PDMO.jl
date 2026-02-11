mutable struct ConsensusSolver <: SpecializedOriginalADMMSubproblemSolver
    nodeID1::String
    nodeID2::String
    edgeID1::String
    edgeID2::String
    buffer1::NumericVariable
    buffer2::NumericVariable
    logLevel::Int64

end 


function ConsensusSolver(nodeID::String, 
    admmGraph::ADMMBipartiteGraph, 
    edgeData::Dict{String, EdgeData}, 
    initialRho::Float64, 
    logLevel::Int64)

    node = admmGraph.nodes[nodeID]
    @assert(isa(node.g, IndicatorSumOfNVariables) && node.g.numberVariables == 2, "ConsensusSolver only supports IndicatorSumOfNVariables with 2 variables")
    @assert(isa(node.f, Zero), "ConsensusSolver only supports Zero objective function")
    @assert(length(node.neighbors) == 2, "ConsensusSolver only supports nodes with 2 neighbors")


    nodeID1 = nothing 
    nodeID2 = nothing 
    edgeID1 = nothing 
    edgeID2 = nothing 
    for edgeID in node.neighbors
        edge = admmGraph.edges[edgeID]
        @assert(edge.nodeID2 == nodeID, "ConsensusSolver only supports edges connected to the node")
        @assert(isa(edge.mappings[edge.nodeID2], LinearMappingExtraction), "ConsensusSolver only supports LinearMappingExtraction mappings")

        if edge.mappings[edge.nodeID2].indexStart == 1 
            nodeID1 = edge.nodeID1 
            edgeID1 = edgeID 
        else 
            nodeID2 = edge.nodeID1 
            edgeID2 = edgeID 
        end 
    end 

    @PDMODebug logLevel "OriginalADMMSubproblemSolve: ADMM node $nodeID initialized with ConsensusSolver."

    return ConsensusSolver(
        nodeID1,
        nodeID2,
        edgeID1,
        edgeID2,
        similar(node.g.rhs), 
        similar(node.g.rhs),
        logLevel)
end 


function solve!(solver::ConsensusSolver,
    nodeID::String, 
    admmGraph::ADMMBipartiteGraph, 
    info::ADMMIterationInfo,
    edgeData::Dict{String, EdgeData}, 
    augmentedLagrangianLinearCoefficientsBuffer::Dict{String, NumericVariable}, 
    enableParallel::Bool = false)

    node = admmGraph.nodes[nodeID]
    @assert !isa(node.g.rhs, Number) "ConsensusSolver expects non-scalar rhs (array-like variable blocks)."
    d1 = size(node.g.rhs, 1)
    rho = info.rhoHistory[end][1]
    fill!(solver.buffer1, 0.0)
    fill!(solver.buffer2, 0.0)
    
    # (lmd1 - lmd2) / (2 * rho)
    copyto!(solver.buffer1, info.dualSol[solver.edgeID1])
    axpy!(-1.0, info.dualSol[solver.edgeID2], solver.buffer1)
    solver.buffer1 ./= 2.0 * rho

    # (Q1x1 -Q2x2 + b)/2
    admmGraph.edges[solver.edgeID2].mappings[solver.nodeID2](info.primalSol[solver.nodeID2], solver.buffer2, true)
    solver.buffer2 .*= -1.0 
    admmGraph.edges[solver.edgeID1].mappings[solver.nodeID1](info.primalSol[solver.nodeID1], solver.buffer2, true)
    axpy!(1.0, node.g.rhs, solver.buffer2)
    solver.buffer2 ./= 2.0 

    # buffer2 = z1 = (Q1x1 -Q2x2 + b)/2 + (lmd1 - lmd2) / (2 * rho)
    axpy!(1.0, solver.buffer2, solver.buffer1)

    # buffer2 = z2 = b-z1 
    copyto!(solver.buffer2, node.g.rhs)
    axpy!(-1.0, solver.buffer1, solver.buffer2)

    # save previous iterate before overwriting
    copyto!(info.primalSolPrev[nodeID], info.primalSol[nodeID])

    # Write the solution into this node's primal variable as a vertical concatenation:
    # info.primalSol[nodeID] = vcat(buffer1, buffer2) along the first dimension.
    tail = ntuple(_ -> Colon(), ndims(info.primalSol[nodeID]) - 1)
    block1 = view(info.primalSol[nodeID], 1:d1, tail...)
    block2 = view(info.primalSol[nodeID], (d1 + 1):(2 * d1), tail...)
    copyto!(block1, solver.buffer1)
    copyto!(block2, solver.buffer2)
end 


function update!(solver::ConsensusSolver, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph, rhoUpdated::Bool)
   return 
end
