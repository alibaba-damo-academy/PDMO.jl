mutable struct OneDimensionalQuadraticSolver <: SpecializedOriginalADMMSubproblemSolver
    nodeID::String
    quadraticCoefficient::Float64 
    linearCoefficient::Float64
    lb::Float64
    ub::Float64 
    currentRho::Float64 

    function OneDimensionalQuadraticSolver(nodeID::String, 
        admmGraph::ADMMBipartiteGraph, 
        edgeData::Dict{String, EdgeData}, 
        rho::Float64, 
        logLevel::Int64)

        node =admmGraph.nodes[nodeID]
        @assert(size(node.val) == (1,) && length(node.val) == 1, "OneDimensionalQuadraticSolver only supports 1-dimensional variables")
        @assert(isa(node.f, QuadraticFunction) || isa(node.f, AffineFunction), "OneDimensionalQuadraticSolver only supports QuadraticFunction or AffineFunction as f.")
        @assert(isa(node.g, IndicatorBox), "OneDimensionalQuadraticSolver only supports IndicatorBox with infinite bounds as g.")
        

        for edgeID in node.neighbors
            mapping = admmGraph.edges[edgeID].mappings[nodeID]
            @assert(isa(mapping, LinearMappingIdentity) || isa(mapping, LinearMappingMatrix), "OneDimensionalQuadraticSolver: only supports LinearMappingIdentity or LinearMappingMatrix as mapping.")
        end 

        if isa(node.f, QuadraticFunction)
            quadraticCoefficient = node.f.Q[1,1]
            linearCoefficient = node.f.q[1]
        elseif isa(node.f, AffineFunction)
            quadraticCoefficient = 0.0
            linearCoefficient = node.f.A[1]
        end 
        lb = node.g.lb[1]
        ub = node.g.ub[1]

        return new(nodeID, quadraticCoefficient, linearCoefficient, lb, ub, rho)

    end 
end 

function solve!(solver::OneDimensionalQuadraticSolver, 
    nodeID::String, 
    admmGraph::ADMMBipartiteGraph, 
    info::ADMMIterationInfo,
    edgeData::Dict{String, EdgeData}, 
    augmentedLagrangianLinearCoefficientsBuffer::Dict{String, NumericVariable},
    enableParallel::Bool = false)

    a = solver.quadraticCoefficient
    b = solver.linearCoefficient + augmentedLagrangianLinearCoefficientsBuffer[nodeID][1]

    for edgeID in admmGraph.nodes[nodeID].neighbors
        mapping = admmGraph.edges[edgeID].mappings[nodeID]
        if isa(mapping, LinearMappingIdentity)
            coe = mapping.coe 
            a += 0.5 * solver.currentRho * coe^2
        elseif isa(mapping, LinearMappingMatrix)
            A = mapping.A
            a += 0.5 * solver.currentRho * dot(A, A)
        end 
    end 

    solution = clamp(-b / (2 * a), solver.lb, solver.ub)
    
    info.primalSolPrev[nodeID][1] = info.primalSol[nodeID][1]
    info.primalSol[nodeID][1] = solution
end 

function update!(solver::OneDimensionalQuadraticSolver, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph, rhoUpdated::Bool)
    if rhoUpdated == false 
        return 
    end 
    solver.currentRho = info.rhoHistory[end][1]
end 