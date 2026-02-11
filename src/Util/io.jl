"""
Save RandomQP instance to JSON;
"""

""" 
Functions
"""
function toDict(f::AffineFunction)
    return Dict("A"=>f.A, "offset"=>f.r, "FunctionType"=>"AffineFunction")
end

function toDict(f::Zero)
    return Dict("FunctionType"=>"Zero")
end 

function toDict(f::IndicatorBox)
    return Dict("lb"=>f.lb, "ub"=>f.ub, "FunctionType"=>"IndicatorBox")
end 

function toDict(f::IndicatorBallL2)
    return Dict("r"=>f.r, "FunctionType"=>"IndicatorBallL2")
end 

function toDict(f::ElementwiseL1Norm)
    return Dict("coefficient"=>f.coefficient, "FunctionType"=>"ElementwiseL1Norm")
end

function toDict(f::IndicatorSumOfNVariables)
    return Dict(
        "numberVariables"=>f.numberVariables,
        "rhs"=>f.rhs,
        "FunctionType"=>"IndicatorSumOfNVariables"
    )
end

function toDict(f::IndicatorHyperplane)
    return Dict(
        "slope"=>f.slope,
        "intercept"=>f.intercept,
        "FunctionType"=>"IndicatorHyperplane"
    )
end

function toDict(f::QuadraticFunction)
    return Dict("Q"=>f.Q, "q"=>f.q, "r"=>f.r, "FunctionType"=>"QuadraticFunction")
end 

function toDict(f::IndicatorLinearSubspace)
    return Dict(
        "A" => f.A,
        "b" => f.b,
        "U" => f.U,
        "S" => f.S,
        "V" => f.V,
        "rank" => f.rank,
        "isFullRank" => f.isFullRank,
        "FunctionType" => "IndicatorLinearSubspace"
    )
end

""" 
Mappings
"""
function toDict(L::LinearMappingMatrix)
    return Dict(
        "A"=>L.A,
        "inputDim"=>L.inputDim,
        "outputDim"=>L.outputDim,
        "MappingType"=>"LinearMappingMatrix"
    )
end

function toDict(L::LinearMappingIdentity)
    return Dict(
        "coe"=>L.coe,
        "MappingType"=>"LinearMappingIdentity"
    )
end

function toDict(L::LinearMappingExtraction)
    return Dict(
        "dim"=>collect(L.dim),  # Convert tuple to array for JSON serialization
        "coe"=>L.coe,
        "indexStart"=>L.indexStart,
        "indexEnd"=>L.indexEnd,
        "MappingType"=>"LinearMappingExtraction"
    )
end

""" 
Block Constraints and Block Variables
"""
function toDict(block::BlockVariable)
    return Dict(
        "id"=>block.id,
        "f"=>toDict(block.f),
        "g"=>toDict(block.g),
        "val"=>block.val
    )
end

function toDict(constr::BlockConstraint)
    return Dict(
        "id"=>constr.id,
        "involvedBlocks"=>constr.involvedBlocks,
        "mappings"=>Dict(id=>toDict(mapping) for (id, mapping) in constr.mappings),
        "rhs"=>constr.rhs
    )
end

""" 
Formulation
"""
function toDict(mbp::MultiblockProblem)
    return Dict(
        "blocks" => [toDict(block) for block in mbp.blocks],
        "constraints" => [toDict(constr) for constr in mbp.constraints],
        "FormulationType" => "MultiblockProblem"
    )
end

"""
    save_multiblock_problem_to_json(mbp::MultiblockProblem, filename::String)

Save a MultiblockProblem instance to a JSON file.

# Arguments
- `mbp::MultiblockProblem`: The problem instance to save
- `filename::String`: The path where the JSON file should be saved

# Example
```julia
save_multiblock_problem_to_json(problem, "problem.json")
```
"""
function saveMultiblockProblemToJSON(mbp::MultiblockProblem, filename::String)
    # Validate the problem before saving
    if !checkMultiblockProblemValidity(mbp, addWrapper=false)
        error("MultiblockProblem: Cannot save invalid problem instance")
    end

    # Convert to dictionary
    json_data = toDict(mbp)

    # Save to file
    open(filename, "w") do io
        JSON.print(io, json_data, 4)  # Use 4 spaces for indentation
    end

    @info "MultiblockProblem saved to $filename"
end 

"""
Graph Components
"""
function toDict(node::Node)
    return Dict(
        "neighbors" => node.neighbors,
        "type" => node.type == VARIABLE_NODE ? "VARIABLE_NODE" : "CONSTRAINT_NODE",
        "source" => node.source,
        "NodeType" => "Node"
    )
end

function toDict(edge::Edge)
    return Dict(
        "nodeID1" => edge.nodeID1,
        "nodeID2" => edge.nodeID2,
        "type" => edge.type == TWO_BLOCK_EDGE ? "TWO_BLOCK_EDGE" : "MULTIBLOCK_EDGE",
        "sourceBlockConstraint" => edge.sourceBlockConstraint,
        "sourceBlockVariable" => edge.sourceBlockVariable,
        "EdgeType" => "Edge"
    )
end

function toDict(graph::MultiblockGraph)
    return Dict(
        "nodes" => Dict(nodeID => toDict(node) for (nodeID, node) in graph.nodes),
        "edges" => Dict(edgeID => toDict(edge) for (edgeID, edge) in graph.edges),
        "colors" => graph.colors,
        "isBipartite" => graph.isBipartite,
        "isConnected" => graph.isConnected,
        "GraphType" => "MultiblockGraph"
    )
end

"""
    save_multiblock_graph_to_json(graph::MultiblockGraph, filename::String)

Save a MultiblockGraph instance to a JSON file.

# Arguments
- `graph::MultiblockGraph`: The graph instance to save
- `filename::String`: The path where the JSON file should be saved
"""
function saveMultiblockGraphToJson(graph::MultiblockGraph, filename::String)
    json_data = toDict(graph)
    
    open(filename, "w") do io
        JSON.print(io, json_data, 4)  # Use 4 spaces for indentation
    end

    @info "MultiblockGraph saved to $filename"
end 

