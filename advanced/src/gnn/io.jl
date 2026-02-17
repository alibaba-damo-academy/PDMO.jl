"""
Save RandomQP instance to JSON;
"""
# using PDMO
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


function readMbpFromJson(filename::String)
    json_data = JSON.parsefile(filename)

    mbp = MultiblockProblem()
    
    numberBlocks = length(json_data["blocks"])
    numberConstraints = length(json_data["constraints"])

    for i in 1:numberBlocks
        block = readBlockFromDict(json_data["blocks"][i])
        addBlockVariable!(mbp, block)
    end
    # println(mbp)
    for j in 1:numberConstraints
        constraint = readConstrFromDict(json_data["constraints"][j])
        addBlockConstraint!(mbp, constraint)
    end
    return mbp
end


function readConstrFromDict(constr_dict::Dict)
    constr = BlockConstraint(constr_dict["id"])
    for k in constr_dict["involvedBlocks"]
        addBlockMappingToConstraint!(constr, k, readMappingsFromDict(constr_dict["mappings"][string(k)]))
    end
    constr.rhs = Float64.(constr_dict["rhs"])
    return constr
end

function readMappingsFromDict(mappings_dict::Dict)
    if mappings_dict["MappingType"] == "LinearMappingMatrix"
        A = [Float64(mappings_dict["A"][i][j]) for i in 1:length(mappings_dict["A"]), j in 1:length(mappings_dict["A"][1])]
        sparse_A = sparse(A)
        transposed_A = copy(transpose(sparse_A))
        return LinearMappingMatrix(transposed_A)
    elseif mappings_dict["MappingType"] == "LinearMappingIdentity"
        error("Not Implemented")
    end

end

function readBlockFromDict(block_dict::Dict)
    block = BlockVariable(block_dict["id"])
    block.f = readFFromDict(block_dict["f"])
    block.g = readFFromDict(block_dict["g"])
    block.val = Float64.(block_dict["val"])
    return block
end


function readFFromDict(f_dict::Dict)
    if f_dict["FunctionType"]=="QuadraticFunction"
        Q = [Float64(f_dict["Q"][i][j]) for i in 1:length(f_dict["Q"]), j in 1:length(f_dict["Q"][1])]
        q = Float64.(f_dict["q"])
        r = Float64(f_dict["r"])
        return QuadraticFunction(sparse(Q),q,r)
    elseif f_dict["FunctionType"]=="IndicatorBox"
        lb = Float64.(f_dict["lb"])
        ub = Float64.(f_dict["ub"])
        return IndicatorBox(lb,ub)
    end
end

function _block_index(data::MultiblockProblem, block_id)
    if block_id isa Integer
        return Int(block_id)
    end
    idx = findfirst(block -> block.id == block_id, data.blocks)
    idx === nothing && error("Block ID not found in MultiblockProblem: $(block_id)")
    return idx
end

function _constraint_index(data::MultiblockProblem, constr_id)
    if constr_id isa Integer
        return Int(constr_id)
    end
    idx = findfirst(constr -> constr.id == constr_id, data.constraints)
    idx === nothing && error("Constraint ID not found in MultiblockProblem: $(constr_id)")
    return idx
end

function _mapping_matrix(mapping_obj)
    if hasproperty(mapping_obj, :A)
        return collect(mapping_obj.A)
    elseif hasproperty(mapping_obj, :coe)
        return [[mapping_obj.coe]]
    end
    error("Unsupported mapping type: $(typeof(mapping_obj))")
end

function toDict(node::PDMO.Node, assignment::Int64, data::MultiblockProblem)
    if node.type == PDMO.VARIABLE_NODE
        block_idx = _block_index(data, node.source)
        return Dict(
            "type"=>"VARIABLE_NODE", 
            "f"=>toDict(data.blocks[block_idx].f), 
            "g"=>toDict(data.blocks[block_idx].g), 
            "assigment"=>assignment,
            "neighboring_edges"=>node.neighbors)
    else 
        constr_idx = _constraint_index(data, node.source)
        number_blocks = length(data.constraints[constr_idx].involvedBlocks)
        rhs = data.constraints[constr_idx].rhs
        return Dict(
            "type"=>"CONSTRAINT_NODE", 
            "f"=>toDict(Zero()), 
            "g"=>toDict(IndicatorSumOfNVariables(number_blocks, rhs)), 
            "assigment"=>assignment, 
            "neighboring_edges"=>node.neighbors)
    end 
end


function toDict(edge::PDMO.Edge, splitting::Tuple{Int64, Int64}, data::MultiblockProblem)
    constr_idx = _constraint_index(data, edge.sourceBlockConstraint)
    node_i = edge.nodeID1 
    node_j = edge.nodeID2 

    if edge.type == PDMO.TWO_BLOCK_EDGE
        block_indices = data.constraints[constr_idx].involvedBlocks
        mapping = data.constraints[constr_idx].mappings
        # println(block_indices)
        # println(collect(mapping[block_indices[1]].A))
        # @assert false
        rhs = collect(data.constraints[constr_idx].rhs)
        return Dict(
            "type"=>"MULTIBLOCK__EDGE", 
            "node_i"=>node_i, 
            "node_j"=>node_j, 
            "constraints"=>Dict(id=>_mapping_matrix(mapping[block_indices[index]]) for (index,id) in enumerate(block_indices)),
            "rhs"=>rhs,
            "should_split"=>splitting[1], 
            "assignment"=>splitting[2]
        )

    else 
        block_indices = data.constraints[constr_idx].involvedBlocks
        mapping = data.constraints[constr_idx].mappings
        rhs = collect(data.constraints[constr_idx].rhs)
        # println(block_indices)
        # println(collect(mapping[block_indices[1]].A))
        # @assert false
        return Dict(
            "type"=>"MULTIBLOCK__EDGE", 
            "node_i"=>node_i, 
            "node_j"=>node_j, 
            "constraints"=>Dict(id=>_mapping_matrix(mapping[block_indices[index]]) for (index,id) in enumerate(block_indices)),
            "rhs"=>rhs,
            "should_split"=>splitting[1], 
            "assignment"=>splitting[2]
        )

        # block_idx = 0
        # block_pos_in_constr = 0 
        # for k in 1:number_blocks
        #     v = block_indices[k]
        #     if node_id(v) == node_i 
        #         block_pos_in_constr = k
        #         block_idx = v
        #         break 
        #     end 
        # end 
        # @assert(block_idx > 0, "ERROR: block not found.")
        
        # rows = Vector{ScalarConstraint}()
        # idx_offset = (block_pos_in_constr - 1) * number_constraints
        # for row_idx in 1:number_constraints
        #     row = data.constraints[constr_idx].constraints[row_idx]
        #     new_row = ScalarConstraint() 
        #     new_row.rhs = 0.0
        #     new_row.coupling_coefficient[node_i] = copy(row.coupling_coefficient[block_idx])
        #     new_row.coupling_coefficient[node_j] = zeros(number_blocks * number_constraints)
        #     new_row.coupling_coefficient[node_j][idx_offset + row_idx] = -1.0
        #     push!(rows, new_row)
        # end 

        # return Dict(
        #     "type"=>"MULTIBLOCK__EDGE", 
        #     "node_i"=>node_i, 
        #     "node_j"=>node_j, 
        #     "constraints"=>[toDict(row) for row in rows],
        #     "should_split"=>splitting[1], 
        #     "assignment"=>splitting[2]
        # )
    end 

end 




