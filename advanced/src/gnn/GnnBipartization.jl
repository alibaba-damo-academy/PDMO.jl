const GNN_ROOT = @__DIR__
const ADVANCED_DIR = abspath(joinpath(GNN_ROOT, "..", ".."))
const GNN_DIR = joinpath(GNN_ROOT, "GNN")
const DEFAULT_MODEL_PATH = joinpath(GNN_DIR, "model_weights_10.pth")
const GNN_MODEL_PATH = get(ENV, "PDMO_GNN_WEIGHTS", DEFAULT_MODEL_PATH)

const DEFAULT_ONNX_MODEL_PATH = joinpath(GNN_DIR, "model.onnx")
const GNN_ONNX_MODEL_PATH = get(ENV, "PDMO_GNN_MODEL_ONNX", DEFAULT_ONNX_MODEL_PATH)

const PDMO_PYTHON = get(ENV, "PDMO_PYTHON", "")

using PyCall
using ONNXRunTime
using NNlib

const PY_INITIALIZED = Ref(false)
const PY_FORCE_CPU = Ref{Union{Nothing, Bool}}(nothing)

function init_python(force_cpu::Bool)
    if PY_INITIALIZED[] && PY_FORCE_CPU[] == force_cpu
        return
    end
    PY_FORCE_CPU[] = force_cpu

    println("PDMO_PYTHON: $(PDMO_PYTHON)")
    if !isempty(PDMO_PYTHON)
        ENV["PYTHON"] = PDMO_PYTHON
    end

    try
        PyCall.pyimport("torch")
        PyCall.pyimport("torch_geometric")
        PyCall.pyimport("numpy")
    catch err
        msg = """
Python deps missing for GNN inference.
Set `PDMO_PYTHON` to a Python with packages: torch, torch_geometric, numpy.
Example:
  python -m venv .venv
  . .venv/bin/activate
  pip install torch numpy
  pip install torch-geometric
Then export PDMO_PYTHON=/path/to/.venv/bin/python and re-run.

PyCall error: $(sprint(showerror, err))
"""
        error(msg)
    end

    PyCall.py"""
import sys
sys.path.append($GNN_ROOT)
from GNN.inference import Inference, read_graph_from_json, get_feature_from_julia_data
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import numpy as np
import random
from GNN.Model import GINE_Net, GCN_Net, GAT_Net

FORCE_CPU = bool($force_cpu)

MODEL_CACHE = {}

def get_inference_model(input_dim, hidden_dim, edge_dim, model_path):
    key = (input_dim, hidden_dim, edge_dim, model_path, FORCE_CPU)
    if key not in MODEL_CACHE:
        import os
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"GNN model weights not found: {model_path}")
        device = torch.device('cpu') if FORCE_CPU else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        MODEL_CACHE[key] = Inference(
            'GINE',
            input_dim,
            hidden_dim,
            edge_dim,
            model_path,
            device = device,
            model_mode='1',
            activation='ELU',
            num_gnn_layers=40,
            average_nodes=10,
            pooling='mincut',
            pool_ratio=0.1
        )
    return MODEL_CACHE[key]

def run_inference_py(data, model_path):
    x, edge_index, edges_attr = get_feature_from_julia_data(data)
    data = Data(x=x, edge_index=edge_index.T, edges_attr=edges_attr)

    input_dim_1= data.x.shape[1]
    hidden_dim = 64
    edge_dim_1 = data.edges_attr.shape[1]

    model = get_inference_model(input_dim_1, hidden_dim, edge_dim_1, model_path)
    output = model.inference(data)

    output = output.cpu().numpy().tolist()
    return output
"""

    PY_INITIALIZED[] = true
end

function apply_gnn_bipartization!(
    gnn_output::Vector{Int64},
    graph::MultiblockGraph,
    node_order::Vector{String},
    edge_order::Vector{String},
    nodesAssignment::Dict{String, Int64},
    edgesSplitting::Dict{String, Tuple{Int64, Int64}})

    empty!(nodesAssignment)
    empty!(edgesSplitting)

    @assert length(gnn_output) == length(node_order) "GNN output length does not match number of nodes."
    for (i, node_id) in enumerate(node_order)
        nodesAssignment[node_id] = gnn_output[i]
    end

    for edge_id in edge_order
        edge = graph.edges[edge_id]
        id1 = edge.nodeID1
        id2 = edge.nodeID2
        if nodesAssignment[id1] != nodesAssignment[id2]
            edgesSplitting[edge_id] = (0, 0)
        else
            if nodesAssignment[id1] == 1
                edgesSplitting[edge_id] = (1, 0)
            else
                edgesSplitting[edge_id] = (1, 1)
            end
        end
    end
end

function gnn_bipartization_impl_pycall(
    graph::MultiblockGraph,
    mbp::MultiblockProblem,
    nodesAssignment::Dict{String, Int64},
    edgesSplitting::Dict{String, Tuple{Int64, Int64}};
    force_cpu::Bool = true,
    model_path::String = GNN_MODEL_PATH)


    node_order = sort(collect(keys(graph.nodes)))
    edge_order = sort(collect(keys(graph.edges)))
    nodes_data = Dict(node_id => toDict(graph.nodes[node_id], 0, mbp) for node_id in node_order)
    edges_data = Dict(edge_id => toDict(graph.edges[edge_id], (0, 0), mbp) for edge_id in edge_order)
    data = Dict(
        "nodes" => nodes_data,
        "edges" => edges_data,
        "node_order" => node_order,
        "edge_order" => edge_order,
    )

    init_python(force_cpu)
    if !isfile(model_path)
        error("GNN model weights not found: $(model_path). Set PDMO_GNN_WEIGHTS to a valid .pth file.")
    end

    output = PyCall.py"run_inference_py"(data, model_path)
    # println("pycall--------")
    # println(output)
    output = Int64.(round.(output))
    apply_gnn_bipartization!(output, graph, node_order, edge_order, nodesAssignment, edgesSplitting)
    return nothing
end


function gnn_bipartization_impl_onnx(graph::MultiblockGraph,
    mbp::MultiblockProblem,
    nodesAssignment::Dict{String, Int64}, 
    edgesSplitting::Dict{String, Tuple{Int64, Int64}};
    model_path::String=GNN_ONNX_MODEL_PATH)


    num_nodes=length(graph.nodes)
    edges=graph.edges
    x = Float32[] 
    edge_index = Int64[]
    edges_attr = Float32[]

    for i in 1:num_nodes
        node_attr = ones(Float32, 20)   
        append!(x, node_attr)
    end
    x = reshape(x, (20, num_nodes))'

    for key in keys(edges)
        edge = edges[key]
        
        edge_attr = Float32[]
        constr_idx = _constraint_index(mbp,edge.sourceBlockConstraint)
        mapping = mbp.constraints[constr_idx].mappings
        # rhs = collect(mbp.constraints[constr_idx].rhs)
        
        key_i = parse(Int, edge.nodeID1[14:end-1])
        key_j = parse(Int, edge.nodeID2[14:end-1])
        
        if typeof(mapping[key_i]) == PDMO.LinearMappingMatrix
            Q_i = collect(mapping[key_i].A)
        else
            Q_i = collect(mapping[key_i].coe)
        end

        if typeof(mapping[key_j]) == PDMO.LinearMappingMatrix
            Q_j = collect(mapping[key_j].A)
        else
            Q_j = collect(mapping[key_j].coe)
        end

        Q_i_array = transpose(convert(Array{Float32, 2}, hcat([Float32.(v) for v in Q_i]...)))
        push!(edge_attr, size(Q_i,1)) 
        append!(edge_attr, get_structure_features(Q_i_array))
        append!(edge_attr, get_geometric_features(Q_i_array))
        push!(edge_attr, cond(Q_i_array))  
        
        Q_j_array = transpose(convert(Array{Float32, 2}, hcat([Float32.(v) for v in Q_j]...)))
        append!(edge_attr, get_structure_features(Q_j_array))
        append!(edge_attr, get_geometric_features(Q_j_array))
        push!(edge_attr, cond(Q_j_array))  
        
        if edge.type == PDMO.TWO_BLOCK_EDGE
            append!(edge_attr, [0.0, 1.0])
        else
            append!(edge_attr, [1.0, 0.0])
        end
        
        append!(edges_attr, edge_attr)
        append!(edges_attr, -edge_attr) 
        
        push!(edge_index, key_i-1)
        push!(edge_index, key_j-1)
        push!(edge_index, key_j-1)
        push!(edge_index, key_i-1)
    end
    num_total_edges = length(edge_index) ÷ 2  
    
    edge_index = reshape(edge_index, (2, num_total_edges))
    edges_attr = reshape(edges_attr, (length(edges_attr) ÷ (num_total_edges), num_total_edges))'
    
    if !isfile(model_path)
        error("GNN model not found: $(model_path). Set PDMO_GNN_MODEL_ONNX to a valid .onnx file.")
    end

    model = ONNXRunTime.load_inference(model_path)

    input = Dict(
        "x" => x,
        "edge_index" => edge_index,
        "edge_attr" => edges_attr
    )

    output = model(input)["output"]

    softmax_output = softmax(output, dims=2)
    argmax_indices = argmax(softmax_output, dims=2) 
    pred_labels = [idx[2] - 1 for idx in argmax_indices]
    # println("onnx---------")
    # println(pred_labels)

    empty!(nodesAssignment)
    empty!(edgesSplitting)

    for i in 1:length(graph.nodes)
        key = "VariableNode("*string(i)*")"
        nodesAssignment[key] = pred_labels[i]
    end

    for key in keys(graph.edges)
        id1 = graph.edges[key].nodeID1
        id2 = graph.edges[key].nodeID2
        if nodesAssignment[id1] != nodesAssignment[id2]
            edgesSplitting[key] = (0,0)
        else 
            if nodesAssignment[id1] == 1
                edgesSplitting[key] = (1,0)
            else
                edgesSplitting[key] = (1,1)
            end
        end   
    end
end

function registerGnnBipartizationImpl!(;
    force_cpu::Bool = true,
    model_path::String = GNN_MODEL_PATH)
    _, ext = splitext(model_path)
    ext = lowercase(ext)
    if ext == ".pth"
        PDMO.registerGnnBipartization!((graph, mbp, nodesAssignment, edgesSplitting; kwargs...) -> begin
            gnn_bipartization_impl_pycall(graph, mbp, nodesAssignment, edgesSplitting;
                force_cpu = force_cpu,
                model_path = model_path)
        end)
    elseif ext == ".onnx"
        PDMO.registerGnnBipartization!((graph, mbp, nodesAssignment, edgesSplitting; kwargs...) -> begin
            gnn_bipartization_impl_onnx(graph, mbp, nodesAssignment, edgesSplitting;
                model_path = model_path)
        end)
    end 
    return nothing
end
