const GNN_ROOT = @__DIR__
const ADVANCED_DIR = abspath(joinpath(GNN_ROOT, "..", ".."))
const GNN_DIR = joinpath(GNN_ROOT, "GNN")
const DEFAULT_MODEL_PATH = joinpath(GNN_DIR, "model_weights_10.pth")
const GNN_MODEL_PATH = get(ENV, "PDMO_GNN_WEIGHTS", DEFAULT_MODEL_PATH)

const DEFAULT_PYTHON = joinpath(ADVANCED_DIR, ".venv-gnn", "bin", "python")
const PDMO_PYTHON = get(ENV, "PDMO_PYTHON", isfile(DEFAULT_PYTHON) ? DEFAULT_PYTHON : "")

using PyCall

const PY_INITIALIZED = Ref(false)
const PY_FORCE_CPU = Ref{Union{Nothing, Bool}}(nothing)

function init_python(force_cpu::Bool)
    if PY_INITIALIZED[] && PY_FORCE_CPU[] == force_cpu
        return
    end
    PY_FORCE_CPU[] = force_cpu
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

function gnn_bipartization_impl(
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
    output = Int64.(round.(output))
    apply_gnn_bipartization!(output, graph, node_order, edge_order, nodesAssignment, edgesSplitting)
    return nothing
end

function registerGnnBipartizationImpl!(;
    force_cpu::Bool = true,
    model_path::String = GNN_MODEL_PATH)
    PDMO.registerGnnBipartization!((graph, mbp, nodesAssignment, edgesSplitting; kwargs...) -> begin
        gnn_bipartization_impl(graph, mbp, nodesAssignment, edgesSplitting;
            force_cpu = force_cpu,
            model_path = model_path)
    end)
    return nothing
end
