#!/usr/bin/env julia

length(ARGS) == 1 || error("Usage: check_pycall.jl /path/to/python")

import Pkg

const ADVANCED_PROJECT = normpath(joinpath(@__DIR__, "..", "..", "advanced"))
Pkg.activate(ADVANCED_PROJECT)

using PyCall

canonical_python(path::AbstractString) = realpath(abspath(expanduser(path)))

selected_python = canonical_python(ARGS[1])
pycall_python = canonical_python(PyCall.python)

println("Selected Python: ", selected_python)
println("PyCall Python:   ", pycall_python)
selected_python == pycall_python || error(
    "PyCall is built against $(pycall_python), not the selected $(selected_python). " *
    "Set PDMO_PYTHON and rerun advanced/warmup.jl before starting fresh GNN experiments."
)

torch = PyCall.pyimport("torch")
torch_geometric = PyCall.pyimport("torch_geometric")
numpy = PyCall.pyimport("numpy")

println(
    "PyCall GNN environment: torch=", torch.__version__,
    ", torch_geometric=", torch_geometric.__version__,
    ", numpy=", numpy.__version__,
)
