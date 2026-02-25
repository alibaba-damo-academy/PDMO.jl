import Pkg
Pkg.activate(@__DIR__)


REMOTE_TEST = true

if REMOTE_TEST == false
    # For local development
    PDMO_PATH = joinpath(@__DIR__, "..")              
    HSL_PATH  = joinpath(PDMO_PATH, "HSL_jll_placeholder")
else
    # For remote test, the script should be run from the dir where PDMO.jl is located
    PDMO_PATH = "PDMO.jl"
    HSL_PATH  = "PDMO.jl/HSL_jll_placeholder" 
end

# If you have HSL: Uncomment and update the path below
# HSL_PATH = "full/path/to/HSL_jll"

Pkg.develop(path=PDMO_PATH)
Pkg.develop(path=HSL_PATH)

Pkg.instantiate()

# If you want to use GNN + PyCall for bipartization, set the PDMO_PYTHON environment variable to 
# the path to the Python executable with the required packages installed:  torch, torch_geometric, numpy.
if haskey(ENV, "PDMO_PYTHON") && !isempty(ENV["PDMO_PYTHON"])
    println("Setting PYTHON environment variable to $(ENV["PDMO_PYTHON"])")
    ENV["PYTHON"] = ENV["PDMO_PYTHON"]
    Pkg.build("PyCall")
end
Pkg.status()