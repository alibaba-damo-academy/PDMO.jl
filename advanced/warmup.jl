import Pkg
Pkg.activate(@__DIR__)


REMOTE_TEST = true

if REMOTE_TEST == false
    # For local development
    PDMO_PATH = joinpath(@__DIR__, "..")              
    HSL_PATH  = joinpath(PDMO_PATH, "..", "HSL_jll_placeholder")
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
Pkg.status()