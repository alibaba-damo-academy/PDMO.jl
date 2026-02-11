using Documenter, TikzPictures

# Load all required packages first
using LinearAlgebra, SparseArrays, Printf, Random, Logging, Dates
using JuMP, Arpack, DataFrames, DataStructures, JSON, MathOptInterface
using HiGHS, Ipopt, FileIO, FilePathsBase, Metis, CodecZlib
using PDMO

# Add the parent directory to the load path so Julia can find the PDMO package
push!(LOAD_PATH, "../")

# Try to load the PDMO package
try
    using PDMO
    @info "Successfully loaded PDMO package for automatic docstring extraction"
catch e
    @error "Failed to load PDMO package: $e"
    @info "Documentation will build with empty @docs blocks"
end

# Build documentation with automatic docstring extraction
makedocs(
    sitename = "PDMO.jl Documentation",
    authors = "PDMO contributors",
    repo = Documenter.Remotes.GitHub("alibaba-damo-academy", "PDMO.jl"),
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://pdmo.readthedocs.io/en/latest/",
        repolink = "https://github.com/alibaba-damo-academy/PDMO.jl",
        assets = ["assets/tikz-support.css"]
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "S1_getting_started.md",
        "Algorithms" => [
            "ADMM" => "S2_algorithms/ADMM.md",
            "AdaPDM" => "S2_algorithms/AdaPDM.md", 
            "BCD" => "S2_algorithms/BCD.md"
        ],
        "Examples" => [
            "Least L1 Norm" => "S3_examples/LeastL1Norm.md",
            "Fused Lasso" => "S3_examples/FusedLasso.md",
            "Dual Lasso" => "S3_examples/DualLasso.md",
            "Dual SVM" => "S3_examples/DualSVM.md",
        ],
        "API Reference" => [
            "Main Algorithm Interface" => "S4_api/main.md"
            , "Functions" => "S4_api/functions.md"
            , "Mappings" => "S4_api/mappings.md"
            , "Formulations" => "S4_api/formulations.md"
            , "ADMM Components" => "S4_api/admm.md"
            , "AdaPDM Components" => "S4_api/pdm.md"
            , "BCD Components" => "S4_api/bcd.md"
            # , "Utilities" => "S4_api/utilities.md"
        ]
    ]
)

deploydocs(
    repo = "github.com/alibaba-damo-academy/PDMO.jl.git",
    devbranch = "main",
    push_preview = true
)