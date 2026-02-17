# include(joinpath(@__DIR__, "../../warmup.jl"))
import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using PDMO

using LinearAlgebra
using SparseArrays
using Random
using Plots
using Printf
using LaTeXStrings
Random.seed!(126)


# Push blocks to very different scales to separate trajectories
R12 = 1.0e-6      # block 1 extremely soft
R23 = 1.0e2       # block 2 medium
R31 = 1.0e8       # block 3 extremely stiff

# Asymmetric, large-magnitude RHS; sum to zero to maintain consistency
b1 = -50.0
b2 = 100.0 
b3 = -50.0  # b1 + b2 + b3 = 0

function demo()
    mbp = MultiblockProblem()

    # variables
    I12 = BlockVariable(1)
    I12.f = QuadraticFunction(sparse([1], [1], [R12], 1,1), zeros(1), 0.0)
    I12.val = zeros(1)

    I23 = BlockVariable(2)
    I23.f = QuadraticFunction(sparse([1], [1], [R23], 1,1), zeros(1), 0.0)
    I23.val = zeros(1)

    I31 = BlockVariable(3)
    I31.f = QuadraticFunction(sparse([1], [1], [R31], 1,1), zeros(1), 0.0)
    I31.val = zeros(1)

    addBlockVariable!(mbp, I12)
    addBlockVariable!(mbp, I23)
    addBlockVariable!(mbp, I31)

    # constraints
    ## I12 - I31 = b1 
    constr1=BlockConstraint(1)
    addBlockMappingToConstraint!(constr1, 1, LinearMappingIdentity(1.0))
    addBlockMappingToConstraint!(constr1, 3, LinearMappingIdentity(-1.0))
    constr1.rhs = b1 

    ## I23 - I12 = b2 
    constr2=BlockConstraint(2)
    addBlockMappingToConstraint!(constr2, 2, LinearMappingIdentity(1.0))
    addBlockMappingToConstraint!(constr2, 1, LinearMappingIdentity(-1.0))
    constr2.rhs = b2 

    ## I31 - I23 = b3 
    constr3=BlockConstraint(3)
    addBlockMappingToConstraint!(constr3, 3, LinearMappingIdentity(1.0))
    addBlockMappingToConstraint!(constr3, 2, LinearMappingIdentity(-1.0))
    constr3.rhs = b3 

    addBlockConstraint!(mbp, constr1)
    addBlockConstraint!(mbp, constr2)
    addBlockConstraint!(mbp, constr3)


    return mbp 
end 

function demo_break12() 
    mbp = MultiblockProblem()

    # variables
    I12 = BlockVariable(1)
    I12.f = QuadraticFunction(sparse([1], [1], [R12], 1,1), zeros(1), 0.0)
    I12.val = zeros(1)

    I23 = BlockVariable(2)
    I23.f = QuadraticFunction(sparse([1], [1], [R23], 1,1), zeros(1), 0.0)
    I23.val = zeros(1)

    I31 = BlockVariable(3)
    I31.f = QuadraticFunction(sparse([1], [1], [R31], 1,1), zeros(1), 0.0)
    I31.val = zeros(1)

    z = BlockVariable(4)
    z.val = zeros(1)

    addBlockVariable!(mbp, I12)
    addBlockVariable!(mbp, I23)
    addBlockVariable!(mbp, I31)
    addBlockVariable!(mbp, z)

    # constraints
    # constraints
    ## z - I12 = 0
    constr4=BlockConstraint(4)
    addBlockMappingToConstraint!(constr4, 4, LinearMappingIdentity(1.0))
    addBlockMappingToConstraint!(constr4, 1, LinearMappingIdentity(-1.0))
    constr4.rhs = zeros(1)

    ## z - I31 = b1
    constr1=BlockConstraint(1)
    addBlockMappingToConstraint!(constr1, 4, LinearMappingIdentity(1.0))
    addBlockMappingToConstraint!(constr1, 3, LinearMappingIdentity(-1.0))
    constr1.rhs = b1 

    ## I23 - I12 = b2 
    constr2=BlockConstraint(2)
    addBlockMappingToConstraint!(constr2, 2, LinearMappingIdentity(1.0))
    addBlockMappingToConstraint!(constr2, 1, LinearMappingIdentity(-1.0))
    constr2.rhs = b2 

    ## I31 - I23 = b3 
    constr3=BlockConstraint(3)
    addBlockMappingToConstraint!(constr3, 3, LinearMappingIdentity(1.0))
    addBlockMappingToConstraint!(constr3, 2, LinearMappingIdentity(-1.0))
    constr3.rhs = b3 

    addBlockConstraint!(mbp, constr4)
    addBlockConstraint!(mbp, constr1)
    addBlockConstraint!(mbp, constr2)
    addBlockConstraint!(mbp, constr3)

    return mbp 
end 



function demo_break23()
    mbp = MultiblockProblem()

    # variables
    I12 = BlockVariable(1)
    I12.f = QuadraticFunction(sparse([1], [1], [R12], 1,1), zeros(1), 0.0)
    I12.val = zeros(1)

    I23 = BlockVariable(2)
    I23.f = QuadraticFunction(sparse([1], [1], [R23], 1,1), zeros(1), 0.0)
    I23.val = zeros(1)

    I31 = BlockVariable(3)
    I31.f = QuadraticFunction(sparse([1], [1], [R31], 1,1), zeros(1), 0.0)
    I31.val = zeros(1)

    z = BlockVariable(4)
    z.val = zeros(1)


    addBlockVariable!(mbp, I12)
    addBlockVariable!(mbp, I23)
    addBlockVariable!(mbp, I31)
    addBlockVariable!(mbp, z)

    # constraints
    ## I12 - I31 = b1 
    constr1=BlockConstraint(1)
    addBlockMappingToConstraint!(constr1, 1, LinearMappingIdentity(1.0))
    addBlockMappingToConstraint!(constr1, 3, LinearMappingIdentity(-1.0))
    constr1.rhs = b1 

    ## z - I23 = 0  
    constr2=BlockConstraint(2)
    addBlockMappingToConstraint!(constr2, 4, LinearMappingIdentity(1.0))
    addBlockMappingToConstraint!(constr2, 2, LinearMappingIdentity(-1.0))
    constr2.rhs = zeros(1)

    ## z - I12 = b2
    constr4=BlockConstraint(3)
    addBlockMappingToConstraint!(constr4, 4, LinearMappingIdentity(1.0))
    addBlockMappingToConstraint!(constr4, 1, LinearMappingIdentity(-1.0))
    constr4.rhs = b2 

    ## I31 - I23 = b3 
    constr3=BlockConstraint(4)
    addBlockMappingToConstraint!(constr3, 3, LinearMappingIdentity(1.0))
    addBlockMappingToConstraint!(constr3, 2, LinearMappingIdentity(-1.0))
    constr3.rhs = b3 

    addBlockConstraint!(mbp, constr1)
    addBlockConstraint!(mbp, constr2)
    addBlockConstraint!(mbp, constr3)
    addBlockConstraint!(mbp, constr4)


    return mbp 
end 



function demo_break31()
    mbp = MultiblockProblem()

    # variables
    I12 = BlockVariable(1)
    I12.f = QuadraticFunction(sparse([1], [1], [R12], 1,1), zeros(1), 0.0)
    I12.val = zeros(1)

    I23 = BlockVariable(2)
    I23.f = QuadraticFunction(sparse([1], [1], [R23], 1,1), zeros(1), 0.0)
    I23.val = zeros(1)

    I31 = BlockVariable(3)
    I31.f = QuadraticFunction(sparse([1], [1], [R31], 1,1), zeros(1), 0.0)
    I31.val = zeros(1)

    z = BlockVariable(4)
    z.val = zeros(1)


    addBlockVariable!(mbp, I12)
    addBlockVariable!(mbp, I23)
    addBlockVariable!(mbp, I31)
    addBlockVariable!(mbp, z)

    # constraints
    ## I12 - I31 = b1 
    constr1=BlockConstraint(1)
    addBlockMappingToConstraint!(constr1, 1, LinearMappingIdentity(1.0))
    addBlockMappingToConstraint!(constr1, 3, LinearMappingIdentity(-1.0))
    constr1.rhs = b1 

    ## I23 - I12 = b2 
    constr2=BlockConstraint(2)
    addBlockMappingToConstraint!(constr2, 2, LinearMappingIdentity(1.0))
    addBlockMappingToConstraint!(constr2, 1, LinearMappingIdentity(-1.0))
    constr2.rhs = b2 

    ## z- I31 = 0
    constr3=BlockConstraint(3)
    addBlockMappingToConstraint!(constr3, 4, LinearMappingIdentity(1.0))
    addBlockMappingToConstraint!(constr3, 3, LinearMappingIdentity(-1.0))
    constr3.rhs = zeros(1)

    ## z- I23 = b3
    constr4=BlockConstraint(4)
    addBlockMappingToConstraint!(constr4, 4, LinearMappingIdentity(1.0))
    addBlockMappingToConstraint!(constr4, 2, LinearMappingIdentity(-1.0))
    constr4.rhs = b3 

    addBlockConstraint!(mbp, constr1)
    addBlockConstraint!(mbp, constr2)
    addBlockConstraint!(mbp, constr3)
    addBlockConstraint!(mbp, constr4)

    return mbp 
end 


function _parse_args(args)
    yscale = "log"
    k = nothing
    rho = 1.0
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--yscale" && i + 1 <= length(args)
            yscale = args[i + 1]
            i += 1
        elseif arg == "--k" && i + 1 <= length(args)
            k = parse(Int, args[i + 1])
            i += 1
        elseif arg == "--rho" && i + 1 <= length(args)
            rho = parse(Float64, args[i + 1])
            i += 1
        end
        i += 1
    end
    return yscale, k, rho
end

function plot_pres_dres(result::Dict{Any, Any}, rho::Float64; yscale::String="log", k::Union{Nothing, Int}=nothing)
    names = ["12", "23", "31"]
    name_plots = Dict(
        "12" => "breaking 1st constraint",
        "23" => "breaking 2nd constraint",
        "31" => "breaking 3rd constraint",
    )
    linestyles = [:solid, :dash, :dot]
    plt = plot(size=(640, 400))

    for (idx, name) in enumerate(names)
        haskey(result, name) || continue
        pres = result[name].iterationInfo.presL2
        dres = result[name].iterationInfo.dresL2
        n = min(length(pres), length(dres))
        if k !== nothing
            n = min(n, k)
        end
        n > 0 || continue
        iters = collect(1:n)
        vals = pres[1:n] .+ dres[1:n]

        if yscale == "log"
            yvals = [v > 0 ? log10(v) : NaN for v in vals]
            ylabel = L"\log_{10}(||\mathrm{Pres}(2)||_2 + ||\mathrm{Dres}(2)||_2)"
        else
            yvals = vals
            ylabel = L"||\mathrm{Pres}(2)||_2 + ||\mathrm{Dres}(2)||_2"
        end

        plot!(
            plt,
            iters,
            yvals,
            label=name_plots[name],
            linestyle=linestyles[mod1(idx, length(linestyles))],
            linewidth=2.5,
        )
        ylabel!(plt, ylabel)
    end

    xlabel!(plt, "Iteration")
    plot!(
        plt,
        grid=true,
        legend=:best,
        legendfontsize=12,
        guidefontsize=12,
        tickfontsize=10,
    )
    rho_int = round(Int, rho)
    rho_tag = isapprox(rho, rho_int; atol=1e-9, rtol=0.0) ? string(rho_int) : replace(@sprintf("%.3f", rho), "." => "p")
    outfile = joinpath(@__DIR__, "demo_plot_$(rho_tag).png")
    savefig(plt, outfile)
    println("[info] saved plot -> $(outfile) (yscale=$(yscale))")
end


if abspath(PROGRAM_FILE) == @__FILE__   
    yscale, k, rho = _parse_args(ARGS)


    param = ADMMParam()
    param.logInterval = 1
    param.initialRho =  rho # smaller rho to let scaled blocks diverge more initially
    param.maxIter = 10000
    param.presTolL2 = 1e-30
    param.dresTolL2 = 1e-30
    param.presTolLInf = 1e-30
    param.dresTolLInf = 1e-30
    param.solver = OriginalADMMSubproblemSolver()

    mbp_list = [("12", demo_break12()), ("23", demo_break23()), ("31", demo_break31())]
    result = Dict()
    for (name, mbp) in mbp_list
        println("Running reformulation breaking-$(name)...")
        result[name] = runBipartiteADMM(mbp, param)
    end 


    for (name, res) in result
        println("name: $(name)")
        println("   objective: $(res.iterationInfo.obj[end])")
        println("   stopIter: $(res.iterationInfo.stopIter)")
    end 

    plot_pres_dres(result, rho; yscale="log", k=k)
end 