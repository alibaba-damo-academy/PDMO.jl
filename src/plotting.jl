using JSON, Plots

function saveJSONL(info , method_name, folder_name)
    folder_name = joinpath("results", folder_name)
    mkpath(folder_name)

    open("$(folder_name)/$(method_name).jsonl", "w") do io
        N = length(info.presL2)
        rhos = fill(info.rhoHistory[1][1], N)
        for (ρ, k) in info.rhoHistory
            for i in (k+1):N
                rhos[i] = ρ
            end
        end

        for i in 1:N
            rec = Dict(
                "iter"     => i,
                "rho"      => rhos[i],
                "presL2"   => info.presL2[i],
                "dresL2"   => info.dresL2[i],
                "presLInf" => info.presLInf[i],
                "dresLInf" => info.dresLInf[i],
                "obj"      => info.obj[i],
                "alObj"    => info.alObj[i],
            )
            println(io, JSON.json(rec; allownan=true))
        end
    end
end

function saveJSONL_adapdm(info , method_name, folder_name)
    folder_name = joinpath("results", folder_name)
    mkpath(folder_name)

    open("$(folder_name)/$(method_name).jsonl", "w") do io
        N = length(info.presL2)

        for i in 1:N
            rec = Dict(
                "iter"     => i,
                "presL2"   => info.presL2[i],
                "dresL2"   => info.dresL2[i],
                "presLInf" => info.presLInf[i],
                "dresLInf" => info.dresLInf[i],
            )
            println(io, JSON.json(rec; allownan=true))
        end
    end
end

Inffilter = x -> x === nothing ? Inf :  x == 0.0  ? eps(Float64) :
x

function load_run(path)
    iters   = Int[]
    time    = Float64[]
    presL2  = Float64[]
    dresL2  = Float64[]
    presLInf  = Float64[]
    dresLInf  = Float64[]
    for line in eachline(path)
        rec = JSON.parse(line; allownan=true)
        push!(iters,  rec["iter"])
        push!(presL2, Inffilter(rec["presL2"]))
        push!(dresL2, Inffilter(rec["dresL2"]))
        push!(presLInf, Inffilter(rec["presLInf"]))
        push!(dresLInf, Inffilter(rec["dresLInf"]))
    end
    return (iters=iters, presL2=presL2, dresL2=dresL2, presLInf=presLInf, dresLInf=dresLInf)
end


function smooth(y::Vector{Float64}, window_size::Int = 50)
    n = length(y)
    smoothed = Vector{Float64}(undef, n)
    half = div(window_size, 2)
    for i in 1:n
        total = 0.0
        count = 0
        for j in max(1, i - half):min(n, i + half)
            total += y[j]
            count += 1
        end
        smoothed[i] = total / count
    end
    return smoothed
end



function plotting(folder_name , normtype ; xlabel = "iter" , smoothing = false) #norm = l2 or linf , xlabel = iter or time
    json_path = joinpath("results", folder_name)
    allfiles = readdir(json_path; join=true)
    jsonl_files = filter(f -> endswith(f, ".jsonl"), allfiles)
    runs = Dict{String,Any}()
    for f in jsonl_files
        label = basename(f)[1:end-6]   
        runs[label] = load_run(f)
    end
    plt = plot(
        xlabel = xlabel == "iter" ?    "Iteration" : "Time",
        ylabel =  normtype == "l2" ? "Residual ∥·∥_2" : "Residual ∥·∥_∞",
        yscale = :log10,
        legend = :topright,
        grid   = true,
        yticks = :log10,
      )
    for (label, run) in runs
        if norm == "l2"
            comb = sqrt.(run.presL2 .^2 .+ run.dresL2 .^2)
            smoothed_comb = smoothing ? smooth(comb) : comb
            if xlabel =="time"
                plot!(plt, run.time, smoothed_comb; label="$label")
            else            
                plot!(plt, run.iters, smoothed_comb; label="$label")
            end
        else
            comb = max.(abs.(run.presLInf), abs.(run.dresLInf))
            smoothed_comb = smoothing ? smooth(comb) : comb
            if xlabel =="time"
                plot!(plt, run.time, smoothed_comb; label="$label")
            else
                plot!(plt, run.iters, smoothed_comb; label="$label")
            end
        end
    end
    title!(plt, folder_name)
    savefig(plt, joinpath("results", folder_name, folder_name))
end
