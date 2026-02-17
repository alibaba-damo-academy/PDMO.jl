struct PowerNetworkInfo
    data::Dict{String, Any}
    BusIDList::Vector{String}
    BrchIDList::Vector{String}
    GenIDList::Vector{String}
    BusToGenDict::Dict{String, Vector{String}}
    BusToShuntDict::Dict{String, Vector{String}}
    BusToLoadDictQ::Dict{String, Vector{Float64}}
    BusToLoadDictP::Dict{String, Vector{Float64}}
    BusFromBrchDict::Dict{String, Vector{String}}
    BusToBrchDict::Dict{String, Vector{String}}
    ReferenceBus::String
    customizedGeneratorCost::Dict{String, CustomizedGeneratorCost}

    function PowerNetworkInfo(dir)
        data = PowerModels.parse_file(dir)

        BusIDList = collect(keys(data["bus"]))

        BrchIDList = Vector{String}()
        tupleList = []
        for l in keys(data["branch"])
            if data["branch"][l]["br_status"] < 0.9
                continue
            end 
            f_bus = data["branch"][l]["f_bus"]
            t_bus = data["branch"][l]["t_bus"]
            if ((f_bus, t_bus) in tupleList) == false && ((t_bus, f_bus) in tupleList) == false
                push!(tupleList, (f_bus, t_bus))
                push!(BrchIDList, l)
            else 
                @warn "PowerNetworkInfo: branch $l is a parallel branch; ignored."
            end 
        end 
        
        GenIDList = collect(Iterators.filter(g->(data["gen"][g]["gen_status"] >= 0.9),
            keys(data["gen"]))) 

        BusToGenDict = Dict{String, Vector{String}}(i => String[] for i in BusIDList)
        for g in GenIDList 
            busid = string(data["gen"][g]["gen_bus"])
            push!(BusToGenDict[busid], g)
        end 

        BusToShuntDict = Dict{String, Vector{String}}(i => String[] for i in BusIDList)
        for sh in keys(data["shunt"])
            if data["shunt"][sh]["status"] >= 0.9
                busid = string(data["shunt"][sh]["shunt_bus"])
                push!(BusToShuntDict[busid], sh)
            end
        end 

        BusToLoadDictQ = Dict{String, Vector{Float64}}(i => Float64[] for i in BusIDList)
        BusToLoadDictP = Dict{String, Vector{Float64}}(i => Float64[] for i in BusIDList)
        for ld in keys(data["load"])
            if data["load"][ld]["status"] >= 0.9
                busid = string(data["load"][ld]["load_bus"])
                push!(BusToLoadDictP[busid], data["load"][ld]["pd"])
                push!(BusToLoadDictQ[busid], data["load"][ld]["qd"])
            end
        end

        BusFromBrchDict = Dict{String, Vector{String}}(i => String[] for i in BusIDList)
        BusToBrchDict = Dict{String, Vector{String}}(i => String[] for i in BusIDList)
        for l in BrchIDList
            f_bus = string(data["branch"][l]["f_bus"])
            t_bus = string(data["branch"][l]["t_bus"])
            push!(BusFromBrchDict[f_bus], l)
            push!(BusToBrchDict[t_bus], l)
        end

        ref_bus = collect(Iterators.filter(i->(data["bus"][i]["bus_type"] == 3), BusIDList))[1]
        
        customizedGeneratorCost = Dict{String, CustomizedGeneratorCost}()
        for g in GenIDList 
            ncost = data["gen"][g]["ncost"]
            cost_coe = data["gen"][g]["cost"]
            if ncost == 1 
                customizedGeneratorCost[g] = CustomizedGeneratorCost(0.0, 0.0, cost_coe[1])
            elseif ncost == 2 
                customizedGeneratorCost[g] = CustomizedGeneratorCost(0.0, cost_coe[1], cost_coe[2])
            elseif ncost == 3 
                customizedGeneratorCost[g] = CustomizedGeneratorCost(cost_coe[1], cost_coe[2], cost_coe[3])
            else 
                @warn "PowerNetworkInfo: generator $g has $(ncost) pieces; skipped."
            end 
        end 

        new(data,
            BusIDList,
            BrchIDList, 
            GenIDList, 
            BusToGenDict, 
            BusToShuntDict, 
            BusToLoadDictQ, 
            BusToLoadDictP, 
            BusFromBrchDict, 
            BusToBrchDict, 
            ref_bus,
            customizedGeneratorCost)
    end 
end  

struct AreaNetworkInfo 
    AreaID::Int64 
    networkInfo::PowerNetworkInfo
    Buses::Vector{String}
    BoundaryBuses::Vector{String}
    NeighboringBuses::Vector{String} 
    TieLines::Vector{String} 
    InternalLines::Vector{String} 
    
    AreaNetworkInfo(area::Int64, networkInfo::PowerNetworkInfo) = new(
        area, 
        networkInfo, 
        Vector{String}(), 
        Vector{String}(), 
        Vector{String}(), 
        Vector{String}(), 
        Vector{String}(), 
    )
end 

function createAreaNetworkInfo(networkInfo::PowerNetworkInfo, bus2Area::Dict{String, Int64}, numberAreas::Int64) 

    info = Dict{Int64, AreaNetworkInfo}(area => AreaNetworkInfo(area, networkInfo) for area in 1:numberAreas)
    tieLines = Vector{String}()

    for i in networkInfo.BusIDList
        area = bus2Area[i]
        push!(info[area].Buses, i)
    end 

    for l in networkInfo.BrchIDList 
        f_bus = string(networkInfo.data["branch"][l]["f_bus"])
        t_bus = string(networkInfo.data["branch"][l]["t_bus"])

        f_bus_area = bus2Area[f_bus]
        t_bus_area = bus2Area[t_bus]

        if f_bus_area == t_bus_area
            push!(info[f_bus_area].InternalLines, l)
        else 
            push!(info[f_bus_area].BoundaryBuses, f_bus)
            push!(info[f_bus_area].NeighboringBuses, t_bus)
            push!(info[f_bus_area].TieLines, l)
            
            push!(info[t_bus_area].BoundaryBuses, t_bus)
            push!(info[t_bus_area].NeighboringBuses, f_bus)
            push!(info[t_bus_area].TieLines, l)

            push!(tieLines, l)
        end 
    end 

    return info, tieLines
end 

function DCModel(networkInfo::PowerNetworkInfo; customizedGeneratorCost::Bool=false)

    model = JuMP.Model(Ipopt.Optimizer)
    # JuMP.set_silent(model)
    if HSL_FOUND 
        JuMP.set_attribute(model, "hsllib", HSL_jll.libhsl_path)
        JuMP.set_attribute(model, "linear_solver", "ma27")
    end 

    varToIdx = Dict{String, Dict{String, Int64}}() 
    number_variables = 0 
    
    varToIdx["theta"] = Dict{String, Int64}()
    for i in networkInfo.BusIDList
        number_variables += 1 
        varToIdx["theta"][i] = number_variables
    end 

    varToIdx["p_g"] = Dict{String, Int64}() 
    for g in networkInfo.GenIDList
        number_variables += 1 
        varToIdx["p_g"][g] = number_variables  
    end 

    varToIdx["p_from"] = Dict{String, Int64}() 
    for l in networkInfo.BrchIDList
        number_variables += 1 
        varToIdx["p_from"][l] = number_variables
    end 

    ## add a vector of variables and set bounds 
    JuMP.@variable(model, x[i in 1:number_variables])
    
    JuMP.set_lower_bound(x[varToIdx["theta"][networkInfo.ReferenceBus]], 0.0)
    JuMP.set_upper_bound(x[varToIdx["theta"][networkInfo.ReferenceBus]], 0.0)

    for g in networkInfo.GenIDList 
        idx = varToIdx["p_g"][g]
        JuMP.set_lower_bound(x[idx], networkInfo.data["gen"][g]["pmin"])
        JuMP.set_upper_bound(x[idx], networkInfo.data["gen"][g]["pmax"])
    end 

    for l in networkInfo.BrchIDList
        if haskey(networkInfo.data["branch"][l], "rate_a")
            idx = varToIdx["p_from"][l]
            rate_a = networkInfo.data["branch"][l]["rate_a"]
            JuMP.set_lower_bound(x[idx], -rate_a)
            JuMP.set_upper_bound(x[idx], rate_a)
        end 
    end 

    ## constraints

    # angle difference 
    for l in networkInfo.BrchIDList
        i = string(networkInfo.data["branch"][l]["f_bus"])
        j = string(networkInfo.data["branch"][l]["t_bus"])

        theta_i_idx = varToIdx["theta"][i]
        theta_j_idx = varToIdx["theta"][j]
        
        JuMP.@constraint(model, x[theta_i_idx] - x[theta_j_idx] <= networkInfo.data["branch"][l]["angmax"])
        JuMP.@constraint(model, x[theta_i_idx] - x[theta_j_idx] >= networkInfo.data["branch"][l]["angmin"])
    end 

    # real power flow 
    for l in networkInfo.BrchIDList
        tm = get(networkInfo.data["branch"][l], "tap", 1.0)  # Default tap ratio to 1.0 if not specified
        y = pinv(networkInfo.data["branch"][l]["br_r"] + im * networkInfo.data["branch"][l]["br_x"])
        g, b = real(y), imag(y)
        
        i = string(networkInfo.data["branch"][l]["f_bus"])
        j = string(networkInfo.data["branch"][l]["t_bus"])

        theta_i_idx = varToIdx["theta"][i]
        theta_j_idx = varToIdx["theta"][j]
        
        JuMP.@constraint(model, x[varToIdx["p_from"][l]] == -b/tm * (x[theta_i_idx] - x[theta_j_idx]))
    end 

    # power flow balance 
    for i in networkInfo.BusIDList
        loadP = isempty(networkInfo.BusToLoadDictP[i]) ? 0.0 : sum(networkInfo.BusToLoadDictP[i])
        shuntG = isempty(networkInfo.BusToShuntDict[i]) ? 0.0 : sum(networkInfo.data["shunt"][sh]["gs"] for sh in networkInfo.BusToShuntDict[i])

        if isempty(networkInfo.BusToGenDict[i])
            JuMP.@constraint(model, 0.0 - loadP - shuntG == 
                sum(x[varToIdx["p_from"][l]] for l in networkInfo.BusFromBrchDict[i]) -
                sum(x[varToIdx["p_from"][l]] for l in networkInfo.BusToBrchDict[i])) # in DC, p_to = -p_from
        else 
            JuMP.@constraint(model, sum(x[varToIdx["p_g"][g]] for g in networkInfo.BusToGenDict[i]) - loadP - shuntG == 
                sum(x[varToIdx["p_from"][l]] for l in networkInfo.BusFromBrchDict[i]) -
                sum(x[varToIdx["p_from"][l]] for l in networkInfo.BusToBrchDict[i]))
        end 
    end 

    # objective 
    expr_obj = JuMP.@expression(model, 0)
    if customizedGeneratorCost
        println("DCModel: using customized generator cost")
        for g in networkInfo.GenIDList
            expr_obj += JuMPAddSmoothFunction(networkInfo.customizedGeneratorCost[g], model, [x[varToIdx["p_g"][g]]])
        end 
    else 
        println("DCModel: using original generator cost")
        for g in networkInfo.GenIDList
            ncost = networkInfo.data["gen"][g]["ncost"]
            cost_coe = networkInfo.data["gen"][g]["cost"]
            if ncost == 1 
                expr_obj += cost_coe[1] # const cost 
            elseif ncost == 2
                expr_obj += cost_coe[1] * x[varToIdx["p_g"][g]] + cost_coe[2]
            elseif ncost == 3
                expr_obj += cost_coe[1] * x[varToIdx["p_g"][g]]^2 + 
                    cost_coe[2] * x[varToIdx["p_g"][g]] + cost_coe[3]
            else 
                @warn "DCModel: generator $g has $(ncost) pieces; skipped. "
            end 
        end 
    end 
    JuMP.@objective(model, Min, expr_obj)
    JuMP.optimize!(model)

    return JuMP.objective_value(model), JuMP.solve_time(model)
end 
