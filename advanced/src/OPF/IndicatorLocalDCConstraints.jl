struct IndicatorLocalDCConstraints <: AbstractFunction 
    model::JuMP.AbstractModel
    numberVariables::Int64
    varToIdx::Dict{String, Dict{String, Int64}}
    areaInfo::AreaNetworkInfo    
end 

function addLocalDCConstraints(model::JuMP.Model, 
    x::Vector{<:JuMP.VariableRef}, 
    varToIdx::Dict{String, Dict{String, Int64}}, 
    areaInfo::AreaNetworkInfo)

     # Fix reference bus angle to 0
     if areaInfo.networkInfo.ReferenceBus in areaInfo.Buses
        idx = varToIdx["theta"][areaInfo.networkInfo.ReferenceBus]
        JuMP.set_lower_bound(x[idx], 0.0)
        JuMP.set_upper_bound(x[idx], 0.0)
    end 

    # Generator bounds
    if isempty(varToIdx["p_g"]) == false
        for g in keys(varToIdx["p_g"])
            idx = varToIdx["p_g"][g]
            JuMP.set_lower_bound(x[idx], areaInfo.networkInfo.data["gen"][g]["pmin"])
            JuMP.set_upper_bound(x[idx], areaInfo.networkInfo.data["gen"][g]["pmax"])
        end 
    end 
    
    # Branch flow limits
    for l in keys(varToIdx["p_from"])
        if haskey(areaInfo.networkInfo.data["branch"][l], "rate_a")
            idx = varToIdx["p_from"][l]
            rate_a = areaInfo.networkInfo.data["branch"][l]["rate_a"]
            JuMP.set_lower_bound(x[idx], -rate_a)
            JuMP.set_upper_bound(x[idx], rate_a)
        end 
    end 
    
    ## constraints
    # Angle difference limits
    for l in keys(varToIdx["p_from"])
        i = string(areaInfo.networkInfo.data["branch"][l]["f_bus"])
        j = string(areaInfo.networkInfo.data["branch"][l]["t_bus"])

        theta_i_idx = varToIdx["theta"][i]
        theta_j_idx = varToIdx["theta"][j]

        JuMP.@constraint(model, x[theta_i_idx] - x[theta_j_idx] <= areaInfo.networkInfo.data["branch"][l]["angmax"])
        JuMP.@constraint(model, x[theta_i_idx] - x[theta_j_idx] >= areaInfo.networkInfo.data["branch"][l]["angmin"])
    end 

    # DC power flow equations
    for l in keys(varToIdx["p_from"])
        tm = get(areaInfo.networkInfo.data["branch"][l], "tap", 1.0)
        y = pinv(areaInfo.networkInfo.data["branch"][l]["br_r"] + im * areaInfo.networkInfo.data["branch"][l]["br_x"])
        b = imag(y)
        
        i = string(areaInfo.networkInfo.data["branch"][l]["f_bus"])
        j = string(areaInfo.networkInfo.data["branch"][l]["t_bus"])

        theta_i_idx = varToIdx["theta"][i]
        theta_j_idx = varToIdx["theta"][j]
        
        JuMP.@constraint(model, x[varToIdx["p_from"][l]] == -b/tm * (x[theta_i_idx] - x[theta_j_idx]))
    end 

    # Power balance at each bus
    for i in areaInfo.Buses
        loadP = isempty(areaInfo.networkInfo.BusToLoadDictP[i]) ? 0.0 : sum(areaInfo.networkInfo.BusToLoadDictP[i])
        shuntG = isempty(areaInfo.networkInfo.BusToShuntDict[i]) ? 0.0 : sum(areaInfo.networkInfo.data["shunt"][sh]["gs"] for sh in areaInfo.networkInfo.BusToShuntDict[i])

        if isempty(areaInfo.networkInfo.BusToGenDict[i])
            JuMP.@constraint(model, 0.0 - loadP - shuntG == 
                sum(x[varToIdx["p_from"][l]] for l in areaInfo.networkInfo.BusFromBrchDict[i]) -
                sum(x[varToIdx["p_from"][l]] for l in areaInfo.networkInfo.BusToBrchDict[i])) # in DC, p_to = -p_from
        else 
            JuMP.@constraint(model, sum(x[varToIdx["p_g"][g]] for g in areaInfo.networkInfo.BusToGenDict[i]) - loadP - shuntG == 
                sum(x[varToIdx["p_from"][l]] for l in areaInfo.networkInfo.BusFromBrchDict[i]) -
                sum(x[varToIdx["p_from"][l]] for l in areaInfo.networkInfo.BusToBrchDict[i]))
        end 
    end 
end 


function IndicatorLocalDCConstraints(areaInfo::AreaNetworkInfo)
    model = JuMP.Model(Ipopt.Optimizer)
    JuMP.set_silent(model)
    if HSL_FOUND 
        JuMP.set_attribute(model, "hsllib", HSL_jll.libhsl_path)
        JuMP.set_attribute(model, "linear_solver", "ma27")
    end 

    varToIdx = Dict{String, Dict{String, Int64}}() 
    varToIdx["theta"] = Dict{String, Int64}()
    varToIdx["p_g"] = Dict{String,Int64}() 
    varToIdx["p_from"] = Dict{String, Int64}() 

    numberVariables = 0 

    # Add voltage angles for buses in area and neighboring buses
    for i in union(areaInfo.Buses, areaInfo.NeighboringBuses)
        numberVariables += 1 
        varToIdx["theta"][i] = numberVariables
    end 

    # Add generator power outputs
    for i in areaInfo.Buses 
        if isempty(areaInfo.networkInfo.BusToGenDict[i]) == false 
            for g in areaInfo.networkInfo.BusToGenDict[i]
                numberVariables += 1 
                varToIdx["p_g"][g] = numberVariables  
            end 
        end 
    end 

    # Add branch power flows
    for l in union(areaInfo.InternalLines, areaInfo.TieLines)
        numberVariables += 1
        varToIdx["p_from"][l] = numberVariables
    end 

    ## add variables and set bounds 
    JuMP.@variable(model, x[i in 1:numberVariables])
   
    # add constraints 
    addLocalDCConstraints(model, x, varToIdx, areaInfo)
   
    return IndicatorLocalDCConstraints(model, numberVariables, varToIdx, areaInfo)
end 


isProximal(::Type{<:IndicatorLocalDCConstraints}) = true 
isConvex(::Type{<:IndicatorLocalDCConstraints}) = true 
isSupportedByJuMP(f::Type{<:IndicatorLocalDCConstraints}) = true 
isSet(::Type{<:IndicatorLocalDCConstraints}) = true 

function (f::IndicatorLocalDCConstraints)(x::NumericVariable, enableParallel::Bool = false)
    if (typeof(x) != Vector{Float64} || length(x) != f.numberVariables)
        error("IndicatorLocalDCConstraints: argument dimension mismatch.")
    end
    
    # Check reference bus angle is 0
    if f.areaInfo.networkInfo.ReferenceBus in f.areaInfo.Buses
        idx = f.varToIdx["theta"][f.areaInfo.networkInfo.ReferenceBus]
        theta = x[idx]
        if abs(theta) > PDMO.FeasTolerance
            return Inf 
        end 
    end 

    # Check generator bounds
    if isempty(f.varToIdx["p_g"]) == false
        for g in keys(f.varToIdx["p_g"])
            idx = f.varToIdx["p_g"][g]
            p_g = x[idx]
            p_min = f.areaInfo.networkInfo.data["gen"][g]["pmin"]
            p_max = f.areaInfo.networkInfo.data["gen"][g]["pmax"]
            if (p_g < p_min - PDMO.FeasTolerance || p_g > p_max + PDMO.FeasTolerance)
                return Inf 
            end 
        end 
    end 
    
    # Check branch flow limits
    for l in keys(f.varToIdx["p_from"])
        if haskey(f.areaInfo.networkInfo.data["branch"][l], "rate_a")
            idx = f.varToIdx["p_from"][l]
            p_from = x[idx]
            rate_a = f.areaInfo.networkInfo.data["branch"][l]["rate_a"]
            if (p_from < -rate_a - PDMO.FeasTolerance || p_from > rate_a + PDMO.FeasTolerance)
                return Inf
            end 
        end 
    end 
    
    # Check angle difference limits
    for l in keys(f.varToIdx["p_from"])
        i = string(f.areaInfo.networkInfo.data["branch"][l]["f_bus"])
        j = string(f.areaInfo.networkInfo.data["branch"][l]["t_bus"])

        theta_i_idx = f.varToIdx["theta"][i]
        theta_j_idx = f.varToIdx["theta"][j]

        theta_diff = x[theta_i_idx] - x[theta_j_idx]
        if (theta_diff < f.areaInfo.networkInfo.data["branch"][l]["angmin"] - PDMO.FeasTolerance)
            return Inf 
        end 
        if (theta_diff > f.areaInfo.networkInfo.data["branch"][l]["angmax"] + PDMO.FeasTolerance)
            return Inf
        end 
    end 

    # Check DC power flow equations
    for l in keys(f.varToIdx["p_from"])
        tm = get(f.areaInfo.networkInfo.data["branch"][l], "tap", 1.0)
        y = pinv(f.areaInfo.networkInfo.data["branch"][l]["br_r"] + im * f.areaInfo.networkInfo.data["branch"][l]["br_x"])
        b = imag(y)
        
        i = string(f.areaInfo.networkInfo.data["branch"][l]["f_bus"])
        j = string(f.areaInfo.networkInfo.data["branch"][l]["t_bus"])

        theta_i_idx = f.varToIdx["theta"][i]
        theta_j_idx = f.varToIdx["theta"][j]
        
        res = x[f.varToIdx["p_from"][l]] + b/tm * (x[theta_i_idx] - x[theta_j_idx])
        if abs(res) > PDMO.FeasTolerance 
            return Inf
        end 
    end 

    # Check power balance at each bus
    for i in f.areaInfo.Buses
        loadP = isempty(f.areaInfo.networkInfo.BusToLoadDictP[i]) ? 0.0 : sum(f.areaInfo.networkInfo.BusToLoadDictP[i])
        shuntG = isempty(f.areaInfo.networkInfo.BusToShuntDict[i]) ? 0.0 : sum(f.areaInfo.networkInfo.data["shunt"][sh]["gs"] for sh in f.areaInfo.networkInfo.BusToShuntDict[i])
        res = loadP + shuntG 

        if isempty(f.areaInfo.networkInfo.BusToGenDict[i]) == false 
            res -= sum(x[f.varToIdx["p_g"][g]] for g in f.areaInfo.networkInfo.BusToGenDict[i])
        end

        if isempty(f.areaInfo.networkInfo.BusFromBrchDict[i]) == false 
            res += sum(x[f.varToIdx["p_from"][l]] for l in f.areaInfo.networkInfo.BusFromBrchDict[i])
        end 

        if isempty(f.areaInfo.networkInfo.BusToBrchDict[i]) == false 
            res -= sum(x[f.varToIdx["p_from"][l]] for l in f.areaInfo.networkInfo.BusToBrchDict[i])
        end 
        
        if abs(res) > PDMO.FeasTolerance 
            return Inf
        end 
    end 

    return 0.0
end 

function proximalOracle!(y::NumericVariable, f::IndicatorLocalDCConstraints, x::NumericVariable, gamma::Float64 = 1.0, enableParallel::Bool = false)
    if (typeof(x) != Vector{Float64} || length(x) != f.numberVariables)
        error("IndicatorLocalDCConstraints: argument dimension mismatch.")
    end

    JuMP.@objective(f.model, Min, sum((f.model[:x][i] - x[i])^2 for i in 1:f.numberVariables))
    JuMP.optimize!(f.model)
    status = JuMP.termination_status(f.model)
    if status != JuMP.MOI.OPTIMAL && status != JuMP.MOI.LOCALLY_SOLVED
        @warn "IndicatorLocalDCConstraints: proximal evaluation not successful; status = $(status)"
    end

    y .= JuMP.value.(f.model[:x])  
end 

function proximalOracle(f::IndicatorLocalDCConstraints, x::NumericVariable, gamma::Float64 = 1.0, enableParallel::Bool = false)
    y = similar(x)
    proximalOracle!(y, f, x, gamma, enableParallel) 
    return y
end 

function JuMPAddProximableFunction(f::IndicatorLocalDCConstraints, model::JuMP.Model, var::Vector{<:JuMP.VariableRef})
    addLocalDCConstraints(model, var, f.varToIdx, f.areaInfo)
    return nothing 
end 