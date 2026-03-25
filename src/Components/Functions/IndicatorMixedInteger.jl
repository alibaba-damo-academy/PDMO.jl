struct IndicatorMixedInteger <: AbstractFunction
    lb::Vector{Float64}
    ub::Vector{Float64}
    isInteger::Vector{Bool}
    function IndicatorMixedInteger(lb::Vector{Float64}, ub::Vector{Float64}, isInteger::Vector{Bool})
        if length(lb) != length(ub) || length(lb) != length(isInteger)
            error("IndicatorMixedInteger: lb, ub, and isInteger must have the same length.")
        end
        if any(lb .> ub)
            error("IndicatorMixedInteger: lb must be less than ub.")
        end

        lb_new = copy(lb) # make sure arguments are not changed
        ub_new = copy(ub)
        for k in eachindex(isInteger)
            if isInteger[k] 
                lb_new[k] = ceil(lb[k])
                ub_new[k] = floor(ub[k])
                if lb_new[k] > ub_new[k]
                    error("IndicatorMixedInteger: Empty domain for index $k")
                end
            end
        end 
        new(lb_new, ub_new, isInteger)
    end
end 

# Override traits for IndicatorMixedInteger
isProximal(f::Type{<:IndicatorMixedInteger}) = true 
isConvex(f::Type{<:IndicatorMixedInteger}) = false
isSet(f::Type{<:IndicatorMixedInteger}) = true
isSupportedByJuMP(f::Type{<:IndicatorMixedInteger}) = true

# function value
function (f::IndicatorMixedInteger)(x::NumericVariable, enableParallel::Bool=false)
    if isa(x, Number)
        if length(f.lb) != 1
            error("IndicatorMixedInteger: scalar input requires scalar (length-1) bounds")
        end
        if x < f.lb[1] - FeasTolerance || x > f.ub[1] + FeasTolerance
            return Inf
        end
        if f.isInteger[1] && abs(x - round(x)) > FeasTolerance
            return Inf
        end
        return 0.0
    end

    if length(x) != length(f.lb)
        error("IndicatorMixedInteger: input dimension must match bounds dimension")
    end

    for k in eachindex(x)
        lb = f.lb[k]
        ub = f.ub[k]

        if x[k] < lb - FeasTolerance || x[k] > ub + FeasTolerance
            return Inf
        end
      
        if f.isInteger[k]
            if abs(x[k] - round(x[k])) > FeasTolerance
                return Inf
            end
        end
    end
    return 0.0
end 


function proximalOracle!(y::NumericVariable, f::IndicatorMixedInteger, x::NumericVariable, gamma::Float64 = 1.0, enableParallel::Bool=false)
    if isa(x, Number)
        error("IndicatorMixedInteger: proximal oracle does not support in-place operations for scalar inputs.")
    end

    if length(x) != length(f.lb) || length(y) != length(f.lb)
        error("IndicatorMixedInteger: input/output dimensions must match bounds dimension")
    end

    y .= clamp.(x, f.lb, f.ub)

    for k in eachindex(x)
        if f.isInteger[k]
            y[k] = round(y[k])
        end
    end
end


function proximalOracle(f::IndicatorMixedInteger, x::NumericVariable, gamma::Float64 = 1.0, enableParallel::Bool=false)
    if isa(x, Number)
        if length(f.lb) != 1
            error("IndicatorMixedInteger: scalar input requires scalar (length-1) bounds")
        end
        y = clamp(x, f.lb[1], f.ub[1])
        return f.isInteger[1] ? round(y) : y
    end

    if length(x) != length(f.lb)
        error("IndicatorMixedInteger: input dimension must match bounds dimension")
    end

    y = similar(x)
    proximalOracle!(y, f, x, gamma, enableParallel)
    return y
end

# JuMP support
function JuMPAddProximableFunction(g::IndicatorMixedInteger, model::JuMP.Model, var::Vector{<:JuMP.VariableRef})
    @assert length(var) == length(g.lb) == length(g.ub) == length(g.isInteger) "IndicatorMixedInteger: variable dimension must match bounds dimension"
    for k in eachindex(var)
        if g.lb[k] > -Inf 
            JuMP.set_lower_bound(var[k], g.lb[k])
        end
        if g.ub[k] < Inf 
            JuMP.set_upper_bound(var[k], g.ub[k])
        end
        if g.isInteger[k]
            JuMP.set_integer(var[k])
        end
    end
    return nothing
end