struct IndicatorBinary <: AbstractFunction 
    function IndicatorBinary()
        new()
    end
end 

# Override traits for IndicatorBinary
isProximal(f::Type{<:IndicatorBinary}) = true 
isConvex(f::Type{<:IndicatorBinary}) = false
isSet(f::Type{<:IndicatorBinary}) = true
isSupportedByJuMP(f::Type{<:IndicatorBinary}) = true

# function value
function (f::IndicatorBinary)(x::NumericVariable, enableParallel::Bool=false)
    for k in eachindex(x)
        if abs(x[k] - 0.0) < FeasTolerance || abs(x[k] - 1.0) < FeasTolerance
            continue
        else
            return Inf
        end
    end
    return 0.0
end 


function proximalOracle!(y::NumericVariable, f::IndicatorBinary, x::NumericVariable, gamma::Float64 = 1.0, enableParallel::Bool=false)
    if isa(x, Number)
        error("IndicatorBinary: proximal oracle does not support in-place operations for scalar inputs.")
    end
    for k in eachindex(x)
        y[k] = x[k] < 0.5 ? 0.0 : 1.0
    end
end


function proximalOracle(f::IndicatorBinary, x::NumericVariable, gamma::Float64 = 1.0, enableParallel::Bool=false)
    if isa(x, Number)
        return x < 0.5 ? 0.0 : 1.0
    end
    
    y = similar(x)
    proximalOracle!(y, f, x, gamma, enableParallel)
    return y
end

# JuMP support
function JuMPAddProximableFunction(g::IndicatorBinary, model::JuMP.Model, var::Vector{<:JuMP.VariableRef})
    dim = length(var)
    for k in 1:dim
        JuMP.set_binary(var[k])
    end
    return nothing
end